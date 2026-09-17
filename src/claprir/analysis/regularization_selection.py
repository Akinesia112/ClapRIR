#!/usr/bin/env python3
"""How good is the inverse operator itself?

The 2026-08-06 meeting recorded that "even deconvolving with the true x is not
[good]", and immediately after that, that lambda "can have quite an impact".
Those two statements have never been separated: every regularized deconvolution
in this repository runs at a hard-coded `relative_regularization=1e-2`, chosen
once and never selected on data.

So before any excitation estimator is blamed for a poor RIR, this audit measures
the ceiling of the inversion itself, and decomposes

    total error = excitation estimation error + inverse-conditioning error

by running the same operator with three excitations: the true clap, the 3 ms
truncation of the observation, and the frozen complex-spectrum regression
estimate. Lambda is selected **on the validation split only** and then applied
unchanged to the test split.

The operator is deterministic, so there is no seed to vary; the only sampling
choice is which records enter the split, and those splits are frozen upstream.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from claprir.metrics.deconvolution import edc_rmse, regularized_deconvolution, rir_metrics

SAMPLE_RATE = 44_100
DEFAULT_DATA_ROOT = Path("data/multiroom_generalization")
REPORT_ROOT = Path("reports/deconvolution_audit")
#: Frozen complex-spectrum regression front end (best excitation estimator on
#: record).  Missing checkpoint is a hard error, never a retrain.
ESTIMATOR_CHECKPOINT = Path(
    "runs/supervised_comparison/complex_spectrum_regression/model_updates1500.pt")
LAMBDA_GRID = tuple(float(x) for x in np.logspace(-5, 0, 11))
#: Mixing time of the hybrid architecture, reused so the early/late split is
#: comparable across the two routes being contrasted.
MIXING_TIME_SAMPLES = 3072


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def truncation_excitation(observation: np.ndarray, length: int,
                          milliseconds: float = 3.) -> np.ndarray:
    """Cut away everything after the first few ms -- the baseline of the meeting."""
    excitation = np.zeros(length, np.float32)
    crop = min(round(milliseconds * SAMPLE_RATE / 1000), len(observation))
    excitation[:crop] = observation[:crop]
    return excitation


def load_estimator(device: torch.device):
    """The frozen y -> x complex-spectrum regressor, on its own 4096-sample contract."""
    if not ESTIMATOR_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"missing excitation estimator checkpoint {ESTIMATOR_CHECKPOINT}. "
            "Validation rules forbid retraining to fill a missing checkpoint; "
            "recover it or drop the estimated-x arm.")
    from claprir.models import excitation_estimator as estimators
    payload = torch.load(ESTIMATOR_CHECKPOINT, map_location=device, weights_only=False)
    config = estimators.ExperimentConfig(**payload["config"])
    model = estimators.Experiment(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config


@torch.no_grad()
def estimated_excitation(model, config, observation: np.ndarray,
                         length: int, device: torch.device) -> np.ndarray:
    """Apply the 4096-sample estimator and zero-pad to the RIR horizon."""
    support = config.signal_length
    window = np.zeros(support, np.float32)
    window[:min(support, len(observation))] = observation[:support]
    tensor = torch.from_numpy(window)[None, None].to(device)
    estimate = model.predict(tensor).cpu().numpy()[0, 0]
    excitation = np.zeros(length, np.float32)
    excitation[:min(length, len(estimate))] = estimate[:length]
    return excitation


def early_late_metrics(reference: np.ndarray, estimate: np.ndarray,
                       edge: int = MIXING_TIME_SAMPLES) -> dict[str, float]:
    def nrmse(a, b):
        return float(np.sqrt(np.mean((b - a) ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-8))
    return {"early_nrmse": nrmse(reference[:edge], estimate[:edge]),
            "late_nrmse": nrmse(reference[edge:], estimate[edge:]),
            "early_edc_rmse_db": edc_rmse(reference[:edge], estimate[:edge])}


def excitations_for(record: dict, length: int, estimator, device: torch.device
                    ) -> dict[str, np.ndarray]:
    padded_true = np.zeros(length, np.float32)
    clean = record["clean"]
    padded_true[:min(length, len(clean))] = clean[:length]
    variants = {"true_x": padded_true,
                "trunc_3ms": truncation_excitation(record["observation"], length)}
    if estimator is not None:
        model, config = estimator
        variants["complex_regression_x"] = estimated_excitation(
            model, config, record["observation"], length, device)
    return variants


def load_records(datasets: tuple[str, ...], split: str, length: int,
                 data_root: Path, max_examples: int) -> list[dict]:
    records = []
    for dataset in datasets:
        data = np.load(data_root / f"{dataset}.npz", mmap_mode="r", allow_pickle=False)
        indices = np.flatnonzero(data["split"].astype(str) == split)[:max_examples]
        for index in indices:
            records.append({
                "dataset": dataset, "room_id": str(data["room_id"][index]),
                "sample_id": str(data["record_id"][index]),
                "rir": np.asarray(data["rir"][index, :length], np.float32),
                "clean": np.asarray(data["clean"][index, 0], np.float32),
                "observation": np.asarray(data["observation"][index, 0, :length], np.float32),
            })
    if not records:
        raise RuntimeError(f"no {split} records in {datasets}")
    return records


def sweep(records: list[dict], length: int, estimator, device: torch.device,
          split: str) -> list[dict]:
    rows = []
    for record in records:
        variants = excitations_for(record, length, estimator, device)
        for name, excitation in variants.items():
            for lam in LAMBDA_GRID:
                estimate = regularized_deconvolution(record["observation"], excitation,
                                                     length, lam)
                rows.append({"excitation": name, "relative_lambda": lam, "split": split,
                             "dataset": record["dataset"], "room_id": record["room_id"],
                             "sample_id": record["sample_id"],
                             **rir_metrics(record["rir"], estimate, record["clean"]),
                             **early_late_metrics(record["rir"], estimate)})
    return rows


def select_lambda(rows: list[dict], criterion: str = "nrmse") -> dict[str, float]:
    """Pick lambda per excitation on the validation split, by median criterion."""
    chosen = {}
    for excitation in sorted({r["excitation"] for r in rows}):
        scores = []
        for lam in LAMBDA_GRID:
            values = [r[criterion] for r in rows
                      if r["excitation"] == excitation and r["relative_lambda"] == lam]
            scores.append((float(np.median(values)), lam))
        chosen[excitation] = min(scores)[1]
    return chosen


def summarize(rows: list[dict], chosen: dict[str, float]) -> list[dict]:
    keys = ("nrmse", "early_nrmse", "late_nrmse", "env_corr", "lsd_db",
            "edc_rmse_db", "roundtrip_nrmse")
    out = []
    for excitation, lam in sorted(chosen.items()):
        for dataset in sorted({r["dataset"] for r in rows}):
            subset = [r for r in rows if r["excitation"] == excitation
                      and r["relative_lambda"] == lam and r["dataset"] == dataset]
            if not subset:
                continue
            row = {"excitation": excitation, "relative_lambda": lam,
                   "dataset": dataset, "n": len(subset)}
            for key in keys:
                values = [r[key] for r in subset if np.isfinite(r[key])]
                if values:
                    row[f"{key}_median"] = float(np.median(values))
            out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["shoebox", "mit", "but", "ace", "openair"])
    parser.add_argument("--horizon-ms", type=int, default=250)
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-estimated-x", action="store_true",
                        help="run only the true-x and truncation arms")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    length = round(SAMPLE_RATE * args.horizon_ms / 1000)
    estimator = None if args.skip_estimated_x else load_estimator(device)

    # OpenAIR has no validation split (it is a held-out provider), so lambda is
    # selected on the providers that do and then applied to everything.
    validation_datasets = tuple(d for d in args.datasets if d != "openair")
    validation = load_records(validation_datasets, "valid", length, args.data_root,
                              args.max_examples)
    validation_rows = sweep(validation, length, estimator, device, "valid")
    chosen = select_lambda(validation_rows)

    test = load_records(tuple(args.datasets), "test", length, args.data_root,
                        args.max_examples)
    test_rows = sweep(test, length, estimator, device, "test")

    write_csv(REPORT_ROOT / "results/lambda_sweep_valid.csv", validation_rows)
    write_csv(REPORT_ROOT / "results/lambda_sweep_test.csv", test_rows)
    summary = summarize(test_rows, chosen)
    write_csv(REPORT_ROOT / "results/test_at_selected_lambda.csv", summary)
    (REPORT_ROOT / "results/selected_lambda.json").write_text(
        json.dumps({"criterion": "median nrmse on valid", "grid": list(LAMBDA_GRID),
                    "selected": chosen, "horizon_ms": args.horizon_ms,
                    "validation_datasets": list(validation_datasets)}, indent=2) + "\n")
    print(json.dumps({"selected_lambda": chosen, "test_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
