#!/usr/bin/env python3
"""Recompute only the EDC column of Table 1 with the Dalsanto EDCLoss.

No training and no reselection: the ORIGINAL matched_reg_1s checkpoints are
re-run on the frozen shards at the frozen supports. The repo's broadband dB
metric is recomputed alongside and checked against reports/single_clap_benchmark/
per_room.csv, so a match there certifies the pipeline before the new column is
read. Measured and simulated supports are both the frozen ones.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / "src"))
from claprir.metrics.energy_decay import EDCLoss
from claprir.metrics.deconvolution import regularized_deconvolution
from claprir.training.train_rir_estimator import RunConfig, load_model
from claprir.metrics.lundeby_truncation import edc_truncated

SR = 44100
PROVIDERS = ("shoebox", "mit", "but", "ace", "openair")
SEEDS = (42001, 42002, 42003)
CROP3, CROP6 = 132, 265
LAM_TIK, LAM_C3 = 1e-5, 10 ** -3.5
LAM_C6 = 0.00031622776601683794
REG_ROOT, REG_NAME = "runs/matched_reg_1s", "hybrid_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{}"
ARMS = ("known_clap_unregularized", "known_clap_tikhonov", "crop_3ms_tikhonov",
        "crop_6ms_tikhonov", "regression_1s")
OUT = ROOT / "reports/energy_decay_metric_recompute"


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def old_edc_rmse_db(target: np.ndarray, estimate: np.ndarray) -> float:
    a = edc_truncated(np.asarray(estimate, np.float64), len(estimate))
    b = edc_truncated(np.asarray(target, np.float64), len(target))
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def load_provider(provider: str):
    with np.load(ROOT / f"data/multiroom_generalization/{provider}.npz",
                 allow_pickle=False) as d:
        splits = d["split"].astype(str)
        idx = np.flatnonzero(splits == "test")
        ids, rooms = d["record_id"].astype(str), d["room_id"].astype(str)
        return (ids[idx], rooms[idx], np.asarray(d["rir"][idx, :SR], np.float32),
                np.asarray(d["observation"][idx, 0, :SR], np.float32),
                np.asarray(d["clean"][idx, 0], np.float32))


def analytic(y: np.ndarray, clap: np.ndarray) -> dict[str, np.ndarray]:
    known = np.zeros(SR, np.float32); known[:882] = clap[:882]
    c3, c6 = np.zeros(SR, np.float32), np.zeros(SR, np.float32)
    c3[:CROP3], c6[:CROP6] = y[:CROP3], y[:CROP6]
    return {ARMS[0]: regularized_deconvolution(y, known, SR, 0.0),
            ARMS[1]: regularized_deconvolution(y, known, SR, LAM_TIK),
            ARMS[2]: regularized_deconvolution(y, c3, SR, LAM_C3),
            ARMS[3]: regularized_deconvolution(y, c6, SR, LAM_C6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report-root", type=Path, default=OUT)
    ap.add_argument("--band-selection", default="intended",
                    choices=("intended", "upstream"))
    args = ap.parse_args()
    out = args.report_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    primary = EDCLoss(SR, band_selection=args.band_selection)
    secondary = EDCLoss(SR, band_selection="upstream" if args.band_selection == "intended"
                        else "intended")

    data = {p: load_provider(p) for p in PROVIDERS}
    # One model load per seed; predict every provider before moving on.
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for seed in SEEDS:
        name = REG_NAME.format(seed)
        run_root = ROOT / REG_ROOT
        ckpt = run_root / name / "model_updates20000.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint {ckpt}; this harness will not train one.")
        cfg = json.loads((run_root / name / "config.resolved.json").read_text())
        cfg["training_datasets"] = tuple(cfg["training_datasets"])
        config = RunConfig(**cfg)
        assert config.run_name == name
        model = load_model(config, 20_000, device, run_root, run_root)
        print(f"loaded ORIGINAL checkpoint seed {seed}", flush=True)
        with torch.no_grad():
            for p in PROVIDERS:
                obs = torch.from_numpy(data[p][3])[:, None].to(device)
                chunks = []
                for s in range(0, len(obs), 8):
                    chunks.append(model.predict(obs[s:s + 8]).cpu().numpy()[:, 0])
                predictions[(p, seed)] = np.concatenate(chunks, 0)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for p in PROVIDERS:
        ids, rooms, targets, obs, cleans = data[p]
        for i in range(len(ids)):
            t = targets[i]
            nz = np.flatnonzero(t)
            bound = 4096 if p == "shoebox" else min(SR, int(nz[-1]) + 1)
            ref = t[:bound].astype(np.float64)
            base = dict(provider=p, room_id=rooms[i], sample_id=ids[i], bound=bound)
            est = analytic(obs[i], cleans[i])
            for arm in ARMS[:4]:
                e = est[arm][:bound]
                rows.append(dict(**base, arm=arm, seed="analytic",
                                 edc_rmse_db=old_edc_rmse_db(ref, e),
                                 edc_dalsanto=primary(e, ref),
                                 edc_dalsanto_alt=secondary(e, ref)))
            for seed in SEEDS:
                e = predictions[(p, seed)][i][:bound]
                rows.append(dict(**base, arm=ARMS[4], seed=seed,
                                 edc_rmse_db=old_edc_rmse_db(ref, e),
                                 edc_dalsanto=primary(e, ref),
                                 edc_dalsanto_alt=secondary(e, ref)))
        print(f"scored {p}: {len(ids)} records", flush=True)
    write_csv(out / "per_example.csv", rows)

    METRICS = ("edc_rmse_db", "edc_dalsanto", "edc_dalsanto_alt")
    # records -> (arm, provider, room, seed) medians -> seed medians -> rooms
    g = defaultdict(list)
    for r in rows:
        g[(r["arm"], r["provider"], r["room_id"], str(r["seed"]))].append(r)
    seed_rows = [dict(arm=a, provider=p, room_id=rm, seed=s,
                      **{m: float(np.median([x[m] for x in v])) for m in METRICS})
                 for (a, p, rm, s), v in sorted(g.items())]
    gr = defaultdict(list)
    for r in seed_rows:
        gr[(r["arm"], r["provider"], r["room_id"])].append(r)
    room_rows = [dict(arm=a, provider=p, room_id=rm,
                      **{m: float(np.median([x[m] for x in v])) for m in METRICS})
                 for (a, p, rm), v in sorted(gr.items())]
    write_csv(out / "per_room.csv", room_rows)

    summary = []
    for population in ("measured_pooled", "shoebox"):
        for arm in ARMS:
            sel = [r for r in room_rows if r["arm"] == arm and
                   (r["provider"] != "shoebox" if population == "measured_pooled"
                    else r["provider"] == "shoebox")]
            row = dict(population=population, arm=arm, n_rooms=len(sel))
            for m in METRICS:
                v = np.array([r[m] for r in sel])
                row[m] = float(np.mean(v)); row[f"{m}_median"] = float(np.median(v))
                row[f"{m}_std"] = float(np.std(v, ddof=1))
            summary.append(row)
    write_csv(out / "summary.csv", summary)

    # Certify against the frozen artifact before the new column is trusted.
    frozen = {(r["arm"], r["provider"], r["room_id"]): float(r["edc_rmse_db"])
              for r in csv.DictReader((ROOT / "reports/single_clap_benchmark/per_room.csv").open())}
    frozen.update({(r["arm"], r["provider"], r["room_id"]): float(r["edc_rmse_db"])
                   for r in csv.DictReader(
                       (ROOT / "reports/cropped_excitation_baseline/per_room.csv").open())})
    deltas = [abs(r["edc_rmse_db"] - frozen[(r["arm"], r["provider"], r["room_id"])])
              for r in room_rows if (r["arm"], r["provider"], r["room_id"]) in frozen]
    max_delta = max(deltas) if deltas else float("nan")
    (out / "verification.json").write_text(json.dumps(dict(
        compared_room_units=len(deltas), max_abs_old_edc_difference=max_delta,
        reproduces_frozen_old_edc=bool(max_delta < 1e-6),
        band_selection=args.band_selection,
        bands=len(primary.centers), center_frequencies=primary.centers,
        note="edc_dalsanto is a dimensionless normalised ratio, not decibels.",
        no_training_or_reselection=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")

    hdr = f"{'population':<16}{'arm':<28}{'EDC dB (old)':>14}{'EDCLoss (new)':>15}{'alt bands':>11}"
    lines = [hdr, "-" * len(hdr)]
    for r in summary:
        lines.append(f"{r['population']:<16}{r['arm']:<28}"
                     f"{r['edc_rmse_db']:>14.3f}{r['edc_dalsanto']:>15.4f}"
                     f"{r['edc_dalsanto_alt']:>11.4f}")
    lines += ["", f"old-metric reproduction vs frozen per_room.csv: "
                  f"max |Δ| = {max_delta:.3g} over {len(deltas)} room units"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nCOMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
