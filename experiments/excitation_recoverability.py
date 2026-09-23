#!/usr/bin/env python3
"""E1 and E2: the recoverability ceiling and the naive baseline, rescored.

E1  true clap -> Tikhonov inversion       is the RIR recoverable at all?
E2  3 ms crop -> the identical inversion  can the first few ms stand in for the clap?

RESCORE_ONLY in publication/AUDIT.md. No model, no training, no sampler -- these
are analytic inversions of stored data. What changes from the old publication
tree is the scoring: the frozen support-aware protocol, and the metric set the
paper now reports, instead of NRMSE alone.

Both use the VALIDATION-SELECTED regularisation from
reports/deconvolution_audit, never the 1e-2 default. That default is three orders
of magnitude above the validation optimum for true x and degrades the ceiling by
12x; computing this table at it once produced two published conclusions that had
to be retracted. The lambdas are read from disk rather than restated here so they
cannot drift from the audit that selected them.

Horizon is 1 s, matching E3, so the recoverability ceiling and the estimators are
scored over the same support. Bound per room is min(1 s, T_valid): past a
record's last real sample the target is padding, not decay.

CPU only.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from claprir.metrics.lundeby_truncation import edc_truncated                                  # noqa: E402
from claprir.metrics.room_acoustics import diagnosis_metrics
from claprir.metrics.deconvolution import regularized_deconvolution

SR = 44_100
PROVIDERS = ("shoebox", "mit", "but", "ace", "openair")
SYNTHETIC = {"shoebox"}
CROP_MS = 3.0


def crop_excitation(observation: np.ndarray, length: int) -> np.ndarray:
    """E2's excitation: the first 3 ms of the recording, zero-padded."""
    clean = np.zeros(length, np.float32)
    crop = min(round(CROP_MS / 1000 * SR), len(observation))
    clean[:crop] = observation[:crop]
    return clean


def select_lambda(data_root: Path, length: int, grid, datasets) -> dict:
    """Re-select on validation, at THIS horizon, by median NRMSE.

    Same criterion, same grid and same split as the original audit; only the
    horizon moves. A lambda is a property of the inverse problem at a given
    length, not a constant of the excitation.
    """
    obs, exc_true, exc_crop, tgt = [], [], [], []
    for prov in datasets:
        d = np.load(data_root / f"{prov}.npz", mmap_mode="r", allow_pickle=False)
        sp = d["split"].astype(str)
        idx = np.flatnonzero((sp == "valid") | (sp == "validation"))
        for i in idx:
            o = np.asarray(d["observation"][i, 0, :length], np.float32)
            c = np.asarray(d["clean"][i, 0], np.float32)
            pad = np.zeros(length, np.float32); pad[:min(len(c), length)] = c[:length]
            obs.append(o); exc_true.append(pad); exc_crop.append(crop_excitation(o, length))
            tgt.append(np.asarray(d["rir"][i, :length], np.float64))
    if not obs:
        raise SystemExit("no validation examples; cannot select lambda")
    print(f"  selecting on {len(obs)} validation examples from {list(datasets)}")
    out = {}
    for key, excs in (("true_x", exc_true), ("trunc_3ms", exc_crop)):
        best, best_v = None, np.inf
        for g in grid:
            v = np.median([np.linalg.norm(
                regularized_deconvolution(o, x, length, g)[:length] - t)
                / (np.linalg.norm(t) + 1e-12)
                for o, x, t in zip(obs, excs, tgt)])
            if v < best_v:
                best, best_v = g, v
        out[key] = best
    return out


def score(t, e, bound):
    t, e = np.asarray(t, np.float64), np.asarray(e, np.float64)
    a, b = edc_truncated(e[:bound], bound), edc_truncated(t[:bound], bound)
    n = min(len(a), len(b))
    out = {"edc_rmse_db": float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2))) if n else np.nan,
           "nrmse": float(np.linalg.norm(e[:bound] - t[:bound])
                          / (np.linalg.norm(t[:bound]) + 1e-12))}
    out.update(diagnosis_metrics(t[:bound].astype(np.float32),
                                 e[:bound].astype(np.float32), SR))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    ap.add_argument("--data-root", type=Path, default=Path("data/multiroom_generalization"))
    ap.add_argument("--report-root", type=Path, default=Path("reports/excitation_recoverability"))
    ap.add_argument("--horizon-ms", type=int, default=1000)
    a = ap.parse_args()
    length = round(SR * a.horizon_ms / 1000)
    prior = json.loads(
        (a.repo_root / "reports/deconvolution_audit/results/selected_lambda.json").read_text())
    print(f"the audit selected its lambdas at {prior['horizon_ms']} ms: "
          f"true_x={prior['selected']['true_x']:.3g} "
          f"trunc_3ms={prior['selected']['trunc_3ms']:.3g}")
    if prior["horizon_ms"] == a.horizon_ms:
        lam = prior["selected"]
        print("  horizon matches; reusing them")
    else:
        print(f"  horizon differs ({a.horizon_ms} ms), so they are RE-SELECTED here on the")
        print("  same grid and the same validation split. Carrying a lambda across a")
        print("  horizon is the shortcut that produced two retractions in this repo.")
        lam = select_lambda(a.data_root, length, prior["grid"],
                            tuple(prior["validation_datasets"]))
        for k, v in lam.items():
            print(f"    {k:<22}{v:.3g}   (was {prior['selected'][k]:.3g} at "
                  f"{prior['horizon_ms']} ms)")
    rows = []
    for prov in PROVIDERS:
        d = np.load(a.data_root / f"{prov}.npz", mmap_mode="r", allow_pickle=False)
        idx = np.flatnonzero(d["split"].astype(str) == "test")
        for i in idx:
            t = np.asarray(d["rir"][i, :length], np.float64)
            obs = np.asarray(d["observation"][i, 0, :length], np.float32)
            clean = np.asarray(d["clean"][i, 0], np.float32)
            padded = np.zeros(length, np.float32)
            padded[:min(len(clean), length)] = clean[:length]
            nz = np.nonzero(t)[0]
            tv = length if prov in SYNTHETIC else ((int(nz[-1]) + 1) if len(nz) else 1)
            bound = max(min(length, tv), 1)
            base = {"provider": prov, "room_id": str(d["room_id"][i]),
                    "sample_id": str(d["record_id"][i]), "bound": bound}
            for arm, exc, key in (("E1_true_clap", padded, "true_x"),
                                  ("E2_crop_3ms", crop_excitation(obs, length), "trunc_3ms")):
                est = regularized_deconvolution(obs, exc, length, lam[key])
                rows.append({**base, "arm": arm, "lambda": lam[key],
                             **score(t, est[:length], bound)})
        print(f"  {prov}: {len(rows)} rows", flush=True)
    (a.report_root / "results").mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = a.report_root / "results/per_example.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
