#!/usr/bin/env python3
"""Objective consistency metric for the real-world transfer section (4.5).

Section 4.5 argues from plots that the inferred RIRs vary less across handclap
modes than the recordings do. This measures that instead of asserting it, and
gives it a baseline: a six-millisecond cropped-excitation inversion of the same
recording, which needs no trained model.

There is no paired reference RIR for phone recordings, so nothing here measures
reconstruction accuracy. What it measures is agreement: if an estimator is
insensitive to which handclap produced the recording, its estimates for
different modes in one room must agree as closely as its estimates for repeats
of a single mode. That comparison is internal to each room and needs no truth.

Decay is compared through the normalised energy decay curve over 0-500 ms, the
window Fig. 3 uses, as an RMS difference in dB.

No training and no new inference: regressor estimates come from the frozen
phone_deployment_evaluation prediction cache.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from claprir.metrics.deconvolution import regularized_deconvolution
from claprir.metrics.lundeby_truncation import edc_truncated

SR = 44100
HORIZON = 44100
SUPPORT_MS = 500
SUPPORT = round(SUPPORT_MS / 1000 * SR)
CROP3, CROP6 = 132, 265
LAM_CROP3 = 10 ** -3.5
LAM_CROP6 = 0.00031622776601683794
SEEDS = (42001, 42002, 42003)
CORPUS = ROOT / "reports/phone_clap_demo"
CACHE = ROOT / "runs/e6_phone_raw_cache"
OUT = ROOT / "reports/phone_transfer_consistency"


def load_order() -> list[dict]:
    """The exact record order phone_deployment_evaluation batched, recovered from its output."""
    rows = [r for r in csv.DictReader((ROOT / "reports/phone_deployment_evaluation/results/per_clap.csv").open())
            if int(r["seed"]) == SEEDS[0]]
    meta = {r["segment_path"]: r for r in csv.DictReader((CORPUS / "results/metadata.csv").open())}
    out = []
    for r in rows:
        m = meta[r["sample_id"]]
        out.append(dict(sample_id=r["sample_id"], room_id=r["room_id"], room_name=r["room_name"],
                        clap_mode=r["clap_mode"], repeat_id=r["repeat_id"],
                        pre_pad_s=float(m["pre_pad_s"])))
    return out


def observation(record: dict) -> np.ndarray:
    """Exactly phone_deployment_evaluation's preprocessing: resample, drop pre-pad, no gain."""
    rate, x = wavfile.read(CORPUS / "results/segments" / record["sample_id"])
    assert rate == 48000 and x.ndim == 1 and np.issubdtype(x.dtype, np.floating)
    x = resample_poly(x.astype(np.float64), 147, 160)
    x = x[round(record["pre_pad_s"] * SR):]
    y = np.zeros(HORIZON, np.float32)
    y[:min(len(x), HORIZON)] = x[:HORIZON].astype(np.float32)
    return y


def crop_estimate(y: np.ndarray, crop: int, lam: float) -> np.ndarray:
    x = np.zeros(HORIZON, np.float32)
    x[:crop] = y[:crop]
    return regularized_deconvolution(y, x, HORIZON, lam)


def decay_curve(signal: np.ndarray) -> np.ndarray:
    """Normalised EDC over the display support, in dB."""
    return edc_truncated(np.asarray(signal, np.float64)[:SUPPORT], SUPPORT)


def curve_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def load_regressor(order: list[dict]) -> dict[int, np.ndarray]:
    """Frozen predictions, reassembled in the order they were batched."""
    out = {}
    for seed in SEEDS:
        chunks = []
        for start in range(0, len(order), 8):
            path = CACHE / f"raw_v1_seed{seed}_batch{start:03d}.npy"
            if not path.exists():
                raise SystemExit(f"missing prediction cache {path}; this harness will not infer.")
            chunks.append(np.load(path))
        stacked = np.concatenate(chunks, 0)
        assert len(stacked) == len(order), (len(stacked), len(order))
        out[seed] = stacked
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-root", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.report_root.resolve(); out.mkdir(parents=True, exist_ok=True)

    order = load_order()
    print(f"{len(order)} phone recordings", flush=True)
    regressor = load_regressor(order)

    curves = defaultdict(dict)     # arm -> index -> curve
    for i, record in enumerate(order):
        y = observation(record)
        curves["recording"][i] = decay_curve(y)
        curves["crop_3ms"][i] = decay_curve(crop_estimate(y, CROP3, LAM_CROP3))
        curves["crop_6ms"][i] = decay_curve(crop_estimate(y, CROP6, LAM_CROP6))
        curves["regressor"][i] = decay_curve(np.median(
            np.stack([regressor[s][i] for s in SEEDS]), axis=0))
        if (i + 1) % 40 == 0:
            print(f"  processed {i+1}/{len(order)}", flush=True)

    by_room = defaultdict(list)
    for i, r in enumerate(order):
        by_room[(r["room_id"], r["room_name"])].append(i)

    pair_rows, room_rows = [], []
    for (room_id, room_name), idx in sorted(by_room.items(), key=lambda kv: int(kv[0][0])):
        mode = {i: order[i]["clap_mode"] for i in idx}
        for arm in ("recording", "crop_6ms", "crop_3ms", "regressor"):
            within, across = [], []
            for a, b in combinations(idx, 2):
                d = curve_distance(curves[arm][a], curves[arm][b])
                (within if mode[a] == mode[b] else across).append(d)
                pair_rows.append(dict(room_id=room_id, room_name=room_name, arm=arm,
                                      sample_a=order[a]["sample_id"], sample_b=order[b]["sample_id"],
                                      same_mode=mode[a] == mode[b], distance_db=d))
            w, c = float(np.mean(within)), float(np.mean(across))
            room_rows.append(dict(room_id=room_id, room_name=room_name, arm=arm,
                                  n_recordings=len(idx), n_within_mode_pairs=len(within),
                                  n_cross_mode_pairs=len(across),
                                  within_mode_db=w, cross_mode_db=c,
                                  mode_sensitivity_db=c - w))
        print(f"  room {room_id} {room_name}: {len(idx)} recordings", flush=True)

    def write(path, rows):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", newline="") as f:
            wtr = csv.DictWriter(f, keys, lineterminator="\n")
            wtr.writeheader(); wtr.writerows(rows)
    write(out / "pairwise.csv", pair_rows)
    write(out / "per_room.csv", room_rows)

    summary = []
    for arm in ("recording", "crop_6ms", "crop_3ms", "regressor"):
        sel = [r for r in room_rows if r["arm"] == arm]
        summary.append(dict(arm=arm, n_rooms=len(sel),
            within_mode_db=float(np.mean([r["within_mode_db"] for r in sel])),
            within_mode_db_std=float(np.std([r["within_mode_db"] for r in sel], ddof=1)),
            cross_mode_db=float(np.mean([r["cross_mode_db"] for r in sel])),
            cross_mode_db_std=float(np.std([r["cross_mode_db"] for r in sel], ddof=1)),
            mode_sensitivity_db=float(np.mean([r["mode_sensitivity_db"] for r in sel])),
            mode_sensitivity_db_std=float(np.std([r["mode_sensitivity_db"] for r in sel], ddof=1))))
    write(out / "summary.csv", summary)

    hdr = (f"{'estimator':<12}{'within-mode':>13}{'cross-mode':>13}"
           f"{'mode sensitivity':>19}")
    lines = [hdr, "-" * len(hdr)]
    for r in summary:
        lines.append(f"{r['arm']:<12}{r['within_mode_db']:>8.2f} dB   "
                     f"{r['cross_mode_db']:>8.2f} dB   "
                     f"{r['mode_sensitivity_db']:>10.2f} dB")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")

    (out / "result.json").write_text(json.dumps(dict(
        id="PHONE-TRANSFER-CONSISTENCY",
        x_decision="Whether section 4.5 can state real-world mode-invariance as a measured "
                   "result with a baseline, instead of an appearance-based claim about plots.",
        question="Do estimates for different handclap modes in one room agree as closely as "
                 "estimates for repeats of a single mode?",
        metric="RMS difference in dB between normalised energy decay curves over 0-500 ms; "
               "mode_sensitivity = cross-mode mean minus within-mode mean. Zero means the "
               "estimator does not care which handclap it heard.",
        baseline="crop_6ms, a six-millisecond cropped-excitation inversion needing no model",
        no_reference_rir="Phone recordings have no paired reference RIR. This measures "
                         "agreement and mode-invariance, NOT reconstruction accuracy.",
        recordings=len(order), rooms=len(by_room), seeds=list(SEEDS),
        regressor_prediction_source="reports/phone_deployment_evaluation cache; no new inference",
        summary=summary, per_room=room_rows,
        completed_at_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nCOMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
