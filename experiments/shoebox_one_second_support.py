#!/usr/bin/env python3
"""Regenerate the four Shoebox evaluation rooms with a true one-second support.

Diagnostic only. Writes reports/shoebox_one_second_support; touches no frozen
shard, no checkpoint, no manuscript. The four analytic arms are recomputed under
four support definitions so the 4096-sample result and the one-second result can
be read side by side.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import scipy.signal as sps

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / "src"))
from claprir.datasets.shoebox_rirs import delay_trim
from claprir.metrics.room_acoustics import edt_seconds, stft_magnitude_errors
from claprir.metrics.deconvolution import regularized_deconvolution
from claprir.metrics.lundeby_truncation import edc_truncated, lundeby

SR = 44100
BANK_SEED = 23400
TEST_ROOMS = (28, 29, 30, 31)
CROP3, CROP6 = 132, 265
LAM_TIKHONOV = 1e-5
LAM_CROP3 = 10 ** -3.5
LAM_CROP6 = 0.00031622776601683794
ARMS = ("known_clap_unregularized", "known_clap_tikhonov",
        "crop_3ms_tikhonov", "crop_6ms_tikhonov")
METRICS = ("edc_rmse_db", "abs_edt_error_ms", "lsd_db", "nrmse")
OUT = ROOT / "reports/shoebox_one_second_support"


def room_parameters() -> dict[int, dict]:
    """Replay build_shoebox_random's RandomState draw order exactly."""
    rng = np.random.RandomState(BANK_SEED)
    rooms = {}
    for i in range(32):
        L, W, H = rng.uniform(3, 10), rng.uniform(3, 8), rng.uniform(2.4, 4)
        rt60 = rng.uniform(0.2, 0.7)
        e_abs, pra_max_order = pra.inverse_sabine(rt60, [L, W, H])
        src = [rng.uniform(.5, L - .5), rng.uniform(.5, W - .5), rng.uniform(1, H - .5)]
        mic = [rng.uniform(.5, L - .5), rng.uniform(.5, W - .5), rng.uniform(1, H - .5)]
        rooms[i] = dict(index=i, dims=[L, W, H], rt60=rt60, e_absorption=float(e_abs),
                        pra_max_order=int(pra_max_order), source=src, receiver=mic)
    return rooms


def simulate(room: dict, max_order: int) -> np.ndarray:
    """One ShoeBox ISM run. Only max_order differs from the frozen generator."""
    r = pra.ShoeBox(room["dims"], fs=SR, materials=pra.Material(room["e_absorption"]),
                    max_order=int(max_order))
    r.add_source(room["source"])
    r.add_microphone(np.array(room["receiver"]).reshape(3, 1))
    r.compute_rir()
    h = r.rir[0][0].astype(np.float32)
    return h / (np.max(np.abs(h)) + 1e-8)


def order_for_one_second(room: dict) -> tuple[int, np.ndarray]:
    """Smallest ISM order whose onset-trimmed RIR reaches the full second."""
    for order in range(8, 241):
        h = delay_trim(simulate(room, order))
        if len(h) >= SR:
            return order, h
    raise RuntimeError(f"room {room['index']}: no order reaches 1 s")


def frame(trimmed: np.ndarray) -> np.ndarray:
    """The frozen normalize_rir contract, at a 44100-sample horizon."""
    out = np.zeros(SR, np.float32)
    out[:min(len(trimmed), SR)] = trimmed[:SR]
    return out / (np.max(np.abs(out)) + 1e-8)


def observe(clap882: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The frozen synthesize() contract: noiseless convolution, peak normalized."""
    y = sps.fftconvolve(clap882, h)[:SR].astype(np.float32)
    return y / (np.max(np.abs(y)) + 1e-8)


def score(target: np.ndarray, estimate: np.ndarray) -> dict:
    t, e = np.asarray(target, np.float64), np.asarray(estimate, np.float64)
    t32, e32 = t.astype(np.float32), e.astype(np.float32)
    ref_edc, est_edc = edc_truncated(t, len(t)), edc_truncated(e, len(e))
    logmag = stft_magnitude_errors(t32, e32)["stft_logmag_mse"]
    return dict(edc_rmse_db=float(np.sqrt(np.mean((est_edc - ref_edc) ** 2))),
                abs_edt_error_ms=abs(float(edt_seconds(e32, SR) - edt_seconds(t32, SR))) * 1000.0,
                lsd_db=20.0 * math.sqrt(logmag),
                nrmse=float(np.linalg.norm(e - t) / (np.linalg.norm(t) + 1e-12)),
                reference_edc_floor_db=float(ref_edc[-1]),
                estimate_edc_floor_db=float(est_edc[-1]))


def estimates(y: np.ndarray, clap882: np.ndarray) -> dict[str, np.ndarray]:
    known = np.zeros(SR, np.float32)
    known[:len(clap882)] = clap882
    crop3, crop6 = np.zeros(SR, np.float32), np.zeros(SR, np.float32)
    crop3[:CROP3], crop6[:CROP6] = y[:CROP3], y[:CROP6]
    return {ARMS[0]: regularized_deconvolution(y, known, SR, 0.0),
            ARMS[1]: regularized_deconvolution(y, known, SR, LAM_TIKHONOV),
            ARMS[2]: regularized_deconvolution(y, crop3, SR, LAM_CROP3),
            ARMS[3]: regularized_deconvolution(y, crop6, SR, LAM_CROP6)}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-root", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.report_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    frozen = np.load(ROOT / "data/multiroom_generalization/shoebox.npz", allow_pickle=False)
    splits = frozen["split"].astype(str)
    record_ids = frozen["record_id"].astype(str)
    rooms = room_parameters()

    geometry, per_example, reproduction = [], [], []
    for k in TEST_ROOMS:
        room = rooms[k]
        rid = f"shoebox_seed23400_{k:02d}"
        j = int(np.flatnonzero((splits == "test") & (record_ids == rid))[0])
        clap = np.asarray(frozen["clean"][j, 0][:882], np.float32)
        frozen_h = np.asarray(frozen["rir"][j], np.float32)
        frozen_y = np.asarray(frozen["observation"][j, 0], np.float32)

        # --- the frozen order-8 room, regenerated, and checked bit for bit ------
        h8_trimmed = delay_trim(simulate(room, min(room["pra_max_order"], 8)))
        old_4096 = np.zeros(4096, np.float32)
        old_4096[:min(len(h8_trimmed), 4096)] = h8_trimmed[:4096]
        old_4096 /= np.max(np.abs(old_4096)) + 1e-8
        rir_match = float(np.max(np.abs(old_4096 - frozen_h[:4096])))
        h8 = frame(h8_trimmed)
        y8 = observe(clap, h8)
        obs_match = float(np.max(np.abs(observe(clap, frame(h8_trimmed[:4096])) - frozen_y)))

        # --- the same room resimulated until the ISM genuinely spans one second -
        order_1s, h1s_trimmed = order_for_one_second(room)
        h1s = frame(h1s_trimmed)
        y1s = observe(clap, h1s)

        tail = h1s[4096:SR]
        cross, noise_db, slope = lundeby(h1s, SR)
        native8 = min(len(h8_trimmed), SR)
        e = np.asarray(h1s, np.float64) ** 2
        edc_db = 10 * np.log10(np.maximum(np.cumsum(e[::-1])[::-1] / (e.sum() + 1e-30), 1e-30))
        decay60 = int(np.argmax(edc_db <= -60.0)) if np.any(edc_db <= -60.0) else SR

        geometry.append(dict(room_index=k, record_id=rid,
            length_m=room["dims"][0], width_m=room["dims"][1], height_m=room["dims"][2],
            rt60_target_s=room["rt60"], e_absorption=room["e_absorption"],
            source_xyz=";".join(f"{v:.6f}" for v in room["source"]),
            receiver_xyz=";".join(f"{v:.6f}" for v in room["receiver"]),
            pra_inverse_sabine_max_order=room["pra_max_order"],
            frozen_max_order=min(room["pra_max_order"], 8),
            frozen_native_samples=len(h8_trimmed),
            frozen_native_ms=1000.0 * len(h8_trimmed) / SR,
            one_second_max_order=order_1s,
            one_second_native_samples=len(h1s_trimmed),
            regenerated_4096_max_abs_diff_vs_frozen=rir_match,
            regenerated_observation_max_abs_diff_vs_frozen=obs_match,
            tail_4096_44100_nonzero=int(np.count_nonzero(tail)),
            tail_4096_44100_total=int(len(tail)),
            tail_is_all_simulated=bool(np.all(tail != 0)),
            tail_rms_dbfs=float(20 * np.log10(np.sqrt(np.mean(tail.astype(np.float64) ** 2)) + 1e-300)),
            edc_db_at_4096=float(edc_db[4096]),
            lundeby_crossing_sample=int(cross), lundeby_noise_db=noise_db,
            decay_minus60db_sample=decay60))

        supports = (("old_4096", h8, y8, 4096),
                    ("native_ism_order8", h8, y8, native8),
                    ("one_second_full", h1s, y1s, SR),
                    ("one_second_to_minus60db", h1s, y1s, decay60))
        for name, h, y, bound in supports:
            arm_estimates = estimates(y, clap)
            for arm in ARMS:
                per_example.append(dict(support=name, room_index=k, record_id=rid,
                    bound=bound, bound_ms=1000.0 * bound / SR, arm=arm,
                    **score(h[:bound], arm_estimates[arm][:bound])))
        reproduction.append(dict(room_index=k, rir_4096_max_abs_diff=rir_match,
                                 observation_max_abs_diff=obs_match))
        print(f"room {k}: order8 native {len(h8_trimmed)} samples, "
              f"1 s at order {order_1s} ({len(h1s_trimmed)} samples), "
              f"regen diff {rir_match:.3g}/{obs_match:.3g}", flush=True)

    write_csv(out / "room_geometry.csv", geometry)
    write_csv(out / "per_example.csv", per_example)

    summary = []
    for name in ("old_4096", "native_ism_order8", "one_second_full", "one_second_to_minus60db"):
        for arm in ARMS:
            group = [r for r in per_example if r["support"] == name and r["arm"] == arm]
            row = dict(support=name, arm=arm, n_rooms=len(group),
                       mean_bound_ms=float(np.mean([r["bound_ms"] for r in group])))
            for m in METRICS:
                vals = np.array([r[m] for r in group if np.isfinite(r[m])])
                row[m] = float(np.mean(vals)) if len(vals) else float("nan")
                row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
                row[f"n_finite_{m}"] = int(len(vals))
            summary.append(row)
    write_csv(out / "summary.csv", summary)

    max_rir = max(r["rir_4096_max_abs_diff"] for r in reproduction)
    max_obs = max(r["observation_max_abs_diff"] for r in reproduction)
    (out / "verification.json").write_text(json.dumps(dict(
        generator_reproduced_bit_exact=bool(max_rir == 0.0),
        max_abs_difference_regenerated_vs_frozen_rir=max_rir,
        max_abs_difference_regenerated_vs_frozen_observation=max_obs,
        all_one_second_tails_fully_simulated=all(g["tail_is_all_simulated"] for g in geometry),
        tail_rms_dbfs_range=[min(g["tail_rms_dbfs"] for g in geometry),
                             max(g["tail_rms_dbfs"] for g in geometry)],
        rooms=len(geometry), rows=len(per_example),
        completed_at_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")

    header = f"{'support':<26}{'arm':<28}{'ms':>8}{'EDC':>9}{'|dEDT|ms':>10}{'LSD':>8}{'NRMSE':>8}"
    lines = [header, "-" * len(header)]
    for r in summary:
        lines.append(f"{r['support']:<26}{r['arm']:<28}{r['mean_bound_ms']:>8.1f}"
                     f"{r['edc_rmse_db']:>9.3f}{r['abs_edt_error_ms']:>10.2f}"
                     f"{r['lsd_db']:>8.3f}{r['nrmse']:>8.4f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nCOMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
