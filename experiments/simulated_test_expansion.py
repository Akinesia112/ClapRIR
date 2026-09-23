#!/usr/bin/env python3
"""Enlarge the simulated test set from 4 held-out RIRs to 216, and rescore.

Table 1's Simulated block rests on four rooms with one RIR each, so its means and
standard deviations are over n=4. This generates 216 further held-out rooms from
the same procedure and a different bank seed, and re-runs every arm on them.

No retraining. The regressor saw rooms 0-23 of build_shoebox_bank(32, seed=23400);
these rooms come from seed 770216 and were never seen in training or validation,
so they are held out in the same sense the original four are. Generation matches
the frozen bank exactly -- in particular max_order = min(pra_max_order, 8) and the
4096-sample crop -- because the regressor was trained on targets of that form and
any other choice would score it out of distribution.

Protocol fixed in reports/simulated_test_expansion/preregistration.json before
generation: 216 rooms, seed 770216, every generated room kept.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import scipy.signal as sps
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from claprir.datasets.shoebox_rirs import delay_trim
from claprir.metrics.room_acoustics import edt_seconds, stft_magnitude_errors
from claprir.metrics.deconvolution import regularized_deconvolution
from claprir.metrics.energy_decay import _backward_int, _discard_last_n_percent
from claprir.datasets.controlled_shards import load_claps
from claprir.training.train_rir_estimator import RunConfig, load_model

SR = 44100
SUPPORT = 4096            # the frozen simulated scoring support
CLAP_SUPPORT = 882
CROP3, CROP6 = 132, 265
LAM_TIK, LAM_C3 = 1e-5, 10 ** -3.5
LAM_C6 = 0.00031622776601683794
SEEDS = (42001, 42002, 42003)
CLAP_SEED_NAMESPACE = 42000
REG_ROOT = ROOT / "runs/matched_reg_1s"
REG_NAME = "hybrid_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{}"
ARMS = ("known_clap_unregularized", "known_clap_tikhonov", "crop_3ms_tikhonov",
        "crop_6ms_tikhonov", "regression_1s")
METRICS = ("edc_broadband_norm_db", "abs_edt_error_ms", "stft_logmag_mse", "nrmse")
OUT = ROOT / "reports/simulated_test_expansion"


def generate_rooms(n: int, bank_seed: int) -> list[dict]:
    """build_shoebox_random's draw order, with the frozen order-8 cap."""
    rng = np.random.RandomState(bank_seed)
    rooms = []
    for i in range(n):
        L, W, H = rng.uniform(3, 10), rng.uniform(3, 8), rng.uniform(2.4, 4)
        rt60 = rng.uniform(0.2, 0.7)
        e_abs, pra_max_order = pra.inverse_sabine(rt60, [L, W, H])
        src = [rng.uniform(.5, L - .5), rng.uniform(.5, W - .5), rng.uniform(1, H - .5)]
        mic = [rng.uniform(.5, L - .5), rng.uniform(.5, W - .5), rng.uniform(1, H - .5)]
        rooms.append(dict(index=i, dims=[L, W, H], rt60=float(rt60),
                          e_absorption=float(e_abs), max_order=int(min(pra_max_order, 8)),
                          source=src, receiver=mic))
    return rooms


def simulate(room: dict) -> np.ndarray:
    r = pra.ShoeBox(room["dims"], fs=SR, materials=pra.Material(room["e_absorption"]),
                    max_order=room["max_order"])
    r.add_source(room["source"])
    r.add_microphone(np.array(room["receiver"]).reshape(3, 1))
    r.compute_rir()
    h = r.rir[0][0].astype(np.float32)
    h = h / (np.max(np.abs(h)) + 1e-8)
    trimmed = delay_trim(h)
    out = np.zeros(SUPPORT, np.float32)
    out[:min(len(trimmed), SUPPORT)] = trimmed[:SUPPORT]
    return out / (np.max(np.abs(out)) + 1e-8)


def broadband_norm_db(pred, true) -> float:
    def curve(x):
        x = _discard_last_n_percent(np.asarray(x, np.float64), 0.5)
        return 10 * np.log10(_backward_int(x) + 1e-32)
    p, t = curve(pred), curve(true)
    p, t = p - p[0], t - t[0]
    ratio = float(np.mean((p - t) ** 2) / np.mean(t ** 2))
    return 10.0 * np.log10(ratio) if ratio > 0 else float("-inf")


def score(target: np.ndarray, estimate: np.ndarray) -> dict:
    t = np.asarray(target, np.float64); e = np.asarray(estimate, np.float64)
    t32, e32 = t.astype(np.float32), e.astype(np.float32)
    return dict(edc_broadband_norm_db=broadband_norm_db(e, t),
                abs_edt_error_ms=abs(float(edt_seconds(e32, SR) - edt_seconds(t32, SR))) * 1000.0,
                stft_logmag_mse=float(stft_magnitude_errors(t32, e32)["stft_logmag_mse"]),
                nrmse=float(np.linalg.norm(e - t) / (np.linalg.norm(t) + 1e-12)))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report-root", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.report_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    prereg = json.loads((out / "preregistration.json").read_text())
    n_rooms = prereg["committed_before_seeing_results"]["n_rooms"]
    bank_seed = prereg["committed_before_seeing_results"]["bank_seed"]

    # Held-out check: none of these rooms may coincide with the trained bank.
    trained = {tuple(np.round(r["dims"] + r["source"] + r["receiver"], 9))
               for r in generate_rooms(32, 23400)}
    rooms = generate_rooms(n_rooms, bank_seed)
    clash = [r["index"] for r in rooms
             if tuple(np.round(r["dims"] + r["source"] + r["receiver"], 9)) in trained]
    assert not clash, f"generated rooms coincide with the trained bank: {clash}"
    print(f"{n_rooms} rooms from seed {bank_seed}; none coincide with the trained bank",
          flush=True)

    claps = load_claps()
    pool = [c for c in claps if c["participant_split"] == "test"]
    print(f"held-out clap pool: {len(pool)}", flush=True)

    targets = np.zeros((n_rooms, SUPPORT), np.float32)
    observations = np.zeros((n_rooms, SR), np.float32)
    excitations = np.zeros((n_rooms, SR), np.float32)
    manifest = []
    for i, room in enumerate(rooms):
        h = simulate(room)
        record_id = f"shoebox_seed{bank_seed}_{i:04d}"
        # Same deterministic clap draw as the frozen materializer.
        seed = int(hashlib.sha256(f"{CLAP_SEED_NAMESPACE}:{record_id}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        pick = int(rng.choice(len(pool), 5, replace=False)[0])
        x = pool[pick]["waveform"][:CLAP_SUPPORT]
        # The frozen materializer convolves against the 44100-sample padded RIR,
        # not the 4096 crop, so the observation carries the full convolution tail.
        h_full = np.zeros(SR, np.float32); h_full[:SUPPORT] = h
        y = sps.fftconvolve(x, h_full)[:SR].astype(np.float32)
        y /= np.max(np.abs(y)) + 1e-8
        targets[i] = h; observations[i] = y
        excitations[i, :CLAP_SUPPORT] = x
        manifest.append(dict(record_id=record_id, room_id=f"expanded_room_{i:04d}",
                             clap_id=pool[pick]["id"], rt60=room["rt60"],
                             length_m=room["dims"][0], width_m=room["dims"][1],
                             height_m=room["dims"][2], max_order=room["max_order"]))
        if (i + 1) % 50 == 0:
            print(f"  generated {i+1}/{n_rooms}", flush=True)
    write_csv(out / "manifest.csv", manifest)

    device = torch.device(args.device)
    predictions = {}
    for seed in SEEDS:
        name = REG_NAME.format(seed)
        cfg = json.loads((REG_ROOT / name / "config.resolved.json").read_text())
        cfg["training_datasets"] = tuple(cfg["training_datasets"])
        model = load_model(RunConfig(**cfg), 20_000, device, REG_ROOT, REG_ROOT)
        obs = torch.from_numpy(observations)[:, None].to(device)
        with torch.no_grad():
            predictions[seed] = np.concatenate(
                [model.predict(obs[s:s + 8]).cpu().numpy()[:, 0] for s in range(0, len(obs), 8)], 0)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  predicted seed {seed}", flush=True)

    rows = []
    for i in range(n_rooms):
        h, y, x = targets[i], observations[i], excitations[i]
        c3 = np.zeros(SR, np.float32); c3[:CROP3] = y[:CROP3]
        c6 = np.zeros(SR, np.float32); c6[:CROP6] = y[:CROP6]
        est = {ARMS[0]: regularized_deconvolution(y, x, SR, 0.0),
               ARMS[1]: regularized_deconvolution(y, x, SR, LAM_TIK),
               ARMS[2]: regularized_deconvolution(y, c3, SR, LAM_C3),
               ARMS[3]: regularized_deconvolution(y, c6, SR, LAM_C6)}
        base = dict(room_id=manifest[i]["room_id"], record_id=manifest[i]["record_id"])
        for arm in ARMS[:4]:
            rows.append(dict(**base, arm=arm, seed="analytic", **score(h, est[arm][:SUPPORT])))
        for seed in SEEDS:
            rows.append(dict(**base, arm=ARMS[4], seed=seed,
                             **score(h, predictions[seed][i][:SUPPORT])))
    write_csv(out / "per_example.csv", rows)

    # record -> seed median (regressor) -> mean and sd across room units
    summary = []
    for arm in ARMS:
        per_room = []
        for i in range(n_rooms):
            vals = [r for r in rows if r["arm"] == arm
                    and r["room_id"] == manifest[i]["room_id"]]
            per_room.append({m: float(np.median([v[m] for v in vals
                             if np.isfinite(v[m])])) for m in METRICS})
        row = dict(arm=arm, n_rooms=n_rooms)
        for m in METRICS:
            v = np.array([p[m] for p in per_room if np.isfinite(p[m])])
            row[m] = float(v.mean()); row[f"{m}_std"] = float(v.std(ddof=1))
            row[f"{m}_median"] = float(np.median(v)); row[f"n_finite_{m}"] = int(len(v))
        summary.append(row)
    write_csv(out / "summary.csv", summary)

    hdr = f"{'arm':<26}{'EDC dB':>10}{'dEDT ms':>10}{'LSE':>10}{'NRMSE':>9}"
    lines = [hdr, "-" * len(hdr)]
    for r in summary:
        lines.append(f"{r['arm']:<26}{r['edc_broadband_norm_db']:>10.2f}"
                     f"{r['abs_edt_error_ms']:>10.1f}{r['stft_logmag_mse']:>10.3f}"
                     f"{r['nrmse']:>9.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "result.json").write_text(json.dumps(dict(
        id="SIMULATED-TEST-EXPANSION", n_rooms=n_rooms, bank_seed=bank_seed,
        rooms_kept="all generated, none dropped", scoring_support=SUPPORT,
        no_training="regressor is the original matched_reg_1s checkpoints",
        held_out_verified="no generated room coincides with the trained bank",
        summary=summary, prediction=prereg["prediction"],
        completed_at_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nCOMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
