#!/usr/bin/env python3
"""Within-room spectral dispersion of real-world RIR estimates (section 4.5).

The question this answers, in Eloi Moliner's phrasing: in one room, 40 different
handclaps should ideally all yield the same estimated RIR -- do they?

Handclap-mode labels are deliberately ignored. An earlier version of this
analysis split the pairs into within-mode and cross-mode groups, which measures
how much extra variation the mode LABEL explains; that is a different and
narrower question. Here every pair of recordings in a room counts equally, so
the number is plain within-room consistency.

Spectra are Fig. 3's, imported from the figure script rather than
re-implemented, so the metric cannot drift from the plot it quantifies:
0-0.9 s, 1/12-octave averaged over 0.1-15 kHz, each signal divided by its own
mean FFT-bin power over 0.1-10 kHz, in dB.

For each room and estimator, the distance between recordings i and j is

    d_ij = sqrt( mean_f ( S_i(f) - S_j(f) )^2 )     [dB]

over all C(40,2) = 780 pairs, summarised per room and then across the 7 rooms.

Phone recordings have no paired reference RIR. This measures agreement between
estimates, NOT reconstruction accuracy. No training and no new inference.
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
sys.path.insert(0, str(ROOT / "figures"))
from claprir.metrics.deconvolution import regularized_deconvolution
from phone_qualitative_figure import spectrum as figure_spectrum, N as FIG_N, HI as FIG_HI

SR = 44100
HORIZON = 44100
CROP3, CROP6 = 132, 265
LAM_CROP3 = 10 ** -3.5
LAM_CROP6 = 0.00031622776601683794
SEEDS = (42001, 42002, 42003)
ARMS = ("recording", "crop_3ms", "crop_6ms", "regressor")
LABELS = {"recording": "Recorded handclaps", "crop_3ms": "Crop 3 ms",
          "crop_6ms": "Crop 6 ms", "regressor": "Neural regressor"}
CORPUS = ROOT / "reports/phone_clap_demo"
CACHE = ROOT / "runs/e6_phone_raw_cache"
OUT = ROOT / "reports/phone_spectral_consistency"


def load_order() -> list[dict]:
    rows = [r for r in csv.DictReader(
        (ROOT / "reports/phone_deployment_evaluation/results/per_clap.csv").open())
        if int(r["seed"]) == SEEDS[0]]
    meta = {r["segment_path"]: r for r in csv.DictReader((CORPUS / "results/metadata.csv").open())}
    return [dict(sample_id=r["sample_id"], room_id=r["room_id"], room_name=r["room_name"],
                 clap_mode=r["clap_mode"], repeat_id=r["repeat_id"],
                 pre_pad_s=float(meta[r["sample_id"]]["pre_pad_s"])) for r in rows]


def observation(record: dict) -> np.ndarray:
    """phone_deployment_evaluation's preprocessing: resample, drop pre-pad, no gain."""
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


def load_regressor(n: int) -> np.ndarray:
    """Frozen predictions, median across the three training seeds."""
    per_seed = []
    for seed in SEEDS:
        chunks = []
        for start in range(0, n, 8):
            path = CACHE / f"raw_v1_seed{seed}_batch{start:03d}.npy"
            if not path.exists():
                raise SystemExit(f"missing prediction cache {path}; this harness will not infer.")
            chunks.append(np.load(path))
        stacked = np.concatenate(chunks, 0)
        assert len(stacked) == n
        per_seed.append(stacked)
    return np.median(np.stack(per_seed), axis=0)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-root", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.report_root.resolve(); out.mkdir(parents=True, exist_ok=True)

    order = load_order()
    n = len(order)
    print(f"{n} phone recordings; Fig. 3 spectra: {FIG_N} samples "
          f"({FIG_N/SR:.2f} s), up to {FIG_HI} Hz", flush=True)
    regressor = load_regressor(n)

    signals = {arm: np.zeros((n, HORIZON), np.float32) for arm in ARMS}
    for i, record in enumerate(order):
        y = observation(record)
        signals["recording"][i] = y
        signals["crop_3ms"][i] = crop_estimate(y, CROP3, LAM_CROP3)
        signals["crop_6ms"][i] = crop_estimate(y, CROP6, LAM_CROP6)
        signals["regressor"][i] = regressor[i]
        if (i + 1) % 70 == 0:
            print(f"  prepared {i+1}/{n}", flush=True)

    # One call per arm, so every spectrum goes through Fig. 3's own function.
    spectra = {}
    for arm in ARMS:
        centers, values = figure_spectrum(signals[arm])
        spectra[arm] = values
        assert values.shape == (n, len(centers)), values.shape
    n_bands = len(centers)
    print(f"1/12-octave bands: {n_bands} ({centers[0]:.0f}-{centers[-1]:.0f} Hz)", flush=True)

    by_room = defaultdict(list)
    for i, r in enumerate(order):
        by_room[(int(r["room_id"]), r["room_name"])].append(i)

    pair_rows, room_rows = [], []
    for (room_id, room_name), idx in sorted(by_room.items()):
        for arm in ARMS:
            S = spectra[arm][idx]
            d = [float(np.sqrt(np.mean((S[a] - S[b]) ** 2)))
                 for a, b in combinations(range(len(idx)), 2)]
            for (a, b), value in zip(combinations(range(len(idx)), 2), d):
                pair_rows.append(dict(room_id=room_id, room_name=room_name, arm=arm,
                                      sample_a=order[idx[a]]["sample_id"],
                                      sample_b=order[idx[b]]["sample_id"],
                                      spectral_distance_db=value))
            d = np.asarray(d)
            room_rows.append(dict(room_id=room_id, room_name=room_name, arm=arm,
                                  n_claps=len(idx), n_pairs=len(d),
                                  mean_db=float(d.mean()), median_db=float(np.median(d)),
                                  p90_db=float(np.quantile(d, .9)), max_db=float(d.max())))
        print(f"  room {room_id} {room_name}: {len(idx)} claps, "
              f"{len(idx)*(len(idx)-1)//2} pairs per arm", flush=True)
    write_csv(out / "pairwise.csv", pair_rows)
    write_csv(out / "per_room.csv", room_rows)

    summary = []
    for arm in ARMS:
        sel = [r for r in room_rows if r["arm"] == arm]
        v = np.array([r["mean_db"] for r in sel])
        summary.append(dict(arm=arm, label=LABELS[arm], n_rooms=len(sel),
                            mean_db=float(v.mean()), std_db=float(v.std(ddof=1)),
                            min_room_db=float(v.min()), max_room_db=float(v.max()),
                            median_of_room_medians_db=float(np.median(
                                [r["median_db"] for r in sel]))))
    write_csv(out / "summary.csv", summary)

    hdr = f"{'estimator':<20}{'mean':>10}{'sd across rooms':>18}{'best room':>12}{'worst room':>12}"
    lines = [hdr, "-" * len(hdr)]
    for r in summary:
        lines.append(f"{r['label']:<20}{r['mean_db']:>7.2f} dB{r['std_db']:>15.2f} dB"
                     f"{r['min_room_db']:>9.2f} dB{r['max_room_db']:>9.2f} dB")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")

    (out / "result.json").write_text(json.dumps(dict(
        id="PHONE-SPECTRAL-CONSISTENCY",
        question="In one room, 40 different handclaps should ideally all yield the same "
                 "estimated RIR. Do they?",
        metric="d_ij = sqrt(mean_f (S_i(f) - S_j(f))^2) in dB over all C(40,2)=780 pairs "
               "per room; S is Fig. 3's spectrum (0-0.9 s, 1/12-octave, 0.1-15 kHz, each "
               "signal normalised by its own 0.1-10 kHz mean FFT-bin power).",
        spectra_source="figures/phone_qualitative_figure.py::spectrum, imported",
        mode_labels_ignored="Deliberate. Every pair in a room counts equally; this is "
                            "within-room consistency, not a mode-label effect.",
        no_reference_rir="Phone recordings have no paired reference RIR. Measures agreement "
                         "between estimates, NOT reconstruction accuracy.",
        recordings=n, rooms=len(by_room), claps_per_room=40, pairs_per_room_per_arm=780,
        bands=n_bands, seeds=list(SEEDS),
        regressor_prediction_source="reports/phone_deployment_evaluation cache; no new inference",
        summary=summary, per_room=room_rows,
        supersedes="reports/phone_transfer_consistency (mode-label framing; kept for the "
                   "record, not referenced by the manuscript)",
        completed_at_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nCOMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
