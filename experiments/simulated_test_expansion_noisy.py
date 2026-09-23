#!/usr/bin/env python3
"""The unregularized row of the expanded simulated set, at 30 dB as Table 1 uses it.

Table 1's known-excitation (unregularized) rows are not noiseless: result.json
sources them from the 30 dB additive-noise benchmark while the rest of the block
is noiseless. simulated_test_expansion.py scored everything noiseless, which is
right for the other four arms and wrong for this one. This redoes it with the
frozen noise generator, three draws per record, on the same 216 rooms.

Analytic only. No training, no model inference.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import scipy.signal as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import noisy_benchmark_common as common
import simulated_test_expansion as base

SR, SUPPORT, CLAP_SUPPORT = base.SR, base.SUPPORT, base.CLAP_SUPPORT
SNR, DRAWS = 30.0, (0, 1, 2)
OUT = ROOT / "reports/simulated_test_expansion"


def main() -> None:
    prereg = json.loads((OUT / "preregistration.json").read_text())
    n_rooms = prereg["committed_before_seeing_results"]["n_rooms"]
    bank_seed = prereg["committed_before_seeing_results"]["bank_seed"]
    rooms = base.generate_rooms(n_rooms, bank_seed)
    manifest = {r["record_id"]: r for r in csv.DictReader((OUT / "manifest.csv").open())}
    claps = base.load_claps()
    pool = {c["id"]: c for c in claps if c["participant_split"] == "test"}

    rows = []
    for i, room in enumerate(rooms):
        record_id = f"shoebox_seed{bank_seed}_{i:04d}"
        m = manifest[record_id]
        h = base.simulate(room)
        x = pool[m["clap_id"]]["waveform"][:CLAP_SUPPORT]
        h_full = np.zeros(SR, np.float32); h_full[:SUPPORT] = h
        clean_y = sps.fftconvolve(x, h_full)[:SR].astype(np.float32)
        clean_y /= np.max(np.abs(clean_y)) + 1e-8
        exc = np.zeros(SR, np.float32); exc[:CLAP_SUPPORT] = x
        key = f"test:shoebox_expanded:{record_id}:clap0"
        for draw in DRAWS:
            y = common.noisy_observation(clean_y, key, draw, SNR)
            est = base.regularized_deconvolution(y, exc, SR, 0.0)[:SUPPORT]
            rows.append(dict(room_id=m["room_id"], record_id=record_id,
                             arm="known_clap_unregularized", snr_db=int(SNR),
                             noise_draw=draw, **base.score(h, est)))
        if (i + 1) % 50 == 0:
            print(f"  scored {i+1}/{n_rooms}", flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "noisy_per_example.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)

    # median over the three draws per record, then mean and sd across rooms
    summary = {}
    for metric in base.METRICS:
        per_room = []
        for i in range(n_rooms):
            rid = f"shoebox_seed{bank_seed}_{i:04d}"
            v = [r[metric] for r in rows if r["record_id"] == rid and np.isfinite(r[metric])]
            if v:
                per_room.append(float(np.median(v)))
        a = np.asarray(per_room)
        summary[metric] = float(a.mean())
        summary[metric + "_std"] = float(a.std(ddof=1))
        summary["n_finite_" + metric] = int(len(a))
    (OUT / "noisy_summary.json").write_text(json.dumps(dict(
        arm="known_clap_unregularized", snr_db=SNR, draws=list(DRAWS),
        n_rooms=n_rooms, summary=summary,
        note="Matches Table 1's source condition for the unregularized rows; the other "
             "four arms stay noiseless.",
        completed_at_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
