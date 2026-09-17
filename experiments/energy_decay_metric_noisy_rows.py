#!/usr/bin/env python3
"""EDC column, 30 dB block: the two unregularized Table 1 rows.

Table 1's known-exc.(unreg.) rows are NOT noiseless -- result.json sources them
from reports/noisy_controlled_benchmark at 30 dB additive Gaussian noise, while
every other row in the same block is noiseless. This recomputes that block with
the Dalsanto EDCLoss, reusing the frozen noise draws exactly. Analytic arms only;
no training, no model inference, no reselection.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import noisy_benchmark_common as common
from claprir.metrics.energy_decay import EDCLoss
from claprir.metrics.deconvolution import regularized_deconvolution
from claprir.metrics.lundeby_truncation import edc_truncated

SR = 44100
SNR = 30.0
DRAWS = (0, 1, 2)
PROVIDERS = ("shoebox", "mit", "but", "ace", "openair")
LAM = {"known_clap_unregularized": 0.0, "known_clap_tikhonov": 0.0003,
       "crop_3ms_tikhonov": 0.003}
CROP3 = 132
OUT = ROOT / "reports/energy_decay_metric_recompute"
FROZEN = ROOT / "reports/noisy_controlled_benchmark/clean_trained_diagnostic/table1/per_room.csv"


def old_edc(target, estimate) -> float:
    a = edc_truncated(np.asarray(estimate, np.float64), len(estimate))
    b = edc_truncated(np.asarray(target, np.float64), len(target))
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def main() -> None:
    primary = EDCLoss(SR, band_selection="intended")
    secondary = EDCLoss(SR, band_selection="upstream")
    rows = []
    for provider in PROVIDERS:
        with np.load(ROOT / f"data/multiroom_generalization/{provider}.npz",
                     allow_pickle=False) as d:
            idx = np.flatnonzero(d["split"].astype(str) == "test")
            ids, rooms = d["record_id"].astype(str), d["room_id"].astype(str)
            targets = np.asarray(d["rir"][idx, :SR], np.float32)
            obs = np.asarray(d["observation"][idx, 0, :SR], np.float32)
            cleans = np.asarray(d["clean"][idx, 0], np.float32)
        for j, i in enumerate(idx):
            t = targets[j]
            nz = np.flatnonzero(t)
            bound = 4096 if provider == "shoebox" else min(SR, int(nz[-1]) + 1)
            ref = t[:bound].astype(np.float64)
            key = f"test:{provider}:{ids[i]}:clap0"
            known = np.zeros(SR, np.float32); known[:882] = cleans[j][:882]
            for draw in DRAWS:
                y = common.noisy_observation(obs[j], key, draw, SNR)
                c3 = np.zeros(SR, np.float32); c3[:CROP3] = y[:CROP3]
                for arm, lam in LAM.items():
                    x = c3 if arm == "crop_3ms_tikhonov" else known
                    e = regularized_deconvolution(y, x, SR, lam)[:bound]
                    rows.append(dict(snr_db=int(SNR), arm=arm, provider=provider,
                        room_id=rooms[i], sample_id=ids[i], noise_draw=draw, bound=bound,
                        edc_rmse_db=old_edc(ref, e), edc_dalsanto=primary(e, ref),
                        edc_dalsanto_alt=secondary(e, ref)))
        print(f"scored {provider}: {len(idx)} records x {len(DRAWS)} draws", flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "noisy_per_example.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)

    METRICS = ("edc_rmse_db", "edc_dalsanto", "edc_dalsanto_alt")
    # median 3 draws per record -> median records per provider/room
    rec = defaultdict(list)
    for r in rows:
        rec[(r["arm"], r["provider"], r["room_id"], r["sample_id"])].append(r)
    record_rows = [dict(arm=a, provider=p, room_id=rm, sample_id=s,
                        **{m: float(np.median([x[m] for x in v])) for m in METRICS})
                   for (a, p, rm, s), v in sorted(rec.items())]
    rm_g = defaultdict(list)
    for r in record_rows:
        rm_g[(r["arm"], r["provider"], r["room_id"])].append(r)
    room_rows = [dict(arm=a, provider=p, room_id=rm,
                      **{m: float(np.median([x[m] for x in v])) for m in METRICS})
                 for (a, p, rm), v in sorted(rm_g.items())]
    with (OUT / "noisy_per_room.csv").open("w", newline="") as f:
        k2 = list(room_rows[0]); w = csv.DictWriter(f, k2, lineterminator="\n")
        w.writeheader(); w.writerows(room_rows)

    frozen = {(r["arm"], r["provider"], r["room_id"]): float(r["edc_rmse_db"])
              for r in csv.DictReader(FROZEN.open()) if int(r["snr_db"]) == 30}
    deltas = [abs(r["edc_rmse_db"] - frozen[(r["arm"], r["provider"], r["room_id"])])
              for r in room_rows if (r["arm"], r["provider"], r["room_id"]) in frozen]
    max_delta = max(deltas) if deltas else float("nan")

    summary = []
    for population in ("measured_pooled", "shoebox"):
        for arm in LAM:
            sel = [r for r in room_rows if r["arm"] == arm and
                   (r["provider"] != "shoebox" if population == "measured_pooled"
                    else r["provider"] == "shoebox")]
            row = dict(snr_db=int(SNR), population=population, arm=arm, n_rooms=len(sel))
            for m in METRICS:
                v = np.array([r[m] for r in sel])
                row[m] = float(np.mean(v)); row[f"{m}_median"] = float(np.median(v))
                row[f"{m}_std"] = float(np.std(v, ddof=1))
            summary.append(row)
    with (OUT / "noisy_summary.csv").open("w", newline="") as f:
        k3 = list(summary[0]); w = csv.DictWriter(f, k3, lineterminator="\n")
        w.writeheader(); w.writerows(summary)
    (OUT / "noisy_verification.json").write_text(json.dumps(dict(
        snr_db=SNR, draws=list(DRAWS), compared_room_units=len(deltas),
        max_abs_old_edc_difference=max_delta,
        reproduces_frozen_noisy_old_edc=bool(max_delta < 1e-6),
        noise_source="experiments/noisy_benchmark_common.noisy_observation, "
                     "record_key test:<provider>:<sample_id>:clap0",
        analytic_only=True, completed_at_utc=datetime.now(timezone.utc).isoformat()),
        indent=2) + "\n")
    for r in summary:
        print(f"{r['population']:<16}{r['arm']:<28} old={r['edc_rmse_db']:>8.3f} "
              f"(med {r['edc_rmse_db_median']:>7.3f})  new={r['edc_dalsanto']:>8.4f} "
              f"(med {r['edc_dalsanto_median']:>7.4f})", flush=True)
    print(f"\nfrozen noisy reproduction: max |Δ| = {max_delta:.3g} over {len(deltas)} room units")
    print(f"COMPLETE {OUT}")


if __name__ == "__main__":
    main()
