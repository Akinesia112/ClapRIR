#!/usr/bin/env python3
"""The 'Norm. broadband EDC' column for Table 1, all arms, both blocks.

EDCLoss's exact formula with the filterbank removed: discard the last 0.5 %,
backward integral with the 1/N factor, dB with a 1e-32 floor, subtract the t=0
level, divide by mean(true**2). One broadband curve, so it isolates the
normalisation from the third-octave decomposition.

Uses Table 1's own aggregation and each row's own source condition: the
unregularized rows at 30 dB with three noise draws, everything else noiseless;
records -> (seed, provider, room) median -> seed median -> across room units.
No training, no reselection.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "experiments"))
import noisy_benchmark_common as common
from claprir.metrics.energy_decay import _backward_int, _discard_last_n_percent
from claprir.metrics.deconvolution import regularized_deconvolution
from claprir.training.train_rir_estimator import RunConfig, load_model

SR = 44100
PROVIDERS = ("shoebox", "mit", "but", "ace", "openair")
SEEDS = (42001, 42002, 42003)
DRAWS = (0, 1, 2)
CROP3, CROP6 = 132, 265
RUN_ROOT = ROOT / "runs/matched_reg_1s"
NAME = "hybrid_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{}"
OUT = ROOT / "reports/energy_decay_metric_recompute"
# Noiseless lambdas (frozen Table 1) and the 30 dB selection for the noisy rows.
LAM_CLEAN = {"known_clap_unregularized": 0.0, "known_clap_tikhonov": 1e-5,
             "crop_3ms_tikhonov": 10 ** -3.5, "crop_6ms_tikhonov": 0.00031622776601683794}
LAM_NOISY = {"known_clap_unregularized": 0.0}
# Each row's source condition and displayed central statistic, as Table 1 uses them.
SPEC = (("known_clap_unregularized", "30 dB noise", "median"),
        ("known_clap_tikhonov", "noiseless", "median"),
        ("crop_3ms_tikhonov", "noiseless", "median"),
        ("crop_6ms_tikhonov", "noiseless", "mean"),
        ("regression_1s", "noiseless", "median"))


def broadband_metrics(pred, true) -> dict:
    """Both readings of the same comparison.

    ``norm``  is Eq. (edc): MSE(Dt_pred, Dt_true) / mean(Dt_true**2), dimensionless.
    ``rms_db`` is its numerator square-rooted: the RMS deviation between the two
    level-normalised decay curves, in decibels. Same curves, same discard, same
    epsilon; only the trailing normalisation by the reference variance differs.
    """
    def curve(x):
        x = _discard_last_n_percent(np.asarray(x, np.float64), 0.5)
        return 10 * np.log10(_backward_int(x) + 1e-32)
    p, t = curve(pred), curve(true)
    p, t = p - p[0], t - t[0]
    mse = float(np.mean((p - t) ** 2))
    ratio = mse / float(np.mean(t ** 2))
    # The ratio is a quotient of mean squares, so 10*log10 is the power-like dB
    # convention (as for a normalised MSE). Converted PER RECORD, before any
    # aggregation, because the log is nonlinear -- the repo already requires this
    # for LSD ("transform before aggregation").
    return dict(edc_broadband_norm=ratio,
                edc_broadband_norm_db=10.0 * np.log10(ratio) if ratio > 0 else float("-inf"),
                edc_broadband_rms_db=float(np.sqrt(mse)))


def excitations(y, clap):
    known = np.zeros(SR, np.float32); known[:882] = clap[:882]
    c3 = np.zeros(SR, np.float32); c3[:CROP3] = y[:CROP3]
    c6 = np.zeros(SR, np.float32); c6[:CROP6] = y[:CROP6]
    return {"known_clap_unregularized": known, "known_clap_tikhonov": known,
            "crop_3ms_tikhonov": c3, "crop_6ms_tikhonov": c6}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {}
    for p in PROVIDERS:
        with np.load(ROOT / f"data/multiroom_generalization/{p}.npz", allow_pickle=False) as d:
            idx = np.flatnonzero(d["split"].astype(str) == "test")
            data[p] = (d["record_id"].astype(str)[idx], d["room_id"].astype(str)[idx],
                       np.asarray(d["rir"][idx, :SR], np.float32),
                       np.asarray(d["observation"][idx, 0, :SR], np.float32),
                       np.asarray(d["clean"][idx, 0], np.float32))

    preds = {}
    for seed in SEEDS:
        name = NAME.format(seed)
        cfg = json.loads((RUN_ROOT / name / "config.resolved.json").read_text())
        cfg["training_datasets"] = tuple(cfg["training_datasets"])
        model = load_model(RunConfig(**cfg), 20_000, device, RUN_ROOT, RUN_ROOT)
        with torch.no_grad():
            for p in PROVIDERS:
                obs = torch.from_numpy(data[p][3])[:, None].to(device)
                out = [model.predict(obs[s:s + 8]).cpu().numpy()[:, 0]
                       for s in range(0, len(obs), 8)]
                preds[(p, seed)] = np.concatenate(out, 0)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"predicted with ORIGINAL seed {seed}", flush=True)

    rows = []
    for p in PROVIDERS:
        ids, rooms, targets, obs, cleans = data[p]
        for i in range(len(ids)):
            t = targets[i]
            bound = 4096 if p == "shoebox" else min(SR, int(np.flatnonzero(t)[-1]) + 1)
            ref = t[:bound].astype(np.float64)
            base = dict(provider=p, room_id=rooms[i], sample_id=ids[i], bound=bound)
            exc = excitations(obs[i], cleans[i])
            for arm, lam in LAM_CLEAN.items():
                e = regularized_deconvolution(obs[i], exc[arm], SR, lam)[:bound]
                rows.append(dict(**base, arm=arm, condition="noiseless", seed="analytic",
                                 draw="", **broadband_metrics(e, ref)))
            for seed in SEEDS:
                e = preds[(p, seed)][i][:bound]
                rows.append(dict(**base, arm="regression_1s", condition="noiseless",
                                 seed=seed, draw="", **broadband_metrics(e, ref)))
            key = f"test:{p}:{ids[i]}:clap0"
            for draw in DRAWS:
                y = common.noisy_observation(obs[i], key, draw, 30.0)
                exc_n = excitations(y, cleans[i])
                for arm, lam in LAM_NOISY.items():
                    e = regularized_deconvolution(y, exc_n[arm], SR, lam)[:bound]
                    rows.append(dict(**base, arm=arm, condition="30 dB noise",
                                     seed="analytic", draw=draw,
                                     **broadband_metrics(e, ref)))
        print(f"scored {p}", flush=True)

    keys = list(rows[0])
    with (OUT / "broadband_per_example.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)

    # records/draws -> (seed, room) -> seeds -> room units, exactly as Table 1 does.
    KEYS = ("edc_broadband_norm", "edc_broadband_norm_db", "edc_broadband_rms_db")
    g = defaultdict(list)
    for r in rows:
        g[(r["arm"], r["condition"], r["provider"], r["room_id"], str(r["seed"]))].append(r)
    seed_rows = [dict(arm=a, condition=c, provider=p, room_id=rm, seed=s,
                      **{k: float(np.median([x[k] for x in v])) for k in KEYS})
                 for (a, c, p, rm, s), v in sorted(g.items())]
    gr = defaultdict(list)
    for r in seed_rows:
        gr[(r["arm"], r["condition"], r["provider"], r["room_id"])].append(r)
    room_rows = [dict(arm=a, condition=c, provider=p, room_id=rm,
                      **{k: float(np.median([x[k] for x in v])) for k in KEYS})
                 for (a, c, p, rm), v in sorted(gr.items())]
    with (OUT / "broadband_per_room.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(room_rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(room_rows)

    out_rows = []
    for population in ("measured_pooled", "shoebox"):
        for arm, cond, stat in SPEC:
            sel = [r for r in room_rows if r["arm"] == arm and r["condition"] == cond
                   and (r["provider"] != "shoebox" if population == "measured_pooled"
                        else r["provider"] == "shoebox")]
            row = dict(population=population, arm=arm, condition=cond,
                       statistic=stat, n_rooms=len(sel))
            for k in KEYS:
                v = np.array([r[k] for r in sel])
                row[k] = float(np.mean(v) if stat == "mean" else np.median(v))
                row[k + "_std"] = float(np.std(v, ddof=1))
            out_rows.append(row)
    with (OUT / "broadband_column.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(out_rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(out_rows)
    (OUT / "broadband_column.json").write_text(json.dumps(dict(
        id="EDC-BROADBAND-NORM-COLUMN",
        definition="EDCLoss formula with the filterbank removed; dimensionless ratio",
        aggregation="records/draws -> (seed, provider, room) median -> seed median -> "
                    "room units; central statistic per row as Table 1 displays it",
        rows=out_rows, no_training_no_reselection=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")

    hdr = (f"{'population':<16}{'arm':<26}{'stat':<7}"
           f"{'ratio':>9}{'  ratio dB':>11}{'sd dB':>9}{'RMSdB':>8}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in out_rows:
        print(f"{r['population']:<16}{r['arm']:<26}{r['statistic']:<7}"
              f"{r['edc_broadband_norm']:>9.4f}{r['edc_broadband_norm_db']:>11.2f}"
              f"{r['edc_broadband_norm_db_std']:>9.2f}{r['edc_broadband_rms_db']:>8.2f}")
    print(f"\nCOMPLETE {OUT / 'broadband_column.csv'}")


if __name__ == "__main__":
    main()
