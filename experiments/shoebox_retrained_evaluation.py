#!/usr/bin/env python3
"""Score the retrained 1-s Regression against the regenerated Shoebox reference.

Four analytic arms plus the neural regressor, on both the frozen 4096-sample
benchmark and the regenerated one-second benchmark, so the old and new numbers
can be read side by side. Also scores the ORIGINAL checkpoints on the new
reference, which isolates how much of any change is the retraining and how much
is the reference itself. Measured-RIR evaluation is untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / "src"))
from claprir.metrics.room_acoustics import edt_seconds, stft_magnitude_errors
from claprir.metrics.deconvolution import regularized_deconvolution
from claprir.training.train_rir_estimator import RunConfig, load_model
from claprir.metrics.lundeby_truncation import edc_truncated

SR = 44100
SEEDS = (42001, 42002, 42003)
CROP3, CROP6 = 132, 265
LAM_TIKHONOV, LAM_CROP3 = 1e-5, 10 ** -3.5
LAM_CROP6 = 0.00031622776601683794
RUN_NAME = "hybrid_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{}"
OLD_RUN_ROOT, NEW_RUN_ROOT = "runs/matched_reg_1s", "runs/shoebox1s_reg"
OLD_SHARD = "data/multiroom_generalization/shoebox.npz"
NEW_SHARD = "data/multiroom_generalization_shoebox1s/shoebox.npz"
ANALYTIC = ("known_clap_unregularized", "known_clap_tikhonov",
            "crop_3ms_tikhonov", "crop_6ms_tikhonov")
METRICS = ("edc_rmse_db", "abs_edt_error_ms", "lsd_db", "nrmse")


def load_split(path: Path):
    d = np.load(path, allow_pickle=False)
    idx = np.flatnonzero(d["split"].astype(str) == "test")
    assert len(idx) == 4, (path, len(idx))
    return (d["record_id"].astype(str)[idx], d["room_id"].astype(str)[idx],
            np.asarray(d["rir"][idx, :SR], np.float32),
            np.asarray(d["observation"][idx, 0, :SR], np.float32),
            np.asarray(d["clean"][idx, 0], np.float32))


def minus60_bound(h: np.ndarray) -> int:
    e = np.asarray(h, np.float64) ** 2
    edc = 10 * np.log10(np.maximum(np.cumsum(e[::-1])[::-1] / (e.sum() + 1e-30), 1e-30))
    return int(np.argmax(edc <= -60.0)) if np.any(edc <= -60.0) else len(h)


def score(target: np.ndarray, estimate: np.ndarray) -> dict:
    t, e = np.asarray(target, np.float64), np.asarray(estimate, np.float64)
    t32, e32 = t.astype(np.float32), e.astype(np.float32)
    ref, est = edc_truncated(t, len(t)), edc_truncated(e, len(e))
    logmag = stft_magnitude_errors(t32, e32)["stft_logmag_mse"]
    return dict(edc_rmse_db=float(np.sqrt(np.mean((est - ref) ** 2))),
                abs_edt_error_ms=abs(float(edt_seconds(e32, SR) - edt_seconds(t32, SR))) * 1000.0,
                lsd_db=20.0 * math.sqrt(logmag),
                nrmse=float(np.linalg.norm(e - t) / (np.linalg.norm(t) + 1e-12)))


def analytic_estimates(y: np.ndarray, clap882: np.ndarray) -> dict[str, np.ndarray]:
    known = np.zeros(SR, np.float32); known[:len(clap882)] = clap882
    c3, c6 = np.zeros(SR, np.float32), np.zeros(SR, np.float32)
    c3[:CROP3], c6[:CROP6] = y[:CROP3], y[:CROP6]
    return {ANALYTIC[0]: regularized_deconvolution(y, known, SR, 0.0),
            ANALYTIC[1]: regularized_deconvolution(y, known, SR, LAM_TIKHONOV),
            ANALYTIC[2]: regularized_deconvolution(y, c3, SR, LAM_CROP3),
            ANALYTIC[3]: regularized_deconvolution(y, c6, SR, LAM_CROP6)}


def regressor_predictions(run_root: Path, observations: np.ndarray, device) -> dict:
    """One checkpoint per seed; no training, no reselection."""
    out = {}
    obs = torch.from_numpy(observations)[:, None].to(device)
    for seed in SEEDS:
        name = RUN_NAME.format(seed)
        cfg_path = run_root / name / "config.resolved.json"
        ckpt = run_root / name / "model_updates20000.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint {ckpt}; this harness will not train one.")
        cfg = json.loads(cfg_path.read_text())
        cfg["training_datasets"] = tuple(cfg["training_datasets"])
        config = RunConfig(**cfg)
        assert config.run_name == name, (config.run_name, name)
        model = load_model(config, 20_000, device, run_root, run_root)
        with torch.no_grad():
            out[seed] = model.predict(obs).cpu().numpy()[:, 0]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--new-run-root", type=Path, default=None,
                    help="where the retrained checkpoints live (default: repo/runs/shoebox1s_reg)")
    ap.add_argument("--report-root", type=Path,
                    default=ROOT / "reports/shoebox_one_second_support")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    repo, out = args.repo.resolve(), args.report_root.resolve()
    new_root = (args.new_run_root or repo / NEW_RUN_ROOT).resolve()
    device = torch.device(args.device)
    out.mkdir(parents=True, exist_ok=True)

    old_ids, old_rooms, old_h, old_y, old_clap = load_split(repo / OLD_SHARD)
    new_ids, new_rooms, new_h, new_y, new_clap = load_split(repo / NEW_SHARD)
    assert list(old_ids) == list(new_ids), "record identity drift"
    assert np.array_equal(old_clap, new_clap), "clap assignment drift"
    assert np.all(old_h[:, 4096:] == 0), "frozen shard is not zero past 4096"
    assert np.all(new_h[:, 4096:] != 0), "regenerated shard tail is not fully simulated"

    print("loading ORIGINAL checkpoints", flush=True)
    old_pred_on_old = regressor_predictions(repo / OLD_RUN_ROOT, old_y, device)
    old_pred_on_new = regressor_predictions(repo / OLD_RUN_ROOT, new_y, device)
    print("loading RETRAINED checkpoints", flush=True)
    new_pred_on_new = regressor_predictions(new_root, new_y, device)

    rows = []
    for i, rid in enumerate(old_ids):
        b60 = minus60_bound(new_h[i])
        blocks = (
            # name,                      target,   observation, clap,        bound, regressor
            ("old_4096",                 old_h[i], old_y[i], old_clap[i], 4096,  old_pred_on_old),
            ("one_second_to_minus60db",  new_h[i], new_y[i], new_clap[i], b60,   new_pred_on_new),
            ("one_second_full",          new_h[i], new_y[i], new_clap[i], SR,    new_pred_on_new),
            ("minus60db_original_model", new_h[i], new_y[i], new_clap[i], b60,   old_pred_on_new),
        )
        for support, h, y, clap, bound, preds in blocks:
            base = dict(support=support, room_id=old_rooms[i], record_id=rid,
                        bound=bound, bound_ms=1000.0 * bound / SR)
            if support != "minus60db_original_model":
                for arm, est in analytic_estimates(y, clap[:882]).items():
                    rows.append(dict(**base, arm=arm, seed="analytic",
                                     **score(h[:bound], est[:bound])))
            arm = ("neural_regressor_original" if support == "minus60db_original_model"
                   else "neural_regressor")
            for seed in SEEDS:
                rows.append(dict(**base, arm=arm, seed=seed,
                                 **score(h[:bound], preds[seed][i][:bound])))
    write_csv(out / "retrained_per_example.csv", rows)

    # median across seeds within room, then mean +/- sd across the four rooms
    summary = []
    for support in ("old_4096", "one_second_to_minus60db", "one_second_full",
                    "minus60db_original_model"):
        arms = [a for a in dict.fromkeys(r["arm"] for r in rows if r["support"] == support)]
        for arm in arms:
            per_room = []
            for rid in old_ids:
                vals = [r for r in rows if r["support"] == support
                        and r["arm"] == arm and r["record_id"] == rid]
                per_room.append({m: float(np.median([v[m] for v in vals
                                 if np.isfinite(v[m])])) for m in METRICS})
            row = dict(support=support, arm=arm, n_rooms=len(per_room),
                       mean_bound_ms=float(np.mean([r["bound_ms"] for r in rows
                           if r["support"] == support and r["arm"] == arm])))
            for m in METRICS:
                v = np.array([p[m] for p in per_room])
                row[m] = float(np.mean(v))
                row[f"{m}_std"] = float(np.std(v, ddof=1))
            summary.append(row)
    write_csv(out / "retrained_summary.csv", summary)

    hdr = f"{'support':<28}{'arm':<28}{'ms':>8}{'EDC':>9}{'|dEDT|ms':>10}{'LSD':>8}{'NRMSE':>8}"
    lines = [hdr, "-" * len(hdr)]
    for r in summary:
        lines.append(f"{r['support']:<28}{r['arm']:<28}{r['mean_bound_ms']:>8.1f}"
                     f"{r['edc_rmse_db']:>9.3f}{r['abs_edt_error_ms']:>10.2f}"
                     f"{r['lsd_db']:>8.3f}{r['nrmse']:>8.4f}")
    (out / "retrained_summary.txt").write_text("\n".join(lines) + "\n")
    (out / "retrained_result.json").write_text(json.dumps(dict(
        id="SHOEBOX-1S-RETRAINED-EVALUATION",
        x_decision="Whether the manuscript's Simulated block, and its claim that the "
                   "neural regressor is the best blind method there, survives moving from "
                   "the 4096-sample support to the regenerated one-second reference.",
        adopted_support="per-room -60 dB EDC point of the regenerated 1 s target",
        new_run_root=str(new_root), old_run_root=OLD_RUN_ROOT,
        summary=summary, completed_at_utc=datetime.now(timezone.utc).isoformat()),
        indent=2) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nCOMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
