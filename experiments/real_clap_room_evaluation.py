#!/usr/bin/env python3
"""E5: does the chosen estimator transfer to actually recorded human claps?

Single clap only. The K = 1/2/3/5 question is removed from this paper
(publication/EXPERIMENTS.md), so one clap per position is used and repeats are
not averaged.

The reference is the sweep-derived RIR that ships with the corpus, so unlike E6
this comparison HAS a ground truth and quantitative error is admissible.

TWO LIMITATIONS, both structural and both reported rather than worked around:

  * The Spheres corpus is 250 ms and the chosen estimator is a 1 s model. The
    observation is zero-padded to 1 s to be fed in, which is a domain shift: a
    real 1 s observation continues to reverberate where this one is padded with
    silence. Scores are therefore taken on the 250 ms the reference actually
    covers, and this table cannot be compared against E3's native-support
    numbers.
  * Padding is applied to the INPUT, not the target, so it cannot be handled by
    the validity mask, which is a training-time construct. This is the input-side
    missingness caveat already registered in the project.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from claprir.metrics.lundeby_truncation import edc_truncated                                  # noqa: E402
from claprir.metrics.room_acoustics import diagnosis_metrics
from claprir.training.train_rir_estimator import RunConfig, load_model

SR = 44_100
SEEDS = (42_001, 42_002, 42_003)
ARM = "runs/matched_reg_1s", "hybrid_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{}"
CLAP = 0                      # single clap: the first of the 12 at each position


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
    ap.add_argument("--npz", type=Path,
                    default=ROOT / "data/real_clap_multiclap/spheres_session1.npz")
    ap.add_argument("--report-root", type=Path, default=ROOT / "reports/real_clap_room_evaluation")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.npz = (a.repo_root / a.npz).resolve()
    a.report_root = (a.repo_root / a.report_root).resolve()
    if not a.npz.is_file(): raise FileNotFoundError(a.npz)
    for seed in SEEDS:
        if not (a.repo_root / ARM[0] / ARM[1].format(seed) / "model_updates20000.pt").is_file():
            raise FileNotFoundError(f"missing E5 checkpoint for seed {seed}")
    dev = torch.device(a.device)
    d = np.load(a.npz, allow_pickle=False)
    obs_all, ref_all = d["observation"], d["rir"]
    n_pos, n_clap, ref_len = obs_all.shape[0], obs_all.shape[1], ref_all.shape[1]
    print(f"{n_pos} positions x {n_clap} claps, reference {ref_len} samples "
          f"({ref_len/SR*1000:.0f} ms); using clap index {CLAP} only")
    rows = []
    for seed in SEEDS:
        root = a.repo_root / ARM[0]; name = ARM[1].format(seed)
        cfg = {k: v for k, v in json.loads(
            (root / name / "config.resolved.json").read_text()).items() if v is not None}
        cfg["training_datasets"] = tuple(cfg["training_datasets"])
        c = RunConfig(**cfg)
        if c.run_name != name:
            raise SystemExit(f"run_name drift: {c.run_name} != {name}")
        m = load_model(c, 20_000, dev, root, root)
        L = c.signal_length
        for s in range(0, n_pos, 8):
            sl = slice(s, min(s + 8, n_pos))
            o = np.zeros((sl.stop - sl.start, 1, L), np.float32)
            o[:, 0, :min(ref_len, L)] = obs_all[sl, CLAP, :L]
            with torch.no_grad():
                est = m.predict(torch.from_numpy(o).to(dev)).cpu().numpy()[:, 0]
            for k, i in enumerate(range(sl.start, sl.stop)):
                t = ref_all[i].astype(np.float64)
                bound = min(ref_len, L)
                rows.append({"seed": seed, "position": str(d["position"][i]),
                             "mic": str(d["mic"][i]), "index": int(i),
                             "clap": CLAP, "bound": bound,
                             "input_padded_from_ms": ref_len / SR * 1000,
                             **score(t, est[k], bound)})
        print(f"  seed {seed}: {len(rows)} rows", flush=True)
        del m
        torch.cuda.empty_cache()
    (a.report_root / "results").mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = a.report_root / "results/per_example.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
