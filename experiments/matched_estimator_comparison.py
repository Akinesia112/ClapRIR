#!/usr/bin/env python3
"""E3: the matched 1 s Regression against the matched 1 s Flow.

publication/EXPERIMENTS.md registers this as the paper's only model-winner table.
Everything except the estimator is matched; see that file for the contract and
for why the sampler is deliberately NOT matched.

Two things this file refuses to do, both registered in advance:

  * Flow noise draws are sampling uncertainty, NOT extra rooms. The five draws of
    one checkpoint are aggregated within (training_seed, room) BEFORE any
    interval is computed, so the statistical unit stays the room and the Flow
    does not get five times the sample size for having a sampler.
  * NRMSE takes part in no verdict. It is computed and printed as a companion
    because it is the number the old paper turned on, and it is excluded from the
    decision because that paper already contains a case where waveform error
    improved while decay got worse.

Decision order, frozen: EDC -> C50/EDT -> EDP -> log-spectrum.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from claprir.models.flow_sampler import logsnr_grid, _churn_step                  # noqa: E402
from claprir.metrics.lundeby_truncation import edc_truncated                                  # noqa: E402
from claprir.metrics.room_acoustics import diagnosis_metrics
from claprir.training.train_rir_estimator import RunConfig, load_model

SR = 44_100
PROVIDERS = ("shoebox", "mit", "but", "ace", "openair")
SYNTHETIC = {"shoebox"}
SEEDS = (42_001, 42_002, 42_003)
DRAWS = 5
NFE, GAMMA = 20, 0.5
JUDGEABLE = ("mit", "openair")        # the only providers with rooms to resample
BOOTSTRAP = 10_000
FLOW = "runs/flow_1s_aux", "hybrid_flow_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{}"
REG = "runs/matched_reg_1s", "hybrid_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{}"
#: frozen hierarchy; the bool is "participates in the verdict"
METRICS = (("edc_rmse_db", "EDC RMSE (dB)", True),
           ("abs_c50_error_db", "|C50 error| (dB)", True),
           ("abs_edt_error_s", "|EDT error| (s)", True),
           ("echo_density_rmse", "EDP RMSE", True),
           ("stft_logmag_mse", "STFT log-mag MSE", True),
           ("nrmse", "NRMSE (companion)", False))


@torch.no_grad()
def flow_sample(model, obs, sigma_d, gen, *, state_callback=None, stop_index=None):
    """Frozen E3 sampling; optional read-only trace or detached prefix return."""
    ts = logsnr_grid(NFE, sigma_d=sigma_d)
    x = torch.randn((obs.shape[0], 1, model.config.signal_length),
                    device=obs.device, generator=gen)
    for i in range(len(ts) - 1):
        if stop_index is not None and i == stop_index:
            return x
        t = float(ts[i])
        v = model.velocity(x, obs, torch.full((obs.shape[0],), t, device=obs.device))
        if state_callback is not None:
            state_callback(t, x, v)
        x = _churn_step(x, -v, 1.0 - ts[i], 1.0 - ts[i + 1], GAMMA, generator=gen)
    return x


def load(root: Path, name: str, dev):
    cfg = json.loads((root / name / "config.resolved.json").read_text())
    cfg["training_datasets"] = tuple(cfg["training_datasets"])
    c = RunConfig(**cfg)
    if c.run_name != name:
        raise SystemExit(f"run_name drift: {c.run_name} != {name}")
    if not (root / name / "model_updates20000.pt").exists():
        raise SystemExit(f"missing checkpoint for {name}; this harness will not train one.")
    return load_model(c, 20_000, dev, root, root), c


def score(target, est, bound) -> dict:
    t, e = np.asarray(target, np.float64), np.asarray(est, np.float64)
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
    ap.add_argument("--report-root", type=Path, default=Path("reports/matched_estimator_comparison"))
    ap.add_argument("--sigma-d", type=float, default=0.027279)   # validity-aware, 1 s
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (a.report_root / "results").mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        fm, _ = load(a.repo_root / FLOW[0], FLOW[1].format(seed), dev)
        rm, _ = load(a.repo_root / REG[0], REG[1].format(seed), dev)
        gen = torch.Generator(device=dev).manual_seed(20260907 + seed)
        for prov in PROVIDERS:
            d = np.load(a.data_root / f"{prov}.npz", mmap_mode="r", allow_pickle=False)
            idx = np.flatnonzero(d["split"].astype(str) == "test")
            for s in range(0, len(idx), 4):
                ch = idx[s:s + 4]
                tgt = np.asarray(d["rir"][ch, :SR], np.float32)
                obs = torch.from_numpy(
                    np.asarray(d["observation"][ch, 0, :SR], np.float32))[:, None].to(dev)
                reg = rm.predict(obs).cpu().numpy()[:, 0]
                flows = [flow_sample(fm, obs, a.sigma_d, gen).cpu().numpy()[:, 0]
                         for _ in range(DRAWS)]
                for k, i in enumerate(ch):
                    t = tgt[k].astype(np.float64)
                    nz = np.nonzero(t)[0]
                    tv = SR if prov in SYNTHETIC else ((int(nz[-1]) + 1) if len(nz) else 1)
                    bound = max(min(SR, tv), 1)
                    base = {"seed": seed, "provider": prov,
                            "room_id": str(d["room_id"][i]),
                            "sample_id": str(d["record_id"][i]), "bound": bound}
                    rows.append({**base, "arm": "regression", "draw": 0,
                                 **score(t, reg[k], bound)})
                    for j, f in enumerate(flows):
                        rows.append({**base, "arm": "flow", "draw": j,
                                     **score(t, f[k], bound)})
            print(f"  seed {seed} {prov}: {len(rows)} rows", flush=True)
        del fm, rm
        torch.cuda.empty_cache()
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (a.report_root / "results/per_example.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, keys, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows "
          f"({len(SEEDS)} seeds x 220 rooms x (1 regression + {DRAWS} flow draws))")


if __name__ == "__main__":
    main()
