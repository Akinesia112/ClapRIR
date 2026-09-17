#!/usr/bin/env python3
"""Supervised comparison supervised experiments.

This module deliberately excludes Joint (x,h), DPS-h, and the H-coordinate
transform.  It provides one data contract and one training path for:

* deterministic y -> x regression;
* time/complex-spectrum conditional flow matching;
* deterministic complex-spectrum regression;
* direct y -> h prediction; and
* one or more clap observations of the same (or perturbed) RIR -> h.

The input dataset is an ``.npz`` shard with float arrays:

``clean[N,T_x]``, ``rir[N,T_h]``, ``observation[N,K,T_y]``.

K may be one.  A ``split[N]`` string array is strongly recommended and must
contain ``train``, ``valid``, or ``test``.  Dataset construction is kept
separate from training so room-disjoint splits can be audited before a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import scipy.signal as sps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _fit_length(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] >= length:
        return x[..., :length]
    return F.pad(x, (0, length - x.shape[-1]))


class ComplexRFFT:
    """Invertible real/imaginary two-channel representation."""

    def __init__(self, signal_length: int):
        self.signal_length = signal_length

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.fft.rfft(_fit_length(x, self.signal_length), n=self.signal_length)
        return torch.cat((z.real, z.imag), dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[1] != 2:
            raise ValueError(f"complex tensor must have two channels, got {z.shape}")
        c = torch.complex(z[:, :1], z[:, 1:2])
        return torch.fft.irfft(c, n=self.signal_length)


def perturb_rir(
    rir: np.ndarray,
    rng: np.random.RandomState,
    max_delay: float = 0.75,
    eq_db: float = 1.5,
    noise_db: float = -55.0,
) -> np.ndarray:
    """Apply mild delay/EQ/noise without changing array length."""
    h = np.asarray(rir, np.float32)
    n = h.size
    # Fractional delay in the Fourier domain.
    delay = rng.uniform(-max_delay, max_delay)
    spec = np.fft.rfft(h)
    freq = np.fft.rfftfreq(n)
    spec *= np.exp(-2j * np.pi * freq * delay)
    # Smooth tilt is a conservative proxy for source-position / mic EQ change.
    tilt = rng.uniform(-eq_db, eq_db)
    spec *= 10.0 ** ((tilt * np.linspace(-0.5, 0.5, spec.size)) / 20.0)
    out = np.fft.irfft(spec, n=n).astype(np.float32)
    rms = np.sqrt(np.mean(out**2) + 1e-12)
    out += rng.randn(n).astype(np.float32) * rms * 10.0 ** (noise_db / 20.0)
    return out


class SupervisedShard(Dataset):
    """Auditable NPZ-backed samples for all Supervised comparison arms."""

    def __init__(
        self,
        path: str,
        split: str,
        target: str,
        num_claps: int = 1,
        perturb: bool = False,
        seed: int = 0,
    ):
        data = np.load(path, allow_pickle=False)
        required = {"clean", "rir", "observation"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        obs = data["observation"]
        if obs.ndim == 2:
            obs = obs[:, None, :]
        if obs.ndim != 3:
            raise ValueError("observation must have shape [N,K,T] or [N,T]")
        if num_claps < 1 or num_claps > obs.shape[1]:
            raise ValueError(f"requested {num_claps} claps, shard has {obs.shape[1]}")
        if target not in {"clean", "rir"}:
            raise ValueError("target must be clean or rir")
        splits = data["split"].astype(str) if "split" in data else np.full(len(obs), "train")
        self.indices = np.flatnonzero(splits == split)
        if not len(self.indices):
            raise ValueError(f"{path}: split {split!r} is empty")
        self.clean = data["clean"]
        self.rir = data["rir"]
        self.obs = obs
        self.target = target
        self.num_claps = num_claps
        self.perturb = perturb
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        i = int(self.indices[item])
        cond = self.obs[i, : self.num_claps].astype(np.float32, copy=True)
        if self.perturb and self.num_claps > 1:
            # Re-synthesize each observation from its clean excitation if the
            # shard provides clean_multi. Otherwise fail instead of silently
            # perturbing an already-convolved waveform.
            rng = np.random.RandomState(self.seed + i)
            clean_i = self.clean[i]
            if clean_i.ndim == 1:
                raise ValueError("perturbed multi-clap requires clean[N,K,T_x]")
            rows = []
            for k in range(self.num_claps):
                hk = perturb_rir(self.rir[i], rng)
                rows.append(sps.fftconvolve(clean_i[k], hk)[: cond.shape[-1]])
            cond = np.asarray(rows, np.float32)
        target = self.clean[i]
        if self.target == "clean" and target.ndim == 2:
            target = target[0]
        if self.target == "rir":
            target = self.rir[i]
        return {
            "condition": torch.from_numpy(cond),
            "target": torch.from_numpy(np.asarray(target, np.float32))[None],
            "rir": torch.from_numpy(np.asarray(self.rir[i], np.float32))[None],
        }


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(width, 2 * width, 3, padding=dilation, dilation=dilation)
        self.proj = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(min(8, width), width)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.conv(F.silu(self.norm(x)) + context[..., None])
        gate, value = h.chunk(2, dim=1)
        return x + self.proj(torch.sigmoid(gate) * torch.tanh(value))


class ConditionalNet(nn.Module):
    """Fully convolutional backbone shared by regression and flow arms."""

    def __init__(self, state_channels: int, condition_channels: int, width=64, blocks=11):
        super().__init__()
        self.state_channels = state_channels
        self.condition_channels = condition_channels
        self.in_proj = nn.Conv1d(state_channels + condition_channels, width, 1)
        self.time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(
            ResidualBlock(width, 2 ** (i % 10)) for i in range(blocks)
        )
        self.out = nn.Conv1d(width, state_channels, 1)

    def forward(self, state: torch.Tensor, condition: torch.Tensor, t: torch.Tensor):
        if condition.shape[-1] != state.shape[-1]:
            condition = F.interpolate(condition, size=state.shape[-1], mode="linear",
                                      align_corners=False)
        h = self.in_proj(torch.cat((state, condition), dim=1))
        context = self.time(t[:, None])
        for block in self.blocks:
            h = block(h, context)
        return self.out(F.silu(h))


@dataclass
class ExperimentConfig:
    name: str
    shard: str
    output_dir: str
    target: str = "clean"
    representation: str = "time"
    method: str = "regression"
    num_claps: int = 1
    perturbed_h: bool = False
    signal_length: int = 4096
    target_length: int = 4096
    batch_size: int = 16
    epochs: int = 100
    lr: float = 2e-4
    stft_weight: float = 1.0
    width: int = 64
    blocks: int = 11
    seed: int = 20260728
    flow_steps: int = 50

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        return cls(**json.loads(Path(path).read_text()))

    def validate(self) -> None:
        if self.target not in {"clean", "rir"}:
            raise ValueError("target must be clean or rir")
        if self.representation not in {"time", "complex"}:
            raise ValueError("representation must be time or complex")
        if self.method not in {"regression", "flow"}:
            raise ValueError("method must be regression or flow")
        if self.perturbed_h and (self.target != "rir" or self.num_claps < 2):
            raise ValueError("perturbed_h is only valid for multi-clap y -> h")


def compressed_stft_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n_fft = min(510, a.shape[-1])
    if n_fft % 2:
        n_fft -= 1
    hop = max(1, n_fft // 4)
    win = torch.hann_window(n_fft, device=a.device)
    aa = torch.stft(a.squeeze(1), n_fft, hop, window=win, return_complex=True)
    bb = torch.stft(b.squeeze(1), n_fft, hop, window=win, return_complex=True)
    return F.mse_loss((aa.abs() + 1e-8) ** 0.667, (bb.abs() + 1e-8) ** 0.667)


class Experiment(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.rep = ComplexRFFT(cfg.target_length) if cfg.representation == "complex" else None
        state_channels = 2 if self.rep else 1
        condition_channels = cfg.num_claps * (2 if self.rep else 1)
        self.net = ConditionalNet(state_channels, condition_channels, cfg.width, cfg.blocks)

    def prepare(self, target: torch.Tensor, condition: torch.Tensor):
        target = _fit_length(target, self.cfg.target_length)
        condition = _fit_length(condition, self.cfg.signal_length)
        if self.rep:
            target = self.rep.encode(target)
            parts = [self.rep.encode(condition[:, k : k + 1]) for k in range(condition.shape[1])]
            condition = torch.cat(parts, dim=1)
        return target, condition

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.rep.decode(state) if self.rep else state

    def loss(self, target: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        target_state, cond_state = self.prepare(target, condition)
        batch = target.shape[0]
        if self.cfg.method == "regression":
            pred_state = self.net(torch.zeros_like(target_state), cond_state,
                                  torch.ones(batch, device=target.device))
            primary = F.mse_loss(pred_state, target_state)
        else:
            t = torch.rand(batch, device=target.device)
            noise = torch.randn_like(target_state)
            state = t[:, None, None] * target_state + (1 - t[:, None, None]) * noise
            velocity = self.net(state, cond_state, t)
            primary = F.mse_loss(velocity, target_state - noise)
            pred_state = state + (1 - t[:, None, None]) * velocity
        pred = self.decode(pred_state)
        truth = _fit_length(target, self.cfg.target_length)
        spectral = compressed_stft_loss(pred, truth)
        total = primary + self.cfg.stft_weight * spectral
        return total, {"primary": float(primary.detach()), "stft": float(spectral.detach())}

    @torch.no_grad()
    def predict(self, condition: torch.Tensor) -> torch.Tensor:
        dummy = torch.zeros(condition.shape[0], 1, self.cfg.target_length,
                            device=condition.device)
        state, cond_state = self.prepare(dummy, condition)
        if self.cfg.method == "regression":
            state = self.net(state, cond_state, torch.ones(condition.shape[0],
                                                           device=condition.device))
        else:
            state = torch.randn_like(state)
            step = 1.0 / self.cfg.flow_steps
            for k in range(self.cfg.flow_steps):
                t = torch.full((condition.shape[0],), k * step, device=condition.device)
                state = state + step * self.net(state, cond_state, t)
        return self.decode(state)


def metrics(reference: torch.Tensor, estimate: torch.Tensor) -> Dict[str, float]:
    r = reference.flatten(1)
    e = estimate.flatten(1)
    err = e - r
    nrmse = torch.sqrt(torch.mean(err**2, 1)) / (torch.sqrt(torch.mean(r**2, 1)) + 1e-8)
    alpha = torch.sum(e * r, 1, keepdim=True) / (torch.sum(r * r, 1, keepdim=True) + 1e-8)
    desired = alpha * r
    residual = e - desired
    sisdr = 10 * torch.log10(
        (torch.sum(desired**2, 1) + 1e-8) / (torch.sum(residual**2, 1) + 1e-8)
    )
    R = torch.fft.rfft(r)
    E = torch.fft.rfft(e)
    lsd = torch.sqrt(torch.mean(
        (20 * torch.log10(E.abs() + 1e-7) - 20 * torch.log10(R.abs() + 1e-7)) ** 2, 1
    ))
    complex_rmse = torch.sqrt(torch.mean(torch.view_as_real(E - R) ** 2, (1, 2)))
    return {
        "nrmse": float(nrmse.mean()),
        "si_sdr_db": float(sisdr.mean()),
        "lsd_db": float(lsd.mean()),
        "complex_rmse": float(complex_rmse.mean()),
    }


def _loader(cfg: ExperimentConfig, split: str, shuffle=False) -> DataLoader:
    ds = SupervisedShard(cfg.shard, split, cfg.target, cfg.num_claps,
                         cfg.perturbed_h, cfg.seed)
    return DataLoader(ds, cfg.batch_size, shuffle=shuffle, num_workers=0)


def train(cfg: ExperimentConfig, device: torch.device) -> Path:
    cfg.validate()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    model = Experiment(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    log_path = out / "train.csv"
    with log_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", "loss", "primary", "stft"])
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            rows = []
            for batch in _loader(cfg, "train", shuffle=True):
                target = batch["target"].to(device)
                condition = batch["condition"].to(device)
                loss, parts = model.loss(target, condition)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                rows.append((float(loss.detach()), parts["primary"], parts["stft"]))
            mean = np.mean(rows, axis=0)
            writer.writerow(dict(epoch=epoch, loss=mean[0], primary=mean[1], stft=mean[2]))
    checkpoint = out / "model.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, checkpoint)
    return checkpoint


def evaluate(cfg: ExperimentConfig, checkpoint: str, device: torch.device) -> Dict:
    model = Experiment(cfg).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    aggregate = []
    start = time.perf_counter()
    count = 0
    with torch.no_grad():
        for batch in _loader(cfg, "test"):
            target = _fit_length(batch["target"].to(device), cfg.target_length)
            estimate = model.predict(batch["condition"].to(device))
            aggregate.append(metrics(target, estimate))
            count += len(target)
    elapsed = time.perf_counter() - start
    keys = aggregate[0]
    result = {k: float(np.mean([row[k] for row in aggregate])) for k in keys}
    result.update(examples=count, runtime_ms_per_example=1000 * elapsed / count,
                  name=cfg.name, method=cfg.method, representation=cfg.representation,
                  target=cfg.target, num_claps=cfg.num_claps,
                  perturbed_h=cfg.perturbed_h)
    out = Path(cfg.output_dir) / "test_metrics.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_smoke_shard(path: str, seed=0) -> None:
    """Small deterministic fixture; explicitly not a scientific dataset."""
    rng = np.random.RandomState(seed)
    n, k, tx, th, ty = 18, 3, 128, 96, 192
    clean = np.zeros((n, k, tx), np.float32)
    rir = np.zeros((n, th), np.float32)
    obs = np.zeros((n, k, ty), np.float32)
    for i in range(n):
        for j in range(k):
            clean[i, j, 8:20] = rng.randn(12) * np.hanning(12)
        rir[i, 0] = 1
        rir[i, 12:72] = rng.randn(60) * np.exp(-np.arange(60) / 14) * 0.1
        for j in range(k):
            obs[i, j] = np.convolve(clean[i, j], rir[i])[:ty]
    split = np.array(["train"] * 12 + ["valid"] * 3 + ["test"] * 3)
    np.savez(path, clean=clean, rir=rir, observation=obs, split=split)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("make-smoke-shard")
    smoke.add_argument("path")
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.command == "make-smoke-shard":
        make_smoke_shard(args.path)
        return
    cfg = ExperimentConfig.load(args.config)
    device = torch.device(args.device)
    checkpoint = Path(cfg.output_dir) / "model.pt"
    if not args.eval_only:
        checkpoint = train(cfg, device)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    print(json.dumps(evaluate(cfg, str(checkpoint), device), indent=2))


if __name__ == "__main__":
    main()
