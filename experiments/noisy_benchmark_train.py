#!/usr/bin/env python3
"""Noise-only augmentation of the selected one-second Regression training.

Imports the original architecture, sampler, objective and optimizer settings.
The local loop adds durable complete resume state and noise provenance only.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
import numpy as np
import torch
from claprir.training.train_rir_estimator import (
    RunConfig, SingleClapStore, build_model, write_csv)
from scripts.queue.noisy_benchmark_common import (
    ROOT, REPORT, read_json, write_json, sha, training_observation, verify_lock)

STOP = False


def stopping(signum, frame):
    global STOP
    STOP = True


class NoisyStore(SingleClapStore):
    def __init__(self, config, data_root):
        super().__init__(config.training_datasets, 'train', config.signal_length,
                         data_root, config.subset, config.sampling,
                         validity_masking=config.validity_masking)
        self.training_seed = config.seed
        self.ordinal = 0
        self.last_snrs = []

    def sample(self, batch_size, rng):
        target, observation, validity = super().sample(batch_size, rng)
        augmented, snrs = [], []
        for row in observation.numpy()[:, 0]:
            value, snr = training_observation(row, self.training_seed, self.ordinal)
            augmented.append(value[None]); snrs.append(snr); self.ordinal += 1
        self.last_snrs = snrs
        return target, torch.from_numpy(np.stack(augmented)), validity


def copy_atomic(source, target):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.copy.tmp')
    shutil.copy2(source, temporary); temporary.replace(target)


def train(args):
    if not Path.cwd().resolve().is_relative_to(Path('/tmp')):
        raise RuntimeError('GPU processes must run from the local /tmp checkout')
    torch.set_num_threads(2)
    report = args.durable / 'reports/noisy_controlled_benchmark'
    lock = verify_lock()
    protocol = read_json(REPORT / 'protocol.json')
    parent_sha = sha(REPORT / 'protocol_lock.json')
    if parent_sha != sha(report / 'protocol_lock.json'):
        raise RuntimeError('Local and durable protocol locks differ')
    settings = protocol['regression']['configs'][str(args.seed)]
    cfg = RunConfig(**dict(settings, training_datasets=tuple(settings['training_datasets'])))
    output = args.runtime / cfg.run_name
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / 'training.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    durable_run = args.durable / 'runs/noisy_regression_1s' / cfg.run_name
    durable_run.mkdir(parents=True, exist_ok=True)
    state_path = report / 'training' / f'seed{args.seed}' / 'state.json'
    contract = dict(parent_lock_sha256=parent_sha, seed=args.seed,
                    config=settings, noise=protocol['noise'], data_sha256=lock['data_sha256'])
    for folder in (output, durable_run):
        for name, value in [('config.resolved.json', settings), ('noise_contract.json', contract)]:
            path = folder / name
            if path.exists() and read_json(path) != value:
                raise RuntimeError('Existing run contract mismatch: ' + str(path))
            write_json(path, value)
    for provider in cfg.training_datasets:
        relative = f'data/multiroom_generalization/{provider}.npz'
        if sha(ROOT / relative) != lock['data_sha256'][relative]:
            raise RuntimeError('Training data changed: ' + provider)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    # Preserve original backend policy; do not silently introduce a new
    # determinism/precision intervention. Complete RNG state is still saved.
    device = torch.device('cuda')
    store = NoisyStore(cfg, ROOT / 'data/multiroom_generalization')
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.learning_rate)
    rng = np.random.RandomState(cfg.seed)
    history, start = [], 0
    last = output / 'model_rolling.pt'
    if not last.exists() and (durable_run / last.name).exists():
        copy_atomic(durable_run / last.name, last)
    if last.exists():
        checkpoint = torch.load(last, map_location='cpu', weights_only=False)
        if checkpoint['noise_contract'] != contract or checkpoint['config'] != asdict(cfg):
            raise RuntimeError('Resume checkpoint contract mismatch')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, history = checkpoint['update'], checkpoint['history']
        rng.set_state(checkpoint['sampler_rng'])
        store.ordinal = checkpoint['noise_ordinal']
        torch.set_rng_state(checkpoint['rng']['torch'])
        torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        np.random.set_state(checkpoint['rng']['numpy'])
        random.setstate(checkpoint['rng']['python'])
        if store.ordinal != start * cfg.batch_size * cfg.accumulate:
            raise RuntimeError('Resume augmentation stream mismatch')
        del checkpoint
    model.train()
    if hasattr(model, 'backbone'):
        model.backbone.tf_stage.eval()
    begun = time.monotonic()
    write_json(state_path, dict(status='training', update=start, target_updates=cfg.updates,
        pid=os.getpid(), seed=args.seed, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
        cwd=str(Path.cwd()), parent_lock_sha256=parent_sha))
    end = min(args.stop_after or cfg.updates, cfg.updates)
    for update in range(start + 1, end + 1):
        optimizer.zero_grad()
        total, mean_snr, logparts = 0., 0., {}
        for micro in range(cfg.accumulate):
            target, observation, validity = store.sample(cfg.batch_size, rng)
            loss, components = model.loss(target.to(device), observation.to(device),
                                          validity_mask=validity.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite loss at update {update}')
            (loss / cfg.accumulate).backward()
            total += float(loss.detach()) / cfg.accumulate
            mean_snr += float(np.mean(store.last_snrs)) / cfg.accumulate
            for key, value in components.items():
                logparts[key] = logparts.get(key, 0.) + float(value) / cfg.accumulate
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 25 == 0 or update == end:
            row = dict(update=update, loss=total, gradient_norm=float(grad),
                mean_snr_db=mean_snr, elapsed_session_seconds=time.monotonic()-begun, **logparts)
            history.append(row); print(json.dumps(dict(seed=args.seed, **row)), flush=True)
            write_json(state_path, dict(status='training', update=update, target_updates=cfg.updates,
                seed=args.seed, pid=os.getpid(), gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
                latest_loss=total, mean_snr_db=mean_snr, elapsed_session_seconds=time.monotonic()-begun,
                parent_lock_sha256=parent_sha))
        if update == 1 or update % 100 == 0 or update == end or STOP:
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                config=asdict(cfg), update=update, parameters=sum(p.numel() for p in model.parameters()),
                noise_contract=contract, sampler_rng=rng.get_state(), noise_ordinal=store.ordinal,
                history=history, rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
            temporary = last.with_suffix('.tmp'); torch.save(payload, temporary); temporary.replace(last)
            copy_atomic(last, durable_run / last.name)
            checkpoint_sha = sha(last)
            if sha(durable_run / last.name) != checkpoint_sha:
                raise RuntimeError('Durable checkpoint copy differs')
            if update in (5000, 10000, 20000):
                copy_atomic(last, durable_run / f'model_updates{update}.pt')
                if update == cfg.updates:
                    copy_atomic(last, output / f'model_updates{update}.pt')
            write_csv(output / 'training.csv', history)
            copy_atomic(output / 'training.csv', durable_run / 'training.csv')
            write_json(state_path, dict(status='trained-terminal' if update == cfg.updates else
                'interrupted-checkpointed' if STOP else 'checkpointed', update=update,
                target_updates=cfg.updates, seed=args.seed, pid=os.getpid(),
                gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], latest_loss=total,
                elapsed_session_seconds=time.monotonic()-begun, checkpoint=str(durable_run / last.name),
                checkpoint_sha256=checkpoint_sha, noise_ordinal=store.ordinal,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                backend=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), deterministic=torch.are_deterministic_algorithms_enabled(),
                    cudnn_benchmark=torch.backends.cudnn.benchmark), parent_lock_sha256=parent_sha))
            del payload
        if STOP:
            return 75
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(42001,42002,42003), required=True)
    parser.add_argument('--durable', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stopping); signal.signal(signal.SIGINT, stopping)
    raise SystemExit(train(args))
