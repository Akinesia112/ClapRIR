"""Frozen noise generation shared by every noisy controlled estimator."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/noisy_controlled_benchmark'
SR = LENGTH = 44100
EVAL_SNRS = (20., 30., 40.)
LAMBDA_GRID = (0., 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6,
               1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
PROVIDERS = ('shoebox', 'mit', 'but', 'ace', 'openair')
TRAIN_PROVIDERS = PROVIDERS[:4]
SEEDS = (42001, 42002, 42003)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rng_for(key):
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')
    return np.random.Generator(np.random.PCG64(seed))


def add_noise(clean, rng, snr):
    clean = np.asarray(clean, dtype=np.float64)
    if clean.shape != (LENGTH,) or not np.isfinite(clean).all():
        raise ValueError('Noise requires exactly 44100 finite observation samples')
    rms = float(np.sqrt(np.mean(clean ** 2)))
    if rms <= 0 or not np.isfinite(snr):
        raise ValueError('Noise scaling requires a nonzero clean observation and finite SNR')
    noise = rng.standard_normal(LENGTH)
    noise -= noise.mean()
    noise *= rms * 10. ** (-float(snr) / 20.) / np.sqrt(np.mean(noise ** 2))
    return (clean + noise).astype(np.float32)


def noisy_observation(clean, record_key, draw, snr):
    if int(draw) not in (0, 1, 2):
        raise ValueError('Evaluation has exactly three predeclared draws')
    return add_noise(clean, rng_for(f'noisy-controlled-v1:{record_key}:draw{int(draw)}'), snr)


def training_observation(clean, training_seed, ordinal):
    rng = rng_for(f'noisy-controlled-train-v1:seed{int(training_seed)}:example{int(ordinal)}')
    snr = float(rng.uniform(20., 40.))
    return add_noise(clean, rng, snr), snr


def noise_metadata(clean, noisy, snr):
    clean, noisy = np.asarray(clean, np.float64), np.asarray(noisy, np.float64)
    residual = noisy - clean
    a, b = np.sqrt(np.mean(clean ** 2)), np.sqrt(np.mean(residual ** 2))
    actual = float(20 * np.log10(a / b))
    if abs(actual - float(snr)) > 1e-3:
        raise RuntimeError(f'Noise SNR mismatch: {actual} vs {snr}')
    return dict(requested_snr_db=float(snr), actual_snr_db=actual,
                clean_rms=float(a), noise_rms=float(b), noise_mean=float(residual.mean()),
                input_sha256=array_sha(noisy))


def verify_lock(report=None):
    report = Path(report or REPORT)
    lock = read_json(report / 'protocol_lock.json')
    if sha(report / 'protocol.json') != lock['protocol_sha256']:
        raise RuntimeError('Frozen noisy protocol changed')
    for relative, digest in lock['source_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError('Frozen noisy source changed: ' + relative)
    return lock
