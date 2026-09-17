"""Canonical, provenance-preserving metadata for measured mono RIR channels."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import scipy.signal as sps
import soundfile as sf

PROJECT_SAMPLE_RATE = 44_100


@dataclass(frozen=True)
class RIRRecord:
    record_id: str
    dataset: str
    room_id: str
    configuration_id: str
    source_id: str
    receiver_id: str
    channel_id: str
    original_file: str
    original_sample_rate: int
    channel_format: str
    split: str
    split_type: str
    rir_duration: float
    rt60: float | None = None
    source_position: tuple[float, ...] | None = None
    receiver_position: tuple[float, ...] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    environment_id: str | None = None
    normalized_sample_rate: int = PROJECT_SAMPLE_RATE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_split(key: str, split_type: str, train: int = 70, validation: int = 15) -> str:
    """Deterministic semantic-unit split; callers pass room or configuration IDs."""
    value = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return ("train" if value < train else
            "validation" if value < train + validation else "test")


def estimate_rt60(waveform: np.ndarray, sample_rate: int) -> float | None:
    """T30-derived RT60 estimate; return None if a usable -5..-35 dB span is absent."""
    energy = np.asarray(waveform, np.float64) ** 2
    if not np.any(energy):
        return None
    edc = np.cumsum(energy[::-1])[::-1]
    db = 10 * np.log10(np.maximum(edc / edc[0], 1e-12))
    use = np.flatnonzero((db <= -5) & (db >= -35))
    if len(use) < 20:
        return None
    slope, _ = np.polyfit(use / sample_rate, db[use], 1)
    value = -60 / slope if slope < 0 else np.nan
    return float(value) if np.isfinite(value) and 0.03 <= value <= 20 else None


def load_mono(record: RIRRecord, target_rate: int = PROJECT_SAMPLE_RATE) -> np.ndarray:
    """Load the recorded physical channel deterministically and peak-scale per RIR."""
    channel = int(record.channel_id)
    if Path(record.original_file).suffix.lower() == ".sofa":
        from claprir.rir.sofa import load_sofa_channel
        mono, sr = load_sofa_channel(record)
    else:
        x, sr = sf.read(record.original_file, always_2d=True)
        if channel >= x.shape[1]:
            raise ValueError(f"channel {channel} absent from {record.original_file}")
        mono = np.asarray(x[:, channel], np.float64)
    if sr != target_rate:
        mono = sps.resample_poly(mono, target_rate, sr)
    peak = np.max(np.abs(mono))
    if not np.isfinite(peak) or peak <= 1e-12:
        raise ValueError(f"silent/non-finite RIR: {record.original_file}")
    return (mono / peak).astype(np.float32)


def manifest_hash(records: list[RIRRecord]) -> str:
    payload = json.dumps([r.to_dict() for r in records], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
