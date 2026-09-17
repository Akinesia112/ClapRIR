"""Measured-RIR registry with local adapters and reproducible acquisition status."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import soundfile as sf

from claprir.rir.metadata import RIRRecord, estimate_rt60, stable_split
from claprir.rir.providers import PROVIDER_SCANNERS
from claprir.rir.sofa import scan_mrtd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: str
    url: str
    adapter: str
    expected_format: str
    access: str
    diversity: str


SPECS = {
    "arni": DatasetSpec("arni", "data/IR_Arni_upload_numClosed_0-5",
        "https://zenodo.org/records/6985104", "arni", "mono WAV", "public-large",
        "one-room/configuration-held-out"),
    "mit": DatasetSpec("mit", "data/MIT_RIR/Audio",
        "https://mcdermottlab.mit.edu/Reverb/IR_Survey.html", "mit", "mono WAV",
        "public", "multi-room/room-group-held-out"),
    "motus": DatasetSpec("motus", "data/MOTUS/extracted",
        "https://zenodo.org/records/4923187", "motus", "Ambisonic/SOFA",
        "public-large", "one-room/configuration-held-out"),
    "but": DatasetSpec("but", "data/BUT_ReverbDB/extracted",
        "https://speech.fit.vut.cz/software/but-speech-fit-reverb-database",
        "but", "WAV", "manual-license-or-provider", "multi-room"),
    "ace": DatasetSpec("ace", "data/ACE/extracted",
        "http://www.ee.ic.ac.uk/naylor/ACEweb/index.html", "ace", "WAV/MAT",
        "manual-license", "multi-room"),
    "mrtd": DatasetSpec("mrtd", "data/MRTD",
        "https://zenodo.org/records/13341566", "mrtd", "SOFA",
        "public-large", "three-environment/spatial"),
    "openair": DatasetSpec("openair", "data/OpenAIR/extracted",
        "https://www.openairlib.net/", "openair", "B-format/WAV",
        "provider-specific", "multi-environment"),
}


def _record(path: Path, dataset: str, room: str, config: str, source: str,
            receiver: str, channel: int, split_unit: str, split_type: str,
            *, analyze_decay: bool = False) -> RIRRecord:
    """Catalog one physical channel without decoding audio by default.

    Full-corpus scans must remain header-only: large, low-diversity providers such
    as Arni contain thousands of redundant measurements. Decay descriptors are
    computed later for the balanced materialized subset, where the waveform is
    already being decoded.
    """
    info = sf.info(path)
    split = stable_split(f"{dataset}:{split_unit}", split_type)
    rt60 = None
    if analyze_decay:
        x, sr = sf.read(path, always_2d=True)
        rt60 = estimate_rt60(x[:, channel], sr)
    return RIRRecord(
        record_id=f"{dataset}:{path.stem}:ch{channel}", dataset=dataset,
        room_id=room, configuration_id=config, source_id=source,
        receiver_id=receiver, channel_id=str(channel), original_file=str(path),
        original_sample_rate=info.samplerate,
        channel_format="mono" if info.channels == 1 else f"physical-channel/{info.channels}",
        split=split, split_type=split_type, rir_duration=info.duration,
        rt60=rt60,
        provenance={"adapter": SPECS[dataset].adapter,
                    "source_url": SPECS[dataset].url,
                    "decay_descriptor": "estimated-after-balanced-selection"},
    )


def scan_arni(root: Path, limit: int | None = None) -> list[RIRRecord]:
    pattern = re.compile(r"IR_numClosed_(\d+)_numComb_(\d+)_mic_(\d+)_sweep_(\d+)")
    out = []
    for path in sorted(root.glob("*.wav")):
        match = pattern.fullmatch(path.stem)
        if not match:
            continue
        closed, combination, mic, sweep = match.groups()
        config = f"closed{closed}_combination{combination}"
        out.append(_record(path, "arni", "aalto_arni", config, f"sweep{sweep}",
                           f"mic{mic}", 0, config, "configuration-held-out"))
        if limit and len(out) >= limit:
            break
    return out


def mit_room_group(stem: str) -> str:
    # Conservative filename grouping used by the supervised flow evaluation: strip terminal measurement indices.
    return re.sub(r"([_-]?(ir|rir|mic|src)?\d+)+$", "", stem.lower()).strip("_-") or stem.lower()


def scan_mit(root: Path, limit: int | None = None) -> list[RIRRecord]:
    out = []
    for path in sorted(root.glob("*.wav")):
        room = mit_room_group(path.stem)
        out.append(_record(path, "mit", room, "measured", "unknown", "unknown",
                           0, room, "room-group-held-out"))
        if limit and len(out) >= limit:
            break
    return out


def scan_generic(name: str, root: Path, limit: int | None = None) -> list[RIRRecord]:
    out = []
    for path in sorted(root.rglob("*.wav")):
        info = sf.info(path)
        room = path.parent.name or "unknown"
        for channel in range(info.channels):
            out.append(_record(path, name, room, path.stem, "unknown", "unknown",
                               channel, room, "room-group-held-out"))
            if limit and len(out) >= limit:
                return out
    return out


def scan_dataset(name: str, limit: int | None = None) -> list[RIRRecord]:
    spec = SPECS[name]
    root = Path(spec.root)
    if not root.exists():
        return []
    if spec.adapter == "arni":
        return scan_arni(root, limit)
    if spec.adapter == "mit":
        return scan_mit(root, limit)
    if spec.adapter == "mrtd":
        return scan_mrtd(root, limit)
    if spec.adapter in PROVIDER_SCANNERS:
        records = PROVIDER_SCANNERS[spec.adapter](root)
        return records[:limit] if limit else records
    return scan_generic(name, root, limit)
