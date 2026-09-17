"""Provider-specific, provenance-preserving measured-RIR adapters."""
from __future__ import annotations

from pathlib import Path
import hashlib
import re

import soundfile as sf

from claprir.rir.metadata import PROJECT_SAMPLE_RATE, RIRRecord, stable_split


def _id(dataset: str, path: Path, channel: int) -> str:
    digest = hashlib.sha256(f"{dataset}:{path}:{channel}".encode()).hexdigest()[:16]
    return f"{dataset}:{digest}:ch{channel}"


def _record(path: Path, dataset: str, room: str, environment: str,
            configuration: str, source: str, receiver: str, channel: int,
            split_unit: str, split_type: str, channel_format: str,
            provenance: dict) -> RIRRecord:
    info = sf.info(path)
    return RIRRecord(
        record_id=_id(dataset, path, channel), dataset=dataset, room_id=room,
        configuration_id=configuration, source_id=source, receiver_id=receiver,
        channel_id=str(channel), original_file=str(path),
        original_sample_rate=info.samplerate, channel_format=channel_format,
        split=stable_split(f"{dataset}:{split_unit}", split_type),
        split_type=split_type, rir_duration=info.duration, rt60=None,
        provenance=provenance, environment_id=environment,
        normalized_sample_rate=PROJECT_SAMPLE_RATE,
    )


def scan_but(root: Path) -> list[RIRRecord]:
    """BUT top-level site/room is the room unit; mic/position remain subordinate."""
    records = []
    for path in sorted(root.rglob("*.wav")):
        if path.parent.name != "RIR":
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) < 6:
            continue
        room, microphone, source, position = parts[0], parts[1], parts[2], parts[3]
        records.append(_record(
            path, "but", room, room, position, source, microphone, 0, room,
            "room-held-out", "mono",
            {"adapter": "but", "relative_path": str(rel),
             "room_rule": "top-level BUT site directory"},
        ))
    return records


def scan_ace(root: Path) -> list[RIRRecord]:
    """ACE room directory is the room unit; array and position are not rooms."""
    records = []
    for path in sorted(root.rglob("*_RIR.wav")):
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) < 4:
            continue
        array, room, position = parts[0], parts[1], parts[2]
        info = sf.info(path)
        for channel in range(info.channels):
            records.append(_record(
                path, "ace", room, room, f"{array}:{position}", "unknown",
                array, channel, room, "room-held-out",
                "mono" if info.channels == 1 else f"physical-channel/{info.channels}",
                {"adapter": "ace", "relative_path": str(rel),
                 "array": array, "position": position,
                 "room_rule": "ACE corpus room directory"},
            ))
    return records


def scan_openair(root: Path) -> list[RIRRecord]:
    """Each downloaded OpenAIR archive directory is one environment/room."""
    records = []
    for environment_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        environment = environment_dir.name
        for path in sorted(environment_dir.rglob("*.wav")):
            rel = path.relative_to(environment_dir)
            lower_parts = {p.lower() for p in rel.parts}
            if "examples" in lower_parts:
                continue
            try:
                info = sf.info(path)
            except RuntimeError:
                continue
            configuration = str(rel.parent)
            match = re.search(r"[Ss](\d+)[_ -]?[Rr](\d+)", path.stem)
            source = f"S{match.group(1)}" if match else "unknown"
            receiver = f"R{match.group(2)}" if match else "unknown"
            for channel in range(info.channels):
                records.append(_record(
                    path, "openair", environment, environment, configuration,
                    source, receiver, channel, environment, "dataset-held-out",
                    "mono" if info.channels == 1 else f"physical-channel/{info.channels}",
                    {"adapter": "openair", "relative_path": str(rel),
                     "room_rule": "one downloaded OpenAIR environment archive"},
                ))
    return records


def scan_motus(root: Path) -> list[RIRRecord]:
    """MOTUS is one room; use Ambisonic channel 0 (W) deterministically."""
    records = []
    pattern = re.compile(r"(\d+)_(\d+)_raw_rirs")
    for path in sorted(root.glob("*.wav")):
        match = pattern.fullmatch(path.stem)
        if not match:
            continue
        configuration, source = match.groups()
        records.append(_record(
            path, "motus", "motus_single_room", "motus_single_room",
            configuration, f"loudspeaker{source}", "hoa_receiver", 0,
            configuration, "configuration-held-out-not-room-held-out",
            "HOA-32ch/channel0-W",
            {"adapter": "motus", "selected_channel": 0,
             "channel_rule": "first-order-independent HOA W channel",
             "room_rule": "MOTUS is one physical room"},
        ))
    return records


PROVIDER_SCANNERS = {
    "but": scan_but,
    "ace": scan_ace,
    "openair": scan_openair,
    "motus": scan_motus,
}
