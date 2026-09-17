"""SOFA adapters for measured room impulse response datasets."""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from claprir.rir.metadata import PROJECT_SAMPLE_RATE, RIRRecord, stable_split


MRTD_PATTERN = re.compile(
    r"(?P<environment>.+)_(?P<receiver>kemar|zoom)_ls_(?P<source>\d+)\.sofa"
)


def scan_mrtd(root: Path, limit: int | None = None) -> list[RIRRecord]:
    """Expose each MRTD measurement/receiver channel as one RIR record."""
    records: list[RIRRecord] = []
    for path in sorted(root.glob("*.sofa")):
        match = MRTD_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        environment = match["environment"]
        receiver = match["receiver"]
        source = f"loudspeaker{match['source']}"
        with h5py.File(path, "r") as sofa:
            measurements, channels, samples = map(int, sofa["Data.IR"].shape)
            sample_rate = int(np.asarray(sofa["Data.SamplingRate"])[0])
            source_positions = np.asarray(sofa["SourcePosition"])
            receiver_positions = np.asarray(sofa["ReceiverPosition"])
        for measurement in range(measurements):
            source_position = tuple(float(v) for v in source_positions[measurement])
            for channel in range(channels):
                receiver_position = tuple(
                    float(v) for v in receiver_positions[min(channel, len(receiver_positions) - 1)]
                )
                records.append(RIRRecord(
                    record_id=(
                        f"mrtd:{environment}:{receiver}:{source}:"
                        f"measurement{measurement}:channel{channel}"
                    ),
                    dataset="mrtd",
                    room_id=environment,
                    environment_id=environment,
                    configuration_id=f"{receiver}:{source}:measurement{measurement}",
                    source_id=source,
                    receiver_id=receiver,
                    channel_id=str(channel),
                    original_file=str(path),
                    original_sample_rate=sample_rate,
                    normalized_sample_rate=PROJECT_SAMPLE_RATE,
                    channel_format=f"SOFA/{receiver}/channel{channel}",
                    split=stable_split(f"mrtd:{environment}", "environment-held-out"),
                    split_type="environment-held-out",
                    rir_duration=samples / sample_rate,
                    source_position=source_position,
                    receiver_position=receiver_position,
                    provenance={
                        "adapter": "mrtd_sofa",
                        "measurement_index": measurement,
                        "source_url": "https://zenodo.org/records/13341566",
                    },
                ))
                if limit and len(records) >= limit:
                    return records
    return records


def load_sofa_channel(record: RIRRecord) -> tuple[np.ndarray, int]:
    """Load the exact SOFA measurement and physical receiver channel."""
    measurement = int(record.provenance["measurement_index"])
    channel = int(record.channel_id)
    with h5py.File(record.original_file, "r") as sofa:
        waveform = np.asarray(sofa["Data.IR"][measurement, channel], np.float64)
        sample_rate = int(np.asarray(sofa["Data.SamplingRate"])[0])
    return waveform, sample_rate
