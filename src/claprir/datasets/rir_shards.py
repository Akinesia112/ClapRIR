#!/usr/bin/env python3
"""Freeze semantic provider manifests and materialize the one-second study contract."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import scipy.signal as sps
import torch

from claprir.datasets.controlled_shards import load_claps
from claprir.datasets.shoebox_rirs import delay_trim
from claprir.rir.metadata import RIRRecord, estimate_rt60, load_mono
from claprir.rir.registry import scan_dataset

FS = 44100
LENGTH = 44100
K_MAX = 5
SEED = 42000
DATA = Path("data/multiroom_generalization")
ROOT = Path("reports/multiroom_generalization")
RESULTS = ROOT / "results"
PROVIDERS = ("mit", "but", "ace", "openair", "arni", "motus", "mrtd")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def semantic_partitions(units: list[str]) -> dict[str, str]:
    ordered = sorted(set(units), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    n = len(ordered)
    n_train = max(1, int(round(.65 * n)))
    n_valid = max(1, int(round(.15 * n))) if n >= 3 else 0
    if n_train + n_valid >= n and n > 1:
        n_train = n - 1 - n_valid
    return {unit: ("train" if i < n_train else "validation" if i < n_train + n_valid else "test")
            for i, unit in enumerate(ordered)}


def assign_splits(dataset: str, records: list[RIRRecord]) -> list[RIRRecord]:
    if dataset == "openair":
        return [dataclasses.replace(r, split="test", split_type="provider-held-out") for r in records]
    if dataset == "mit":
        rows = list(csv.DictReader(open("configs/rir_datasets/mit_room_split.csv")))
        mapping = {row["normalized_room_group"]: row["partition"] for row in rows}
        return [dataclasses.replace(r, split=mapping.get(r.room_id, r.split),
                                    split_type="conservative-room-group-disjoint") for r in records]
    if dataset in {"but", "ace"}:
        mapping = semantic_partitions([r.room_id for r in records])
        return [dataclasses.replace(r, split=mapping[r.room_id], split_type="room-disjoint")
                for r in records]
    return records


def scan_all() -> dict[str, list[RIRRecord]]:
    return {dataset: assign_splits(dataset, scan_dataset(dataset)) for dataset in PROVIDERS}


def assert_no_leakage(records: list[RIRRecord]) -> None:
    units = defaultdict(set)
    for r in records:
        unit = r.configuration_id if "configuration-held-out" in r.split_type else r.room_id
        units[(r.dataset, unit)].add(r.split)
    leaked = {k: sorted(v) for k, v in units.items() if len(v) > 1}
    if leaked:
        raise RuntimeError(f"semantic leakage: {leaked}")
    if any(r.dataset == "openair" and r.split != "test" for r in records):
        raise RuntimeError("OpenAIR provider-held-out isolation violated")


def cap_records(records: list[RIRRecord], cap_per_room: int = 16) -> list[RIRRecord]:
    groups = defaultdict(list)
    for r in records:
        groups[(r.dataset, r.room_id, r.split)].append(r)
    selected = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: hashlib.sha256(r.record_id.encode()).hexdigest())
        selected.extend(rows[:cap_per_room])
    return selected


def fit_rir(record: RIRRecord) -> tuple[np.ndarray, float | None]:
    full = load_mono(record, FS)
    rt60 = estimate_rt60(full, FS)
    full = delay_trim(full)
    out = np.zeros(LENGTH, np.float32)
    out[:min(LENGTH, len(full))] = full[:LENGTH]
    out /= np.max(np.abs(out)) + 1e-8
    return out, rt60


def synthesize(h: np.ndarray, claps: list[dict], split: str,
               rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray, list[str]]:
    participant = "train" if split in {"train", "validation"} else "test"
    pool = [c for c in claps if c["participant_split"] == participant]
    picks = rng.choice(len(pool), K_MAX, replace=False)
    clean = np.stack([pool[int(i)]["waveform"] for i in picks]).astype(np.float32)
    observation = []
    for x in clean:
        y = sps.fftconvolve(x[:882], h)[:LENGTH].astype(np.float32)
        y /= np.max(np.abs(y)) + 1e-8
        observation.append(y)
    return clean, np.stack(observation), [pool[int(i)]["id"] for i in picks]


def materialize_provider(dataset: str, records: list[RIRRecord], claps: list[dict]) -> list[dict]:
    clean_rows, h_rows, y_rows, split_rows, room_rows, config_rows, id_rows = [], [], [], [], [], [], []
    manifest = []
    for index, record in enumerate(records):
        h, rt60 = fit_rir(record)
        seed = int(hashlib.sha256(f"{SEED}:{record.record_id}".encode()).hexdigest()[:8], 16)
        clean, observation, clap_ids = synthesize(h, claps, record.split, np.random.RandomState(seed))
        clean_rows.append(clean); h_rows.append(h); y_rows.append(observation)
        split_rows.append("valid" if record.split == "validation" else record.split)
        room_rows.append(record.room_id); config_rows.append(record.configuration_id)
        id_rows.append(record.record_id)
        manifest.append({"dataset": dataset, "record_id": record.record_id,
                         "room_id": record.room_id, "configuration_id": record.configuration_id,
                         "partition": record.split, "split_type": record.split_type,
                         "clap_ids": ";".join(clap_ids), "rt60": rt60,
                         "source_path": record.original_file})
    np.savez(DATA / f"{dataset}.npz", clean=np.asarray(clean_rows), rir=np.asarray(h_rows),
             observation=np.asarray(y_rows), split=np.asarray(split_rows),
             dataset=np.full(len(records), dataset), room_id=np.asarray(room_rows),
             configuration_id=np.asarray(config_rows), record_id=np.asarray(id_rows))
    return manifest


def materialize_shoebox(claps: list[dict]) -> list[dict]:
    sources = [("train", "data/supervised_comparison/train.npz"),
               ("valid", "data/supervised_comparison/val.npz"),
               ("test", "data/supervised_comparison/test_sim_heldout.npz")]
    rows = []
    arrays = {k: [] for k in ("clean", "rir", "observation", "split", "dataset",
                              "room_id", "configuration_id", "record_id")}
    for split, path in sources:
        data = np.load(path, allow_pickle=False)
        seen = set()
        for i, room in enumerate(data["group_id"].astype(str)):
            if room in seen:
                continue
            seen.add(room)
            source_h = np.asarray(data["rir"][i], np.float32)
            h = np.zeros(LENGTH, np.float32)
            h[:min(LENGTH, len(source_h))] = source_h[:LENGTH]
            record_id = str(data["rir_id"][i])
            seed = int(hashlib.sha256(f"{SEED}:{record_id}".encode()).hexdigest()[:8], 16)
            clean, observation, clap_ids = synthesize(h, claps, split, np.random.RandomState(seed))
            for key, value in (("clean", clean), ("rir", h), ("observation", observation),
                               ("split", split), ("dataset", "shoebox"), ("room_id", room),
                               ("configuration_id", "procedural"), ("record_id", record_id)):
                arrays[key].append(value)
            rows.append({"dataset": "shoebox", "record_id": record_id, "room_id": room,
                         "configuration_id": "procedural", "partition": split,
                         "split_type": "room-disjoint", "clap_ids": ";".join(clap_ids),
                         "rt60": "", "source_path": path})
    np.savez(DATA / "shoebox.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    if (RESULTS / "data_summary.json").exists() and not args.force:
        raise FileExistsError("frozen manifest exists; pass --force only for deterministic verification")
    catalogs = scan_all()
    all_records = [r for values in catalogs.values() for r in values]
    assert_no_leakage(all_records)
    record_rows = [r.to_dict() for r in all_records]
    write_csv(RESULTS / "rir_manifest.csv", record_rows)
    split_rows = []
    for r in all_records:
        unit = r.configuration_id if "configuration-held-out" in r.split_type else r.room_id
        split_rows.append({"dataset": r.dataset, "semantic_unit": unit,
                           "room_id": r.room_id, "configuration_id": r.configuration_id,
                           "partition": r.split, "split_type": r.split_type})
    unique = {(r["dataset"], r["semantic_unit"], r["partition"]): r for r in split_rows}
    write_csv(RESULTS / "split_manifest.csv", list(unique.values()))
    inventory = []
    for dataset, records in catalogs.items():
        inventory.append({"dataset": dataset, "independent_rooms": len({r.room_id for r in records}),
                          "environments": len({r.environment_id for r in records}),
                          "configurations": len({r.configuration_id for r in records}),
                          "rirs": len(records), "channels": ";".join(sorted({r.channel_format for r in records})),
                          "sample_rates": ";".join(map(str, sorted({r.original_sample_rate for r in records}))),
                          "split_type": ";".join(sorted({r.split_type for r in records}))})
    write_csv(RESULTS / "provider_inventory.csv", inventory)
    write_csv(RESULTS / "room_manifest.csv", [{"dataset": d, "room_id": room,
              "split": next(r.split for r in records if r.room_id == room)}
              for d, records in catalogs.items() for room in sorted({r.room_id for r in records})])
    claps = load_claps()
    data_rows = materialize_shoebox(claps)
    selected_counts = {"shoebox": len(data_rows)}
    for dataset, records in catalogs.items():
        selected = cap_records(records)
        part = materialize_provider(dataset, selected, claps)
        data_rows.extend(part); selected_counts[dataset] = len(part)
    write_csv(RESULTS / "data_manifest.csv", data_rows)
    digest = hashlib.sha256()
    for name in ("rir_manifest.csv", "split_manifest.csv", "data_manifest.csv"):
        digest.update((RESULTS / name).read_bytes())
    summary = {"schema_version": 1, "seed": SEED, "sample_rate": FS,
               "signal_length": LENGTH, "k_max": K_MAX,
               "manifest_sha256": digest.hexdigest(), "provider_counts": {d: len(r) for d,r in catalogs.items()},
               "materialized_counts": selected_counts, "dataset_held_out": "openair",
               "normalization": "deterministic physical mono channel, polyphase resample, onset trim, per-RIR and per-observation peak scale",
               "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                           "cuda": torch.cuda.is_available(), "numpy": np.__version__, "scipy": scipy.__version__}}
    (RESULTS / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
