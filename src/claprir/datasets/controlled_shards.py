#!/usr/bin/env python3
"""Materialize frozen, leakage-controlled Supervised comparison NPZ shards."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import scipy.signal as sps
import soundfile as sf
import torch

from claprir.rir.signals import FS, convolve_clap_rir, make_rt60_rir
from claprir.datasets.shoebox_rirs import build_shoebox_bank, delay_trim

ROOT = Path("reports/supervised_comparison")
RESULTS = ROOT / "results"
DATA = Path("data/supervised_comparison")
SEED = 2342026
SIGNAL_LENGTH = 4096
CLAP_SUPPORT = 882
RIR_LENGTH = 4096
K_MAX = 3


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_claps() -> list[dict]:
    split = json.loads(Path("data/real_claps/split.json").read_text())
    participant_split = {int(p): "train" for p in split["train"]}
    participant_split.update({int(p): "test" for p in split["test"]})
    rows = []
    for row in csv.DictReader(open("data/real_claps/metadata.csv")):
        if row["tier"] != "clean" or int(row["primary_channel"]) != 3:
            continue
        participant = int(row["participant"])
        path = Path(
            f"data/real_claps/participant{participant:02d}/"
            f"block{int(row['block_idx']):02d}/clap{int(row['clap_idx']):02d}.wav"
        )
        x, sr = sf.read(path, always_2d=True)
        if sr != FS:
            raise RuntimeError(f"unexpected clap rate {sr}: {path}")
        x = x[:CLAP_SUPPORT, 3].astype(np.float32)
        x /= np.max(np.abs(x)) + 1e-8
        padded = np.zeros(SIGNAL_LENGTH, np.float32)
        padded[:len(x)] = x
        rows.append({
            "id": f"P{participant:02d}_B{int(row['block_idx']):02d}_C{int(row['clap_idx']):02d}",
            "participant": participant,
            "participant_split": participant_split[participant],
            "path": str(path),
            "waveform": padded,
        })
    return rows


def load_wav_rir(path: Path) -> np.ndarray:
    x, sr = sf.read(path, always_2d=True)
    x = x[:, 0]
    if sr != FS:
        x = sps.resample_poly(x, FS, sr)
    x = delay_trim(np.asarray(x, np.float32))
    out = np.zeros(RIR_LENGTH, np.float32)
    out[:min(len(x), RIR_LENGTH)] = x[:RIR_LENGTH]
    out /= np.max(np.abs(out)) + 1e-8
    return out


def rir_catalog() -> dict[str, list[dict]]:
    catalog: dict[str, list[dict]] = defaultdict(list)
    # Fresh procedural rooms are split by room bank, never by observation.
    for index, rir in enumerate(build_shoebox_bank(32, seed=23400)):
        split = "train" if index < 24 else "valid" if index < 28 else "test"
        catalog["shoebox"].append({
            "dataset": "shoebox", "rir_id": f"shoebox_seed23400_{index:02d}",
            "group_id": f"shoebox_room_{index:02d}", "split": split,
            "split_type": "room-disjoint", "path": "procedural", "waveform": delay_trim(rir),
        })
    mit_rows = list(csv.DictReader(open("configs/rir_datasets/mit_room_split.csv")))
    for row in mit_rows:
        partition = row["partition"]
        split = "valid" if partition == "validation" else partition
        catalog["mit"].append({
            "dataset": "mit", "rir_id": row["rir_identifier"],
            "group_id": row["normalized_room_group"], "split": split,
            "split_type": "conservative-room-group-disjoint", "path": row["path"],
        })
    # Keep the frozen namespace so renaming the experiment does not silently
    # change the published Arni split.
    legacy_arni_split_namespace = "priority234-arni"
    # Strongest internal Arni unit: acoustic configuration (closed, combination).
    import re
    pattern = re.compile(
        r"IR_numClosed_(?P<closed>\d+)_numComb_(?P<comb>\d+)_mic_(?P<mic>\d+)_sweep_(?P<sweep>\d+)\.wav"
    )
    for path in sorted(Path("data/IR_Arni_upload_numClosed_0-5").glob("*.wav")):
        match = pattern.fullmatch(path.name)
        if not match or match["sweep"] != "2":
            continue
        group = f"closed{match['closed']}_comb{match['comb']}"
        value = int(_sha(f"{legacy_arni_split_namespace}:{group}")[:8], 16) % 10
        split = "train" if value < 6 else "valid" if value < 8 else "test"
        catalog["arni"].append({
            "dataset": "arni", "rir_id": path.stem, "group_id": group, "split": split,
            "split_type": "configuration-held-out-not-room-held-out", "path": str(path),
        })
    for index, rt60 in enumerate(np.linspace(.2, .8, 16)):
        catalog["diffuse"].append({
            "dataset": "diffuse", "rir_id": f"diffuse_{index:02d}_rt60_{rt60:.3f}",
            "group_id": f"diffuse_seed_{index:02d}", "split": "test",
            "split_type": "OOD-only", "path": "procedural",
            "waveform": make_rt60_rir(rt60, n=RIR_LENGTH,
                                      rng=np.random.RandomState(SEED + index)),
        })
    return catalog


def normalize_rir(item: dict) -> np.ndarray:
    if "waveform" in item:
        source = np.asarray(item["waveform"], np.float32)
        source = delay_trim(source)
        out = np.zeros(RIR_LENGTH, np.float32)
        out[:min(len(source), RIR_LENGTH)] = source[:RIR_LENGTH]
        out /= np.max(np.abs(out)) + 1e-8
        return out
    return load_wav_rir(Path(item["path"]))


def materialize(name: str, rirs: list[dict], claps: list[dict], n: int,
                split_label: str, seed: int) -> tuple[Path, list[dict]]:
    rng = np.random.RandomState(seed)
    selected_rirs = [r for r in rirs if r["split"] == split_label]
    if not selected_rirs:
        raise RuntimeError(f"{name}: no RIRs in {split_label}")
    clap_pool = [c for c in claps if c["participant_split"] ==
                 ("train" if split_label in {"train", "valid"} else "test")]
    clean = np.zeros((n, K_MAX, SIGNAL_LENGTH), np.float32)
    rir_array = np.zeros((n, RIR_LENGTH), np.float32)
    observation = np.zeros((n, K_MAX, SIGNAL_LENGTH), np.float32)
    split = np.full(n, split_label)
    rir_ids, group_ids, example_ids = [], [], []
    manifest = []
    for index in range(n):
        item = selected_rirs[index % len(selected_rirs)]
        h = normalize_rir(item)
        indices = rng.choice(len(clap_pool), K_MAX, replace=False)
        for k, clap_index in enumerate(indices):
            x = clap_pool[int(clap_index)]["waveform"]
            y = sps.fftconvolve(x[:CLAP_SUPPORT], h)[:SIGNAL_LENGTH]
            y /= np.max(np.abs(y)) + 1e-8
            clean[index, k] = x
            observation[index, k] = y
        rir_array[index] = h
        rir_ids.append(item["rir_id"])
        group_ids.append(item["group_id"])
        example_id = f"{name}_{index:04d}"
        example_ids.append(example_id)
        manifest.append({
            "shard": name, "example_id": example_id, "partition": split_label,
            "dataset": item["dataset"], "rir_id": item["rir_id"],
            "group_id": item["group_id"], "split_type": item["split_type"],
            "source_path": item["path"], "clap_ids": ";".join(clap_pool[int(i)]["id"] for i in indices),
            "sample_rate": FS, "signal_length": SIGNAL_LENGTH, "rir_length": RIR_LENGTH,
        })
    path = DATA / f"{name}.npz"
    np.savez(path, clean=clean, rir=rir_array, observation=observation, split=split,
             rir_id=np.asarray(rir_ids), group_id=np.asarray(group_ids),
             example_id=np.asarray(example_ids), dataset=np.full(n, rirs[0]["dataset"]))
    return path, manifest


def assert_no_leakage(rows: list[dict]) -> None:
    locations: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        locations[(row["dataset"], row["group_id"])].add(row["partition"])
    leaked = {key: values for key, values in locations.items() if len(values) > 1}
    if leaked:
        raise RuntimeError(f"RIR/room group leakage: {leaked}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    claps = load_claps()
    catalog = rir_catalog()
    plan = [
        ("train", "shoebox", "train", 384, 10),
        ("val", "shoebox", "valid", 64, 11),
        ("test_sim_seen", "shoebox", "train", 64, 12),
        ("test_sim_heldout", "shoebox", "test", 64, 13),
        ("train_mit", "mit", "train", 256, 14),
        ("val_mit_room_disjoint", "mit", "valid", 64, 15),
        ("test_mit_room_disjoint", "mit", "test", 64, 16),
        ("train_arni", "arni", "train", 256, 17),
        ("test_arni_configuration", "arni", "test", 64, 18),
        ("test_diffuse_rt60", "diffuse", "test", 64, 19),
    ]
    rows = []
    for name, source, partition, count, offset in plan:
        path = DATA / f"{name}.npz"
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists; use --force for deterministic rebuild")
        _, part = materialize(name, catalog[source], claps, count, partition, SEED + offset)
        rows.extend(part)
    # Leakage applies within each dataset and semantic group. Seen sanity intentionally
    # reuses training rooms, so exclude it from disjoint-partition assertion.
    assert_no_leakage([r for r in rows if r["shard"] != "test_sim_seen"])
    _write_csv(RESULTS / "data_manifest.csv", rows)
    split_rows = [{
        "dataset": item["dataset"], "rir_id": item["rir_id"], "group_id": item["group_id"],
        "partition": item["split"], "split_type": item["split_type"], "source_path": item["path"],
    } for values in catalog.values() for item in values]
    _write_csv(RESULTS / "split_manifest.csv", split_rows)
    summary = {
        "seed": SEED, "sample_rate": FS, "signal_length": SIGNAL_LENGTH,
        "clap_support": CLAP_SUPPORT, "rir_length": RIR_LENGTH, "k_max": K_MAX,
        "normalization": "clean/h/y independently peak-normalized; onset trim at 50% RIR peak",
        "claps_available": len(claps),
        "shards": {name: count for name, _, _, count, _ in plan},
        "catalog": {name: len(items) for name, items in catalog.items()},
        "manifest_sha256": hashlib.sha256((RESULTS / "data_manifest.csv").read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.cuda.is_available(), "numpy": np.__version__,
                    "scipy": scipy.__version__},
    }
    (RESULTS / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
