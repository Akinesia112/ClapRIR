#!/usr/bin/env python3
"""Regenerate the Shoebox shard with true one-second ISM targets.

The frozen generator caps the image-source order at min(max_order, 8), which
yields 163-226 ms of RIR; materialize.py then truncates to RIR_LENGTH=4096 and
manifest.py zero-pads that out to 44100. This rebuilds all 32 rooms with the
order raised until the onset-trimmed response spans the full 44100 samples.

Every room parameter, source/receiver position, absorption value, split and
held-out clap assignment is preserved: max_order is not drawn from the room RNG,
and the clap draw depends only on the record id. Writes a NEW data root; the
frozen shard is never touched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyroomacoustics as pra

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / "src"))
from claprir.datasets.shoebox_rirs import delay_trim
from claprir.datasets.rir_shards import (LENGTH, SEED,
                                                                   synthesize)
from claprir.datasets.controlled_shards import load_claps

BANK_SEED = 23400
N_ROOMS = 32
FROZEN = ROOT / "data/multiroom_generalization/shoebox.npz"


def split_for(index: int) -> str:
    """rir_catalog()'s room-disjoint bank split, as materialize.py assigns it."""
    return "train" if index < 24 else "valid" if index < 28 else "test"


def room_parameters() -> list[dict]:
    """Replay build_shoebox_random's RandomState draw order exactly."""
    rng = np.random.RandomState(BANK_SEED)
    rooms = []
    for i in range(N_ROOMS):
        L, W, H = rng.uniform(3, 10), rng.uniform(3, 8), rng.uniform(2.4, 4)
        rt60 = rng.uniform(0.2, 0.7)
        e_abs, pra_max_order = pra.inverse_sabine(rt60, [L, W, H])
        src = [rng.uniform(.5, L - .5), rng.uniform(.5, W - .5), rng.uniform(1, H - .5)]
        mic = [rng.uniform(.5, L - .5), rng.uniform(.5, W - .5), rng.uniform(1, H - .5)]
        rooms.append(dict(index=i, dims=[L, W, H], rt60=float(rt60),
                          e_absorption=float(e_abs), pra_max_order=int(pra_max_order),
                          source=src, receiver=mic))
    return rooms


def simulate(room: dict, max_order: int) -> np.ndarray:
    r = pra.ShoeBox(room["dims"], fs=LENGTH, materials=pra.Material(room["e_absorption"]),
                    max_order=int(max_order))
    r.add_source(room["source"])
    r.add_microphone(np.array(room["receiver"]).reshape(3, 1))
    r.compute_rir()
    h = r.rir[0][0].astype(np.float32)
    return h / (np.max(np.abs(h)) + 1e-8)


def frame(trimmed: np.ndarray) -> np.ndarray:
    out = np.zeros(LENGTH, np.float32)
    out[:min(len(trimmed), LENGTH)] = trimmed[:LENGTH]
    return out / (np.max(np.abs(out)) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path,
                    default=ROOT / "data/multiroom_generalization_shoebox1s")
    ap.add_argument("--report-root", type=Path,
                    default=ROOT / "reports/shoebox_one_second_support")
    args = ap.parse_args()
    out_root, report = args.out_root.resolve(), args.report_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / "shoebox.npz"
    if target.exists():
        raise FileExistsError(f"regenerated shard exists; preserve it: {target}")

    frozen = np.load(FROZEN, allow_pickle=False)
    frozen_ids = frozen["record_id"].astype(str)
    claps = load_claps()
    rooms = room_parameters()

    arrays = {k: [] for k in ("clean", "rir", "observation", "split", "dataset",
                              "room_id", "configuration_id", "record_id")}
    manifest, checks = [], []
    for room in rooms:
        i = room["index"]
        record_id = f"shoebox_seed23400_{i:02d}"
        split = split_for(i)

        # The frozen order-8 room, regenerated, as a bit-exactness control.
        h8 = delay_trim(simulate(room, min(room["pra_max_order"], 8)))
        old = np.zeros(4096, np.float32)
        old[:min(len(h8), 4096)] = h8[:4096]
        old /= np.max(np.abs(old)) + 1e-8

        # The same room, resimulated until the ISM spans the full second.
        order = None
        for candidate in range(8, 241):
            trimmed = delay_trim(simulate(room, candidate))
            if len(trimmed) >= LENGTH:
                order, h1s_trimmed = candidate, trimmed
                break
        if order is None:
            raise RuntimeError(f"room {i}: no ISM order reaches one second")
        h = frame(h1s_trimmed)

        seed = int(hashlib.sha256(f"{SEED}:{record_id}".encode()).hexdigest()[:8], 16)
        clean, observation, clap_ids = synthesize(h, claps, split,
                                                  np.random.RandomState(seed))
        for key, value in (("clean", clean), ("rir", h), ("observation", observation),
                           ("split", split), ("dataset", "shoebox"),
                           ("room_id", f"shoebox_room_{i:02d}"),
                           ("configuration_id", "procedural"), ("record_id", record_id)):
            arrays[key].append(value)

        j = int(np.flatnonzero(frozen_ids == record_id)[0])
        assert str(frozen["split"][j]) == split, (i, split, frozen["split"][j])
        rir_delta = float(np.max(np.abs(old - frozen["rir"][j][:4096])))
        clap_delta = float(np.max(np.abs(np.asarray(clean) - frozen["clean"][j])))
        tail = h[4096:]
        checks.append(dict(room_index=i, record_id=record_id, split=split,
            rt60_s=room["rt60"], pra_inverse_sabine_max_order=room["pra_max_order"],
            frozen_max_order=min(room["pra_max_order"], 8),
            frozen_native_samples=len(h8), frozen_native_ms=1000.0 * len(h8) / LENGTH,
            one_second_max_order=order, one_second_native_samples=len(h1s_trimmed),
            order8_4096_max_abs_diff_vs_frozen=rir_delta,
            clap_assignment_max_abs_diff_vs_frozen=clap_delta,
            tail_4096_nonzero=int(np.count_nonzero(tail)), tail_4096_total=int(len(tail)),
            tail_fully_simulated=bool(np.all(tail != 0)),
            energy_fraction_past_4096=float(np.sum(h[4096:].astype(np.float64) ** 2)
                                            / np.sum(h.astype(np.float64) ** 2))))
        manifest.append(dict(dataset="shoebox", record_id=record_id,
            room_id=f"shoebox_room_{i:02d}", configuration_id="procedural",
            partition=split, split_type="room-disjoint", clap_ids=";".join(clap_ids),
            rt60=room["rt60"], source_path="procedural:one_second_ism",
            max_order=order, length_m=room["dims"][0], width_m=room["dims"][1],
            height_m=room["dims"][2], e_absorption=room["e_absorption"],
            source_xyz=";".join(f"{v:.6f}" for v in room["source"]),
            receiver_xyz=";".join(f"{v:.6f}" for v in room["receiver"])))
        print(f"room {i:>2} [{split:<5}] rt60={room['rt60']:.3f} order 8->{order:<3} "
              f"native {len(h8):>5}->{len(h1s_trimmed):>6} samples, "
              f"past-4096 energy {checks[-1]['energy_fraction_past_4096']:.3e}, "
              f"control diff {rir_delta:g}/{clap_delta:g}", flush=True)

    np.savez(target, **{k: np.asarray(v) for k, v in arrays.items()})

    max_rir = max(c["order8_4096_max_abs_diff_vs_frozen"] for c in checks)
    max_clap = max(c["clap_assignment_max_abs_diff_vs_frozen"] for c in checks)
    assert max_rir == 0.0, f"generator drift: {max_rir}"
    assert max_clap == 0.0, f"clap assignment drift: {max_clap}"
    assert all(c["tail_fully_simulated"] for c in checks)

    report.mkdir(parents=True, exist_ok=True)
    keys = list(checks[0])
    with (report / "regenerated_shard_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader(); w.writerows(checks)
    keys = list(manifest[0])
    with (report / "regenerated_data_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader(); w.writerows(manifest)
    summary = dict(shard=str(target.relative_to(ROOT)), rooms=N_ROOMS,
        horizon_samples=LENGTH, bank_seed=BANK_SEED, clap_seed_namespace=SEED,
        splits={s: sum(c["split"] == s for c in checks) for s in ("train", "valid", "test")},
        ism_order_range=[min(c["one_second_max_order"] for c in checks),
                         max(c["one_second_max_order"] for c in checks)],
        order8_control_bit_exact=True, clap_assignment_bit_exact=True,
        all_tails_fully_simulated=True,
        energy_past_4096=dict(
            min=min(c["energy_fraction_past_4096"] for c in checks),
            median=float(np.median([c["energy_fraction_past_4096"] for c in checks])),
            max=max(c["energy_fraction_past_4096"] for c in checks)),
        rooms_with_over_1pct_past_4096=sum(
            c["energy_fraction_past_4096"] > 0.01 for c in checks),
        frozen_shard_untouched=True,
        shard_bytes=target.stat().st_size,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    (report / "regenerated_shard.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"COMPLETE {target}", flush=True)


if __name__ == "__main__":
    main()
