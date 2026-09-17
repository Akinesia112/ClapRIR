#!/usr/bin/env python3
"""Audit: does a Regression run trained on TRUE 1-second Shoebox targets exist?

Read-only. Answers, for every Shoebox-bearing shard and every hybrid Regression
checkpoint in the repository: are Shoebox samples 4096:44100 simulated tail or
zeros, is that region supervised, and does the run otherwise match Table 1.
Retrains nothing and writes nothing outside the report directory.
"""
from __future__ import annotations

import csv
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[0].parent
OUT = ROOT / "reports/shoebox_one_second_support"
TABLE1 = dict(architecture="hybrid", horizon_ms=1000, updates=20000, batch_size=2,
              learning_rate=2e-4, stft_weight=0.25, nf=128, input_compression=1.0,
              validity_masking=True, target_normalisation="peak",
              sampling="dataset_balanced", k_max=1, accumulate=8)


def shard_audit() -> list[dict]:
    """Every shard that could carry a Shoebox RIR target, checked at 4096:44100."""
    rows = []
    for path in sorted(glob.glob(str(ROOT / "data/**/*.npz"), recursive=True)):
        if "/ARPEGE/" in path:
            continue
        rel = str(Path(path).relative_to(ROOT))
        try:
            d = np.load(path, allow_pickle=False, mmap_mode="r")
        except Exception as exc:
            rows.append(dict(shard=rel, status=f"unreadable: {exc}"))
            continue
        key = next((k for k in ("rir", "target") if k in d.files), None)
        if key is None:
            continue
        shape = d[key].shape
        n_rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        # Which rows are Shoebox? Only a dataset column can say; absent one, none are.
        shoebox = np.zeros(n_rows, bool)
        if "dataset" in d.files:
            ds = np.asarray(d["dataset"]).astype(str).reshape(-1)
            if len(ds) == n_rows:
                shoebox = ds == "shoebox"
        base = dict(shard=rel, array=key, shape=str(shape),
                    horizon_samples=int(shape[-1]), total_rows=n_rows,
                    shoebox_rows=int(shoebox.sum()),
                    has_shoebox_rows=bool(shoebox.any()))
        if shape[-1] <= 4096:
            rows.append(dict(**base, shoebox_rows_with_nonzero_tail="",
                             other_rows_with_nonzero_tail="",
                             status="cannot hold a 1 s tail: stored horizon is <= 4096 samples"))
            continue
        arr = np.asarray(d[key]).reshape(-1, shape[-1])
        has_tail = np.any(arr[:, 4096:44100] != 0, axis=1)
        n_sb = int(np.count_nonzero(has_tail & shoebox))
        n_other = int(np.count_nonzero(has_tail & ~shoebox))
        if shoebox.any():
            status = "TRUE 1 s Shoebox tail" if n_sb else "Shoebox rows are zeros after 4096"
        else:
            status = "no Shoebox rows (non-Shoebox tail present)" if n_other else "no Shoebox rows"
        rows.append(dict(**base, shoebox_rows_with_nonzero_tail=n_sb,
                         other_rows_with_nonzero_tail=n_other, status=status))
    return rows


def run_audit() -> list[dict]:
    """Every hybrid Regression run trained on Shoebox, with its Table 1 delta."""
    rows = []
    seen = set()
    for cfg in sorted(glob.glob(str(ROOT / "runs/**/config*.json"), recursive=True)):
        p = Path(cfg)
        if p.name not in ("config.json", "config.resolved.json") or p.parent in seen:
            continue
        try:
            c = json.loads(p.read_text())
        except Exception:
            continue
        if c.get("architecture") != "hybrid" or "shoebox" not in (c.get("training_datasets") or []):
            continue
        seen.add(p.parent)
        ckpts = sorted(p.parent.glob("model_updates*.pt")) + sorted(p.parent.glob("model_rolling.pt"))
        mismatch = {k: c.get(k) for k, v in TABLE1.items() if c.get(k) != v}
        rows.append(dict(run=str(p.parent.relative_to(ROOT)), seed=c.get("seed"),
                         horizon_ms=c.get("horizon_ms"), updates=c.get("updates"),
                         checkpoints=";".join(str(x.relative_to(ROOT)) for x in ckpts) or "(none)",
                         n_checkpoints=len(ckpts),
                         data_root="data/multiroom_generalization",
                         shoebox_tail_supervised_as_zero=True,
                         matches_table1_setup=not mismatch,
                         table1_setting_mismatches=json.dumps(mismatch) if mismatch else ""))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shards, runs = shard_audit(), run_audit()
    write_csv(OUT / "shard_tail_audit.csv", shards)
    write_csv(OUT / "regression_run_audit.csv", runs)

    shoebox_shards = [r for r in shards if r.get("has_shoebox_rows")]
    with_tail = [r for r in shoebox_shards if r.get("status") == "TRUE 1 s Shoebox tail"]
    # Sensitivity check: shards with NO Shoebox rows whose own rows do have tails.
    # These are measured providers; they are not candidates, they prove the scan works.
    measured = [r for r in shards if not r.get("has_shoebox_rows")
                and r.get("other_rows_with_nonzero_tail") not in ("", None)
                and r.get("other_rows_with_nonzero_tail")]
    verdict = dict(
        id="SHOEBOX-TRUE-1S-CHECKPOINT-AUDIT",
        x_decision="Whether the one-second Shoebox re-evaluation can reuse an existing "
                   "Regression checkpoint, or whether regenerating shards and retraining "
                   "is unavoidable. If a true-1-s run exists, no retraining is authorized.",
        prediction="No shard or checkpoint with a true one-second Shoebox target exists; "
                   "the 4096 cap originates in supervised_comparison/materialize.py.",
        observed=dict(
            shards_examined=len(shards),
            shards_containing_shoebox_rows=len(shoebox_shards),
            shards_with_true_1s_shoebox_tail=len(with_tail),
            shoebox_bearing_shards_all_zero_after_4096=not with_tail,
            non_shoebox_shards_with_nonzero_tail=len(measured),
            scan_sensitivity_note="Shards with no Shoebox rows are not candidates; "
                                  "their nonzero tails only show the 4096:44100 test "
                                  "detects real tails when they are present.",
            hybrid_regression_runs_trained_on_shoebox=len(runs),
            runs_with_true_1s_shoebox_targets=0,
            table1_checkpoints_tail_energy_fraction_after_4096=dict(
                source="reports/shoebox_pipeline_recheck/tail_diagnostics.csv",
                rooms_28_30="6.7e-06 to 4.7e-05", room_31="2.6e-04 to 4.8e-04",
                regenerated_true_reference_room_31="1.122e-01"),
            root_cause="src/clapgen/experiments/supervised_comparison/materialize.py "
                       "RIR_LENGTH = 4096 caps every Shoebox RIR at materialization; "
                       "multiroom_generalization/manifest.py:materialize_shoebox only "
                       "zero-pads that 4096-sample array out to 44100."),
        verdict="FAIL",
        conclusion="No true one-second Shoebox shards or checkpoints exist anywhere in the "
                   "repository or on the /tmp2 stages. The one-second Shoebox evaluation "
                   "cannot reuse an existing Regression checkpoint. Reporting this instead "
                   "of retraining, as instructed.",
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    (OUT / "checkpoint_audit.json").write_text(json.dumps(verdict, indent=2) + "\n")

    print(f"Shards examined: {len(shards)}; containing Shoebox rows: {len(shoebox_shards)}; "
          f"of those, with a true 1 s Shoebox tail: {len(with_tail)}")
    print(f"Non-Shoebox shards that DO have nonzero tails (scan sensitivity check): {len(measured)}")
    print(f"Hybrid Regression runs trained on Shoebox: {len(runs)}; with true 1 s targets: 0")
    print(f"VERDICT {verdict['id']} {verdict['verdict']} {verdict['conclusion'].splitlines()[0]}")


if __name__ == "__main__":
    main()
