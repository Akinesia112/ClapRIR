#!/usr/bin/env python3
"""Frozen post-meeting Table 1: three analytic controls and selected Regression.

CPU only. Reads frozen shards and prior matching Regression scores; writes only
reports/single_clap_benchmark. Never invokes training or another estimator.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from claprir.metrics.room_acoustics import (echo_density_profile, edt_seconds,
                                         stft_magnitude_errors)
from claprir.metrics.deconvolution import c50_db, regularized_deconvolution
from claprir.metrics.lundeby_truncation import edc_truncated

SR = 44100
PROVIDERS = ("shoebox", "mit", "but", "ace", "openair")
SEEDS = (42001, 42002, 42003)
ARMS = ("known_clap_unregularized", "known_clap_tikhonov",
        "crop_3ms_tikhonov", "regression_1s")
LABELS = dict(zip(ARMS, ("Known clap, unregularized", "Known clap, Tikhonov",
                         "3 ms crop + Tikhonov", "1 s Regression")))
METRICS = ("edc_rmse_db", "abs_c50_error_db", "abs_edt_error_s",
           "echo_density_rmse", "stft_logmag_mse", "nrmse")
HEADERS = ("EDC RMSE (dB)", "C50 abs. error (dB)", "EDT abs. error (s)",
           "EDP RMSE", "STFT log-mag MSE", "NRMSE")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def reference_features(t):
    t32 = t.astype(np.float32)
    return dict(edc=edc_truncated(t, len(t)),
                eta=echo_density_profile(t32, SR)[1],
                c50=c50_db(t32, SR), edt=edt_seconds(t32, SR))


def score(t, e, features):
    t, e = np.asarray(t, np.float64), np.asarray(e, np.float64)
    t32, e32 = t.astype(np.float32), e.astype(np.float32)
    est_edc = edc_truncated(e, len(e))
    eta = echo_density_profile(e32, SR)[1]
    return dict(edc_rmse_db=float(np.sqrt(np.mean((est_edc - features["edc"]) ** 2))),
                abs_c50_error_db=abs(float(c50_db(e32, SR) - features["c50"])),
                abs_edt_error_s=abs(float(edt_seconds(e32, SR) - features["edt"])),
                echo_density_rmse=float(np.sqrt(np.mean((eta - features["eta"]) ** 2))),
                stft_logmag_mse=stft_magnitude_errors(t32, e32)["stft_logmag_mse"],
                nrmse=float(np.linalg.norm(e - t) / (np.linalg.norm(t) + 1e-12)))


def summarize(rows, out):
    # Preserve the established order: records within seed/room, then seeds.
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["arm"], r["provider"], r["room_id"], str(r["seed"]))].append(r)
    seed_rows = []
    for (arm, provider, room, seed), group in sorted(grouped.items()):
        row = dict(arm=arm, provider=provider, room_id=room, seed=seed,
                   n_records=len(group))
        for m in METRICS:
            vals = [float(r[m]) for r in group if np.isfinite(float(r[m]))]
            row[m] = float(np.median(vals)) if vals else np.nan
            row[f"n_finite_{m}"] = len(vals)
        seed_rows.append(row)
    write_csv(out / "per_seed_room.csv", seed_rows)
    rooms = defaultdict(list)
    for r in seed_rows:
        rooms[(r["arm"], r["provider"], r["room_id"])].append(r)
    room_rows = []
    for (arm, provider, room), group in sorted(rooms.items()):
        row = dict(arm=arm, provider=provider, room_id=room,
                   statistical_unit=f"{provider}:{room}", n_seeds=len(group),
                   n_records=group[0]["n_records"])
        for m in METRICS:
            vals = [r[m] for r in group if np.isfinite(r[m])]
            row[m] = float(np.median(vals)) if vals else np.nan
            row[f"n_finite_seeds_{m}"] = len(vals)
        room_rows.append(row)
    write_csv(out / "per_room.csv", room_rows)
    rng = np.random.default_rng(20260911)
    summaries = []
    for population in ("measured_pooled", *PROVIDERS):
        for arm in ARMS:
            group = [r for r in room_rows if r["arm"] == arm and
                     (r["provider"] != "shoebox" if population == "measured_pooled"
                      else r["provider"] == population)]
            row = dict(population=population, arm=arm, n_rooms=len(group),
                       n_records=sum(r["n_records"] for r in group))
            for m in METRICS:
                vals = np.array([r[m] for r in group if np.isfinite(r[m])])
                row[m] = float(np.median(vals)) if len(vals) else np.nan
                row[f"n_finite_rooms_{m}"] = len(vals)
                # Descriptive room bootstrap only. No provider-population claim.
                boot = np.median(vals[rng.integers(0, len(vals), (10000, len(vals)))], axis=1)
                row[f"{m}_ci95_low"], row[f"{m}_ci95_high"] = np.quantile(boot, [.025, .975])
            summaries.append(row)
    write_csv(out / "summary.csv", summaries)
    main_rows = [r for r in summaries if r["population"] in ("measured_pooled", "shoebox")]
    def markdown(table_rows):
        lines = ["| Population | Method | Rooms | " + " | ".join(HEADERS) + " |",
                 "|---|---|---:|" + "---:|" * len(METRICS)]
        for r in table_rows:
            lines.append("| " + " | ".join([r["population"], LABELS[r["arm"]],
                str(r["n_rooms"])] + [f"{r[m]:.6g}" for m in METRICS]) + " |")
        return "\n".join(lines) + "\n"
    (out / "table1.md").write_text(markdown(main_rows))
    (out / "provider_breakdown.md").write_text(markdown([r for r in summaries if r["population"] != "measured_pooled"]))
    tex = ["% Generated by experiments/single_clap_benchmark.py; no provider medians are averaged.",
           r"\begin{table*}[t]", r"\centering\scriptsize",
           r"\caption{Single-clap estimation on the frozen one-second benchmark. Measured RIRs pool 54 independent provider/room units (216 RIR records); Shoebox is separate (4 rooms). Record medians are computed within each training seed and room, then medians across the three fixed Regression seeds, then across rooms. Analytic controls have no training seed. Every metric uses $\min(44100,T_{\rm valid})$ samples; Shoebox uses its retained 4096 samples (92.88 ms), not the padded tail. EDT uses 51 measured rooms because three MIT records have undefined target EDT; all other metrics use all 54 measured rooms. All errors are lower-is-better; NRMSE is a companion metric. Known-clap controls use privileged excitation.}",
           r"\label{tab:post-meeting-table1}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Population & Method & EDC (dB) & $|\Delta C_{50}|$ (dB) & $|\Delta\mathrm{EDT}|$ (s) & EDP & Log-mag MSE & NRMSE \\", r"\midrule"]
    for r in main_rows:
        pop = "Measured pooled" if r["population"] == "measured_pooled" else "Shoebox (92.88 ms)"
        tex.append(" & ".join([pop, LABELS[r["arm"]]] + [f"{r[m]:.4g}" for m in METRICS]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "table1.tex").write_text("\n".join(tex) + "\n")
    return main_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-root", type=Path, default=ROOT / "reports/single_clap_benchmark")
    a = ap.parse_args()
    out = a.report_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Completed output exists; preserve it: {result_path}")
    source_paths = ["experiments/excitation_recoverability.py", "experiments/matched_estimator_comparison.py",
        "experiments/matched_estimator_report.py", "reports/excitation_recoverability/result.json",
        "reports/deconvolution_audit/results/selected_lambda.json",
        "reports/matched_estimator_comparison/result.json", "reports/matched_estimator_comparison/results/per_example.csv",
        "reports/excitation_recoverability/results/per_example.csv", "reports/shoebox_pipeline_recheck/support_corrected.csv",
        "reports/shoebox_pipeline_recheck/result.json", "reports/shoebox_pipeline_recheck/source_reconstruction.json",
        "reports/multiroom_generalization/results/data_manifest.csv",
        "reports/multiroom_generalization/results/split_manifest.csv", "configs/rir_datasets/mit_room_split.csv",
        "data/real_claps/split.json", "src/clapgen/evaluation/metrics.py", "src/clapgen/evaluation/diagnosis.py",
        "src/clapgen/experiments/multiroom_generalization/manifest.py",
        "src/clapgen/experiments/supervised_comparison/materialize.py", "eloi_flow_debug/lundeby.py"]
    source_hashes = {p: sha(ROOT / p) for p in source_paths}
    prior = json.loads((ROOT / "reports/excitation_recoverability/result.json").read_text())
    lam = prior["lambda_selected_at_1s"]
    assert np.isclose(lam["true_x"], 1e-5) and np.isclose(lam["trunc_3ms"], 10 ** -3.5)
    checkpoint_source = json.loads((ROOT / "reports/shoebox_pipeline_recheck/result.json").read_text())
    checkpoints = [c for c in checkpoint_source["checkpoints"] if c["arm"] == "regression"]
    assert len(checkpoints) == 3
    protocol = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="Table 1 only: three analytic controls and existing selected one-second Regression; CPU only",
        sample_rate=SR, horizon_samples=SR, observation_index=0,
        measured_providers=list(PROVIDERS[1:]), synthetic_provider="shoebox",
        support="Shoebox: retained4096. Measured: last nonzero stored target sample+1, capped44100; validated against matching per-example Regression bounds. All metrics slice target and estimate before any integration/STFT/EDP.",
        lambdas=dict(known_clap_unregularized=0.0, known_clap_tikhonov=lam["true_x"], crop_3ms_tikhonov=lam["trunc_3ms"]),
        regularization_provenance="Existing independent one-second validation selections in reports/excitation_recoverability/result.json; criterion pooled validation-example median NRMSE, validation providers Shoebox/MIT/BUT/ACE. Retained unchanged; no new selection/tuning. Historical selection used legacy full padded horizon; this report corrects TEST scoring only and does not claim newly support-selected lambdas.",
        inversion="Existing regularized_deconvolution: FFT length131072; Y*conj(X)/(abs(X)^2+lambda_rel*max(abs(X)^2)+1e-20); truncate to44100; float32; peak-normalize with+1e-8. Lambda0 retains numerical1e-20 guard, no tuned regularizer. Crop excitation: first132 observation samples (2.993ms), remainder zeros.",
        input_target_preprocessing="Frozen shards unchanged: clean human excitation first882 samples, peak normalized, participant-held-out test; target onset trim at first>=50% peak and peak normalization; noiseless fftconvolve then observation peak normalization. First stored clap only.",
        training="Selected hybrid Regression, k_max1,1000ms,input_compression1,validity_masking true,peak targets,masked compressed-STFT auxiliary0.25,20k updates,lr2e-4,batch2*accumulate8,training providers Shoebox/MIT/BUT/ACE,dataset-balanced,three seeds42001/02/03. Historical synthetic training mask is preserved.",
        aggregation="Median records within each(seed,provider,room), then median seeds within(provider,room), then median pooled provider/room values. Each room gets one vote; training seeds/channels/records are not independent rooms. No averaging of provider medians. Analytic controls have one seedless estimate per record.",
        intervals="10000 bootstrap resamples of provider/room units, random seed20260911, conditional on observed provider mixture; descriptive, no generalization to unobserved providers. BUT2/ACE1 rooms remain descriptive.",
        metrics=list(METRICS), checkpoints=checkpoints, sources_sha256=source_hashes)
    write_json(out / "protocol.json", protocol)
    original = read_csv(ROOT / "reports/matched_estimator_comparison/results/per_example.csv")
    corrected = read_csv(ROOT / "reports/shoebox_pipeline_recheck/support_corrected.csv")
    regression = [r for r in original if r["arm"] == "regression" and r["provider"] != "shoebox"]
    regression += [r for r in corrected if r["arm"] == "regression"]
    reg_map = {(r["provider"], r["sample_id"], int(r["seed"])): r for r in regression}
    assert len(reg_map) == 660
    manifest = {(r["dataset"], r["record_id"]): r for r in read_csv(ROOT / "reports/multiroom_generalization/results/data_manifest.csv")}
    clap_split = json.loads((ROOT / "data/real_claps/split.json").read_text())
    assert not set(clap_split["train"]) & set(clap_split["test"])
    old_controls = {(r["provider"], r["sample_id"], r["arm"]): r for r in read_csv(ROOT / "reports/excitation_recoverability/results/per_example.csv")}
    rows, examples, split_audit, comparisons = [], [], [], []
    for provider in PROVIDERS:
        path = ROOT / "data/multiroom_generalization" / f"{provider}.npz"
        print(f"Loading frozen {provider} shard", flush=True)
        with np.load(path, allow_pickle=False) as d:
            splits, rooms, ids = d["split"].astype(str), d["room_id"].astype(str), d["record_id"].astype(str)
            idx = np.flatnonzero(splits == "test")
            # NPZ members are materialized once, not repeatedly for each record.
            targets, observations, cleans = d["rir"][idx, :SR], d["observation"][idx, 0, :SR], d["clean"][idx, 0]
        room_parts = defaultdict(set)
        for room, split in zip(rooms, splits):
            room_parts[room].add(split)
        assert all(len(s) == 1 for s in room_parts.values()), provider
        if provider == "openair": assert set(splits) == {"test"}
        split_audit.append(dict(provider=provider, n_test_records=len(idx),
            n_test_rooms=len(set(rooms[idx])), no_room_split_leakage=True,
            split_ids_sha256=hashlib.sha256(json.dumps(list(zip(ids.tolist(), rooms.tolist(), splits.tolist()))).encode()).hexdigest(),
            shard_path=str(path), shard_bytes=path.stat().st_size))
        part = []
        for j, i in enumerate(idx):
            target, obs, clean = targets[j], observations[j], cleans[j]
            nz = np.flatnonzero(target)
            bound = 4096 if provider == "shoebox" else min(SR, int(nz[-1]) + 1)
            assert np.all(target[bound:] == 0) and np.all(clean[882:] == 0)
            m = manifest[(provider, ids[i])]
            assert m["room_id"] == rooms[i] and m["partition"] == "test"
            clap_id = m["clap_ids"].split(";")[0]
            assert int(clap_id[1:3]) in {int(p) for p in clap_split["test"]}
            base = dict(provider=provider, room_id=rooms[i], sample_id=ids[i],
                        bound=bound, observation_index=0, split="test")
            examples.append(dict(**base, clap_id=clap_id, source_path=m["source_path"],
                split_type=m["split_type"], shard_index=int(i),
                target_sha256=array_sha(target), observation_sha256=array_sha(obs), clean_sha256=array_sha(clean)))
            target = target[:bound].astype(np.float64)
            features = reference_features(target)
            padded = np.zeros(SR, np.float32)
            padded[:min(len(clean), SR)] = clean[:SR]
            cropped = np.zeros(SR, np.float32)
            cropped[:132] = obs[:132]
            for arm, excitation, strength in ((ARMS[0], padded, 0.),
                    (ARMS[1], padded, lam["true_x"]), (ARMS[2], cropped, lam["trunc_3ms"])):
                est = regularized_deconvolution(obs, excitation, SR, strength)
                scores = score(target, est[:bound], features)
                assert all(np.isfinite(v) for k, v in scores.items() if k != "abs_edt_error_s"), (provider, ids[i], arm, scores)
                part.append(dict(**base, arm=arm, seed="analytic", lambda_relative=strength,
                                 score_source="recomputed_post_meeting", **scores))
                if provider != "shoebox" and arm in ARMS[1:3]:
                    old_arm = "E1_true_clap" if arm == ARMS[1] else "E2_crop_3ms"
                    old = old_controls[(provider, ids[i], old_arm)]
                    assert int(old["bound"]) == bound
                    for metric in METRICS:
                        historical = float(old[metric])
                        missing = np.isnan(scores[metric]) and np.isnan(historical)
                        assert missing or (np.isfinite(scores[metric]) and np.isfinite(historical))
                        delta = 0.0 if missing else abs(scores[metric] - historical)
                        comparisons.append(dict(provider=provider, sample_id=ids[i], arm=arm, metric=metric, both_unmeasurable=missing, absolute_difference=delta))
            for seed in SEEDS:
                old = reg_map[(provider, ids[i], seed)]
                assert old["room_id"] == rooms[i] and int(old["bound"]) == bound
                part.append(dict(**base, arm=ARMS[3], seed=seed, lambda_relative="",
                    score_source="shoebox_support_corrected" if provider == "shoebox" else "existing_matched_e3",
                    **{metric: float(old[metric]) for metric in METRICS}))
            if (j + 1) % 10 == 0 or j + 1 == len(idx):
                print(f"{provider}: scored {j+1}/{len(idx)} records", flush=True)
        rows.extend(part)
        write_csv(out / f"per_example_{provider}.csv", part)
        write_csv(out / "example_manifest.csv", examples)
    assert len(examples) == 220 and len(rows) == 1320
    write_csv(out / "per_example.csv", rows)
    write_csv(out / "split_audit.csv", split_audit)
    write_csv(out / "historical_measured_reproduction.csv", comparisons)
    max_delta = max(r["absolute_difference"] for r in comparisons)
    assert max_delta < 1e-8, f"Metric implementation drift: {max_delta}"
    main_rows = summarize(rows, out)
    validation = dict(status="passed", test_records=220, measured_records=216,
        measured_rooms=54, shoebox_rooms=4, rows=1320, analytic_rows=660, regression_rows=660,
        measured_baseline_metric_comparisons=len(comparisons), max_historical_measured_metric_difference=max_delta,
        all_regression_ids_rooms_bounds_matched=True, all_test_claps_in_frozen_test_participants=True,
        room_split_leakage=False, pooled_provider_medians_averaged=False,
        checkpoint_reuse="Prior exact-checkpoint scores; provenance hashes retained. No new model inference needed.",
        nonfinite_metric_rows={m: sum(not np.isfinite(float(r[m])) for r in rows) for m in METRICS},
        no_training_or_out_of_scope_inference=True,
        generated_source_sha256=sha(Path(__file__)))
    write_json(out / "verification.json", validation)
    write_json(result_path, dict(status="completed", main_table=main_rows, protocol=str(out / "protocol.json"),
                               verification=validation, completed_at_utc=datetime.now(timezone.utc).isoformat()))
    lines = ["# Post-meeting Table 1", "", "Completed the four frozen methods on 220 identical test records. The primary measured pool contains 216 records from 54 provider/room units; Shoebox contains four rooms and is reported separately.", "",
        "## Protocol and provenance", "", "One-second model and observations at 44.1 kHz, first stored clap only, unchanged room-disjoint/provider-held-out splits and participant-held-out test claps. All six metrics crop both signals to actual retained target support before calculation. Shoebox uses 4096 samples (92.88 ms); measured support is the stored target endpoint capped at 44100. This describes retained support, not an assumption that the room has finished decaying.", "",
        "The selected estimator is the existing 20,000-update, k_max=1 Regression family with seeds 42001/02/03. Its measured per-example scores are reused from the exact matching E3 run; Shoebox scores come from the documented independent support correction. Every record ID, room and support bound was matched to the current frozen shards. No estimator was trained or reselected.", "",
        "All three analytic controls were recomputed. The existing relative lambdas are 0, 1e-5, and 3.1622776601683794e-4 respectively. The two nonzero values were selected independently on the one-second validation data before this task; no test tuning or re-selection occurred. Those historical validation choices used the old padded-horizon criterion; the values are retained while this task corrects test scoring. Inversion uses the established 131072-point Fourier solve and peak normalization. The crop uses 132 samples (2.993 ms). Lambda zero retains only the numerical 1e-20 denominator guard.", "",
        "`protocol.json` records equations, model/checkpoint provenance, preprocessing, metric definitions by source hashes, and the original validation-selection sources. `example_manifest.csv` records each test record, clap, original source, support and hashes of the evaluated input/target/excitation arrays. `split_audit.csv` validates room isolation and test participant membership. Frozen source shards, checkpoints and prior results are unchanged.", "",
        "## Aggregation", "", "For each metric, take the median over RIR records within each training seed and provider/room, then the median over the three Regression seeds, then the median across rooms. Analytic controls have a single seedless estimate per record. Pooling concatenates the 54 measured provider/room values; it does not average four provider medians and does not count channels or training seeds as independent rooms. The pool is dominated numerically by MIT (42 rooms), with BUT 2, ACE 1 and OpenAIR 9; it describes this provider mixture. `per_seed_room.csv` and `per_room.csv` preserve the aggregation units, and `summary.csv` includes finite counts and descriptive 10,000-resample room bootstrap intervals. BUT and ACE cannot support broad within-provider inference.", "",
        (out / "table1.md").read_text().rstrip(), "",
        "## Verification and limits", "", f"All {len(comparisons)} recomputed measured E1/E2 metric values reproduce the historical values within {max_delta:.3g}. Every Regression score is joined to the same sample/room/bound. The new unregularized row is recomputed at one second, and every Shoebox analytic metric is corrected independently. Undefined EDT values are preserved as NaN and excluded only from that metric's aggregation; all per-record/room/summary finite counts are retained. `verification.json` records these checks.", "",
        "Known-clap rows use privileged excitation on noiseless synthetic convolutions with measured RIRs. They establish recoverability in that observation model, not robustness to phone noise. The Regression checkpoint was trained with the historical synthetic support mask; this task corrects evaluation without changing training. Finite-support C50/EDT characterize the retained window. NRMSE remains a companion metric.", "",
        "## Artifacts and manuscript alignment", "", "Primary table: `table1.tex` and `table1.md`. Provider detail: `provider_breakdown.md`; all values/intervals: `summary.csv`; raw scores: `per_example.csv`. Reproduce from any local working directory `python experiments/single_clap_benchmark.py --report-root /tmp/table1_reproduction`.", "",
        "The generic publication/configs/frozen_protocol.json still says 250 ms, and publication/EXPERIMENTS.md / the active main table still include superseded experiment scope. The authorized post-meeting table is supplied here for manuscript integration; AfterMeeting.pdf and manuscript narrative need alignment with Regression-only, no Flow/Spheres/ARPEGE/multi-clap mainline and the single qualitative phone figure. No out-of-scope experiment or phone quantitative analysis was run."]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print((out / "table1.md").read_text(), flush=True)
    print(f"COMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
