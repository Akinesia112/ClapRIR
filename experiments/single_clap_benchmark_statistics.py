#!/usr/bin/env python3
"""Recover each Table 1 row by its printed numbers, then report room mean +/- SD."""
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/single_clap_benchmark"
AUDIT = REPORT / "mean_std_recovery"
PUB = ROOT / "publication/paper/tables"
METRICS = ("edc_rmse_db", "abs_c50_error_db", "abs_edt_error_s",
           "echo_density_rmse", "stft_logmag_mse", "nrmse")
LABELS = {
    "known_clap_unregularized": "Known clap, unregularized",
    "known_clap_tikhonov": "Known clap, Tikhonov",
    "crop_3ms_tikhonov": "3 ms crop + Tikhonov",
    "regression_1s": "1 s Regression",
}
CLEAN_ROOM = "reports/single_clap_benchmark/per_room.csv"
CLEAN_RAW = "reports/single_clap_benchmark/per_example.csv"
NOISY_ROOM = "reports/noisy_controlled_benchmark/clean_trained_diagnostic/table1/per_room.csv"
NOISY_RAW = "reports/noisy_controlled_benchmark/clean_trained_diagnostic/table1/per_example.csv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def median_reduce(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in grouped.items():
        row = dict(zip(keys, key))
        for metric in METRICS:
            values = [float(r[metric]) for r in group if math.isfinite(float(r[metric]))]
            row[metric] = statistics.median(values) if values else math.nan
        result.append(row)
    return result


def verify_rooms(derived, stored):
    key = lambda r: (r["arm"], r["provider"], r["room_id"])
    index = {key(r): r for r in stored}
    assert set(index) == {key(r) for r in derived}
    diffs = []
    for row in derived:
        ref = index[key(row)]
        for metric in METRICS:
            a, b = float(row[metric]), float(ref[metric])
            assert math.isfinite(a) == math.isfinite(b)
            if math.isfinite(a):
                diffs.append(abs(a-b))
    assert max(diffs) == 0
    return dict(rooms=len(derived), finite_metric_checks=len(diffs), max_absolute_difference=max(diffs))


def pm(mean, std):
    largest = max(abs(mean), abs(std))
    if largest and (largest < .01 or largest >= 1000):
        exponent = math.floor(math.log10(largest))
        scale = 10 ** exponent
        return r"$(" + f"{mean/scale:.3g}" + r"\pm" + f"{std/scale:.3g}" + r")\times10^{" + str(exponent) + "}$"
    return "$" + f"{mean:.3g}" + r"\pm" + f"{std:.3g}" + "$"


def main():
    target = PUB / "single_clap_benchmark.tex"
    original = target.read_text()
    before_sha = sha(target)
    if AUDIT.exists():
        raise RuntimeError("Recovery already exists; preserve its frozen input and provenance.")
    label_to_arm = {label: arm for arm, label in LABELS.items()}
    targets = []
    for line in original.splitlines():
        if line.startswith(("Measured pooled &", "Shoebox (92.88 ms) &")):
            fields = [s.strip() for s in line.removesuffix("\\\\").split("&")]
            targets.append(dict(population="measured_pooled" if fields[0]=="Measured pooled" else "shoebox",
                                arm=label_to_arm[fields[1]], old_values=list(map(float, fields[2:]))))
    assert len(targets) == 8 and all(len(r["old_values"]) == 6 for r in targets)
    clean = read(ROOT/CLEAN_ROOM)
    noisy_all = read(ROOT/NOISY_ROOM)
    noisy = [r for r in noisy_all if r["snr_db"]=="30" and r["arm"]=="known_clap_unregularized"]
    clean_derived = median_reduce(median_reduce(read(ROOT/CLEAN_RAW),
        ("arm", "provider", "room_id", "seed")), ("arm", "provider", "room_id"))
    noisy_raw = [r for r in read(ROOT/NOISY_RAW)
                 if r["snr_db"]=="30" and r["arm"]=="known_clap_unregularized"]
    assert len(noisy_raw) == 660
    assert all(r["seed"]=="analytic" for r in noisy_raw)
    draw_groups = defaultdict(set)
    for r in noisy_raw:
        draw_groups[(r["provider"], r["room_id"], r["sample_id"])].add(r["noise_draw"])
    assert len(draw_groups)==220 and all(v=={"0","1","2"} for v in draw_groups.values())
    noisy_derived = median_reduce(median_reduce(noisy_raw,
        ("arm", "provider", "room_id", "sample_id")), ("arm", "provider", "room_id"))
    checks = dict(local_original=verify_rooms(clean_derived, clean),
                  local_30db_unregularized=verify_rooms(noisy_derived, noisy))
    matches, summary, selected, comparison = [], [], [], []
    candidates = {"local_no_added_noise": clean,
                  **{f"local_{snr}db": [r for r in noisy_all if r["snr_db"]==str(snr)] for snr in (20,30,40)}}
    for target_row in targets:
        population, arm = target_row["population"], target_row["arm"]
        def in_group(r):
            return r["arm"]==arm and (r["provider"]!="shoebox" if population=="measured_pooled" else r["provider"]=="shoebox")
        for name, candidate in candidates.items():
            group = [r for r in candidate if in_group(r)]
            match_count = 0
            for metric, old in zip(METRICS, target_row["old_values"]):
                v = [float(r[metric]) for r in group if math.isfinite(float(r[metric]))]
                match_count += float(f"{statistics.median(v):.4g}")==old
            comparison.append(dict(population=population, arm=arm, candidate=name,
                                   matched_published_median_cells=match_count, total_cells=6))
        is_noisy = arm=="known_clap_unregularized"
        source = NOISY_ROOM if is_noisy else CLEAN_ROOM
        condition = "30 dB additive Gaussian noise" if is_noisy else "no added observation noise"
        group = [r for r in (noisy if is_noisy else clean) if in_group(r)]
        assert len(group)==(54 if population=="measured_pooled" else 4)
        row = dict(population=population, arm=arm, n_rooms=len(group),
                   n_records=sum(int(r.get("n_records", r.get("records", 0))) for r in group),
                   central_statistic="mean_across_room_summaries", dispersion="sample_sd_across_room_summaries",
                   sd_ddof=1, source_per_room=source, source_condition=condition,
                   training_seeds="42001,42002,42003" if arm=="regression_1s" else "analytic",
                   noise_draws_per_record=3 if is_noisy else 1)
        for metric, old in zip(METRICS, target_row["old_values"]):
            vals = [float(r[metric]) for r in group if math.isfinite(float(r[metric]))]
            median = statistics.median(vals)
            assert float(f"{median:.4g}")==old, (population, arm, metric, median, old)
            mean, std = statistics.mean(vals), statistics.stdev(vals)
            assert np.isclose(mean, np.mean(vals), rtol=1e-14, atol=1e-14)
            assert np.isclose(std, np.std(vals, ddof=1), rtol=1e-14, atol=1e-14)
            assert len(vals)==(51 if population=="measured_pooled" and metric=="abs_edt_error_s" else len(group))
            row.update({metric: mean, metric+"_std": std, metric+"_source_median": median,
                        "n_finite_rooms_"+metric: len(vals)})
            matches.append(dict(population=population, arm=arm, metric=metric, old_table_value=old,
                                recovered_median=median, matched_at_4_significant_digits=True,
                                mean=mean, sample_std=std, finite_rooms=len(vals), source_per_room=source,
                                source_condition=condition))
        selected.extend(dict(population=population, source_per_room=source, source_condition=condition,
                             **{k:r[k] for k in ("arm","provider","room_id",*METRICS)}) for r in group)
        summary.append(row)
    assert len(matches)==48
    # No mutation of the publication table occurs until every row is recovered and verified.
    AUDIT.mkdir(parents=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup=REPORT/"history"/("before_mean_std_"+stamp)
    for source in [target, *[PUB/("single_clap_benchmark."+ext) for ext in ("csv","md","pdf")],
                   *[REPORT/name for name in ("table1.tex","table1.csv","table1.md","table1.pdf",
                                             "summary.csv","result.json","README.md","canonical_exports.json")]]:
        if source.exists():
            dest=backup/("publication" if source.parent==PUB else "report")/source.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest)
    (AUDIT/"original_table.tex").write_text(original)
    caption = (
        r"Single-clap RIR estimation: mean $\pm$ sample standard deviation across provider/room summaries. "
        r"Measured results use 54 rooms (51 for EDT); Shoebox uses four rooms and 4096-sample (92.88-ms) valid support. "
        r"Within-room record, noise-draw, and training-seed medians follow the respective source protocols; "
        r"Regression uses three training seeds. Unregularized rows use 30-dB noisy observations; "
        r"the remaining rows use the original no-added-noise results. All errors are lower-is-better.")
    tex = ["% Generated by experiments/single_clap_benchmark_statistics.py; per-row sources and original median matches are recorded in reports/single_clap_benchmark/mean_std_recovery.",
           r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{"+caption+"}", r"\label{tab:post-meeting-table1}",
           r"\resizebox{\textwidth}{!}{%",r"\begin{tabular}{llrrrrrr}",r"\toprule",
           r"Population & Method & EDC (dB) & $|\Delta C_{50}|$ (dB) & $|\Delta\mathrm{EDT}|$ (s) & EDP & Log-mag MSE & NRMSE \\",
           r"\midrule"]
    md = ["| Population | Method | EDC (dB) | C50 error (dB) | EDT error (s) | EDP | Log-mag MSE | NRMSE |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        pop = "Measured pooled" if r["population"]=="measured_pooled" else "Shoebox (92.88 ms)"
        tex.append(" & ".join([pop,LABELS[r["arm"]],*[pm(r[m],r[m+"_std"]) for m in METRICS]])+r" \\")
        md.append("| "+" | ".join([pop,LABELS[r["arm"]],*[f"{r[m]:.6g} ± {r[m+'_std']:.6g}" for m in METRICS]])+" |")
    tex += [r"\bottomrule",r"\end{tabular}%",r"}",r"\end{table*}"]
    text="\n".join(tex)+"\n"
    assert sha(target)==before_sha, "Table changed during recovery."
    target.write_text(text)
    (REPORT/"table1.tex").write_text(text)
    (REPORT/"table1.md").write_text("\n".join(md)+"\n\n"+caption+"\n")
    write_csv(REPORT/"table1.csv", summary)
    write_csv(REPORT/"summary.csv", summary)
    write_csv(AUDIT/"row_metric_recovery.csv", matches)
    write_csv(AUDIT/"candidate_comparison.csv", comparison)
    write_csv(AUDIT/"recovered_room_scores.csv", selected)
    sources = (CLEAN_ROOM,CLEAN_RAW,NOISY_ROOM,NOISY_RAW,
               "reports/single_clap_benchmark/noise_provenance_audit.json",
               "reports/single_clap_benchmark/external_rerun_import.json")
    provenance=dict(status="completed", original_table_sha256=before_sha,
        current_table_sha256=sha(target), backup=str(backup.relative_to(ROOT)),
        matched_rows=8, matched_cells=48, matching_rule="Exact reproduction at original four-significant-digit precision",
        user_authorized_mixed_sources=True, original_values_are_medians=True,
        new_values_are="arithmetic mean +/- sample standard deviation across finite room summaries",
        sd_ddof=1, preserved_inner_aggregation=True, no_provider_median_averaging=True,
        no_training_or_inference=True, row_sources=summary,
        source_sha256={p:sha(ROOT/p) for p in sources}, raw_to_room_checks=checks,
        independent_numpy_mean_std_check=True,
        source_identity_limit="Numerical recovery from the local source rows, authorized by the user; not a claim that the separate external machine's raw run was imported.",
        source_conditions="Unregularized: 30 dB, 3 noise draws; other rows: historical no-added-noise scores, Regression 3 seeds.")
    write_json(AUDIT/"provenance.json",provenance)
    write_json(REPORT/"result.json",dict(status="completed-recovered-room-mean-and-sd", main_table=summary,
        statistic="mean +/- sample SD of room summaries", source_conditions=provenance["source_conditions"],
        matched_source_cells=48, mixed_sources_user_authorized=True, local_inference_run=False,
        provenance="mean_std_recovery/provenance.json", historical_external_import="external_rerun_import.json"))
    exports=[]
    for ext in ("tex","csv","md"):
        source=REPORT/("table1."+ext);dest=PUB/("single_clap_benchmark."+ext)
        if source!=dest:shutil.copyfile(source,dest)
        exports.append(dict(source=str(source),destination=str(dest),sha256=sha(dest)))
    write_json(REPORT/"canonical_exports.json",dict(status="text-exports-synchronized-pdf-pending",
        source_type="per-row-numerically-recovered-room-mean-and-sample-sd",exports=exports))
    (REPORT/"README.md").write_text(
        "# Table 1: recovered sources, room mean and standard deviation\n\n"
        "The user authorized combining row sources matched to the existing table's numbers. All 48 printed values were recovered as medians, not means: the two unregularized rows match the local 30-dB diagnostic, and the other six rows match the original Table 1 room scores.\n\n"
        "The current table reports the arithmetic mean and sample SD (ddof=1) across room summaries. This changes the central values from the original medians. Existing within-room record/noise-draw/Regression-seed medians are preserved; rooms are equally weighted, provider medians are never averaged, and SD is not an SE, CI, or cross-training-seed deviation.\n\n"
        "Sources retain different conditions. Unregularized uses 30-dB added Gaussian noise with three draws per record. Other rows use the historical no-added-noise source results; Regression uses seeds 42001/42002/42003. The publication caption states these conditions. Their combination was explicitly authorized and is not represented as a common-noise comparison.\n\n"
        "Measured pooling has 54 provider/rooms and 216 records; EDT has 51 finite rooms. Shoebox has four rooms on 4096 valid samples. Missing descriptors remain missing. Every finite room/outlier contributes to the mean and sample SD.\n\n"
        "Current exports: table1.tex/csv/md/pdf and publication/paper/tables/single_clap_benchmark.tex/csv/md/pdf. Full precision means, SDs, source medians, and finite counts are in table1.csv.\n\n"
        "Source recovery: mean_std_recovery/provenance.json, row_metric_recovery.csv, candidate_comparison.csv, and recovered_room_scores.csv. Raw scores independently reproduced all stored room summaries with maximum difference zero; mean/SD also agree between Python statistics and NumPy.\n\n"
        "Original table/export backup: "+str(backup.relative_to(ROOT))+". Earlier external_rerun_import.json and provider_breakdown.* retain their historical meanings. The separate external raw run was not imported; this is a local per-row numerical reconstruction approved by the user.\n")
    print(json.dumps(dict(status="updated",matched_cells=48,table=str(target),backup=str(backup),
                          edc_examples=[dict(population=r["population"],arm=r["arm"],
                                             mean=r["edc_rmse_db"],std=r["edc_rmse_db_std"]) for r in summary])))


if __name__=="__main__":
    main()
