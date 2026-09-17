#!/usr/bin/env python3
"""Isolated 6-ms analytic control; never trains or re-evaluates existing arms."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import single_clap_benchmark as base
import single_clap_benchmark_statistics as original

OUT = ROOT / 'reports/cropped_excitation_baseline'
PUB = ROOT / 'publication/paper/tables'
ARM = 'crop_6ms_tikhonov'
LENGTH = 44100
CROP = round(.006 * LENGTH)
METRICS = (*base.METRICS, 'lsd_db')
ORDER = ('known_clap_unregularized', 'known_clap_tikhonov',
         'crop_3ms_tikhonov', ARM, 'regression_1s')
LABELS = dict(zip(ORDER, ('Known excitation, unregularized',
    'Known excitation, Tikhonov', 'Cropped-excitation (3 ms) + Tikhonov',
    'Cropped-excitation (6 ms) + Tikhonov', 'Neural regressor')))


def freeze():
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Preserve the already frozen protocol.')
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((ROOT / 'reports/deconvolution_audit/results/selected_lambda.json').read_text())
    source_names = [
        'experiments/cropped_excitation_baseline.py', 'experiments/single_clap_benchmark.py',
        'experiments/excitation_recoverability.py', 'experiments/single_clap_benchmark_statistics.py',
        'src/clapgen/evaluation/metrics.py', 'src/clapgen/evaluation/diagnosis.py',
        'eloi_flow_debug/lundeby.py', 'reports/deconvolution_audit/results/selected_lambda.json',
        'reports/excitation_recoverability/result.json', 'reports/single_clap_benchmark/protocol.json',
        'reports/single_clap_benchmark/example_manifest.csv', 'reports/single_clap_benchmark/per_example.csv',
        'reports/single_clap_benchmark/per_room.csv', 'reports/single_clap_benchmark/table1.csv',
        'reports/noisy_controlled_benchmark/clean_trained_diagnostic/table1/per_example.csv',
        'data/real_claps/split.json']
    hashes = {name: base.sha(ROOT / name) for name in source_names}
    for provider in base.PROVIDERS:
        name = f'data/multiroom_generalization/{provider}.npz'
        hashes[name] = base.sha(ROOT / name)
        print('Frozen source', provider, flush=True)
    for ext in ('tex', 'csv', 'md', 'pdf'):
        for directory, label in ((PUB, 'publication'), (ROOT / 'reports/single_clap_benchmark', 'report')):
            name = ('single_clap_benchmark.' if label == 'publication' else 'table1.') + ext
            dest = OUT / 'original' / label / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / name, dest)
    examples = base.read_csv(ROOT / 'reports/single_clap_benchmark/example_manifest.csv')
    selections = [min((r for r in examples if r['provider'] == p),
                     key=lambda r: hashlib.sha256(r['sample_id'].encode()).hexdigest())
                  for p in base.PROVIDERS]
    protocol = dict(id='CROPPED_EXCITATION_6MS', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        x_decision='Whether the manuscript can attribute the weak crop baseline solely to choosing 3 ms; report 6 ms before any narrative change.',
        prediction='A longer crop may improve excitation coverage but also include more room response; no preferred result ordering is required.',
        scope='Only compute crop_6ms_tikhonov; preserve existing results and checkpoints; no neural inference or training.',
        sample_rate=LENGTH, horizon_samples=LENGTH, crop_samples=CROP,
        crop_ms_actual=1000 * CROP / LENGTH,
        crop_rule='First round(.006*44100)=265 samples of stored observation index 0, remainder zero; no new onset alignment or taper.',
        observation_condition='Same stored no-added-observation-noise inputs as the current 3-ms row. No new noise construction.',
        selection=dict(grid=prior['grid'], providers=prior['validation_datasets'],
            criterion='Pooled validation-record median NRMSE on full 44100-sample horizon, exactly as original 3-ms selection.',
            support_caveat='Historical lambda selection includes the padded horizon. Preserve it for the controlled crop comparison; all TEST metrics crop to valid support.',
            tie_break='First minimum in original ascending candidate order', test_selection=False),
        inversion='Reuse regularized_deconvolution unchanged: FFT131072, relative lambda*max(|X|^2), guard1e-20, float32 estimate, peak normalization+1e-8.',
        test='Exact original 220-record manifest: 216 measured records in54 provider/rooms plus4 Shoebox records/rooms; observation_index0.',
        valid_support='min(44100,last nonzero target+1) for measured;4096 for Shoebox. Slice reference and estimate before all metrics.',
        metrics=list(METRICS),
        lsd_definition='20*sqrt(stft_logmag_mse) PER RECORD, i.e. RMS dB difference over existing STFT bins/frames (Hann510,hop127,reference-peak -80dB floor); transform before aggregation.',
        aggregation='Median records within(seed,provider,room), median seeds withinroom; arithmetic mean and sampleSD(ddof1) across finite room summaries. Existing unregularized also median3noisedraws perrecord first. Never average provider medians.',
        examples=selections, sources_sha256=hashes)
    base.write_json(OUT / 'protocol.json', protocol)
    # Register the pre-result contract without modifying HEAD or the user's index.
    def git(*args, data=None):
        return subprocess.check_output(['git', *args], cwd=ROOT, input=data).decode().strip()
    entries = []
    for name, path in [('protocol.json', OUT / 'protocol.json'),
                       ('cropped_excitation_baseline.py', Path(__file__))]:
        oid = git('hash-object', '-w', '--stdin', data=path.read_bytes())
        entries.append(f'100644 blob {oid}\t{name}\n')
    tree = git('mktree', data=''.join(sorted(entries)).encode())
    parent = git('rev-parse', 'HEAD')
    commit = git('commit-tree', tree, '-p', parent,
                 data=b'Freeze six-millisecond cropped-excitation control\n')
    git('update-ref', 'refs/experiments/cropped-excitation-6ms', commit)
    base.write_json(OUT / 'registration.json', dict(commit=commit, parent=parent,
        ref='refs/experiments/cropped-excitation-6ms', protocol_sha256=base.sha(OUT/'protocol.json')))
    print('Protocol frozen and registered', commit, flush=True)


def load_records(provider, split):
    with np.load(ROOT / f'data/multiroom_generalization/{provider}.npz', allow_pickle=False) as d:
        partitions = d['split'].astype(str)
        idx = np.flatnonzero(np.isin(partitions, ['valid', 'validation'] if split == 'valid' else ['test']))
        ids, rooms = d['record_id'].astype(str), d['room_id'].astype(str)
        observation = d['observation'][idx, 0, :LENGTH]
        target = d['rir'][idx, :LENGTH]
    for j, i in enumerate(idx):
        yield dict(provider=provider, sample_id=ids[i], room_id=rooms[i], shard_index=int(i)), observation[j], target[j]


def excitation(observation):
    x = np.zeros(LENGTH, np.float32)
    x[:CROP] = observation[:CROP]
    return x


def select():
    if (OUT / 'selected_lambda.json').exists():
        raise FileExistsError('Lambda already selected; do not reselect.')
    protocol = json.loads((OUT / 'protocol.json').read_text())
    rows = []
    for provider in protocol['selection']['providers']:
        for meta, y, target in load_records(provider, 'valid'):
            t = target.astype(np.float64)
            x = excitation(y)
            for value in protocol['selection']['grid']:
                estimate = base.regularized_deconvolution(y, x, LENGTH, value)
                error = float(np.linalg.norm(estimate - t) / (np.linalg.norm(t) + 1e-12))
                rows.append(dict(**meta, split='validation', lambda_relative=value, nrmse=error))
        print('Validation complete:', provider, flush=True)
    aggregate = [dict(lambda_relative=g, median_nrmse=float(np.median(
        [r['nrmse'] for r in rows if r['lambda_relative'] == g])))
        for g in protocol['selection']['grid']]
    best = min(aggregate, key=lambda r: r['median_nrmse'])
    base.write_csv(OUT / 'validation_per_example.csv', rows)
    base.write_csv(OUT / 'validation_grid.csv', aggregate)
    base.write_json(OUT / 'selected_lambda.json', dict(**best,
        n_validation_records=len(rows)//len(aggregate),
        source='Validation only; all 11 candidates, unchanged 3-ms criterion.',
        frozen_before_test_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=base.sha(OUT/'protocol.json')))
    print('Selected lambda', best, flush=True)


def evaluate():
    if (OUT / 'per_example.csv').exists():
        raise FileExistsError('Test evaluation already exists; preserve it.')
    value = json.loads((OUT/'selected_lambda.json').read_text())['lambda_relative']
    manifest = {(r['provider'], r['sample_id']): r for r in base.read_csv(
        ROOT/'reports/single_clap_benchmark/example_manifest.csv')}
    (OUT/'estimates').mkdir(exist_ok=True)
    rows = []
    for provider in base.PROVIDERS:
        estimates, ids = [], []
        for meta, y, t32 in load_records(provider, 'test'):
            ref = manifest[(provider, meta['sample_id'])]
            assert base.array_sha(y) == ref['observation_sha256']
            assert base.array_sha(t32) == ref['target_sha256']
            bound = int(ref['bound'])
            assert bound == (4096 if provider == 'shoebox' else min(LENGTH, int(np.flatnonzero(t32)[-1]) + 1))
            estimate = base.regularized_deconvolution(y, excitation(y), LENGTH, value)
            target = t32[:bound].astype(np.float64)
            metrics = base.score(target, estimate[:bound], base.reference_features(target))
            metrics['lsd_db'] = 20 * math.sqrt(metrics['stft_logmag_mse'])
            rows.append(dict(**meta, arm=ARM, seed='analytic', split='test',
                bound=bound, observation_index=0, lambda_relative=value, **metrics))
            estimates.append(estimate); ids.append(meta['sample_id'])
            if len(rows) % 20 == 0:
                print(f'Scored {len(rows)}/220', flush=True)
        np.savez_compressed(OUT/'estimates'/f'{provider}.npz',
            sample_id=np.asarray(ids), estimate=np.asarray(estimates))
        print('Test provider complete:', provider, flush=True)
    assert len(rows) == len(manifest) == 220
    base.write_csv(OUT/'per_example.csv', rows)


def reduce_rows(rows, keys, metrics=METRICS):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    result = []
    for key, group in groups.items():
        row = dict(zip(keys, key), n_records=len(group))
        for m in metrics:
            values = [float(r[m]) for r in group if math.isfinite(float(r[m]))]
            row[m] = statistics.median(values) if values else math.nan
        result.append(row)
    return result


def summarize():
    raw = base.read_csv(OUT/'per_example.csv')
    rooms = reduce_rows(raw, ('arm', 'provider', 'room_id'))
    assert len(rooms) == 58
    base.write_csv(OUT/'per_room.csv', rooms)
    summary = []
    for population in ('measured_pooled', *base.PROVIDERS):
        group = [r for r in rooms if (r['provider'] != 'shoebox' if population == 'measured_pooled' else r['provider'] == population)]
        row = dict(population=population, arm=ARM, n_rooms=len(group),
            n_records=sum(r['n_records'] for r in group),
            source_per_room='reports/cropped_excitation_baseline/per_room.csv',
            source_condition='no added observation noise', central_statistic='mean_across_room_summaries',
            dispersion='sample_sd_across_room_summaries', sd_ddof=1,
            training_seeds='analytic', noise_draws_per_record=1)
        for metric in METRICS:
            vals = [r[metric] for r in group if math.isfinite(r[metric])]
            row[metric] = statistics.mean(vals) if vals else math.nan
            row[metric+'_std'] = statistics.stdev(vals) if len(vals)>1 else math.nan
            row[metric+'_source_median'] = statistics.median(vals) if vals else math.nan
            row['n_finite_rooms_'+metric] = len(vals)
            if len(vals)>1:
                assert np.isclose(row[metric+'_std'], np.std(vals, ddof=1), rtol=1e-14)
        summary.append(row)
    base.write_csv(OUT/'summary.csv', summary)
    print('Room mean and sample SD written.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'select', 'evaluate', 'summarize'])
    args = parser.parse_args()
    globals()[args.action]()


if __name__ == '__main__':
    main()
