#!/usr/bin/env python3
"""Verify the completed crop control and synchronize its report indexes."""
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
import soundfile as sf
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cropped_excitation_baseline as task
from claprir.metrics.room_acoustics import _stft_magnitude

ROOT, OUT, PUB, base = task.ROOT, task.OUT, task.PUB, task.base
protocol = json.loads((OUT/'protocol.json').read_text())
result = json.loads((OUT/'result.json').read_text())
rows = base.read_csv(OUT/'per_example.csv')
old = base.read_csv(OUT/'original/publication/single_clap_benchmark.csv')
current = base.read_csv(PUB/'single_clap_benchmark.csv')
assert len(rows)==220 and len(current)==10
assert {r['arm'] for r in rows}=={task.ARM}
assert len({(r['provider'],r['room_id']) for r in rows})==58
for r in old:
    new = next(v for v in current if (v['population'],v['arm'])==(r['population'],r['arm']))
    assert all(new[k]==v for k,v in r.items())
for r in current:
    assert int(r['n_rooms'])==(54 if r['population']=='measured_pooled' else 4)
    assert int(r['n_finite_rooms_abs_edt_error_s'])==(51 if r['population']=='measured_pooled' else 4)
    assert float(r['abs_edt_error_ms'])==1000*float(r['abs_edt_error_s'])
validation = base.read_csv(OUT/'validation_per_example.csv')
assert len(validation)==92*11
v_ids={(r['provider'],r['sample_id']) for r in validation}
t_ids={(r['provider'],r['sample_id']) for r in rows}
assert not (v_ids&t_ids)
checks=[]
for selection in protocol['examples']:
    provider=selection['provider']
    target,sr=sf.read(OUT/'audio'/f'{provider}_rir_reference.wav',dtype='float64')
    estimate,sre=sf.read(OUT/'audio'/f'{provider}_rir_6ms.wav',dtype='float64')
    assert sr==sre==44100 and len(target)==len(estimate)==int(selection['bound'])
    r=next(r for r in rows if (r['provider'],r['sample_id'])==(provider,selection['sample_id']))
    waveform_error=float(np.linalg.norm(estimate-target)/(np.linalg.norm(target)+1e-12))
    ref=_stft_magnitude(target);est=_stft_magnitude(estimate)
    floor=float(ref.max())*1e-4
    db_error=20*np.log10(np.maximum(est,floor))-20*np.log10(np.maximum(ref,floor))
    lsd=float(np.sqrt(np.mean(db_error**2)))
    assert math.isclose(waveform_error,float(r['nrmse']),rel_tol=1e-13)
    assert math.isclose(lsd,float(r['lsd_db']),rel_tol=1e-13)
    checks.append(dict(provider=provider,sample_id=r['sample_id'],
        saved_audio_matches_scored_nrmse=True,direct_rms_db_matches_lsd=True))
source_checks={}
for name, digest in protocol['sources_sha256'].items():
    if name.startswith('data/multiroom_generalization/'):
        continue  # Every test observation and target was checked against frozen array hashes during scoring.
    source=(OUT/'original/report/table1.csv') if name=='reports/single_clap_benchmark/table1.csv' else ROOT/name
    source_checks[name]=base.sha(source)==digest
assert all(source_checks.values())
pdfs={}
for name in ('single_clap_benchmark.pdf','post_meeting_table1_six_metrics.pdf'):
    document=PdfReader(PUB/name)
    assert len(document.pages)==1
    pdfs[name]=dict(pages=1,sha256=base.sha(PUB/name))
tex=(PUB/'single_clap_benchmark.tex').read_text()
assert tex.count(r'\multirow{5}')==2
assert '\\begin{table*}' not in tex and '\\begin{table}[t]' in tex
for name in ('compile.log','six_metrics_compile.log'):
    log=(OUT/'table_preview'/name).read_text()
    assert 'Overfull' not in log and 'Undefined control sequence' not in log
audit=dict(status='passed',old_rows=8,old_means_preserved=48,old_sample_sds_preserved=48,
    test_records=220,validation_records=92,lambda_candidates=11,validation_test_id_disjoint=True,
    example_checks=checks,static_source_hash_checks=source_checks,pdfs=pdfs,
    original_six_metric_cells_verbatim_in_companion=True,
    visual_review='pending',publication_tex_sha256=base.sha(PUB/'single_clap_benchmark.tex'))
base.write_json(OUT/'validation.json',audit)
report=ROOT/'reports/single_clap_benchmark'
for name in ('result.json','summary.csv','canonical_exports.json','README.md'):
    dest=OUT/'original/report'/name
    if not dest.exists():shutil.copyfile(report/name,dest)
shutil.copyfile(OUT/'table1.csv',report/'summary.csv')
previous=json.loads((report/'result.json').read_text())
previous['main_table']=current
previous['crop_6ms_addition']=dict(report='reports/cropped_excitation_baseline/result.json',
    selected_lambda=result['selected_lambda']['lambda_relative'],
    existing_rows_unchanged=True,compact_table_metrics=['edc_rmse_db','abs_edt_error_ms','lsd_db','nrmse'],
    original_six_metrics_preserved=True)
base.write_json(report/'result.json',previous)
base.write_json(report/'canonical_exports.json',dict(status='synchronized',
    report='reports/cropped_excitation_baseline',
    exports=[dict(source=str(OUT/('table1.'+ext)),destination=str(PUB/('single_clap_benchmark.'+ext)),
                  sha256=base.sha(PUB/('single_clap_benchmark.'+ext))) for ext in ('tex','csv','md','pdf')],
    six_metric_companion='publication/paper/tables/post_meeting_table1_six_metrics.tex'))
readme=(report/'README.md').read_text()
addition='\n## Six-millisecond control added\n\nThe current publication table adds a separately validation-selected6-ms crop baseline. All previous means and sampleSDs are preserved in CSV and the six-metric companion. The compact one-column table reports EDC, EDT(ms), LSD(dB), andNRMSE. LSD is derived perrecord before aggregation. Full selection, scores, comparison, and audits: `../cropped_excitation_baseline/README.md`.\n'
if '## Six-millisecond control added' not in readme:(report/'README.md').write_text(readme+addition)
print(json.dumps(audit,indent=2))
