#!/usr/bin/env python3
"""Deterministic ARPEGE figures from frozen predictions, with no inference."""
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'experiments')]
from arpege_external_evaluation import REPORT, METRICS, read_csv, write_csv, write_json, sha, verify_lock
from claprir.metrics.lundeby_truncation import edc_truncated

OUT = ROOT/'publication/figures'
COLORS = dict(regression='#245b85', flow='#bd5425', target='#222222')
NAMES = dict(regression='1 s Regression', flow='1 s Flow', target='Measured RIR reference')
LABELS = ['EDC RMSE (dB)', r'$|C_{50}$ error$|$ (dB)', '|EDT error| (s)',
          'EDP RMSE', 'STFT log-mag MSE', 'NRMSE (companion)']
TABLE_LABELS = ['EDC RMSE (dB)', 'Absolute C50 error (dB)', 'Absolute EDT error (s)',
                'EDP RMSE', 'STFT log-mag MSE', 'NRMSE (companion)']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})


def reduce_rows(rows, keys, reducer=np.median):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for group, subset in sorted(groups.items()):
        r=dict(zip(keys,group)); r['n_children']=len(subset)
        for metric in METRICS:
            a=np.array([float(s[metric]) if s[metric] not in ('',None) else np.nan for s in subset])
            a=a[np.isfinite(a)]
            r[metric]=float(reducer(a)) if len(a) else float('nan')
            r[f'{metric}_n_valid']=len(a)
        out.append(r)
    return out


def save(fig, name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        legend=ax.get_legend()
        if legend and not np.isfinite(legend.get_window_extent(renderer).get_points()).all():
            raise RuntimeError('Invalid legend geometry')
    for ext in ('pdf','png'):
        dst=OUT/f'{name}.{ext}'
        fig.savefig(dst,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
        shutil.copy2(dst,REPORT/'figures'/dst.name)
    plt.close(fig)


def main():
    lock, contract=verify_lock()
    rows=read_csv(REPORT/'per_example.csv')
    protocol=json.loads((REPORT/'protocol.json').read_text())
    expected=len(lock['events'])*len(protocol['training_seeds'])*(1+protocol['flow_draws'])
    if len(rows)!=expected: raise RuntimeError('Incomplete comparison')
    keys=[(r['pairing_id'],r['arm'],r['training_seed'],r['draw']) for r in rows]
    if len(set(keys))!=len(keys): raise RuntimeError('Duplicated scoring records')
    OUT.mkdir(exist_ok=True); (REPORT/'figures').mkdir(exist_ok=True)
    seeds=reduce_rows(rows,['room','position','array','pairing_id','arm','training_seed'],np.mean)
    records=reduce_rows(seeds,['room','position','array','pairing_id','arm'])
    positions=reduce_rows(records,['room','position','arm'])
    rooms=reduce_rows(positions,['room','arm'])
    overall=reduce_rows(rooms,['arm'])
    write_csv(REPORT/'record_summary.csv',records)
    write_csv(REPORT/'room_position_summary.csv',positions)
    write_csv(REPORT/'room_summary.csv',rooms)
    room_ids=sorted({r['room'] for r in records},key=int)
    fig,axes=plt.subplots(2,3,figsize=(12.8,7.1),layout='constrained')
    for ax,metric,label in zip(axes.flat,METRICS,LABELS):
        for xi,room in enumerate(room_ids):
            ids=sorted({r['pairing_id'] for r in records if r['room']==room})
            jitter=dict(zip(ids,np.linspace(-.27,.27,len(ids))))
            for arm,offset in [('regression',-.025),('flow',.025)]:
                rr=[r for r in records if r['room']==room and r['arm']==arm]
                ax.scatter([xi+jitter[r['pairing_id']]+offset for r in rr],
                           [r[metric] for r in rr],s=17,color=COLORS[arm],alpha=.38,linewidths=0)
                pp=[r for r in positions if r['room']==room and r['arm']==arm]
                ax.scatter(np.linspace(xi-.24,xi+.24,len(pp))+offset,
                           [r[metric] for r in pp],s=65,marker='_',color=COLORS[arm],alpha=.8)
                summary=next(r for r in rooms if r['room']==room and r['arm']==arm)
                ax.scatter(xi+offset*4,summary[metric],s=46,marker='D',
                           color=COLORS[arm],edgecolor='white',linewidth=.5,zorder=5)
        ax.set_xticks(range(len(room_ids)),[f'Room {r}' for r in room_ids])
        ax.set_ylabel(label); ax.set_ylim(bottom=0); ax.grid(axis='y',alpha=.16)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fig.suptitle('ARPEGE: paired external transfer against measured RIR references',fontsize=13)
    handles=[Line2D([],[],color=COLORS[a],lw=2,label=NAMES[a]) for a in ('regression','flow')]
    handles += [Line2D([],[],color='.4',marker=m,ls='',alpha=alpha,ms=size,label=text)
        for m,alpha,size,text in [('o',.4,4,f'Receiver recording ({len(records)//2} / model)'),
                                 ('_',.8,8,f'Position median ({len(positions)//2} / model)'),
                                 ('D',1,5,f'Room median ({len(room_ids)} / model)')]]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    save(fig,'arpege_external_transfer')

    with np.load(REPORT/'examples.npz') as f: ex={k:f[k] for k in f.files}
    if ex['pairing_id'].astype(str).tolist()!=lock['examples']: raise RuntimeError('Representative membership changed')
    n=len(ex['pairing_id'])
    fig,axes=plt.subplots(n,4,figsize=(13.5,3.05*n),squeeze=False,layout='constrained')
    for i in range(n):
        bound=int(ex['bound'][i]); time=np.arange(bound)/44.1
        title=f"Room {ex['room'][i]}, position {ex['position'][i]}, {ex['array'][i]}"
        a=axes[i,0]
        a.plot(np.arange(44100)/44.1,ex['observation'][i],color='#555555',lw=.45)
        a.set_title(title+'\nSingle-clap observation'); a.set_ylabel('Peak-normalized amplitude')
        a.set_xlim(0,1000); a.set_xlabel('Time (ms)')
        for arm in ('regression','flow','target'):
            signal=ex[arm][i,:bound]
            style=dict(color=COLORS[arm],lw=.55 if arm=='target' else .7,
                       ls='--' if arm=='target' else '-',label=NAMES[arm],alpha=.85 if arm=='target' else .9)
            axes[i,1].plot(time,signal,**style)
            axes[i,2].plot(time,edc_truncated(signal,bound),**{**style,'lw':.8 if arm=='target' else 1.25})
            early=min(bound,round(.030*44100))
            axes[i,3].plot(time[:early],signal[:early],**style)
        for j in (1,2,3):
            axes[i,j].set_xlabel('Time after independent onset (ms)')
            axes[i,j].legend(loc='best',framealpha=.9,fontsize=7)
        axes[i,1].set_title('RIR: measured reference and predictions')
        axes[i,1].set_ylabel('Amplitude (reference peak units)'); axes[i,1].set_xlim(0,1000)
        axes[i,2].set_title('Energy decay over measured support')
        axes[i,2].set_ylabel('EDC (dB)'); axes[i,2].set_xlim(0,1000); axes[i,2].set_ylim(-70,1)
        axes[i,3].set_title('Early RIR detail (0–30 ms)')
        axes[i,3].set_ylabel('Amplitude (reference peak units)'); axes[i,3].set_xlim(0,30)
        for a in axes[i]: a.grid(alpha=.15); a.xaxis.set_minor_locator(AutoMinorLocator(2))
    fig.suptitle('ARPEGE: original 1 s models and same-location measured references\n'
                 'Fixed seed 42001; one fixed Flow draw; examples selected before inference',fontsize=12)
    save(fig,'arpege_examples')

    sources=[REPORT/n for n in ['pairing_manifest.csv','pairing_audit.json','event_manifest.csv',
        'evaluation_lock.json','protocol.json','model_provenance.json','inference_backend.json',
        'execution_amendment.json','per_example.csv','examples.npz']]
    inputs={str(p.relative_to(ROOT)):sha(p) for p in sources}
    figures={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.glob('arpege_*')) if p.suffix in ('.png','.pdf')}
    result=dict(status='scored-and-rendered',scope=protocol['scope'],evaluation_only=True,
        n_paired_recordings=len(lock['events']),n_position_configurations=len(positions)//2,n_rooms=len(room_ids),
        n_prediction_rows=len(rows),n_training_seeds=len(protocol['training_seeds']),flow_draws=protocol['flow_draws'],
        excluded_events=lock['excluded'],overall_medians_across_room_medians=overall,
        aggregation=protocol['aggregation'],representatives=lock['examples'],input_sha256=inputs,
        figure_sha256=figures,figure_script_sha256=sha(Path(__file__)),checkpoint_provenance=contract['models'],
        reference_directivity_matched=False,statistical_inference='descriptive only; no significance test')
    write_json(REPORT/'result.json',result)
    lines=['# ARPEGE external paired-clap evaluation','',
        f"Completed: {len(lock['events'])} recorded-clap/receiver pairs, {len(positions)//2} position configurations, {len(room_ids)} rooms; {len(rows)} predictions.",'',
        'Original final E3 hybrid/U-Net Regression and Flow, seed 42001 and fixed 20k checkpoints. '
        'Flow uses the imported E3 log-SNR sampler: 20 NFE, churn 0.5, sigma_d 0.027279, projection OFF. '
        'One frozen Flow draw per record; no output gain/delay fitting or best-of-K selection. '
        'ARPEGE is external evaluation only and is absent from training/fine-tuning.','',
        'Both final arms run on the same CUDA device. An incomplete CPU execution was retained as diagnostics '
        'after a throughput check; execution_amendment.json preserves its original lock and explains the backend switch. '
        'Only the matched CUDA predictions enter these figures. The data, checkpoints, sampler and selection rules did not change.','',
        '## Room-level descriptive summaries','',
        '| Metric | Regression | Flow |','|---|---:|---:|']
    summaries={r['arm']:r for r in overall}
    for k,label in zip(METRICS,TABLE_LABELS):
        lines.append(f"| {label} | {summaries['regression'][k]:.4g} | {summaries['flow'][k]:.4g} |")
    lower = {arm:[TABLE_LABELS[j] for j,k in enumerate(METRICS)
                  if summaries[arm][k] < summaries['flow' if arm=='regression' else 'regression'][k]]
             for arm in ('regression','flow')}
    lines += ['', 'At the median-across-room-median level, Regression has lower '+', '.join(lower['regression'])+
        '; Flow has lower '+', '.join(lower['flow'])+'. This is a metric-specific descriptive result, '
        'and does not establish superiority on every position or across training seeds.','',
        'Every headline number is a median across room medians. Within each room, arrays are first summarized within position. '
        'The CSVs preserve valid counts for each metric; missing EDT is not converted to zero. '
        'This single-checkpoint comparison does not estimate training-seed or posterior uncertainty.','',
        '## Pairing and preprocessing','',
        'The official annotation supplies source coordinates, receiver coordinates and receiver orientation. '
        'All retained pairs pass exact metadata equality. Front32/EM32 and front64/EM64 follow the official notebook. '
        'The first physical capsule (zero-based channel 0) is used in both recordings, with no channel averaging. '
        'Clap annotation paths lack release suffixes; each maps to exactly one WAV in its annotated directory. '
        'Path aliases are retained in pairing_manifest.csv, and source/receiver equality is checked independently.','',
        'Each long recording supplies its first eligible isolated clap, selected entirely from the input before inference. '
        'The existing Spheres envelope detector is reused with all detected peaks retained for adjacency checks: '
        '0.5 s preceding gap, 1.05 s following gap, 5 ms prepad, full 1 s after alignment. '
        'Both observation and reference independently follow the E3/E5 44.1 kHz resampling, first half-peak onset and crop peak-normalization convention. '
        'There is no denoising or reference-guided observation alignment. Output waveforms retain model amplitude.','',
        'The reference is measured with a loudspeaker at the annotated source position. '
        'Hand-clap directivity is unknown and differs from the loudspeaker; waveform identity is not guaranteed by geometric pairing. '
        'There is no released clean clap excitation here, so these results do not evaluate blind x recovery.','',
        '## Figures and captions','',
        '- `publication/figures/arpege_external_transfer.{pdf,png}`: external transfer against same-location measured RIRs. '
        'Six acoustic/waveform panels show receiver recordings, position medians and room medians for both original 1 s models. '
        'All finite outliers remain visible; NRMSE is a companion. Descriptive evidence from six rooms.','',
        '- `publication/figures/arpege_examples.{pdf,png}`: one-clap observation, measured/predicted RIR, EDC and a separate 0–30 ms RIR detail. '
        'Black thin dashed curves are measured references; blue/orange are Regression/Flow. EDC display is limited to -70 dB, while scoring uses full valid support. '
        'Rooms are the first/middle/last sorted room IDs, and the record with smallest SHA256(pairing_id) within each is selected before predictions. '
        'Seed 42001 and Flow draw 0 are fixed for every panel.','',
        'Allowed claims: descriptive external RIR transfer against paired measured references. '
        'Unsupported: a new global model winner, matched source directivity, absolute calibrated gain, significance across 56 independent rooms, or clean-excitation recovery.','',
        '## Reproduction and exact inputs','',
        'Metadata: `python experiments/arpege_pairing.py`.',
        
        'Run `experiments/arpege_external_evaluation.py prepare`, then `score --threads 2`, then `figures/arpege_external_figures.py`. '
        'Plotting needs only committed CSV/NPZ/report artifacts and never reruns inference. Full prediction caches live under `$CLAPRIR_RUNTIME/arpege_external_evaluation/`.','',
        'For scoring set `CUDA_VISIBLE_DEVICES=$CLAPRIR_GPU_UUID`, '
        '`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OPENBLAS_NUM_THREADS=2`, and `OMP_NUM_THREADS=2`. '
        'The evaluator caps its process at 15% of the existing physical GPU memory and refuses another device.','',
        'Exact source artifacts and SHA256 hashes are in result.json, including pairing manifest, evaluation lock, checkpoint paths, model and sampler sources, per-example CSV and selected waveform NPZ. '
        'Raw data provenance: https://zenodo.org/records/20622134 and https://github.com/jdpascal/ARPEGE.','']
    (REPORT/'README.md').write_text('\n'.join(lines))
    write_json(REPORT/'state.json',dict(status='scored-and-rendered-awaiting-visual-review',records=len(rows)))
    print(json.dumps({'figures':figures,'summary':overall},indent=2))


if __name__=='__main__': main()
