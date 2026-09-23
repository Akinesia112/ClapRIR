#!/usr/bin/env python3
"""Describe corrected diagnostics without near-terminal or ratio overclaims."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/flow_audit_corrected'; P=['shoebox','mit','but','ace','openair']
def read(pattern):
    rows=[]
    for p in sorted(OUT.glob(pattern)): rows.extend(csv.DictReader(p.open()))
    return rows

def main():
    assert (OUT/'completion.json').exists()
    traj=read('*_trajectory.csv'); spec=read('*_spectral.csv'); result={}; summaries=[]
    keys=['teacher_Ev','shuffled_teacher_Ev','D_x','D_v','teacher_endpoint','rollout_endpoint']
    for provider in P:
        full=[r for r in traj if r['provider']==provider and r['window_ms']=='0-1000']
        ts=sorted({float(r['t']) for r in full}); noise=[r for r in full if float(r['t'])==0.]
        s=[r for r in spec if r['provider']==provider and r['window_ms']=='0-1000' and int(r['bins'])>0]
        magnitude=sum(float(r['magnitude_sum']) for r in s); total=sum(float(r['complex_sum']) for r in s)
        middle=[t for t in ts if .5<=t<=.95]
        means=[float(np.mean([float(r['rollout_endpoint']) for r in full if float(r['t'])==t])) for t in middle]
        result[provider]=dict(magnitude_share=magnitude/total,phase_timing_share=1-magnitude/total,
            noise_teacher_Ev=float(np.mean([float(r['teacher_Ev']) for r in noise])),
            noise_shuffled_Ev=float(np.mean([float(r['shuffled_teacher_Ev']) for r in noise])),
            mid_times=middle,mid_rollout_endpoint_mse=means,mid_last_first_ratio=means[-1]/means[0],
            n_examples_per_seed=len(noise)//3,n_rooms=len({r['room_id'] for r in full}))
        for window in ['0-70','70-250','250-500','500-1000','0-1000']:
            for t in ts:
                rows=[r for r in traj if r['provider']==provider and r['window_ms']==window and float(r['t'])==t]
                valid=[r for r in rows if int(r['reportable'])]
                # at least two physical examples, all training seeds share support
                reportable=len(valid)>=6
                summary=dict(provider=provider,window_ms=window,t=t,n_valid_examples_per_seed=len(valid)//3,n_valid_samples_per_seed=sum(int(r['valid_samples']) for r in valid)//3,reportable=int(reportable))
                for k in keys: summary[k]=float(np.mean([float(r[k]) for r in valid])) if reportable else float('nan')
                summaries.append(summary)
    result['interpretation']='Post-hoc descriptive audit; initial noise paired but later churn contributes to D_x/D_v. Teacher endpoint convergence is geometric, not evidence of field accuracy. Conditioning and rollout behavior may coexist; no global causal claim.'
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'window_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(summaries[0]),lineterminator="\n"); w.writeheader(); w.writerows(summaries)
    lines=['# Corrected Flow mechanism and rollout audit','','Existing final 1 s Flow checkpoints, seeds 42001/02/03. First up to 16 test examples per provider; post-hoc diagnostic subset, not the full E3 benchmark. Exact E3 sampler is imported: log-SNR, sigma_d=0.027279, gamma=0.5, 20 NFE, no projection. No model training or shipped predict() is used.','','| Provider | Examples/seed | Rooms | Magnitude share | Phase/timing share | Teacher velocity MSE at t=0 | Shuffled at t=0 | Endpoint MSE at t=0.5354 | Endpoint MSE at t=0.9193 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in P:
        v=result[p]; lines.append(f"| {p} | {v['n_examples_per_seed']} | {v['n_rooms']} | {v['magnitude_share']:.3f} | {v['phase_timing_share']:.3f} | {v['noise_teacher_Ev']:.6g} | {v['noise_shuffled_Ev']:.6g} | {v['mid_rollout_endpoint_mse'][0]:.6g} | {v['mid_rollout_endpoint_mse'][-1]:.6g} |")
    lines+=['','The last two columns are absolute endpoint MSE at the actual mid/late sampler points. Ratios remain only in machine-readable diagnostics. Full per-time absolute errors are in window_summary.csv and per-example trajectory files.','','## Interpretation boundaries','','Waveform velocity MSE already supervises timing/phase. Residual phase/timing dominance does not establish absence of phase supervision. Magnitude/phase decomposition is checked numerically and uses STFT of each actual valid crop; padding is not a target. This differs from historical full-padded-window spectral aggregation, so old approximate shares are not copied into this report.','','Teacher endpoint error obeys E_h=(1-t)^2 E_v; its near-terminal decline is not a field-quality result. Noise-side shuffled-condition changes characterize observation dependence; near-data-side changes do not. D_x/D_v use the actual initial rollout noise as teacher epsilon. Later churn adds randomness, so divergence includes stochastic coupling differences and cannot alone prove off-manifold degradation.','','## Support','','All waveform quantities are valid-mask means PER EXAMPLE. Windows: 0–70, 70–250, 250–500, 500–1000 ms plus full support. Per-example windows under 64 samples are N/A; summaries with fewer than two physical examples are N/A. Spectral crops require 510 samples. window_summary.csv reports valid examples and samples per seed; repeating training seeds does not increase support. Provider counts refer to the subset, not all E3 rooms. No bootstrap significance is invented.','','## Historical validity','','The old 250 ms Audit 3 used the shipped uniform-Euler sampler and is invalid for final-sampler conclusions. Its teacher/condition-shuffle analyses are separate. The earlier rollout implementation also sampled independent teacher noise despite its paired-noise description; this rerun fixes that. No historical CSV is overwritten.','','Reproduce: `PYTHONPATH=src python experiments/flow_audit_corrected.py` then `python experiments/flow_mechanism_audit.py`. Completed seed/provider groups are reused. Toy trace/prefix equality and real-checkpoint traced/untraced equality guard the single E3 sampler implementation.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.7),layout='constrained')
    for p in P:
        rows=[r for r in summaries if r['provider']==p and r['window_ms']=='0-1000' and r['t']<=.95]
        for ax,key in zip(axes.flat,['teacher_Ev','rollout_endpoint','D_x','D_v']): ax.plot([r['t'] for r in rows],[r[key] for r in rows],label=p,marker='.',lw=1)
    for ax,label in zip(axes.flat,['Teacher velocity MSE','Own-rollout endpoint MSE','State divergence MSE (includes churn)','Velocity divergence MSE (includes churn)']):
        ax.set_xlabel('Sampler time t'); ax.set_ylabel(label); ax.set_yscale('symlog',linthresh=1e-6); ax.grid(alpha=.15)
    for ax in axes.flat: ax.legend(ncol=2,fontsize=6)
    for ext in ('pdf','png'): fig.savefig(ROOT/f'publication/figures/flow_mechanism_current.{ext}',dpi=220,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
    tex=[r'\begin{table}[t]',r'\centering\small',r'\caption{Corrected final-sampler complex-STFT decomposition on actual valid support. Shares are descriptive on the audit subset, not full-benchmark inference.}',r'\label{tab:phase}',r'\begin{tabular}{lrr}',r'\toprule',r'Provider & Magnitude share & Phase/timing share \\',r'\midrule']
    for p in P: tex.append(f"{p} & {result[p]['magnitude_share']:.3f} & {result[p]['phase_timing_share']:.3f}"+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\begin{figure*}[t]',r'\centering\includegraphics[width=\textwidth]{flow_mechanism_current.pdf}',r'\caption{Corrected existing-checkpoint Flow diagnostics on valid support. Teacher velocity error and own-rollout endpoint error are absolute MSE. Divergence includes later churn randomness; no global causal mechanism is identified. Only nonterminal times through 0.95 are plotted.}',r'\end{figure*}']
    (ROOT/'publication/paper/tables/flow_mechanism_current.tex').write_text('\n'.join(tex)+'\n')
    for p in P: print(p,result[p])
if __name__=='__main__': main()
