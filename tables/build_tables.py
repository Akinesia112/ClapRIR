#!/usr/bin/env python3
"""Build only frozen E1/E2/E3/E5/E6 paper tables; no historical multiclap data."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'experiments'))
from matched_estimator_report import load,collapse
OUT=Path(__file__).resolve().parent/'tables'
P=('shoebox','mit','but','ace','openair'); LABEL={'shoebox':'Shoebox','mit':'MIT','but':'BUT','ace':'ACE','openair':'OpenAIR'}
METRICS=('edc_rmse_db','abs_c50_error_db','abs_edt_error_s','echo_density_rmse','stft_logmag_mse','nrmse')
def main():
    rows=load(ROOT/'reports/matched_estimator_comparison/results/per_example.csv')
    corrected=load(ROOT/'reports/shoebox_pipeline_recheck/support_corrected.csv')
    assert len(corrected)==72 and all(r['provider']=='shoebox' and r['bound']==4096 for r in corrected)
    rows=[r for r in rows if r['provider']!='shoebox']+corrected
    collapsed={m:collapse(rows,m) for m in METRICS}
    lines=[r'% Generated from original E3 per_example.csv + shoebox_pipeline_recheck/support_corrected.csv; aggregation unchanged.',r'\begin{table*}[t]',r'\centering\scriptsize',r'\caption{E3: matched 1 s estimators. Provider medians aggregate draws within seed/room, then seeds within room and rooms within provider. $\dagger$ marks Shoebox: all six metrics use its retained 4096-sample (92.88 ms) support. Source reconstruction proved that longer simulated responses had been cropped before padding to 1 s; scoring was corrected on unchanged model predictions. This is not a full-second synthetic decay benchmark. Measured-provider rows are unchanged. BUT (two rooms) and ACE (one room) are descriptive. NRMSE is a companion.}',r'\label{tab:main}',r'\begin{tabular}{llrrrrrrrr}',r'\toprule',r'Provider & Estimator & EDC RMSE (dB) & $|\Delta C_{50}|$ (dB) & $|\Delta\mathrm{EDT}|$ (s) & EDP RMSE & Log-mag MSE & NRMSE & Active M & Inference \\',r'\midrule']
    for p in P:
        for arm in ('regression','flow'):
            vals=[]
            for metric in METRICS:
                v=[v for (a,provider,room),v in collapsed[metric].items() if a==arm and provider==p]
                vals.append(f'{np.median(v):.3f}'+(r'$\dagger$' if p=='shoebox' and metric=='edc_rmse_db' else ''))
            lines.append(' & '.join([LABEL[p],arm.capitalize()]+vals+(['26.53','1 forward'] if arm=='regression' else ['29.31','20 NFE']))+r' \\')
        lines.append(r'\addlinespace')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (OUT/'main_table.tex').write_text('\n'.join(lines)+'\n')
    contrasts=json.loads((ROOT/'reports/matched_estimator_comparison/results/contrasts.json').read_text())
    names={'edc_rmse_db':'EDC RMSE (dB)','abs_c50_error_db':r'$|\Delta C_{50}|$ (dB)','abs_edt_error_s':r'$|\Delta\mathrm{EDT}|$ (s)','echo_density_rmse':'EDP RMSE','stft_logmag_mse':'Log-mag MSE','nrmse':'NRMSE (companion)'}
    lines=[r'% Generated from frozen E3 contrasts.json; no bootstrap rerun.',r'\begin{table}[t]',r'\centering\scriptsize',r'\caption{Paired room-level Flow-minus-Regression median differences and frozen 95\% room-cluster intervals. Positive favors Regression. MIT has 42 rooms, OpenAIR nine. These are not differences between marginal medians.}',r'\label{tab:e3paired}',r'\begin{tabular}{llr}',r'\toprule',r'Provider & Metric & Delta [95\% interval] \\',r'\midrule']
    for p in ('mit','openair'):
        for m in METRICS:
            r=next(x for x in contrasts if x['provider']==p and x['metric']==m)
            lines.append(f"{LABEL[p]} & {names[m]} & {r['median_delta']:+.6f} [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUT/'e3_paired_contrasts.tex').write_text('\n'.join(lines)+'\n')
    e5=json.loads((ROOT/'reports/real_clap_room_evaluation/result.json').read_text())
    e6=json.loads((ROOT/'reports/phone_deployment_evaluation/result.json').read_text())
    v=e5['mean_of_seed_means']; q=e6['overall_median_within_cell_iqr']
    (OUT/'real_world_numbers.tex').write_text('\n'.join([r'% Generated E5/E6 numbers; no E6 reference error.',f'\\newcommand{{\\EfiveEDC}}{{{v["edc_rmse_db"]:.3f}}}',f'\\newcommand{{\\EfiveC}}{{{v["abs_c50_error_db"]:.3f}}}',f'\\newcommand{{\\EfiveEDT}}{{{v["abs_edt_error_s"]:.3f}}}',f'\\newcommand{{\\EsixC}}{{{q["c50_db"]:.3f}}}',f'\\newcommand{{\\EsixEDT}}{{{q["edt_s"]:.4f}}}'])+'\n')
    print('Rebuilt frozen E3 table and E5/E6 numeric macros')
if __name__=='__main__': main()
