#!/usr/bin/env python3
"""E1/E2 figure from committed 1 s artifacts; no inversion rerun."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    source=ROOT/'reports/excitation_recoverability/result.json'; data=json.loads(source.read_text())
    providers=['mit','but','ace','openair']; metrics=[('edc_rmse_db','EDC RMSE (dB)'),('abs_c50_error_db','|C50 error| (dB)'),('abs_edt_error_s','|EDT error| (s)'),('stft_logmag_mse','STFT log-mag MSE')]
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.1,3.8),layout='constrained')
    for ax,(key,label) in zip(axes.flat,metrics):
        for arm,color,marker,title in [('E1_true_clap','#245b85','o','E1: true clap'),('E2_3ms_crop','#a85326','s','E2: 3 ms surrogate')]:
            # Keep arm names faithful to the result artifact.
            if arm not in data['medians']: arm=next(k for k in data['medians'] if k.startswith('E2'))
            y=[data['medians'][arm][p][key] for p in providers]
            ax.plot(range(4),y,marker=marker,lw=.8,color=color,label=title)
        ax.set_xticks(range(4),['MIT','BUT','ACE','OpenAIR']); ax.set_ylabel(label); ax.set_yscale('log'); ax.grid(axis='y',alpha=.2)
    for ax in axes.flat: ax.legend(fontsize=6);
    for ext in ('pdf','png'): fig.savefig(ROOT/f'publication/figures/excitation_recoverability.{ext}',dpi=220,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
    (ROOT/'reports/excitation_recoverability/figure_provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),statistic='provider medians from frozen 1 s result.json',axes='logarithmic; no statistical significance implied; synthetic exact-silence metrics excluded'),indent=2)+'\n')
if __name__=='__main__': main()
