#!/usr/bin/env python3
"""Current replacement of original fig0/fig1/fig4/fig5 entry point.

Retains waveform-over-decay panels and reference-last black-dashed convention
of the original script (archived with its original checkpoint loader). Uses
only committed current result exports, so regeneration never invokes a stale
250 ms model, a joint-condition model, or Flow.predict().
"""
import argparse,json,sys,shutil,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src')]
from claprir.metrics.lundeby_truncation import edc_truncated
REF=dict(color='black',linestyle=(0,(4,2)),zorder=5)
OUT=ROOT/'publication/figures'
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 for ext in ('png','pdf'): fig.savefig(OUT/f'{name}.{ext}',dpi=200,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
 plt.close(fig)
def panels(d,indices,arms,name,title):
 fig,axes=plt.subplots(2,len(indices),figsize=(max(7.1,3.0*len(indices)),5.4),squeeze=False,layout='constrained')
 for col,i in enumerate(indices):
  b=4096 if str(d['provider'][i])=='shoebox' else int(d['bound'][i]);t=np.arange(b)/44.1
  for key,label,color in arms:
   e=d[key][i,:b];axes[0,col].plot(t,e,lw=.5,color=color,label=label,zorder=2)
   axes[1,col].plot(t,edc_truncated(e,b),lw=1,color=color,label=label,zorder=2)
  for row,values in ((0,d['target'][i,:b]),(1,edc_truncated(d['target'][i,:b],b))):
   ax=axes[row,col];ax.plot(t,values,lw=.3 if row==0 else .65,label='Target RIR',**REF)
   ax.set_xlim(0,t[-1]);ax.set_xlabel('Time (ms)');ax.grid(alpha=.2);ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.legend(fontsize=6,loc='upper right' if row==0 else 'lower left')
  axes[0,col].set_title(str(d['provider'][i]).upper()+' / '+str(d['room_id'][i])+('\n92.88 ms retained crop; corrected scoring support' if str(d['provider'][i])=='shoebox' else ''),fontsize=7)
  axes[0,col].set_ylabel('Amplitude (target peak units)');axes[1,col].set_ylabel('EDC (dB)');axes[1,col].set_ylim(-70,2)
 fig.suptitle(title,fontsize=10);save(fig,name)
def figure_one(d):
 # Original figure_one: one method per column, waveform above EDC.
 i=list(d['provider']).index('mit');b=4096 if str(d['provider'][i])=='shoebox' else int(d['bound'][i]);t=np.arange(b)/44.1
 arms=[('oracle','True-clap inversion','#568a35'),('crop','3 ms inversion','#9c70a7'),('regression','1 s Regression','#245b85'),('flow','1 s Flow','#a85326')]
 fig,axes=plt.subplots(2,4,figsize=(14,5.5),layout='constrained')
 for col,(key,name,color) in enumerate(arms):
  e=d[key][i,:b];h=d['target'][i,:b]
  for row,estimate,target in [(0,e,h),(1,edc_truncated(e,b),edc_truncated(h,b))]:
   ax=axes[row,col];ax.plot(t,estimate,lw=.45 if row==0 else 1,color=color,label=name,zorder=2);ax.plot(t,target,lw=.3 if row==0 else .65,label='Target RIR',**REF)
   ax.set_xlim(0,t[-1]);ax.set_xlabel('Time (ms)');ax.set_ylabel('Amplitude' if row==0 else 'EDC (dB)');ax.grid(alpha=.2);ax.legend(fontsize=6,loc='upper right' if row==0 else 'lower left')
   if row==1:ax.set_ylim(-70,2)
  axes[0,col].set_title(name)
 fig.suptitle('MIT: '+str(d['room_id'][i])+'; original waveform / decay layout; current 1 s models',fontsize=10)
 save(fig,'paper_fig1_oracle_crop_hybrid')
def figure_model_comparison(d):
 # Original figure_m1_vs_m2 four-panel layout, with model identities explicit.
 # k_max=1 checkpoints cannot be relabeled as jointly conditioned models.
 i=list(d['provider']).index('mit');b=4096 if str(d['provider'][i])=='shoebox' else int(d['bound'][i]);t=np.arange(b)/44.1;h=d['target'][i,:b]
 arms=[('regression','1 s Regression','#245b85'),('flow','1 s Flow','#a85326')]
 fig,axes=plt.subplots(2,2,figsize=(12,6),layout='constrained')
 for ax,(key,name,color) in zip(axes[0],arms):
  ax.plot(t,d[key][i,:b],lw=.4,color=color,label=name);ax.plot(t,h,lw=.3,label='Target RIR',**REF);ax.set_title(name);ax.set_ylabel('Amplitude')
 for key,name,color in arms:
  axes[1,0].plot(t,edc_truncated(d[key][i,:b],b),color=color,lw=1,label=name)
  axes[1,1].semilogy(t,np.abs(d[key][i,:b])+1e-7,color=color,lw=.3,alpha=.7,label=name)
 axes[1,0].plot(t,edc_truncated(h,b),lw=.65,label='Target RIR',**REF);axes[1,0].set_ylim(-70,2);axes[1,0].set_ylabel('EDC (dB)')
 axes[1,1].semilogy(t,np.abs(h)+1e-7,lw=.3,label='Target RIR',**REF);axes[1,1].set_ylabel('|RIR amplitude| + 1e-7')
 for ax in axes.flat:ax.set_xlim(0,t[-1]);ax.set_xlabel('Time (ms)');ax.legend(fontsize=7);ax.grid(alpha=.2)
 fig.suptitle('MIT: final 1 s single-clap Regression / Flow / target (original four-panel layout)',fontsize=11)
 save(fig,'fig4c_m1_vs_m2_example')
def figure_deconvolution_rows():
 # Replace old NRMSE-only bars using frozen acoustic metrics and current arms.
 sys.path.insert(0,str(ROOT/'experiments'))
 from matched_estimator_report import load,collapse
 rows=load(ROOT/'reports/matched_estimator_comparison/results/per_example.csv')
 inv=load(ROOT/'reports/excitation_recoverability/results/per_example.csv')
 for r in inv:r['seed']=0
 rows+=inv;providers=['mit','but','ace','openair']
 fig,axes=plt.subplots(2,2,figsize=(9,5),layout='constrained')
 for ax,(metric,label) in zip(axes.flat,[('edc_rmse_db','EDC RMSE (dB)'),('abs_c50_error_db','|C50 error| (dB)'),('abs_edt_error_s','|EDT error| (s)'),('stft_logmag_mse','STFT log-mag MSE')]):
  v=collapse(rows,metric)
  for arm,name,color,marker in [('E1_true_clap','True-clap inversion','#568a35','^'),('E2_crop_3ms','3 ms inversion','#9c70a7','v'),('regression','1 s Regression','#245b85','o'),('flow','1 s Flow','#a85326','s')]:
   vals=[np.median([value for (a,p,r),value in v.items() if a==arm and p==provider]) for provider in providers]
   ax.plot(range(4),vals,label=name,color=color,marker=marker,lw=.7,ms=4)
  ax.set_xticks(range(4),['MIT','BUT','ACE','OpenAIR']);ax.set_ylabel(label);ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=6)
 fig.suptitle('Current 1 s inversion controls and matched learned estimators; provider medians',fontsize=10)
 save(fig,'fig0_deconvolution_rows')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['fig0','fig1','fig2','fig4','fig4c','fig5']);a=ap.parse_args()
 # fig0 carries current acoustic recoverability, not historical NRMSE bars.
 if a.only in (None,'fig0'):
  figure_deconvolution_rows()
 if a.only in (None,'fig1','fig4','fig4c'):
  with np.load(ROOT/'reports/current_benchmark_figures/examples.npz') as d:
   if a.only in (None,'fig1'):
    figure_one(d)
   if a.only in (None,'fig4','fig4c'):
    figure_model_comparison(d)
    indices=list(range(len(d['provider'])))
    panels(d,indices,[('regression','1 s Regression','#245b85'),('flow','1 s Flow','#a85326')],'fig4_single_clap_models','Final 1 s models: five providers; one preselected example each')
 if a.only in (None,'fig2'):
  fig,ax=plt.subplots(figsize=(9,2.7));ax.set_axis_off()
  boxes=[(.03,.42,'One recorded clap y'),(.34,.72,'Regression: y → predicted RIR\n26.53 M; 1 forward; selected'),(.34,.12,'Flow: (y, state, t) → velocity\n29.31 M; 20 NFE; comparator'),(.76,.42,'1 s predicted RIR\nValidity-aware scoring')]
  for x,y,text in boxes:ax.text(x,y,text,transform=ax.transAxes,va='center',fontsize=8,bbox=dict(boxstyle='round,pad=.5',fc='white',ec='.35'))
  for start,end in [((.25,.45),(.34,.72)),((.25,.42),(.34,.12)),((.70,.72),(.76,.47)),((.70,.12),(.76,.39))]:ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=1))
  save(fig,'fig2_architecture')
 if a.only in (None,'fig5'):
  from model_comparison_figures import main as comparisons
  comparisons()
  for ext in ('png','pdf'):shutil.copyfile(OUT/f'e5_examples.{ext}',OUT/f'fig5_real_claps.{ext}')
 print('Current fig0/1/2/4/5 regenerated; historical K-clap plots are archived.')
if __name__=='__main__':main()
