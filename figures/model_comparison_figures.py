#!/usr/bin/env python3
"""User-requested line comparisons; no inference. Both models are final 1 s arms."""
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from claprir.metrics.lundeby_truncation import edc_truncated
from claprir.plotting.style import readable
OUT=ROOT/'publication/figures'; REPORT=ROOT/'reports/real_clap_and_phone_figures'
LABELS={'MediumSizeMeetingRoom':'Meeting room','SmallPCRoom':'Small PC room','ClubroomSmall':'Small clubroom','ClubroomMedium':'Medium clubroom','AcousticsHallway':'Acoustics hallway','Bathroom':'Bathroom','Elevator':'Elevator'}
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    for ext in ('pdf','png'): fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
    plt.close(fig)
def main():
    ex=ROOT/'reports/real_clap_room_evaluation/results/examples.npz'; flow=REPORT/'e5_flow_examples.npz'
    with np.load(ex) as d,np.load(flow) as f:
        assert np.array_equal(d['index'],f['index'])
        fig,axes=plt.subplots(3,3,figsize=(10.8,8.4),layout='constrained')
        for j,quantile in enumerate(('Lower quartile','Median','Upper quartile')):
            t=np.arange(11025)/44.1
            series=[(f['prediction'][j,:11025],'1 s Flow','#a85326','-',2),(d['prediction'][j],'1 s Regression','#245b85','-',3),(d['reference'][j],'Sweep reference','black','--',5)]
            for y,label,color,ls,z in series:
                axes[0,j].plot(t,y,color=color,ls=ls,lw=.45,label=label,zorder=z,alpha=.85)
                axes[1,j].plot(t[:221],y[:221],color=color,ls=ls,lw=.7,label=label,zorder=z)
                axes[2,j].plot(t,edc_truncated(y,11025),color=color,ls=ls,lw=1.1,label=label,zorder=z)
            axes[0,j].set_title(quantile+'\n'+readable(str(d['mic'][j]),str(d['position'][j])).replace(', ','\n'),fontsize=8)
            for row in range(3):
                ax=axes[row,j]
                ax.set_xlabel('Time (ms)'+(' — early 5 ms' if row==1 else ''))
                ax.set_ylabel('EDC (dB)' if row==2 else 'Amplitude (target peak units)')
                ax.set_xlim(0,5 if row==1 else 250)
                ax.xaxis.set_major_locator(MultipleLocator(1 if row==1 else 50))
                ax.xaxis.set_minor_locator(MultipleLocator(.2 if row==1 else 10))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.grid(which='major',alpha=.2); ax.grid(which='minor',alpha=.07)
                ax.legend(fontsize=6,loc='lower left' if row==2 else 'upper right',framealpha=.95)
            axes[2,j].set_ylim(-70,2)
            axes[2,j].yaxis.set_major_locator(MultipleLocator(10))
        fig.suptitle('E5: 1 s models; frozen 250 ms paired sweep-reference support',fontsize=11)
        save(fig,'e5_examples')
    sel=json.loads((REPORT/'provenance.json').read_text())
    regpath=ROOT/'reports/phone_deployment_evaluation/results/edc_curves.npz'; flowpath=REPORT/'e6_flow_comparison_curves.npz'
    with np.load(regpath) as z: reg={(int(s),str(k)):v for s,k,v in zip(z['seed'],z['sample_id'],z['edc_db'])}
    with np.load(flowpath) as z: fm={str(k):v for k,v in zip(z['sample_id'],z['edc_db'])}
    rows=list(csv.DictReader((ROOT/'reports/phone_deployment_evaluation/results/per_clap.csv').open()))
    fig,axes=plt.subplots(3,3,figsize=(10.8,8.0),sharex=True,sharey=True,layout='constrained')
    for i,room in enumerate(sel['E6_example_rooms']):
        for j,mode in enumerate(sel['E6_example_modes'][room]):
            rs=sorted([r for r in rows if r['seed']=='42001' and r['room_name']==room and r['clap_mode']==mode],key=lambda r:int(r['repeat_id'])); assert len(rs)==5
            ax=axes[i,j]
            for lookup,label,color in ((reg,'1 s Regression','#245b85'),(fm,'1 s Flow','#a85326')):
                y=np.array([lookup[42001,r['sample_id']] if label=='1 s Regression' else lookup[r['sample_id']] for r in rs])
                for curve in y: ax.plot(np.arange(1000),curve,color=color,lw=.55,alpha=.3)
                ax.plot(np.arange(1000),np.median(y,axis=0),color=color,lw=1.3,label=label)
            ax.set_title(f'{LABELS[room]} / {mode}',fontsize=9); ax.set_xlim(0,1000); ax.set_ylim(-70,2)
            ax.xaxis.set_major_locator(MultipleLocator(200)); ax.xaxis.set_minor_locator(MultipleLocator(50))
            ax.yaxis.set_major_locator(MultipleLocator(10)); ax.yaxis.set_minor_locator(MultipleLocator(5))
            ax.grid(which='major',alpha=.2); ax.grid(which='minor',alpha=.07)
            ax.legend(fontsize=7,loc='lower left',title='Thick: median; thin: 5 claps',title_fontsize=6)
            if j==0: ax.set_ylabel(('Most stable','Median stability','Least stable')[i]+'\nPredicted EDC (dB)')
            if i==2: ax.set_xlabel('Time (ms)')
    fig.suptitle('E6: predictions only — no reference RIR',fontsize=10)
    save(fig,'e6_phone_examples')
    sources=[ex,flow,regpath,flowpath,REPORT/'model_comparison_provenance.json']
    (REPORT/'comparison_figure_sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print('Redrew 1 s model comparison line figures')
if __name__=='__main__': main()
