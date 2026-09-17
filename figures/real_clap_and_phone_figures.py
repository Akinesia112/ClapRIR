#!/usr/bin/env python3
"""Paper figures from frozen result artifacts; deterministic, no inference.

E6 primary quantitative figures: median descriptor across training seeds for
EACH recorded clap, then keep the five repeats visible. Training seeds are not
additional acoustic observations. E6-C uses fixed training seed 42001.
"""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from claprir.metrics.lundeby_truncation import edc_truncated
OUT=ROOT/'publication/figures'; REPORT=ROOT/'reports/real_clap_and_phone_figures'
MODES=['P1','P2','P3','A1','A2','A3','A1-','A1+']
LABELS={'MediumSizeMeetingRoom':'Meeting room','SmallPCRoom':'Small PC room','ClubroomSmall':'Small clubroom','ClubroomMedium':'Medium clubroom','AcousticsHallway':'Acoustics hallway','Bathroom':'Bathroom','Elevator':'Elevator'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
SOURCES=[ROOT/'reports/real_clap_room_evaluation/results/per_example.csv',ROOT/'reports/phone_deployment_evaluation/results/per_clap.csv',ROOT/'reports/phone_deployment_evaluation/results/edc_curves.npz']

def read(p): return list(csv.DictReader(p.open()))
def save(fig,name):
    for ext in ('pdf','png'): fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
    plt.close(fig)
def main():
    OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    e5,e6=read(SOURCES[0]),read(SOURCES[1]); assert len(e5)==153 and len(e6)==840
    ids=sorted({int(r['index']) for r in e5})
    metrics=[('edc_rmse_db','EDC RMSE (dB)'),('abs_c50_error_db',r'$|C_{50}$ error$|$ (dB)'),('abs_edt_error_s','|EDT error| (s)'),('stft_logmag_mse','STFT log-mag MSE'),('nrmse','NRMSE (companion)')]
    vals={k:np.array([np.median([float(r[k]) for r in e5 if int(r['index'])==i]) for i in ids]) for k,_ in metrics}
    order=np.lexsort((ids,vals['edc_rmse_db']))
    fig,axes=plt.subplots(5,1,figsize=(10.8,9.0),sharex=True,layout='constrained')
    for ax,(key,label) in zip(axes,metrics):
        v=vals[key][order]; ax.scatter(np.arange(1,52),v,s=16,color='#245b85',zorder=3,label='1 s Regression: one position/mic pair\n(point = median across 3 training seeds)')
        ax.axhline(np.median(v),color='#245b85',ls='--',lw=1,label=f'Regression median across 51 pairs: {np.median(v):.3g}')
        ax.set_ylabel(label); ax.set_ylim(bottom=0); ax.grid(axis='y',alpha=.18); ax.legend(loc='upper left',fontsize=7,framealpha=.95)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    axes[-1].set_xlabel('Position/microphone-pair rank by Regression EDC error (51 pairs; 17 positions × 3 mics)')
    axes[0].set_title('E5: 1 s Regression against sweep reference; 250 ms scoring support')
    save(fig,'e5_real_clap_transfer')
    ex=ROOT/'reports/real_clap_room_evaluation/results/examples.npz'
    if ex.exists(): SOURCES.append(ex)
    # Preserve five physical recordings, not fifteen seed-replicates.
    cells={}
    for room in sorted({r['room_name'] for r in e6}):
        for mode in MODES:
            records=[r for r in e6 if r['room_name']==room and r['clap_mode']==mode]
            assert len(records)==15
            cells[room,mode]={key:np.array([np.median([float(r[key]) for r in records if int(r['repeat_id'])==k]) for k in range(1,6)]) for key in ('c50_db','edt_s')}
    rooms=sorted({r['room_name'] for r in e6},key=lambda room:int(next(r['room_id'] for r in e6 if r['room_name']==room)))
    fig,axes=plt.subplots(7,2,figsize=(10.8,15.4),sharex=True,layout='constrained')
    for i,room in enumerate(rooms):
        for j,(key,label) in enumerate((('c50_db',r'Predicted $C_{50}$ (dB)'),('edt_s','Predicted EDT (s)'))):
            ax=axes[i,j]
            for k,mode in enumerate(MODES):
                v=cells[room,mode][key]; ax.scatter(k+np.linspace(-.12,.12,5),v,s=11,color='#245b85',alpha=.65)
                ax.plot([k-.21,k+.21],[np.median(v)]*2,color='#245b85',lw=1.8)
            ax.set_title(LABELS[room],loc='left'); ax.set_ylabel(label); ax.grid(axis='y',alpha=.18)
            ax.set_xticks(range(8),MODES)
            allv=np.concatenate([cells[room,m][key] for m in MODES]); span=max(float(np.ptp(allv)),.12 if key=='edt_s' else 1.)
            ax.set_ylim(float(allv.min())-.65*span,float(allv.max())+.15*span)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            handles=[Line2D([],[],color='#245b85',marker='o',ls='',ms=4,alpha=.65,label='1 s Regression: 5 recorded claps\n(each point = median across 3 seeds)'),Line2D([],[],color='#245b85',lw=1.8,label='Median across the 5 claps')]
            ax.legend(handles=handles,loc='lower right',fontsize=6.5,framealpha=.95)
            ax.tick_params(axis='x',labelbottom=True)
    fig.suptitle('E6: 1 s Regression; no reference RIR — panel-specific vertical ranges',fontsize=10)
    axes[-1,0].set_xlabel('Clap mode'); axes[-1,1].set_xlabel('Clap mode')
    save(fig,'e6_phone_consistency')
    # Mean absolute deviation is not introduced: use the familiar median
    # absolute deviation from that room's median, in descriptor's own units.
    heat={}; summary=[]
    for key in ('c50_db','edt_s'):
        arr=[]
        for room in rooms:
            center=np.median(np.concatenate([cells[room,m][key] for m in MODES]))
            values=[float(np.median(np.abs(cells[room,m][key]-center))) for m in MODES]
            arr.append(values)
            for m,value in zip(MODES,values):
                v=cells[room,m][key]
                summary.append(dict(room=room,mode=m,metric=key,room_median=float(center),cell_median=float(np.median(v)),cell_iqr=float(np.percentile(v,75)-np.percentile(v,25)),deviation_from_room_median=value,n_claps=5))
        heat[key]=np.array(arr)
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.4),layout='constrained')
    for ax,key,title in zip(axes,('c50_db','edt_s'),('Within-room C50 deviation (dB)','Within-room EDT deviation (s)')):
        im=ax.imshow(heat[key],aspect='auto',cmap='magma',vmin=0)
        ax.set_xticks(range(8),MODES,rotation=45); ax.set_yticks(range(7),[LABELS[r] for r in rooms]); ax.set_title(title); ax.set_xlabel('Clap mode')
        for i in range(7):
            for j in range(8): ax.text(j,i,f'{heat[key][i,j]:.2f}',ha='center',va='center',fontsize=6,color='white' if heat[key][i,j]<heat[key].max()*.6 else 'black')
        fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,'e6_phone_robustness')
    with (REPORT/'cell_consistency.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(summary[0]),lineterminator="\n"); w.writeheader(); w.writerows(summary)
    # Rank rooms on equal-weight mean across the 8 C50 cell deviations.
    ranking=sorted(rooms,key=lambda r:(float(heat['c50_db'][rooms.index(r)].mean()),r))
    selected=[ranking[0],ranking[len(ranking)//2],ranking[-1]]
    selected_modes={}
    for room in selected:
        orderm=sorted(MODES,key=lambda m:(float(np.median(cells[room,m]['c50_db'])),MODES.index(m)))
        selected_modes[room]=[orderm[0],orderm[len(orderm)//2],orderm[-1]]
    provenance=dict(sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
        E5_position_order=[ids[i] for i in order],E6_rooms=rooms,E6_example_rooms=selected,E6_example_modes=selected_modes,
        E6_quantitative='per-clap median across 3 seeds; five physical claps retained in each room/mode',E6_qualitative_seed=42001,
        heatmap_equation='c_r = median_(m,k) median_s d_(s,r,m,k); H_(r,m) = median_k |median_s d_(s,r,m,k)-c_r|',
        room_selection='mean over eight C50 H cells; ascending rank, select first/middle/last; ties lexicographic',
        mode_selection='within selected room, sort modes by median C50; select first/middle/last, ties frozen mode order',
        no_ground_truth_for_E6=True)
    (REPORT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Generated figures; E5 examples available:',ex.exists()); print('E6 selected rooms:',selected)

if __name__=='__main__': main()
