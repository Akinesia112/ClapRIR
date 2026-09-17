#!/usr/bin/env python3
"""E0 descriptive on-axis training-support energy/duration, no model metrics."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'reports/e0_characterization'
def main():
    OUT.mkdir(exist_ok=True); base=ROOT/'data/real_claps'; meta=base/'metadata.csv'
    records=[r for r in csv.DictReader(meta.open()) if r['tier']=='clean' and r['primary_channel']=='3']
    table=OUT/'per_clap.csv'
    if table.exists(): rows=list(csv.DictReader(table.open()))
    else:
        rows=[]
        for r in records:
            p=int(r['participant']); folder=base/('_review' if p in (3,6) else '')
            path=folder/f'participant{p:02d}'/f"block{int(r['block_idx']):02d}"/f"clap{int(r['clap_idx']):02d}.wav"
            x,sr=sf.read(path,always_2d=True); assert sr==44100
            x=x[:,3]; start=max(0,int(np.argmax(np.abs(x)))-882//8); x=x[start:start+882]
            energy=np.cumsum(x*x); norm=energy/(energy[-1]+1e-30); i5=int(np.searchsorted(norm,.05)); i95=int(np.searchsorted(norm,.95))
            rows.append(dict(participant=p,block=int(r['block_idx']),clap=int(r['clap_idx']),energy_digital_squared_s=float(energy[-1]/sr),duration_5_95_ms=float((i95-i5)/sr*1000),source=str(path.relative_to(ROOT))))
        with table.open('w',newline='') as f:
            w=csv.DictWriter(f,list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
    assert len(rows)==len(records)
    ps=sorted({int(r['participant']) for r in rows}); fig,axes=plt.subplots(1,3,figsize=(7.1,2.5),layout='constrained')
    counts=[sum(int(r['participant'])==p for r in rows) for p in ps]; axes[0].bar(range(len(ps)),counts,color='#245b85'); axes[0].set_ylabel('Retained claps')
    for ax,key,label in [(axes[1],'energy_digital_squared_s','20 ms support energy\n(digital amplitude squared × s)'),(axes[2],'duration_5_95_ms','5–95% energy duration\nwithin 20 ms support (ms)')]:
        v=[[float(r[key]) for r in rows if int(r['participant'])==p] for p in ps]
        ax.boxplot(v,positions=range(len(ps)),widths=.6,showfliers=True,flierprops={'markersize':1.3},medianprops={'color':'#a85326'}); ax.set_ylabel(label)
        if 'energy' in key: ax.set_yscale('log')
    for ax in axes: ax.set_xticks(range(len(ps)),[f'{p:02d}' for p in ps],rotation=90); ax.set_xlabel('Participant'); ax.grid(axis='y',alpha=.15)
    plt.rcParams['pdf.fonttype']=42
    for ext in ('pdf','png'): fig.savefig(ROOT/f'publication/figures/e0_dataset_statistics.{ext}',dpi=220,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
    summary=dict(n_claps=len(rows),participants=len(ps),metadata_sha256=hashlib.sha256(meta.read_bytes()).hexdigest(),energy='sum x[n]^2 / 44100 on raw on-axis 20 ms training support; digital units, not calibrated acoustic energy',duration='time between 5% and 95% cumulative energy within the same support; not full event duration',mode_labels='Anechoic metadata has no P1/A1 mode column; no mode-to-block mapping is inferred',median_duration_ms=float(np.median([float(r['duration_5_95_ms']) for r in rows])))
    (OUT/'result.json').write_text(json.dumps(summary,indent=2)+'\n'); print(summary)
if __name__=='__main__': main()
