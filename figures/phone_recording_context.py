#!/usr/bin/env python3
"""Observed phone-clap decay for context, explicitly NOT a target RIR.

Reuses the original phone_clap_render_figures reverse-energy integration;
reads frozen M4A-derived segments and metadata, never redetects or normalizes
model inputs. No inference. Each curve stops at its real recorded support.
"""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/real_clap_and_phone_figures';OUT=ROOT/'publication/figures'
def main():
 selection=json.loads((REPORT/'provenance.json').read_text())
 meta=ROOT/'reports/phone_clap_demo/results/metadata.csv';rows=list(csv.DictReader(meta.open()))
 fig,axes=plt.subplots(3,3,figsize=(10.8,8),sharex=True,sharey=True,layout='constrained');used=[]
 for i,room in enumerate(selection['E6_example_rooms']):
  for j,mode in enumerate(selection['E6_example_modes'][room]):
   rs=sorted([r for r in rows if r['source']=='m4a' and r['protocol_status']=='normal' and r['event_type']=='clap' and r['room_name']==room and r['clap_mode']==mode],key=lambda r:int(r['repeat_id']));assert len(rs)==5
   ax=axes[i,j]
   for k,r in enumerate(rs):
    source=ROOT/'reports/phone_clap_demo/results/segments'/r['segment_path'];sr,y=wavfile.read(source);assert y.ndim==1 and np.issubdtype(y.dtype,np.floating)
    x=y[round(float(r['pre_pad_s'])*sr):];x=x[:sr].astype(np.float64)
    e=np.cumsum(x[::-1]**2)[::-1];edc=10*np.log10(e/max(e[0],1e-30)+1e-30)
    ax.plot(np.arange(len(edc))*1000/sr,edc,lw=.65,color='.35',alpha=.65,label='Recorded clap y (5 repetitions)\nObservation, not target RIR' if k==0 else None)
    used.append(dict(sample_id=r['segment_path'],samples=len(x),sample_rate=sr,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
   ax.set_title(room+' / '+mode,fontsize=8);ax.set_xlim(0,1000);ax.set_ylim(-70,2);ax.grid(alpha=.18);ax.legend(loc='lower left',fontsize=6)
   if j==0:ax.set_ylabel('Recorded-clap EDC (dB)')
   if i==2:ax.set_xlabel('Time after clap onset (ms)')
 fig.suptitle('E6 input context: recorded clap + room + ambient noise; no reference RIR',fontsize=10)
 for ext in ('png','pdf'):fig.savefig(OUT/f'e6_phone_recording_context.{ext}',dpi=220,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
 plt.close(fig)
 (REPORT/'phone_recording_context.json').write_text(json.dumps(dict(metadata=str(meta.relative_to(ROOT)),metadata_sha256=hashlib.sha256(meta.read_bytes()).hexdigest(),source_equation='Original phone_clap_render_figures.py reverse cumulative sum of squared recorded signal; EDC normalizes the plotted cumulative energy, not the model input.',role='Observed y, not RIR h; no target/reference or accuracy claim',recording_window='At most 1 s after metadata onset; stop at actual segment end. No padded samples enter this plot. Terminal falls depend on finite-window integration.',sources=used),indent=2)+'\n')
if __name__=='__main__':main()
