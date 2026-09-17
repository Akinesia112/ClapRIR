#!/usr/bin/env python3
"""Improve diagnostic labels only, reusing the frozen STFT arrays unchanged."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/phone_spectral_analysis'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 z=np.load(OUT/'spectrogram_data.npz');t=z['time_seconds'];f=z['frequency_hz'];groups=[z['recorded_db'],z['inferred_db']]
 lock=json.loads((OUT/'selection.json').read_text());old=json.loads((OUT/'result.json').read_text());exports=[];archive=OUT/'figure_layout_drafts';archive.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
 for span in (40,50,60):
  fig=plt.figure(figsize=(15.8,8.4));grid=fig.add_gridspec(6,8,height_ratios=[.18,1,1,.18,1,1],left=.075,right=.935,bottom=.09,top=.925,hspace=.35,wspace=.14)
  for room_index,room_name in enumerate(lock['room_names']):
   title=fig.add_subplot(grid[room_index*3,:]);title.axis('off');title.text(0,.4,room_name,fontsize=12,weight='semibold',ha='left',va='center')
   for kind in range(2):
    for k,mode in enumerate(lock['modes']):
     ax=fig.add_subplot(grid[room_index*3+1+kind,k]);mesh=ax.pcolormesh(t*1000,f/1000,groups[kind][room_index*8+k],shading='nearest',cmap='magma',vmin=30-span,vmax=30,rasterized=True)
     ax.set_xlim(0,900);ax.set_ylim(.1,10);ax.set_xticks([0,450,900]);ax.set_yticks([.1,5,10]);ax.tick_params(labelsize=8)
     if kind==0:ax.set_title(mode,fontsize=10,pad=3);ax.set_xticklabels([])
     if k:ax.set_yticklabels([])
     else:ax.set_ylabel(('Recorded' if kind==0 else 'Inferred')+'\nFrequency (kHz)',fontsize=9)
     if room_index==1 and kind==1:ax.set_xlabel('Time (ms)',fontsize=9)
  cax=fig.add_axes([.95,.14,.013,.72]);fig.colorbar(mesh,cax=cax,label='Power relative to panel mean (dB)')
  fig.suptitle(f'Spectrogram display range: {span} dB',fontsize=13,y=.985)
  for ext in ('pdf','png'):
   source=OUT/'figures'/f'phone_spectrogram_range_{span}db.{ext}';archived=archive/source.name
   if not archived.exists():shutil.copyfile(source,archived)
   fig.savefig(source,dpi=240,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
   target=ROOT/'publication/figures'/source.name;shutil.copyfile(source,target);assert sha(source)==sha(target)
   exports.append(dict(source=str(source),destination=str(target),sha256=sha(source)))
  plt.close(fig)
 names={e['destination'] for e in exports};old['exports']=[e for e in old['exports'] if e['destination'] not in names]+exports
 old['diagnostic_label_review_source_sha256']=sha(Path(__file__));(OUT/'result.json').write_text(json.dumps(old,indent=2)+'\n')
 (OUT/'diagnostic_layout_review.json').write_text(json.dumps(dict(status='completed',change='Room titles have separate grid rows; concise recorded/inferred row labels. No data or normalization changes.',source_sha256=sha(Path(__file__)),spectrogram_array_sha256=sha(OUT/'spectrogram_data.npz'),original_drafts_preserved=str(archive),exports=exports),indent=2)+'\n')
 print('Updated diagnostic labels only',flush=True)
if __name__=='__main__':main()
