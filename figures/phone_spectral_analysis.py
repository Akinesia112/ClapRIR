#!/usr/bin/env python3
"""Frozen input-only two-room phone spectral review; existing Regression only."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly,welch,stft
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/phone_spectral_analysis'
OLD=ROOT/'reports/post_meeting_phone_spectral'
MODES=('P1','P2','P3','A1','A2','A3','A1-','A1+')
SR=44100;N=39690
ROOM_NAMES={1:'Medium-size meeting room',2:'Acoustics hallway'}
EXPECTED='fa83b4108777231ef557379e8f7478aea1ab63f3e95449382edeaf85442d87fc'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def csv_write(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def verify():
 lock=read(OUT/'selection.json');assert sha(OUT/'inputs.npz')==lock['inputs_sha256'];assert sha(Path(__file__))==lock['script_sha256']
 for p,digest in lock['source_sha256'].items():assert sha(ROOT/p)==digest,p
 return lock

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'selection.json').exists():verify();print('Selection already frozen');return
 metadata=ROOT/'reports/phone_clap_demo/results/metadata.csv';rows=list(csv.DictReader(metadata.open()))
 cohort=[r for r in rows if r['source']=='m4a' and r['event_type']=='clap' and r['protocol_status']=='normal']
 eligible=[r for r in cohort if r['clipped']=='0']
 available={int(r['room_id']):r['room_name'] for r in eligible}
 circulation=sorted(i for i,name in available.items() if any(k in name.lower() for k in ('hallway','elevator')) and {r['clap_mode'] for r in eligible if int(r['room_id'])==i}==set(MODES))
 rooms=[1,circulation[0]];assert rooms==[1,2]
 old=read(OLD/'selection.json');old_inference=read(OLD/'inference.json');e6=read(ROOT/'reports/phone_deployment_evaluation/provenance.json')
 assert e6['metadata_sha256']==sha(metadata)==old['metadata_sha256']
 checkpoint=ROOT/old['checkpoint_relative'];assert sha(checkpoint)==old['checkpoint_sha256']==old_inference['checkpoint_sha256']==EXPECTED
 assert Path(e6['checkpoints']['42001']).resolve()==checkpoint.resolve()
 assert sha(OLD/'selection.json')==old_inference['selection_sha256'];assert sha(OLD/'predictions.npz')==old_inference['predictions_sha256']
 original_inputs=np.load(OLD/'inputs.npz')['observation']
 observations=[];backgrounds=[];early=[];records=[]
 for room in rooms:
  for k,mode in enumerate(MODES):
   candidates=[r for r in eligible if int(r['room_id'])==room and r['clap_mode']==mode]
   row=min(candidates,key=lambda r:(-float(r['snr_db']),int(r['event_index']),r['segment_path']))
   assert row['selected_for_demo']=='1'
   source=ROOT/'reports/phone_clap_demo/results/segments'/row['segment_path'];rate,raw=wavfile.read(source)
   assert rate==48000 and raw.ndim==1 and raw.dtype==np.float32
   wave=resample_poly(raw.astype(np.float64),147,160);onset=round(float(row['pre_pad_s'])*SR)
   signal=wave[onset:];valid=min(len(signal),SR);assert valid>=N+512
   observation=np.zeros(SR,np.float32);observation[:valid]=signal[:valid].astype(np.float32)
   background=wave[onset-round(.25*SR):onset-round(.05*SR)];assert len(background)==8820
   signal_early=wave[onset:onset+8820];assert len(signal_early)==8820
   original_index=next(i for i,r in enumerate(cohort) if r['segment_path']==row['segment_path'])
   cache=ROOT/f'runs/e6_phone_raw_cache/raw_v1_seed42001_batch{original_index//8*8:03d}.npy'
   assert cache.exists(),'Missing saved selected inference; membership must remain fixed before any future inference'
   if room==1:
    assert row['segment_path']==Path(old['records'][k]['segment_relative']).relative_to('reports/phone_clap_demo/results/segments').as_posix()
    assert np.array_equal(observation,original_inputs[k]);assert array_sha(observation)==old['records'][k]['input_sha256']
   records.append(dict(room_id=room,room_name=ROOM_NAMES[room],metadata_room_name=row['room_name'],mode=mode,
    repeat_id=int(row['repeat_id']),event_index=int(row['event_index']),metadata_snr_db=float(row['snr_db']),
    segment_relative=str(source.relative_to(ROOT)),segment_sha256=sha(source),pre_pad_seconds=float(row['pre_pad_s']),
    input_sha256=array_sha(observation),valid_input_samples=valid,background_sha256=array_sha(background),
    source_metadata_index=original_index,e6_cache_relative=str(cache.relative_to(ROOT)),e6_cache_sha256=sha(cache),e6_cache_row=original_index%8))
   observations.append(observation);backgrounds.append(background);early.append(signal_early)
 np.savez_compressed(OUT/'inputs.npz',observation=np.asarray(observations),background=np.asarray(backgrounds),early=np.asarray(early),valid_samples=np.array([r['valid_input_samples'] for r in records]))
 sources=[metadata,ROOT/'reports/phone_clap_demo/rooms/room_map.csv',ROOT/'reports/phone_clap_demo/rooms/clap_taxonomy.md',
  ROOT/'reports/phone_deployment_evaluation/provenance.json',ROOT/'experiments/phone_deployment_evaluation.py',
  OLD/'selection.json',OLD/'inference.json',OLD/'predictions.npz',OLD/'inputs.npz',checkpoint,ROOT/old['config_relative'],
  ROOT/'figures/post_meeting_phone_spectrogram.py',ROOT/'src/clapgen/experiments/hybrid_direct_rir/run.py']
 lock=dict(status='frozen-before-reading-selected-predictions',frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
  room_ids=rooms,room_names=[ROOM_NAMES[r] for r in rooms],available_rooms=available,
  room_rule='Keep medium-size meeting room first; choose smallest numeric eligible circulation-space category (hallway/elevator) with all eight modes. Metadata category contrast only; no predicted or measured reverberation ranking.',
  selection_rule='M4A normal protocol claps only, exclude clipped; highest existing pre-clap SNR per room/mode; event-index then filename tie break; all eight modes in protocol order',
  selection_uses_predictions=False,records=records,modes=list(MODES),checkpoint_sha256=EXPECTED,seed=42001,update=20000,
  preprocessing='Original mono float WAV48k -> float64 scipy resample_poly147/160 -> remove rounded metadata pre-pad -> raw float32 first44100, zero-pad only model input beyond valid support; no normalization, filtering, denoising or synthetic noise',
  display_samples=N,display_seconds=.9,normalization_band_hz=[100,10000],
  spectra='Rectangular first39690samples, no detrend; real FFT power; fixed1/12-octave mean linear power; dB after averaging. Each spectrum divided by its own mean FFT-bin power100Hz–10kHz, identical for all rooms/input/output; fixed reference permits axis-extension comparison.',
  frequency_review='Equal200ms postonset[0,.2] and background[-.25,-.05] Welch density: Hann2048,hop512,nfft2048,no detrend. Approximate SNR=10log10(max(integrated signal+noise minus background, tiny)/background), assumes locally stationary background. No denoising of any plotted signal.',
  frequency_decision_rule='Display100Hz–20kHz if BOTH selected-room medians have at least1% estimated signal-excess energy in10–20kHz and approximate high-band SNR>=10dB; otherwise display100Hz–10kHz. Report complete20kHz inspection either way. Selection-conditional diagnostic thresholds, not a model-quality rule.',
  spectrogram='Exact existing Hann1024,hop128,nfft1024,44.1k,boundaryzeros,paddedFalse; show times<=.9s and100Hz–10kHz. Preserve perpanel mean-power normalization; original limits[-60,+30] span90dB. Compare40/50/60dB spans with same upper+30 and lower-10/-20/-30. No signal or STFT change.',
  inputs_sha256=sha(OUT/'inputs.npz'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},script_sha256=sha(Path(__file__)))
 write(OUT/'selection.json',lock);csv_write(OUT/'selection.csv',records)
 print('FROZEN',lock['room_names'],len(records),flush=True)

def reuse():
 lock=verify();predictions=[];evidence=[]
 old=np.load(OLD/'predictions.npz')['inferred_rir']
 for index,row in enumerate(lock['records']):
  cache=ROOT/row['e6_cache_relative'];assert sha(cache)==row['e6_cache_sha256']
  if row['room_id']==1:
   prediction=old[index].copy();source=OLD/'predictions.npz';cache_row=index;origin='original eight-clap qualitative prediction, preserved bit-for-bit'
  else:
   batch=np.load(cache,allow_pickle=False);assert batch.ndim==2 and batch.shape[1]==SR
   prediction=batch[row['e6_cache_row']].copy();source=cache;cache_row=row['e6_cache_row'];origin='existing E6 raw-amplitude seed42001 batch cache; exact metadata-order position'
  assert prediction.dtype==np.float32 and prediction.shape==(SR,) and np.isfinite(prediction).all()
  predictions.append(prediction);evidence.append(dict(room_id=row['room_id'],mode=row['mode'],input_sha256=row['input_sha256'],
   prediction_array_sha256=array_sha(prediction),source=str(source.relative_to(ROOT)),source_sha256=sha(source),cache_row=cache_row,origin=origin))
 predictions=np.asarray(predictions);assert np.array_equal(predictions[:8],old)
 np.savez_compressed(OUT/'predictions.npz',inferred_rir=predictions)
 write(OUT/'prediction_provenance.json',dict(status='reused-existing-predictions',new_inferences=0,training_performed=False,
  checkpoint_sha256=EXPECTED,selection_sha256=sha(OUT/'selection.json'),predictions_sha256=sha(OUT/'predictions.npz'),room1_bit_exact=True,
  cache_preprocessing_check='E6 metadata SHA, checkpoint path/current SHA, metadata order and raw-amplitude preprocessing matched; selected input hashes recorded before prediction extraction',records=evidence))
 print('Reused16 predictions; no model inference',flush=True)

def spectrum(waves,hi):
 f=np.fft.rfftfreq(N,1/SR);power=np.abs(np.fft.rfft(waves[:,:N].astype(np.float64),axis=1))**2
 reference=power[:,(f>=100)&(f<=10000)].mean(axis=1,keepdims=True);power/=reference
 edges=100*2**(np.arange(120)/12);edges=np.r_[edges[edges<hi],hi]
 centers=[];values=[]
 for a,b in zip(edges[:-1],edges[1:]):
  mask=(f>=a)&(f<b);assert mask.any();centers.append(np.sqrt(a*b));values.append(10*np.log10(np.maximum(power[:,mask].mean(axis=1),1e-30)))
 return np.array(centers),np.stack(values,axis=1)

def review():
 lock=verify();data=np.load(OUT/'inputs.npz');early=data['early'];background=data['background'];observation=data['observation']
 kwargs=dict(fs=SR,window='hann',nperseg=2048,noverlap=1536,nfft=2048,detrend=False,scaling='density',axis=-1)
 f,signal=welch(early,**kwargs);_,noise=welch(background,**kwargs);df=f[1]-f[0]
 excess=np.maximum(signal-noise,0);rows=[];room_summaries=[]
 rawf=np.fft.rfftfreq(N,1/SR);rawp=np.abs(np.fft.rfft(observation[:,:N].astype(np.float64),axis=1))**2
 for i,meta in enumerate(lock['records']):
  row={k:meta[k] for k in ('room_id','room_name','mode','event_index','metadata_snr_db')}
  for a,b,label in ((100,10000,'low'),(10000,20000,'high'),(100,20000,'full')):
   mask=(f>=a)&(f<b);sp=float(signal[i,mask].sum()*df);npow=float(noise[i,mask].sum()*df)
   row[f'{label}_signal_plus_background_power']=sp;row[f'{label}_background_power']=npow
   row[f'{label}_post_to_pre_db']=float(10*np.log10(sp/npow))
   row[f'{label}_approx_snr_db']=float(10*np.log10(max(sp-npow,np.finfo(float).tiny)/npow))
  row['high_excess_energy_fraction']=float(excess[i,(f>=10000)&(f<20000)].sum()/excess[i,(f>=100)&(f<20000)].sum())
  row['high_full_900ms_energy_fraction']=float(rawp[i,(rawf>=10000)&(rawf<20000)].sum()/rawp[i,(rawf>=100)&(rawf<20000)].sum())
  row['high_frequency_bins_post_above_pre_10db_fraction']=float(np.mean(signal[i,(f>=10000)&(f<20000)]/noise[i,(f>=10000)&(f<20000)]>=10))
  rows.append(row)
 for room in lock['room_ids']:
  group=[r for r in rows if r['room_id']==room]
  info=dict(room_id=room,room_name=ROOM_NAMES[room],selected_claps=len(group))
  for k in ('high_excess_energy_fraction','high_full_900ms_energy_fraction','high_approx_snr_db','low_approx_snr_db','high_frequency_bins_post_above_pre_10db_fraction'):
   v=[r[k] for r in group];info[k+'_median']=float(np.median(v));info[k+'_min']=min(v);info[k+'_max']=max(v)
  info['passes_high_frequency_display_rule']=info['high_excess_energy_fraction_median']>=.01 and info['high_approx_snr_db_median']>=10
  room_summaries.append(info)
 hi=20000 if all(r['passes_high_frequency_display_rule'] for r in room_summaries) else 10000
 csv_write(OUT/'frequency_quality_per_clap.csv',rows);csv_write(OUT/'frequency_quality_by_selected_room.csv',room_summaries)
 np.savez_compressed(OUT/'frequency_quality_spectra.npz',frequency_hz=f,signal_plus_background_psd=signal,background_psd=noise,approx_frequency_snr_db=10*np.log10(np.maximum(signal-noise,np.finfo(float).tiny)/noise))
 write(OUT/'frequency_decision.json',dict(status='input-only-frequency-review-completed',selection_sha256=sha(OUT/'selection.json'),
  frequency_hz=[100,hi],normalization_hz=[100,10000],rule=lock['frequency_decision_rule'],room_summaries=room_summaries,
  interpretation='Energy and SNR describe these16 selected recordings, not the complete phone corpus; pre-onset background may contain prior-clap decay. Approximate SNR assumes locally stationary background and is not reference-RIR accuracy.'))
 print(json.dumps(dict(frequency_hz=[100,hi],room_summaries=room_summaries),indent=2),flush=True)

def export(fig,stem):
 folder=OUT/'figures';folder.mkdir(exist_ok=True);records=[]
 for ext in ('pdf','png'):
  source=folder/f'{stem}.{ext}';fig.savefig(source,dpi=240,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
  destination=ROOT/'publication/figures'/source.name;shutil.copyfile(source,destination);assert sha(source)==sha(destination)
  records.append(dict(source=str(source),destination=str(destination),sha256=sha(source)))
 return records

def render():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.ticker import FixedLocator,FixedFormatter
 lock=verify();provenance=read(OUT/'prediction_provenance.json');assert provenance['selection_sha256']==sha(OUT/'selection.json');assert provenance['predictions_sha256']==sha(OUT/'predictions.npz')
 data=np.load(OUT/'inputs.npz');inputs=data['observation'];predictions=np.load(OUT/'predictions.npz')['inferred_rir'];hi=read(OUT/'frequency_decision.json')['frequency_hz'][1]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 colors=plt.get_cmap('tab10').colors[:8];exports=[]
 for upper,stem in ((hi,'phone_room_spectra'),(10000 if hi==20000 else 20000,'phone_room_spectra_frequency_comparison')):
  f,before=spectrum(inputs,upper);_,after=spectrum(predictions,upper)
  np.savez_compressed(OUT/f'{stem}_data.npz',frequency_hz=f,recorded_db=before,inferred_db=after)
  fig,axes=plt.subplots(2,2,figsize=(8.2,5.65),sharex=True,sharey=True)
  for room_index,room in enumerate(lock['room_ids']):
   for kind,values in enumerate((before,after)):
    ax=axes[room_index,kind]
    for k,mode in enumerate(MODES):ax.plot(f/1000,values[room_index*8+k],color=colors[k],lw=1.25,ls='-' if k<4 else '--',label=mode)
    ax.set_xscale('log');ax.set_xlim(.1,upper/1000);ticks=[.1,.3,1,3,10]+([20] if upper==20000 else [])
    ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in ticks]));ax.grid(alpha=.2,lw=.5)
    if room_index==1:ax.set_xlabel('Frequency (kHz)')
    if kind==0:ax.set_ylabel('Relative spectral power (dB)')
   axes[room_index,0].text(0,1.035,ROOM_NAMES[room],transform=axes[room_index,0].transAxes,fontsize=11,weight='semibold')
  values=np.r_[before.ravel(),after.ravel()];axes[0,0].set_ylim(5*np.floor(values.min()/5)-1,5*np.ceil(values.max()/5)+1)
  fig.text(.31,.965,'Recorded phone claps',ha='center',fontsize=12,weight='semibold');fig.text(.755,.965,'Inferred RIRs',ha='center',fontsize=12,weight='semibold')
  handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=8,frameon=False,bbox_to_anchor=(.53,.005),handlelength=2,columnspacing=1,handletextpad=.5)
  fig.subplots_adjust(left=.105,right=.99,top=.865,bottom=.135,hspace=.31,wspace=.13)
  exports+=export(fig,stem);plt.close(fig)
 # Compare display ranges using precisely the old STFT and normalization.
 groups=[];statistics=[]
 for kind,waves in enumerate((inputs,predictions)):
  f,t,z=stft(waves.astype(np.float64),fs=SR,window='hann',nperseg=1024,noverlap=896,nfft=1024,axis=-1,boundary='zeros',padded=False)
  fm=(f>=100)&(f<=10000);tm=t<=.9;assert t[tm][-1]+512/SR<data['valid_samples'].min()/SR
  p=np.abs(z[:,fm][:,:,tm])**2;mean=p.mean(axis=(1,2),keepdims=True);db=10*np.log10(np.maximum(p/mean,1e-12));groups.append(db)
  for i,row in enumerate(lock['records']):
   late=t[tm]>=.6
   statistics.append(dict(room_id=row['room_id'],mode=row['mode'],signal='recorded' if kind==0 else 'inferred',
    peak_relative_to_mean_db=float(db[i].max()),late_median_relative_to_mean_db=float(np.median(db[i,:,late])),
    original_90db_fraction_below_floor=float(np.mean(db[i]<-60)),original_fraction_above_ceiling=float(np.mean(db[i]>30)),
    range40_fraction_below_floor=float(np.mean(db[i]<-10)),range50_fraction_below_floor=float(np.mean(db[i]<-20)),range60_fraction_below_floor=float(np.mean(db[i]<-30))))
 f=f[fm];t=t[tm];old=np.load(OLD/'display_spectrograms.npz');assert np.array_equal(groups[0][:8],old['recorded_db']) and np.array_equal(groups[1][:8],old['inferred_db'])
 np.savez_compressed(OUT/'spectrogram_data.npz',frequency_hz=f,time_seconds=t,recorded_db=groups[0],inferred_db=groups[1])
 csv_write(OUT/'spectrogram_display_diagnostics.csv',statistics)
 for span in (40,50,60):
  fig,axes=plt.subplots(4,8,figsize=(15.8,7.4),sharex=True,sharey=True)
  for room_index,room in enumerate(lock['room_ids']):
   for kind in range(2):
    row=room_index*2+kind
    for k,mode in enumerate(MODES):
     ax=axes[row,k];mesh=ax.pcolormesh(t*1000,f/1000,groups[kind][room_index*8+k],shading='nearest',cmap='magma',vmin=30-span,vmax=30,rasterized=True)
     ax.set_xlim(0,900);ax.set_ylim(.1,10);ax.set_xticks([0,450,900]);ax.set_yticks([.1,5,10]);ax.tick_params(labelsize=8)
     if row==0:ax.set_title(mode,fontsize=11)
     if row==3:ax.set_xlabel('Time (ms)',fontsize=9)
     if k==0:ax.set_ylabel('Frequency (kHz)',fontsize=9)
    fig.text(.012,.755-row*.2,('Recorded' if kind==0 else 'Inferred')+'\n'+ROOM_NAMES[room],rotation=90,ha='center',va='center',fontsize=9)
  fig.subplots_adjust(left=.07,right=.935,top=.91,bottom=.12,hspace=.28,wspace=.13)
  cax=fig.add_axes([.95,.18,.012,.67]);fig.colorbar(mesh,cax=cax,label='Power relative to panel mean (dB)')
  fig.suptitle(f'Spectrogram display range: {span} dB',fontsize=13)
  exports+=export(fig,f'phone_spectrogram_range_{span}db');plt.close(fig)
 # Input/background spectral evidence, with all eight selected modes retained.
 quality=np.load(OUT/'frequency_quality_spectra.npz');f=quality['frequency_hz'];mask=(f>=100)&(f<=20000)
 fig,axes=plt.subplots(2,2,figsize=(9,5.5),sharex=True)
 for room_index,room in enumerate(lock['room_ids']):
  for k,mode in enumerate(MODES):
   i=room_index*8+k;signal=quality['signal_plus_background_psd'][i];noise=quality['background_psd'][i]
   axes[room_index,0].plot(f[mask]/1000,10*np.log10(np.maximum(signal[mask],1e-30)),color=colors[k],lw=.9,label=mode)
   axes[room_index,0].plot(f[mask]/1000,10*np.log10(np.maximum(noise[mask],1e-30)),color=colors[k],lw=.65,ls=':',alpha=.55)
   axes[room_index,1].plot(f[mask]/1000,10*np.log10(signal[mask]/noise[mask]),color=colors[k],lw=.9)
  for ax in axes[room_index]:ax.set_xscale('log');ax.set_xlim(.1,20);ax.axvline(10,color='black',lw=.6,ls='--');ax.grid(alpha=.15)
  axes[room_index,0].set_ylabel('PSD (dB/Hz)');axes[room_index,1].set_ylabel('Post / pre power (dB)');axes[room_index,1].axhline(0,color='black',lw=.6)
  axes[room_index,0].set_title(ROOM_NAMES[room],loc='left')
 for ax in axes[1]:ax.set_xlabel('Frequency (kHz)')
 fig.suptitle('Recorded-clap and background spectra: equal 200 ms windows',fontsize=12)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=8,frameon=False)
 fig.subplots_adjust(left=.09,right=.97,top=.86,bottom=.15,hspace=.4,wspace=.26)
 exports+=export(fig,'phone_frequency_signal_background');plt.close(fig)
 write(OUT/'result.json',dict(status='rendered-awaiting-visual-review',selection_sha256=sha(OUT/'selection.json'),prediction_provenance_sha256=sha(OUT/'prediction_provenance.json'),
  frequency_hz=[100,hi],normalization_hz=[100,10000],examples=16,rooms=lock['room_names'],new_inferences=0,
  original_meeting_room_prediction_arrays_unchanged=True,original_meeting_room_spectrogram_arrays_bit_exact=True,
  no_reference_rir=True,reconstruction_accuracy_claim=False,signal_or_noise_suppression_applied=False,exports=exports))
 print('RENDERED',len(exports),'canonical files',flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','reuse','review','render']);args=parser.parse_args()
 {'prepare':prepare,'reuse':reuse,'review':review,'render':render}[args.action]()
if __name__=='__main__':main()
