#!/usr/bin/env python3
"""Render phone qualitative spectra and compact spectrum/time variants from frozen arrays."""
from pathlib import Path
import json, shutil, hashlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.text import Text

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'reports/phone_spectral_analysis'; F=ROOT/'publication/figures'
MODES=('P1','P2','P3','A1','A2','A3','A1-','A1+')
ROOMS=('Medium-size meeting room','Highly reverberant hallway')
SR=44100; N=39690; HI=15000

def spectrum(waves, normalize=True):
 f=np.fft.rfftfreq(N,1/SR); p=np.abs(np.fft.rfft(waves[:,:N].astype(float),axis=1))**2
 if normalize:p/=p[:,(f>=100)&(f<=10000)].mean(axis=1,keepdims=True)
 edges=100*2**(np.arange(120)/12); edges=np.r_[edges[edges<HI],HI]
 c=[]; v=[]
 for a,b in zip(edges[:-1],edges[1:]):
  m=(f>=a)&(f<b); c.append(np.sqrt(a*b));v.append(10*np.log10(np.maximum(p[:,m].mean(1),1e-30)))
 return np.asarray(c),np.stack(v,1)


def digital_spectrum(waves):
 # Common absolute digital power convention; no signal-dependent divisor.
 # One-sided FFT-bin powers sum to mean-square original digital amplitude.
 f=np.fft.rfftfreq(N,1/SR)
 fft=np.fft.rfft(waves[:,:N].astype(float),axis=1)
 power=np.abs(fft)**2/(N*N)
 power[:,1:-1]*=2 # N is even; DC and Nyquist are not doubled.
 assert np.allclose(power.sum(axis=1),np.mean(waves[:,:N].astype(float)**2,axis=1),rtol=1e-12,atol=1e-15)
 edges=100*2**(np.arange(120)/12);edges=np.r_[edges[edges<HI],HI]
 centers=[];values=[]
 for a,b in zip(edges[:-1],edges[1:]):
  mask=(f>=a)&(f<b)
  centers.append(np.sqrt(a*b))
  # Average ORIGINAL-amplitude bin powers, then reference fixed digital power 1.
  values.append(10*np.log10(np.maximum(power[:,mask].mean(axis=1),1e-30)))
 return np.asarray(centers),np.stack(values,axis=1)


def rms_envelope(signal):
 # Centered 221-sample (~5 ms) RMS, sampled every 44 samples (~1 ms).
 # Clip the first window to available signal samples, without zero padding.
 # Read the full saved waveform so the 250-ms endpoint uses real context.
 signal=np.asarray(signal,dtype=float)
 centers=np.unique(np.r_[np.arange(0,round(.25*SR)+1,44),round(.25*SR)]).astype(int)
 half=110
 lo=np.maximum(centers-half,0);hi=np.minimum(centers+half+1,len(signal))
 energy=np.r_[0.,np.cumsum(signal*signal)]
 rms=np.sqrt(np.maximum((energy[hi]-energy[lo])/(hi-lo),0))
 return centers/SR*1000,rms


def save(fig,stem):
 F.mkdir(parents=True,exist_ok=True)
 for ext in ('pdf','png'):
  fig.savefig(F/f'{stem}.{ext}',dpi=240,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {})

def main():
 lock=json.loads((R/'selection.json').read_text()); x=np.load(R/'inputs.npz')['observation']; h=np.load(R/'predictions.npz')['inferred_rir']
 fully_unnormalized='--fully-unnormalized-only' in sys.argv
 calculate_spectrum=digital_spectrum if fully_unnormalized else spectrum
 f,rx=calculate_spectrum(x);_,rh=calculate_spectrum(h); colors=plt.get_cmap('tab10').colors[:8]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 def four(modes,stem):
  modes=tuple(modes)
  # Separate rows for column headings, each room heading, and the whole legend.
  fig=plt.figure(figsize=(3.5,3.5),dpi=180)
  heights=[.20,.19,.90,.25,.90,.40]
  heights += [.35]
  grid=fig.add_gridspec(len(heights),2,height_ratios=heights,
    left=.145,right=.97,top=.98,bottom=.025,hspace=.09,wspace=.23)
  titles=[];plots=[]
  def heading(spec,text,size=7.5,bold=False,vertical=.5):
   header=fig.add_subplot(spec);header.set_axis_off()
   label=header.text(.5,vertical,text,ha='center',va='center',fontsize=size,
     weight='semibold' if bold else 'normal',transform=header.transAxes)
   titles.append(label)
  heading(grid[0,0],'Recorded claps',bold=True)
  heading(grid[0,1],'Inferred RIRs',bold=True)
  for ri,room in enumerate(ROOMS):
   row=2+2*ri;heading(grid[row-1,:],room,bold=True)
   for ci,vals in enumerate((rx,rh)):
    a=fig.add_subplot(grid[row,ci]);plots.append(a)
    for k in modes:
     a.plot(f/1000,vals[ri*8+k],color=colors[k],lw=.8,
       ls='-' if k<4 else '--',label=MODES[k])
    a.set_xscale('log');a.set_xlim(.1,15);a.set_ylim(-40,20)
    a.grid(alpha=.18,lw=.35);a.set_yticks([-40,-20,0,20])
    a.xaxis.set_major_locator(FixedLocator([.1,1,15]))
    a.xaxis.set_major_formatter(FixedFormatter(['0.1','1','15']))
    a.xaxis.set_minor_locator(NullLocator())
    a.tick_params(labelsize=6.5,pad=2,length=2.5,labelbottom=ri==1,labelleft=ci==0)
    if ci==0:a.set_ylabel('Relative power (dB)',fontsize=7,labelpad=2)
  heading(grid[5,:],'Frequency (kHz)',size=7,vertical=.25)
  legend_ax=fig.add_subplot(grid[-1,:]);legend_ax.set_axis_off()
  hs,ls=plots[0].get_legend_handles_labels()
  order=[0,4,1,5,2,6,3,7] if len(modes)==8 else list(range(len(modes)))
  legend=legend_ax.legend([hs[k] for k in order],[ls[k] for k in order],
    loc='center',ncol=4 if len(modes)==8 else 3,fontsize=7,
    frameon=False,handlelength=1.8,handletextpad=.5,columnspacing=1.2,
    borderaxespad=0,borderpad=0,labelspacing=.5)
  finish(fig,stem,plots,titles,legend,ls)
 def finish(fig,stem,plots,titles,legend,ls,details=None):
  fig.canvas.draw();renderer=fig.canvas.get_renderer();frame=fig.bbox
  text_boxes=[]
  for text in fig.findobj(Text):
   if not text.get_visible() or not text.get_text().strip():continue
   box=text.get_window_extent(renderer)
   if box.width<.1 or box.height<.1:continue
   assert box.x0>=frame.x0-.5 and box.x1<=frame.x1+.5 and box.y0>=frame.y0-.5 and box.y1<=frame.y1+.5, ('Clipped text',text.get_text(),box.bounds)
   text_boxes.append((text.get_text(),box))
  for i,(name,a) in enumerate(text_boxes):
   for other,b in text_boxes[i+1:]:
    assert min(min(a.x1,b.x1)-max(a.x0,b.x0),min(a.y1,b.y1)-max(a.y0,b.y0))<=.4, ('Text overlap',name,other)
  for item in [*titles,legend]:
   box=item.get_window_extent(renderer)
   assert all(not box.overlaps(a.bbox) for a in plots),'Text/legend overlaps plot'
  box=legend.get_window_extent(renderer)
  assert frame.contains(box.x0,box.y0) and frame.contains(box.x1,box.y1)
  qa=R/'compact_layout_qa';qa.mkdir(exist_ok=True)
  backup=qa/'before_room_and_legend_fix';backup.mkdir(exist_ok=True)
  for ext in ('pdf','png'):
   source=F/f'{stem}.{ext}';destination=backup/source.name
   if source.exists() and not destination.exists():shutil.copyfile(source,destination)
  save(fig,stem)
  check=dict(status='geometry-passed-awaiting-full-pdf-review',
    headers_outside_plot=True,legend_inside_page=True,no_text_overlaps=True,
    figure_inches=list(fig.get_size_inches()),legend_labels=ls,
    display_hz=[100,15000],normalization_hz=[100,10000],display_db=[-40,20],
    data_unchanged=True,new_inferences=0,
    pdf_sha256=hashlib.sha256((F/f'{stem}.pdf').read_bytes()).hexdigest(),
    png_sha256=hashlib.sha256((F/f'{stem}.png').read_bytes()).hexdigest())
  check.update(details or {})
  (qa/f'{stem}.json').write_text(json.dumps(check,indent=2)+'\n')
  plt.close(fig)
 def reference_eight():
  # Preserve the separately requested original spectrum-only presentation.
  stem='phone_qualitative_spectrum_8modes'
  with plt.rc_context({'font.size':10,'axes.titlesize':11}):
   fig,axes=plt.subplots(2,2,figsize=(8.2,5.65),sharex=True,sharey=True,dpi=180)
   titles=[];plots=list(axes.flat)
   for ri,room in enumerate(ROOMS):
    for ci,values in enumerate((rx,rh)):
     ax=axes[ri,ci]
     for k,mode in enumerate(MODES):
      ax.plot(f/1000,values[ri*8+k],color=colors[k],lw=1.25,
        ls='-' if k<4 else '--',label=mode)
     ax.set_xscale('log');ax.set_xlim(.1,15);ax.set_ylim(-40,20)
     ax.xaxis.set_major_locator(FixedLocator([.1,.3,1,3,10,15]))
     ax.xaxis.set_major_formatter(FixedFormatter(['0.1','0.3','1','3','10','15']))
     ax.set_yticks([-40,-30,-20,-10,0,10,20]);ax.grid(alpha=.2,lw=.5)
     if ri==1:ax.set_xlabel('Frequency (kHz)')
     if ci==0:ax.set_ylabel('Relative spectral power (dB)')
    titles.append(axes[ri,0].text(0,1.035,room,
      transform=axes[ri,0].transAxes,fontsize=11,weight='semibold'))
   fig.subplots_adjust(left=.105,right=.975,top=.865,bottom=.135,hspace=.31,wspace=.13)
   for ci,label in enumerate(('Recorded phone claps','Inferred RIRs')):
    pos=axes[0,ci].get_position()
    titles.append(fig.text((pos.x0+pos.x1)/2,.965,label,ha='center',fontsize=12,weight='semibold'))
   handles,labels=axes[0,0].get_legend_handles_labels()
   legend=fig.legend(handles,labels,loc='lower center',ncol=8,frameon=False,
     bbox_to_anchor=(.53,.005),handlelength=2,columnspacing=1,handletextpad=.5)
   details=dict(layout='wide 2x2 shared spectral axes',layout_reference='publication/figures/phone_room_spectra.pdf',
     frequency_ticks_khz=[.1,.3,1,3,10,15],spectral_ticks_db=list(range(-40,21,10)),
     legend_columns=8,shared_legend_count=1,super_title=False,
     spectral_panel_inches=[[a.get_position().width*8.2,a.get_position().height*5.65] for a in axes.flat])
   finish(fig,stem,plots,titles,legend,labels,details)
 def combined(scale):
  assert scale in ('db','linear')
  stem=('fully_unnormalized_rms' if scale=='db' else 'fully_unnormalized_waveform') if fully_unnormalized else 'phone_qualitative_spectrum_time'+('_linear' if scale=='linear' else '')
  width,height=8.2,10.2
  # Match original spectral panel dimensions; group the spectra and waveforms
  # for each room together while sharing signal-type columns across all rows.
  panel_width=3.3492957746478873
  spectral_height=1.7854978354978353
  xleft=(.105,.975-panel_width/width)
  row_bottom=(7.515,5.7,2.965,1.15)
  # Full-unnormalized mode uses absolute digital bin powers.
  # Other modes retain their separately requested original spectral convention.
  spectra=(rx,rh)
  temporal=[]
  for signals in (x,h):
   temporal.append(np.stack([20*np.log10(np.maximum(rms_envelope(v)[1],1e-12)) for v in signals]))
  def shared_limits(arrays):
   lo=min(float(v.min()) for v in arrays);hi=max(float(v.max()) for v in arrays)
   return [float(10*np.floor(lo/10)),float(10*np.ceil(hi/10))]
  spectral_limits=shared_limits(spectra) if fully_unnormalized else [-40.,20.];rms_limits=shared_limits(temporal)
  for values in (*spectra,*temporal):assert np.all(np.isfinite(values))
  spectral_ticks=np.arange(spectral_limits[0],spectral_limits[1]+1,10).tolist()
  rms_ticks=np.arange(rms_limits[0],rms_limits[1]+1,20).tolist()
  if rms_ticks[-1]!=rms_limits[1]:rms_ticks.append(rms_limits[1])
  with plt.rc_context({'font.size':10,'axes.titlesize':11}):
   fig=plt.figure(figsize=(width,height),dpi=180)
   plots=[];titles=[]
   for row in range(4):
    ri=row//2;is_spectrum=row%2==0
    panel_height=spectral_height if is_spectrum else 1.0
    for ci,signals in enumerate((x,h)):
     ax=fig.add_axes([xleft[ci],row_bottom[row]/height,panel_width/width,panel_height/height])
     plots.append(ax)
     for k,mode in enumerate(MODES):
      if is_spectrum:
       tx=f/1000;values=spectra[ci][ri*8+k];lw=1.25;alpha=1
      else:
       values=signals[ri*8+k,:round(.25*SR)].astype(float)
       tx=np.arange(len(values))/SR*1000;lw=.4;alpha=.65
       if scale=='db':
        tx,values=rms_envelope(signals[ri*8+k])
        values=20*np.log10(np.maximum(values,1e-12));lw=1.0;alpha=1
       # The linear alternative preserves signed digital samples: no gain.
      ax.plot(tx,values,color=colors[k],lw=lw,alpha=alpha,
        ls='-' if k<4 else '--',label=mode)
     if is_spectrum:
      ax.set_xscale('log');ax.set_xlim(.1,15);ax.set_ylim(*spectral_limits)
      ax.xaxis.set_major_locator(FixedLocator([.1,.3,1,3,10,15]))
      ax.xaxis.set_major_formatter(FixedFormatter(['0.1','0.3','1','3','10','15']))
      ax.set_yticks(spectral_ticks);ax.set_xlabel('Frequency (kHz)')
      if ci==0:
       ax.set_ylabel('Spectral power (dBFS)' if fully_unnormalized else 'Relative spectral power (dB)')
       titles.append(ax.text(0,1.035,ROOMS[ri],transform=ax.transAxes,fontsize=11,weight='semibold'))
     else:
      ax.set_xlim(0,250);ax.set_xticks([0,50,100,150,200,250]);ax.set_xlabel('Time (ms)')
      if scale=='db':
       ax.set_ylim(*rms_limits);ax.set_yticks(rms_ticks)
       if ci==0:ax.set_ylabel('RMS level (dBFS)')
      else:
       assert max(np.max(np.abs(x[:,:round(.25*SR)])),np.max(np.abs(h[:,:round(.25*SR)])))<1
       ax.set_ylim(-1,1);ax.set_yticks([-1,-.5,0,.5,1])
       if ci==0:ax.set_ylabel('Digital amplitude')
     ax.tick_params(labelleft=ci==0);ax.grid(alpha=.2,lw=.5)
     titles.append(ax.text(.5,-.55/panel_height,'('+chr(ord('a')+row*2+ci)+')',
       transform=ax.transAxes,ha='center',va='top',fontsize=10))
   for ci,label in enumerate(('Recorded phone claps','Inferred RIRs')):
    titles.append(fig.text(xleft[ci]+panel_width/width/2,.965,label,ha='center',fontsize=12,weight='semibold'))
   handles,labels=plots[0].get_legend_handles_labels()
   legend=fig.legend(handles,labels,loc='lower center',ncol=8,frameon=False,
     bbox_to_anchor=(.53,.005),handlelength=2,columnspacing=1,handletextpad=.5)
   details=dict(layout='Room 1 spectra, Room 1 time, Room 2 spectra, Room 2 time; recorded-clap / inferred-RIR columns throughout',
     layout_reference='publication/figures/phone_room_spectra.pdf',
     panel_labels=[chr(ord('a')+i) for i in range(8)],
     waveform_modes=list(MODES),waveform_curves_per_panel=[8,8,8,8],
     waveform_signals=['recorded_clap','inferred_rir'],
     waveform_time_ms=[0,250],waveform_ticks_ms=[0,50,100,150,200,250],
     waveform_panel_inches=[[panel_width,1.0]]*4,
     spectral_panel_inches=[[panel_width,spectral_height]]*4,
     frequency_ticks_khz=[.1,.3,1,3,10,15],spectral_ticks_db=spectral_ticks,
     normalization_hz=None if fully_unnormalized else [100,10000],display_db=spectral_limits,
     spectrum_normalization=('None per signal: one-sided FFT-bin power c[k]*abs(FFT(s))[k]**2/N**2, N=39690, c=2 except DC/Nyquist=1; mean within 1/12-octave bins, then 10 log10 with fixed digital power reference 1.' if fully_unnormalized else 'Original spectral convention: squared 39690-point FFT magnitude divided by each signal mean FFT-bin power over 0.1--10 kHz; 1/12-octave mean linear power, then 10 log10. No spectral-peak normalization.'),
     waveform_scale=scale,waveform_y_axis='RMS level (dBFS)' if scale=='db' else 'Digital amplitude',
     waveform_y_limits=rms_limits if scale=='db' else [-1,1],
     waveform_normalization='None: RMS of original digital samples, fixed amplitude reference 1' if scale=='db' else 'None: raw signed digital samples, no amplitude scaling',
     rms_window_samples=221 if scale=='db' else None,rms_hop_samples=44 if scale=='db' else None,
     rms_boundary='Centered window clipped to available source support, divided by actual sample count; no zero padding. The 250-ms point uses the full saved waveform.' if scale=='db' else None,
     waveform_db_definition='20 log10(RMS), fixed digital amplitude reference 1 and numerical amplitude floor 1e-12; no envelope-peak division' if scale=='db' else None,
     waveform_alignment='unchanged input-onset alignment; no prediction-dependent shift',
     spectral_curves_below_display_floor=[int((v<spectral_limits[0]).sum()) for v in spectra],
     rms_curves_below_display_floor=[int((v<rms_limits[0]).sum()) for v in temporal],
     display_calibration=('All panels preserve original digital levels. dBFS uses fixed digital power reference 1 and digital amplitude reference 1; no per-signal gain. Not calibrated SPL. Spectral values are mean powers per FFT bin, not integrated octave-band powers or PSD per Hz.' if fully_unnormalized else 'Temporal panels preserve original digital levels, not calibrated SPL; RMS dBFS uses digital full scale 1 for every curve. Spectra retain the original per-signal mean-power normalization.'),
     legend_columns=8,shared_legend_count=1,super_title=False)
   finish(fig,stem,plots,titles,legend,labels,details)
 if fully_unnormalized:
  # Direct sensitivity check: per-signal normalization would erase this gain.
  expected_shift=20*np.log10(2)
  for signals,original in ((x,rx),(h,rh)):
   scaled=digital_spectrum(signals.astype(float)*2)[1]
   assert np.allclose(scaled-original,expected_shift,atol=1e-10)
   for signal in signals:
    _,rms=rms_envelope(signal);_,scaled_rms=rms_envelope(signal.astype(float)*2)
    assert np.min(rms)>1e-12
    assert np.allclose(20*np.log10(scaled_rms)-20*np.log10(rms),expected_shift,atol=1e-10)
  combined('db');combined('linear')
  operations=dict(
   status='rendered-geometry-and-numeric-checks-passed-awaiting-pdf-review',
   source_inputs=str((R/'inputs.npz').relative_to(ROOT)),source_predictions=str((R/'predictions.npz').relative_to(ROOT)),
   source_selection_sha256=hashlib.sha256((R/'selection.json').read_bytes()).hexdigest(),
   source_input_sha256=hashlib.sha256((R/'inputs.npz').read_bytes()).hexdigest(),
   source_prediction_sha256=hashlib.sha256((R/'predictions.npz').read_bytes()).hexdigest(),
   spectral_panels=['a','b','e','f'],temporal_panels=['c','d','g','h'],
   sample_rate=SR,spectral_samples=N,display_hz=[100,15000],
   spectrum_operations=[
    's[n] = original saved digital samples, n=0,...,39689. No taper, filtering, DC subtraction, rescaling or realignment.',
    'S[k] = sum_n s[n] exp(-2*pi*i*k*n/N), N=39690.',
    'P[k] = c[k] * abs(S[k])**2 / N**2, c[k]=2 for interior one-sided bins and 1 for DC/Nyquist.',
    'P_band = arithmetic mean of P[k] for FFT frequencies inside each 1/12-octave display bin (last bin ends at 15 kHz).',
    'L_band = 10*log10(max(P_band,1e-30) / 1). Common digital power reference 1; no per-signal reference.',
   ],
   rms_operations=[
    'Original saved digital samples; unchanged onset alignment.',
    'R[m] = sqrt(sum_{n in W_m} s[n]**2 / len(W_m)), centered 221-sample window, centers every 44 samples plus the exact 250-ms endpoint; boundary windows use available real samples.',
    'L_R[m] = 20*log10(max(R[m],1e-12) / 1). Common digital amplitude reference 1. No division by max(R), and no maximum/mean subtraction.',
   ],
   waveform_operations=['Plot s[n] directly against 1000*n/44100 ms over the existing 0--250-ms display crop; no amplitude operations.'],
   no_per_signal_normalization=True,no_peak_mean_median_subtraction=True,no_output_gain_fitting=True,new_inference=0,
   spectrum_parseval_check='Passed: sum of one-sided bin powers equals mean square original waveform.',
   doubled_input_gain_shift_db=expected_shift,gain_preservation_check='Passed for all 32 spectra and RMS envelopes.',
   power_definition='FFT-bin power averaged inside octave bins, not integrated octave-band power or PSD/Hz. A bin-centered full-scale sine has total power 0.5 (-3.0103 dBFS) with this digital power reference.',
   signal_unit_note='Original saved digital amplitudes, not calibrated SPL; preserving model output gain does not establish physical calibration.'
  )
  (R/'fully_unnormalized_operations.json').write_text(json.dumps(operations,indent=2)+'\n')
  text=['# Fully unnormalized phone figure operations','',
   'All panels use the original frozen digital samples. No per-signal amplitude, mean-power, median, peak or RMS-maximum normalization; no gain fitting. N-dependent FFT power conversion is identical for all curves.','',
   '## Spectral panels (a), (b), (e), (f)','']
  text+=['- '+v for v in operations['spectrum_operations']]
  text+=['','## RMS temporal panels (c), (d), (g), (h)','']
  text+=['- '+v for v in operations['rms_operations']]
  text+=['','## Raw waveform temporal panels (c), (d), (g), (h)','']
  text+=['- '+v for v in operations['waveform_operations']]
  text+=['',operations['power_definition'],operations['signal_unit_note'],'',
   'Checks: Parseval equality passes; doubling any input shifts both spectrum and RMS by 6.0206 dB. Existing samples, predictions, and prior publication figures are preserved.','',
   'Outputs: publication/figures/fully_unnormalized_rms.pdf/.png and fully_unnormalized_waveform.pdf/.png.']
  (R/'fully_unnormalized_operations.md').write_text('\n'.join(text)+'\n')
  print(json.dumps(operations,indent=2))
  return
 if '--paper-only' not in sys.argv:reference_eight()
 if '--8modes-only' in sys.argv:return
 combined('db')
 combined('linear')
 if '--paper-only' in sys.argv:return
 four((0,1,2),'phone_qualitative_spectrum_3modes')
 # Retain the compact spectrum-only candidate separately from the wide reference layout.
 four(range(8),'phone_qualitative_spectrum_only')
 records=[]
 for row in lock['records']:
  row=dict(row);row['room_name']=ROOMS[0] if row['room_id']==1 else ROOMS[1];records.append(row)
 out={'status':'rendered-awaiting-visual-review','rooms':list(ROOMS),'modes':list(MODES),'time_modes':list(MODES),
  'frequency_display_hz':[100,15000],'normalization_hz':[100,10000],'spectrum_window_seconds':[0,.9],
  'spectrum':'All spectral panels use the original mean FFT-bin power normalization over 0.1--10 kHz and shared axes; temporal panels have no per-signal scaling.',
  'time_display_ms':[0,250],'time_display_normalization':'Combined figures: no per-signal scaling. RMS uses 20 log10(RMS) with fixed digital reference 1; linear uses original signed samples.',
  'selection_rule':'unclipped; highest input-only pre-handclap SNR per room/mode before inference',
  'records':records,'new_inference':False,'spectrogram_main_paper':False,
  'outputs':[f'publication/figures/phone_qualitative_spectrum_{x}.{e}' for x in ('8modes','3modes','time','only') for e in ('pdf','png')]}
 (R/'compact_figure_report.json').write_text(json.dumps(out,indent=2)+'\n')
 (R/'compact_figure_report.md').write_text('# Phone qualitative compact figures\n\nDisplay is 0.1--15 kHz. Combined figures retain original mean-power spectral normalization; only temporal panels use unnormalized digital levels. The combined time panels show all eight frozen modes over 0--250 ms as unnormalized RMS levels or original signed samples. All selected recordings are unclipped and use the frozen input-only pre-handclap-SNR rule. No inference was run.\n')
 print('WROTE',len(out['outputs']),'artifacts')
if __name__=='__main__':main()
