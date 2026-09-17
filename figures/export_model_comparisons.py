#!/usr/bin/env python3
"""Missing qualitative Flow predictions only; existing Regression outputs reused.

One fixed training seed and one stochastic draw, NOT a new model-selection
experiment. All Flow calls import the exact E3 sampler. E5 keeps frozen 250 ms
common support. E6 keeps the frozen raw-amplitude input and has no target RIR.
"""
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from matched_estimator_comparison import load,flow_sample
from claprir.metrics.lundeby_truncation import edc_truncated
SEED=42001; SIGMA=.027279

def main():
    torch.set_num_threads(2); dev=torch.device('cuda'); out=ROOT/'reports/real_clap_and_phone_figures'
    model,c=load(ROOT/'runs/flow_1s_aux',f'hybrid_flow_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{SEED}',dev)
    ex=ROOT/'reports/real_clap_room_evaluation/results/examples.npz'
    with np.load(ex) as z: idx=z['index'].tolist()
    path=out/'e5_flow_examples.npz'
    if not path.exists():
        with np.load(ROOT/'data/real_clap_multiclap/spheres_session1.npz') as z:
            obs=np.zeros((len(idx),1,44100),np.float32); obs[:,0,:11025]=z['observation'][idx,0,:11025]
        with torch.no_grad(): pred=flow_sample(model,torch.from_numpy(obs).to(dev),SIGMA,torch.Generator(device=dev).manual_seed(20260907+SEED)).cpu().numpy()[:,0]
        np.savez_compressed(path,index=idx,prediction=pred,seed=SEED,score_support=11025)
    path=out/'e6_flow_comparison_curves.npz'
    if not path.exists():
        selection=json.loads((out/'provenance.json').read_text()); corpus=ROOT/'reports/phone_clap_demo'
        meta=[r for r in csv.DictReader((corpus/'results/metadata.csv').open()) if r['source']=='m4a' and r['event_type']=='clap' and r['protocol_status']=='normal']
        chosen=[r for r in meta if r['room_name'] in selection['E6_example_rooms'] and r['clap_mode'] in selection['E6_example_modes'][r['room_name']]]
        assert len(chosen)==45
        curves=[]; ids=[]
        for start in range(0,len(chosen),5):
            batch=np.zeros((len(chosen[start:start+5]),1,44100),np.float32)
            for j,r in enumerate(chosen[start:start+5]):
                rate,x=wavfile.read(corpus/'results/segments'/r['segment_path']); assert rate==48000 and x.ndim==1 and np.issubdtype(x.dtype,np.floating)
                x=resample_poly(x.astype(np.float64),147,160)[round(float(r['pre_pad_s'])*44100):]
                batch[j,0,:min(len(x),44100)]=x[:44100].astype(np.float32)
                ids.append(r['segment_path'])
            with torch.no_grad(): pred=flow_sample(model,torch.from_numpy(batch).to(dev),SIGMA,torch.Generator(device=dev).manual_seed(20260907+SEED+start)).cpu().numpy()[:,0]
            for e in pred: curves.append(np.interp(np.arange(1000)*44.1,np.arange(44100),edc_truncated(e,44100)).astype(np.float32))
            print('E6 Flow comparison',start+len(batch),'/45',flush=True)
        np.savez_compressed(path,sample_id=np.array(ids),edc_db=curves,time_ms=np.arange(1000),seed=SEED)
    (out/'model_comparison_provenance.json').write_text(json.dumps(dict(training_seed=SEED,sampler='experiments/matched_estimator_comparison.py:flow_sample',sigma_d=SIGMA,nfe=20,gamma=.5,peak_projection=False,noise_seeds='20260907+42001 for E5; +batch_start for E6 (5 per batch)',selection='Unchanged Regression-based representative selections; no Flow-dependent selection',E5_support='frozen 250 ms common support, both models trained at 1 s',E6_input='exact raw-amplitude resample/onset convention of final E6; no GT',purpose='One-seed one-draw qualitative comparison only; no downstream model reselection'),indent=2)+'\n')
    print('Model comparison exports complete',flush=True)
if __name__=='__main__': main()
