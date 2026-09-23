#!/usr/bin/env python3
"""Final E6: raw-amplitude phone deployment; estimates only, no reference.

Absolute paths, preflight of all 280 metadata members, cached batch predictions,
atomic incremental descriptors and EDC artifacts. Existing inference is reused.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from claprir.metrics.lundeby_truncation import edc_truncated
from claprir.metrics.room_acoustics import edt_seconds
from claprir.metrics.deconvolution import c50_db, t30_seconds
from claprir.training.train_rir_estimator import RunConfig, load_model
SR=44100
SEEDS=(42001,42002,42003)


def atomic_csv(path, rows):
    temp=path.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
    temp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',type=Path,default=ROOT)
    ap.add_argument('--corpus',type=Path,default=ROOT/'reports/phone_clap_demo')
    ap.add_argument('--report-root',type=Path,default=ROOT/'reports/phone_deployment_evaluation')
    ap.add_argument('--cache-root',type=Path,default=ROOT/'runs/e6_phone_raw_cache')
    ap.add_argument('--device',default='cuda')
    a=ap.parse_args(); torch.set_num_threads(2)
    for key in ('repo_root','corpus','report_root','cache_root'): setattr(a,key,getattr(a,key).resolve())
    meta_path=a.corpus/'results/metadata.csv'
    meta=[r for r in csv.DictReader(meta_path.open()) if r['source']=='m4a' and r['event_type']=='clap' and r['protocol_status']=='normal']
    assert len(meta)==280 and len({r['segment_path'] for r in meta})==280
    cells={}
    for r in meta:
        cells.setdefault((r['room_name'],r['clap_mode']),[]).append(r)
        assert (a.corpus/'results/segments'/r['segment_path']).is_file(), r['segment_path']
    assert len(cells)==56 and all(len(v)==5 for v in cells.values())
    runroot=a.repo_root/'runs/matched_reg_1s'
    names={s:f'hybrid_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{s}' for s in SEEDS}
    for name in names.values():
        assert (runroot/name/'model_updates20000.pt').is_file()
        assert (runroot/name/'config.resolved.json').is_file()
    a.report_root.mkdir(parents=True,exist_ok=True); a.cache_root.mkdir(parents=True,exist_ok=True)
    results=a.report_root/'results'; results.mkdir(exist_ok=True)
    contract=dict(protocol='raw_amplitude_v1',model='matched 1 s Regression',seeds=list(SEEDS),sample_rate=SR,
                  preprocessing='M4A-derived float audio; resample_poly(147,160); metadata pre-pad removal; float32; no amplitude normalization, filtering or denoising',
                  metadata=str(meta_path),metadata_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                  checkpoints={str(s):str(runroot/names[s]/'model_updates20000.pt') for s in SEEDS},
                  edc='backward-integrated predicted energy, normalized to 0 dB; 1000 points at 0..999 ms; 1 s finite-horizon descriptors; no ground truth',
                  descriptors=['c50_db','edt_s','t30_s','peak','edc_at_100ms_db','edc_at_250ms_db'],
                  t30_limitation='companion only; finite-horizon terminal decay can bias T30; not a stable primary metric')
    cp=a.report_root/'provenance.json'
    if cp.exists(): assert json.loads(cp.read_text())==contract, 'protocol mismatch'
    cp.write_text(json.dumps(contract,indent=2)+'\n')
    out=results/'per_clap.csv'; rows=list(csv.DictReader(out.open())) if out.exists() else []
    done={(int(r['seed']),r['sample_id']) for r in rows}
    curves={}
    curve_path=results/'edc_curves.npz'
    if curve_path.exists():
        with np.load(curve_path) as z:
            curves={(int(s),str(k)):v for s,k,v in zip(z['seed'],z['sample_id'],z['edc_db'])}
    def save():
        # Curves precede CSV so every committed row has a curve after a kill.
        keys=sorted(curves)
        temp=curve_path.with_suffix('.tmp.npz')
        np.savez_compressed(temp,seed=np.array([k[0] for k in keys]),sample_id=np.array([k[1] for k in keys]),
                            time_ms=np.arange(1000),edc_db=np.array([curves[k] for k in keys],np.float32))
        temp.replace(curve_path); atomic_csv(out,rows)
    print('Preflight OK: 280 claps, 56 cells, raw amplitude; resume',len(rows),flush=True)
    for seed in SEEDS:
        cfg=json.loads((runroot/names[seed]/'config.resolved.json').read_text()); cfg['training_datasets']=tuple(cfg['training_datasets'])
        c=RunConfig(**cfg); model=None
        for start in range(0,len(meta),8):
            chunk=meta[start:start+8]
            keys=[(seed,r['segment_path']) for r in chunk]
            if all(k in done and k in curves for k in keys): continue
            cache=a.cache_root/f'raw_v1_seed{seed}_batch{start:03d}.npy'
            if cache.exists():
                est=np.load(cache)
                assert est.shape==(len(chunk),44100) and np.isfinite(est).all()
            else:
                if model is None: model=load_model(c,20000,torch.device(a.device),runroot,runroot)
                batch=np.zeros((len(chunk),1,c.signal_length),np.float32)
                for j,r in enumerate(chunk):
                    rate,x=wavfile.read(a.corpus/'results/segments'/r['segment_path'])
                    assert rate==48000 and x.ndim==1 and np.issubdtype(x.dtype,np.floating), 'expected authoritative true-mono float WAV'
                    x=resample_poly(x.astype(np.float64),147,160)
                    x=x[round(float(r['pre_pad_s'])*SR):]
                    batch[j,0,:min(len(x),c.signal_length)]=x[:c.signal_length].astype(np.float32)
                with torch.no_grad(): est=model.predict(torch.from_numpy(batch).to(a.device)).cpu().numpy()[:,0]
                assert np.isfinite(est).all()
                temp=cache.with_suffix('.tmp.npy'); np.save(temp,est); temp.replace(cache)
            for j,r in enumerate(chunk):
                key=keys[j]; e=est[j].astype(np.float64)
                curve=edc_truncated(e,len(e))
                curves[key]=np.interp(np.arange(1000)*SR/1000,np.arange(len(curve)),curve).astype(np.float32)
                if key in done: continue
                rows.append(dict(seed=seed,room_id=r['room_id'],room_name=r['room_name'],clap_mode=r['clap_mode'],repeat_id=r['repeat_id'],
                    sample_id=r['segment_path'],snr_db=float(r['snr_db']),c50_db=float(c50_db(e,SR)),edt_s=float(edt_seconds(e.astype(np.float32),SR)),
                    t30_s=float(t30_seconds(e,SR)),peak=float(np.abs(e).max()),edc_at_100ms_db=float(curve[4410]),edc_at_250ms_db=float(curve[11025])))
                done.add(key)
            save(); print(f'seed {seed}: saved {len(rows)}/840',flush=True)
        del model; torch.cuda.empty_cache()
    assert len(rows)==840 and len(done)==840 and len(curves)==840
    (a.report_root/'completion.json').write_text(json.dumps(dict(status='completed-valid',rows=len(rows),protocol=contract['protocol']),indent=2)+'\n')
    print('E6 COMPLETE: 840 estimate-only records, no GT',flush=True)

if __name__=='__main__': main()
