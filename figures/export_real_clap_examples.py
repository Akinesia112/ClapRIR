#!/usr/bin/env python3
"""Inference only for missing E5 figure waveforms; never rerun quantitative scores."""
import csv,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from claprir.training.train_rir_estimator import RunConfig,load_model
OUT=ROOT/'reports/real_clap_room_evaluation/results/examples.npz'
def main():
    if OUT.exists(): print('existing examples retained'); return
    torch.set_num_threads(2)
    rows=list(csv.DictReader((ROOT/'reports/real_clap_room_evaluation/results/per_example.csv').open()))
    assert len(rows)==153
    rank=sorted({int(r['index']) for r in rows},key=lambda i:(np.median([float(r['edc_rmse_db']) for r in rows if int(r['index'])==i]),i))
    idx=[rank[int(np.floor((len(rank)-1)*q))] for q in (.25,.5,.75)]
    seed=42001; root=ROOT/'runs/matched_reg_1s'; name=f'hybrid_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{seed}'
    cfg=json.loads((root/name/'config.resolved.json').read_text()); cfg['training_datasets']=tuple(cfg['training_datasets']); c=RunConfig(**cfg)
    with np.load(ROOT/'data/real_clap_multiclap/spheres_session1.npz') as d:
        obs=np.zeros((3,1,c.signal_length),np.float32); obs[:,0,:11025]=d['observation'][idx,0,:11025]
        target=d['rir'][idx,:11025].astype(np.float32); positions=d['position'][idx].astype(str); mics=d['mic'][idx].astype(str)
    model=load_model(c,20000,torch.device('cuda'),root,root)
    with torch.no_grad(): pred=model.predict(torch.from_numpy(obs).cuda()).cpu().numpy()[:,0,:11025]
    np.savez_compressed(OUT,index=idx,reference=target,prediction=pred,position=positions,mic=mics,seed=seed,sample_rate=44100)
    (OUT.parent/'examples_selection.json').write_text(json.dumps(dict(rule='Sort 51 positions by median EDC RMSE across training seeds, index breaks ties; floor((51-1)*q), q=0.25/0.5/0.75. Fixed seed 42001 for qualitative curves.',indices=idx,reason_for_inference='Quantitative scores existed but predicted waveforms were missing; only three selected examples exported.'),indent=2)+'\n')
    print('exported',idx,flush=True)
if __name__=='__main__': main()
