#!/usr/bin/env python3
"""Frozen E3 models on metadata-paired ARPEGE; never trains or selects a model.

Prepare is independent of predictions. Score imports E3's actual sampler and
scorer. A separate plotting script reads only exported artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
REPORT = ROOT / "reports/arpege_external_evaluation"
RUNTIME = Path(os.environ.get("CLAPRIR_RUNTIME",
                              Path(__file__).resolve().parents[1] / "runtime"))\
          / "arpege_external_evaluation"
SR = 44100
SEEDS = (42001,)
FLOW_DRAWS = 1
ARMS = {
    "regression": ("runs/matched_reg_1s", "hybrid_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{}"),
    "flow": ("runs/flow_1s_aux", "hybrid_flow_shoebox_mit_but_ace_1000ms_full_c1_vm_seed{}"),
}
METRICS = ("edc_rmse_db", "abs_c50_error_db", "abs_edt_error_s", "echo_density_rmse",
           "stft_logmag_mse", "nrmse")
PROTOCOL = {
    "role": "external evaluation only; original E3 models; no training or model selection",
    "sample_rate": SR, "horizon_samples": SR, "channel_index": 0,
    "training_seeds": list(SEEDS), "flow_draws": FLOW_DRAWS,
    "scope": "single frozen E3 checkpoint per method; descriptive external comparison, no training-seed uncertainty",
    "flow_sampler": "experiments/matched_estimator_comparison.py:flow_sample",
    "sigma_d": 0.027279, "nfe": 20, "churn": 0.5, "peak_projection": False,
    "coordinate": "original E3 peak-target waveform coordinate, NOT normalized blind/rescue coordinate",
    "draw_seed": "20260907 + training_seed + 100 * sorted_manifest_index + draw_index",
    "point_rule": "one preregistered Flow draw per observation; no best-of-K or output fitting",
    "mono": "physical capsule index 0 in the same released Eigenmike array for clap and reference",
    "event_detector": "existing Spheres clap_onsets: 1 ms absolute envelope, 6% peak threshold, 0.30 s separation; retain all detected peaks for adjacency checks",
    "event_rule": "first chronological peak with >=0.50 s preceding gap and >=1.05 s following gap; 5 ms prepad, 1.04 s raw window; require full 1 s after independent onset alignment",
    "waveform_preprocessing": "resample_poly to 44.1 kHz; independently first sample >=50% peak; crop 1 s; peak-normalize crop by max(abs)+1e-8, same E5/E3 convention",
    "forbidden_preprocessing": "no denoising, high-pass, channel average, fitted gain, reference-guided delay or fitted output alignment",
    "score_support": "common measured post-onset prefix, capped at 44100; no reference zero padding scored",
    "aggregation": "mean Flow draw metrics within training seed/record; median seeds within record; median arrays within room/position; median positions within room; median rooms overall, descriptive only",
    "examples": "sorted available room IDs: first, middle, last; smallest SHA256(pairing_id) within each; fixed training seed42001/draw0; selected before inference",
    "reference": "measured RIR at matched source/receiver coordinates; loudspeaker and hand-clap directivity not matched",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_csv(path):
    with Path(path).open() as f: return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None: fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != clean(value):
            raise RuntimeError(f"Frozen artifact drift: {path}; do not replace after inference")
    else: write_json(path, value)


def read_capsule(path, channel):
    parts = []
    with sf.SoundFile(path) as f:
        rate, channels = f.samplerate, f.channels
        if not 0 <= channel < channels: raise ValueError("invalid physical receiver channel")
        for block in f.blocks(blocksize=65536, dtype="float32", always_2d=True):
            parts.append(block[:, channel].copy())
    return np.concatenate(parts), rate


def canonical(signal, rate, length=SR):
    """E5 align/crop/peak convention generalized only in horizon, not definition."""
    divisor = math.gcd(SR, int(rate))
    x = resample_poly(signal, SR // divisor, int(rate) // divisor).astype(np.float32)
    peak = float(np.max(np.abs(x)))
    if not np.isfinite(x).all() or peak <= 0: raise ValueError("silent/nonfinite waveform")
    onset = int(np.flatnonzero(np.abs(x) >= .5 * peak)[0])
    support = min(length, len(x) - onset)
    out = np.zeros(length, np.float32); out[:support] = x[onset:onset + support]
    out /= np.max(np.abs(out)) + 1e-8
    return out, support, onset


def extract_event(signal, rate):
    from claprir.datasets.handclap_corpus import clap_onsets
    # This upper bound retains every detected event; no strongest-12 selection
    # may conceal the next clap when checking a one-second observation window.
    peaks = clap_onsets(signal, rate, max_claps=len(signal))
    for j, peak in enumerate(peaks):
        before = np.inf if j == 0 else (peak - peaks[j - 1]) / rate
        after = np.inf if j + 1 == len(peaks) else (peaks[j + 1] - peak) / rate
        if before < .5 or after < 1.05: continue
        start = max(0, peak - round(.005 * rate)); end = start + round(1.04 * rate)
        if end > len(signal): continue
        y, support, onset = canonical(signal[start:end], rate)
        if support < SR: continue
        return y, dict(detected_events=len(peaks), event_index=j, peak_frame=peak,
                       raw_start_frame=start, raw_stop_frame=end, source_rate=rate,
                       onset_resampled_frame=onset, preceding_gap_s=before,
                       following_gap_s=after, input_valid_samples=support,
                       raw_peak=float(np.max(np.abs(signal[start:end]))),
                       clipping_fraction=float(np.mean(np.abs(signal[start:end]) >= 32767 / 32768)))
    return None, dict(detected_events=len(peaks), reason="no first eligible isolated complete one-second clap")


def model_contract():
    previous = json.loads((ROOT / "reports/current_benchmark_figures/provenance.json").read_text())
    models = {}
    for arm, (base, name) in ARMS.items():
        for seed in SEEDS:
            folder = ROOT / base / name.format(seed)
            cfgpath = folder / "config.resolved.json"; cfg = json.loads(cfgpath.read_text())
            expected = dict(architecture="hybrid" if arm == "regression" else "hybrid_flow",
                            horizon_ms=1000, updates=20000, nf=128, input_compression=1.0,
                            stft_weight=.25, spectral_weight=0.0, validity_masking=True,
                            target_normalisation="peak", k_max=1, seed=seed)
            for key, value in expected.items():
                if cfg[key] != value: raise RuntimeError(f"Original E3 config mismatch: {folder}/{key}")
            if not np.isclose(cfg["spectral_exponent"], 2/3): raise RuntimeError("cSTFT exponent drift")
            ckpt = folder / "model_updates20000.pt"; digest = sha(ckpt)
            if seed == 42001 and digest != previous["checkpoints"][arm]["sha256"]:
                raise RuntimeError(f"{arm} differs from original publication checkpoint")
            models[f"{arm}_{seed}"] = dict(path=str(ckpt.relative_to(ROOT)), sha256=digest,
                config=cfg, config_sha256=sha(cfgpath), prior_publication_hash_verified=(seed == 42001))
    sources = [Path(__file__), ROOT / "experiments/matched_estimator_comparison.py",
               ROOT / "eloi_flow_debug/flow_sampler.py", ROOT / "eloi_flow_debug/lundeby.py",
               ROOT / "src/clapgen/evaluation/diagnosis.py",
               ROOT / "src/clapgen/models/hybrid_rir.py",
               ROOT / "src/clapgen/experiments/hybrid_direct_rir/run.py",
               ROOT / "src/clapgen/experiments/real_clap_multiclap/extract.py"]
    return dict(models=models, sources={str(p.relative_to(ROOT)): sha(p) for p in sources})


def prepare():
    REPORT.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True)
    manifest = REPORT / "pairing_manifest.csv"
    pairs = sorted(read_csv(manifest), key=lambda r: r["pairing_id"])
    if not pairs: raise RuntimeError("No metadata-verified pairs; inference is not authorized")
    if len({r['pairing_id'] for r in pairs}) != len(pairs): raise RuntimeError("duplicate pair ID")
    freeze_json(REPORT / "protocol.json", PROTOCOL)
    freeze_json(REPORT / "model_provenance.json", model_contract())
    arrays = []; events = []; excluded = []
    for index, r in enumerate(pairs):
        if r["geometry_verified"].lower() != "true" or int(r["channel_index"]) != 0:
            raise RuntimeError("unverified geometry/channel in scoring manifest")
        clap, rate = read_capsule(ROOT / r["clap_file"], 0)
        y, detail = extract_event(clap, rate)
        if y is None:
            excluded.append(dict(pairing_id=r["pairing_id"], **detail)); continue
        raw_h, fs_h = read_capsule(ROOT / r["rir_file"], 0)
        h, bound, onset = canonical(raw_h, fs_h)
        if bound < 2205: raise RuntimeError("Reference support too short for C50")
        event = dict(pairing_id=r["pairing_id"], manifest_index=index, room=r["room"],
                     position=r["position"], array=r["array"], bound=bound,
                     reference_onset_resampled_frame=onset, **detail,
                     clap_sha256=sha(ROOT / r["clap_file"]), rir_sha256=sha(ROOT / r["rir_file"]),
                     input_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                     target_sha256=hashlib.sha256(h.tobytes()).hexdigest())
        arrays.append((y,h)); events.append(event)
        print(f"Prepared {len(events)}/{len(pairs)} {r['pairing_id']} ({detail['detected_events']} detected events)", flush=True)
    if not events: raise RuntimeError("No isolated one-second observations; do not invent/pad a second clap")
    rooms = sorted({e['room'] for e in events})
    selected_rooms = [rooms[i] for i in sorted({0, len(rooms)//2, len(rooms)-1})]
    examples = [min((e for e in events if e['room']==room),
                    key=lambda e: hashlib.sha256(e['pairing_id'].encode()).hexdigest())['pairing_id']
                for room in selected_rooms]
    lock = dict(pairing_manifest_sha256=sha(manifest), protocol_sha256=sha(REPORT/'protocol.json'),
                model_provenance_sha256=sha(REPORT/'model_provenance.json'), events=events,
                excluded=excluded, examples=examples, available_pairs=len(pairs), usable_events=len(events))
    freeze_json(REPORT / "evaluation_lock.json", lock)
    write_csv(REPORT/'event_manifest.csv', events)
    write_csv(REPORT/'event_exclusions.csv', excluded, ['pairing_id','detected_events','reason'])
    cache = RUNTIME/'inputs.npz'
    values = dict(observation=np.stack([a[0] for a in arrays]), target=np.stack([a[1] for a in arrays]),
                  pairing_id=np.asarray([e['pairing_id'] for e in events]))
    if cache.exists():
        with np.load(cache) as old:
            if any(not np.array_equal(old[k], v) for k,v in values.items()): raise RuntimeError("input cache drift")
    else: np.savez_compressed(cache, **values)
    write_json(REPORT/'state.json', dict(status="prepared-no-inference", usable_events=len(events),
                                       rooms=len(rooms), examples=examples))


def verify_lock():
    lock = json.loads((REPORT/'evaluation_lock.json').read_text())
    for stem, key in [('pairing_manifest.csv','pairing_manifest_sha256'),
                      ('protocol.json','protocol_sha256'), ('model_provenance.json','model_provenance_sha256')]:
        if sha(REPORT/stem) != lock[key]: raise RuntimeError(f"Frozen file changed: {stem}")
    contract = json.loads((REPORT/'model_provenance.json').read_text())
    for path, digest in contract['sources'].items():
        if sha(ROOT/path) != digest: raise RuntimeError(f"Evaluation implementation changed: {path}")
    return lock, contract


def amend_execution_to_cuda():
    """Preserve the unfinished CPU attempt; freeze a matched CUDA execution.

    This changes execution backend only, before producing a comparison or
    viewing provider outcomes. All scientific choices and waveform tensors stay
    frozen. No training process, environment, or GPU allocation is modified.
    """
    if (REPORT/'per_example.csv').exists():
        raise RuntimeError('Completed evaluation must not be restarted')
    old = {name:json.loads((REPORT/name).read_text()) for name in
           ['evaluation_lock.json','model_provenance.json','inference_backend.json','state.json']}
    current = model_contract()
    if old['model_provenance.json']['models'] != current['models']:
        raise RuntimeError('Checkpoint/config change is not an execution amendment')
    for path, value in old['model_provenance.json']['sources'].items():
        if path != str(Path(__file__).relative_to(ROOT)) and current['sources'][path] != value:
            raise RuntimeError('Shared sampler/model/scorer changed')
    destination=RUNTIME/'predictions_cpu_attempt'
    if destination.exists(): raise RuntimeError('CPU attempt already archived')
    freeze_json(REPORT/'execution_amendment.json',dict(
        reason='CPU throughput would take over one hour; original E3 used CUDA. Freeze both arms on the same already allocated 48 GB GPU; preserve incomplete CPU diagnostics.',
        before_final_comparison=True, provider_outcomes_not_inspected=True,
        scientific_protocol_changed=False, checkpoint_changed=False, data_changed=False,
        gpu_uuid=os.environ.get('CLAPRIR_GPU_UUID',''),previous=old,
        cpu_cache=str(destination)))
    (RUNTIME/'predictions').rename(destination)
    (REPORT/'inference_backend.json').rename(REPORT/'inference_backend_cpu_attempt.json')
    write_json(REPORT/'model_provenance.json',current)
    lock=old['evaluation_lock.json']
    lock['model_provenance_sha256']=sha(REPORT/'model_provenance.json')
    write_json(REPORT/'evaluation_lock.json',lock)
    write_json(REPORT/'state.json',dict(status='prepared-for-matched-cuda-execution'))


def score_all(device='cuda', threads=2):
    import torch
    from matched_estimator_comparison import load, flow_sample, score, NFE, GAMMA
    from clapgen.experiments.blind_joint.edp_scoring_fast import verified_edp_context
    if NFE != 20 or GAMMA != .5: raise RuntimeError("E3 sampler drift")
    lock, contract = verify_lock(); torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    dev = torch.device(device)
    if dev.type=='cuda':
        expected=os.environ.get('CLAPRIR_GPU_UUID')
        if not expected or os.environ.get('CUDA_VISIBLE_DEVICES') != expected \
                or torch.cuda.device_count() != 1:
            raise RuntimeError('Set CLAPRIR_GPU_UUID and CUDA_VISIBLE_DEVICES to one card')
        free,total=torch.cuda.mem_get_info()
        if free < 8*1024**3: raise RuntimeError('Insufficient free memory for short inference')
        torch.cuda.set_per_process_memory_fraction(.15)
    backend = dict(device=str(dev), torch=torch.__version__, threads=threads,
                   cuda_initialized=torch.cuda.is_initialized(), deterministic_algorithms=True)
    freeze_json(REPORT/'inference_backend.json', backend)
    events = lock['events']; start = time.monotonic()
    with np.load(RUNTIME/'inputs.npz') as d:
        obs=d['observation']; targets=d['target']; ids=d['pairing_id'].astype(str)
    if ids.tolist() != [e['pairing_id'] for e in events]: raise RuntimeError("input identity/order mismatch")
    for i,e in enumerate(events):
        for key, value in [('input_sha256',obs[i]),('target_sha256',targets[i])]:
            if hashlib.sha256(value.tobytes()).hexdigest()!=e[key]: raise RuntimeError("cached waveform drift")
    runtime = RUNTIME/'predictions'; runtime.mkdir(exist_ok=True)
    rows=[]; selected={}
    with verified_edp_context() as edp:
        freeze_json(REPORT/'edp_provenance.json', edp)
        for arm,(base,name) in ARMS.items():
            for seed in SEEDS:
                ck=contract['models'][f'{arm}_{seed}']
                if sha(ROOT/ck['path']) != ck['sha256']: raise RuntimeError("checkpoint changed")
                model, cfg=load(ROOT/base,name.format(seed),dev); model.eval()
                draws=FLOW_DRAWS if arm=='flow' else 1
                for i,e in enumerate(events):
                    x=torch.from_numpy(obs[i])[None,None].to(dev)
                    for draw in range(draws):
                        key=f'{arm}_{seed}_{i:03d}_{draw}'
                        predfile=runtime/f'{key}.npy'; metafile=runtime/f'{key}.json'
                        identity=dict(lock_sha256=sha(REPORT/'evaluation_lock.json'), arm=arm, seed=seed,
                                      draw=draw, input_sha256=e['input_sha256'], checkpoint_sha256=ck['sha256'],
                                      inference_seed=20260907+seed+100*int(e['manifest_index'])+draw,
                                      backend=backend)
                        if predfile.exists() and metafile.exists():
                            saved=json.loads(metafile.read_text())
                            if saved['identity'] != identity or saved['prediction_sha256'] != sha(predfile):
                                raise RuntimeError("Prediction provenance mismatch")
                            prediction=np.load(predfile); metrics=saved['metrics']
                        else:
                            with torch.inference_mode():
                                if arm=='flow':
                                    pred=flow_sample(model,x,.027279,torch.Generator(device=dev).manual_seed(identity['inference_seed']))
                                else: pred=model.predict(x)
                            prediction=pred[0,0].cpu().numpy()
                            if prediction.shape!=(SR,) or not np.isfinite(prediction).all(): raise RuntimeError("invalid model output")
                            metrics=score(targets[i],prediction,int(e['bound']))
                            np.save(predfile,prediction)
                            write_json(metafile,dict(identity=identity,prediction_sha256=sha(predfile),metrics=metrics))
                        rows.append(dict(pairing_id=e['pairing_id'],room=e['room'],position=e['position'],array=e['array'],
                                         arm=arm,training_seed=seed,draw=draw,bound=e['bound'],
                                         **{k:metrics[k] for k in METRICS}))
                        if e['pairing_id'] in lock['examples'] and seed==42001 and draw==0:
                            selected[arm,e['pairing_id']]=prediction
                    write_json(REPORT/'state.json',dict(status="scoring",arm=arm,training_seed=seed,
                        completed_records=len(rows), expected_records=len(events)*len(SEEDS)*(1+FLOW_DRAWS), elapsed_s=time.monotonic()-start,
                        last_pairing_id=e['pairing_id'],pid=os.getpid(),device=str(dev)))
                    print(f"{arm} seed{seed}: {i+1}/{len(events)}, total {len(rows)}/{len(events)*len(SEEDS)*(1+FLOW_DRAWS)}, elapsed {time.monotonic()-start:.1f}s",flush=True)
                del model
    write_csv(REPORT/'per_example.csv',rows)
    chosen=[ids.tolist().index(k) for k in lock['examples']]
    np.savez_compressed(REPORT/'examples.npz',pairing_id=ids[chosen],observation=obs[chosen],target=targets[chosen],
        regression=np.stack([selected['regression',k] for k in lock['examples']]),
        flow=np.stack([selected['flow',k] for k in lock['examples']]),
        bound=np.asarray([events[i]['bound'] for i in chosen]),
        room=np.asarray([events[i]['room'] for i in chosen]), position=np.asarray([events[i]['position'] for i in chosen]),
        array=np.asarray([events[i]['array'] for i in chosen]))
    write_json(REPORT/'state.json',dict(status="scored-awaiting-report",records=len(rows),pid=os.getpid(),device=str(dev)))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['prepare','score','amend-cuda']); ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    if args.stage=='prepare': prepare()
    elif args.stage=='amend-cuda': amend_execution_to_cuda()
    else: score_all(threads=args.threads)


if __name__=='__main__': main()
