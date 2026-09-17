#!/usr/bin/env python3
"""
rir_sources.py — RIR generation + (clean_clap, room_clap) pairing for the
supervised conditional-clap-estimation baseline.

Three RIR sources, by increasing realism:
  1. synthetic RT60 exp×noise   — TRAINING (cheap, unlimited; Kyung's param.)
  2. shoebox (pyroomacoustics)  — TESTING (has early reflections)
  3. MIT measured RIR           — TESTING (real measurement; download if absent)

room_clap = clean_clap ⊛ RIR via the VERIFIED flip-convolution operator.
fs = 44100 everywhere; any RIR at another fs is resampled with an explicit log.
"""

import numpy as np
import torch
import torch.nn.functional as F

FS = 44100


# ── 1. synthetic RT60 RIR (training) ──────────────────────────────────────────

def make_rt60_rir(rt60=0.5, A=1.0, fs=FS, n=None, rng=None):
    """
    Kyung's verbatim parameterization: exponentially-decaying white noise.
        lambda = 6.9078 / (RT60 * fs)     # 6.9078 = ln(1000) → 60 dB decay
        rir    = randn(N) * A * exp(-lambda * k)
    This is a DIFFUSE late-reverb model — NO direct path, NO early reflections.
    It is the hardest test case (the clap is smeared into a structureless tail).
    Earlier versions added a `rir[0] += direct_gain` "direct peak" — REMOVED: it
    was buried in the tail (ratio to tail-RMS only ~3.6, global max not at t=0),
    so it did not constitute a direct sound and did not change results (env_corr
    0.725 with vs 0.713 without). For a real direct path / early reflections use
    the shoebox or MIT RIRs instead.
    Returns float32, raw (room_clap is re-normalized after convolution anyway).
    """
    if rng is None:
        rng = np.random
    lam = 6.9078 / (rt60 * fs)
    if n is None:
        n = int(1.0 * fs)                  # N = fs = 1 s (Kyung)
    k = np.arange(n)
    rir = rng.randn(n).astype(np.float64) * A * np.exp(-lam * k)
    return rir.astype(np.float32)


def make_rt60_rir_random(rt60_range=(0.2, 0.8), fs=FS, rng=None):
    """Training RIR with RT60 randomized over a range for diversity."""
    if rng is None:
        rng = np.random
    rt60 = rng.uniform(*rt60_range)
    return make_rt60_rir(rt60=rt60, fs=fs, rng=rng), rt60


# ── 2. shoebox-simulated RIR (testing) ────────────────────────────────────────

def make_shoebox_rirs(fs=FS, max_order=10):
    """
    Generate a handful of shoebox RIRs with early reflections via
    pyroomacoustics. Returns list of (rir_float32, meta_dict).
    Requires pyroomacoustics.
    """
    import pyroomacoustics as pra
    configs = [
        # (room_dim,        src_pos,        mic_pos,        rt60,  label)
        ([4, 3, 2.5],   [1.0, 1.0, 1.2], [3.0, 2.0, 1.2], 0.3, "small_room"),
        ([7, 5, 3.0],   [1.5, 1.0, 1.5], [5.5, 4.0, 1.5], 0.5, "medium_room"),
        ([12, 9, 4.0],  [2.0, 2.0, 1.7], [10.0, 7.0, 1.7], 0.8, "large_hall"),
        ([5, 4, 2.8],   [1.0, 1.0, 1.4], [4.0, 3.0, 1.4], 0.4, "office"),
    ]
    out = []
    for room_dim, src, mic, rt60, label in configs:
        e_abs, max_o = pra.inverse_sabine(rt60, room_dim)
        room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_abs),
                           max_order=min(max_o, max_order))
        room.add_source(src)
        room.add_microphone(np.array(mic).reshape(3, 1))
        room.compute_rir()
        rir = room.rir[0][0].astype(np.float32)
        rir = rir / np.max(np.abs(rir))
        out.append((rir, {"label": label, "room_dim": room_dim,
                          "rt60_target": rt60, "len": len(rir)}))
    return out


# ── 3. MIT measured RIR (testing) ─────────────────────────────────────────────

MIT_URL = "https://mcdermottlab.mit.edu/Reverb/IR_Survey.html"
MIT_DOWNLOAD = ("MIT Acoustical Reverberation Survey — 271 real RIRs.\n"
                "  Download: https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip\n"
                "  (or per-file from the survey page). 48 kHz wavs → resample to 44.1k.")

def load_mit_rirs(mit_dir, fs=FS, n_max=8):
    """Load + resample MIT RIRs from a directory of wavs. Returns list of (rir, meta)."""
    import soundfile as sf
    from pathlib import Path
    import scipy.signal as sps
    out = []
    for wav in sorted(Path(mit_dir).glob("*.wav"))[:n_max]:
        x, sr = sf.read(str(wav))
        if x.ndim > 1:
            x = x[:, 0]
        if sr != fs:
            x = sps.resample_poly(x, fs, sr)
        x = (x / np.max(np.abs(x))).astype(np.float32)
        out.append((x, {"label": wav.stem, "orig_fs": sr, "len": len(x)}))
    return out


# ── pairing: room_clap = clean_clap ⊛ RIR (verified flip convolution) ──────────

def convolve_clap_rir(clean, rir, out_len=None, device="cpu"):
    """
    room_clap = clean ⊛ rir via F.conv1d(clean, rir.flip) — the operator verified
    against np.convolve to 7e-9. Both at FS. Returns float32 (out_len,) (or full).

    Efficiency: if out_len is set, room_clap[:out_len] only depends on
    rir[:out_len + len(clean)] (since room[n]=Σ_k clean[k] rir[n-k], k<len(clean)),
    so the RIR is truncated before the conv — avoids a full 44100-tap conv for a
    4096-sample output (numerically identical on the kept region).
    """
    clean = np.asarray(clean)
    if out_len is not None:
        keep = out_len + len(clean)
        rir = rir[:keep]
    c = torch.as_tensor(clean, dtype=torch.float32, device=device).view(1, 1, -1)
    h = torch.as_tensor(rir,   dtype=torch.float32, device=device).view(1, 1, -1)
    L_h = h.shape[-1]
    y = F.conv1d(c, h.flip(-1), padding=L_h - 1)   # (1,1, T_c + L_h - 1)
    y = y.squeeze().cpu().numpy()
    if out_len is not None:
        y = y[:out_len]
    return y.astype(np.float32)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    OUT = Path("rir_out"); OUT.mkdir(exist_ok=True)
    rng = np.random.RandomState(0)

    print("=" * 60)
    print("PHASE 0 — RIR sources")
    print("=" * 60)

    # ---- 1. RT60 synthetic ----
    print("\n[1] Synthetic RT60 RIR (training):")
    rt60_examples = []
    for _ in range(3):
        rir, rt60 = make_rt60_rir_random(rng=rng)
        # measure realized T60 from EDC (Schroeder)
        edc = np.cumsum(rir[::-1] ** 2)[::-1]
        edc_db = 10 * np.log10(edc / edc[0] + 1e-12)
        # find -60 dB crossing
        idx = np.where(edc_db <= -60)[0]
        t60_meas = idx[0] / FS if len(idx) else len(rir) / FS
        rt60_examples.append((rt60, t60_meas))
        print(f"    RT60 target={rt60:.3f}s  measured≈{t60_meas:.3f}s  "
              f"std/|mean|={rir[1:].std()/max(abs(rir[1:].mean()),1e-9):.0f}")

    rir_ex, rt60_ex = make_rt60_rir_random(rng=np.random.RandomState(1))
    t = np.arange(len(rir_ex)) / FS * 1000
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(t, rir_ex, lw=0.5); ax[0].set_title(f"RT60 RIR (target {rt60_ex:.2f}s) full")
    ax[1].plot(t[:int(0.05*FS)], rir_ex[:int(0.05*FS)], lw=0.7)
    ax[1].set_title("0-50ms (uniform noise from t=0, NO early reflections)")
    for a in ax: a.set_xlabel("ms"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "rt60_rir.png", dpi=120); plt.close(fig)
    print(f"    → plot: rir_out/rt60_rir.png")

    # ---- 2. shoebox ----
    print("\n[2] Shoebox RIR (testing, has early reflections):")
    try:
        shoe = make_shoebox_rirs()
        for rir, meta in shoe:
            print(f"    {meta['label']:13s}  len={meta['len']:6d}  "
                  f"({meta['len']/FS:.2f}s)  rt60_target={meta['rt60_target']}")
        rir_s, meta_s = shoe[1]  # medium room
        t = np.arange(len(rir_s)) / FS * 1000
        fig, ax = plt.subplots(1, 2, figsize=(12, 3))
        ax[0].plot(t, rir_s, lw=0.5); ax[0].set_title(f"Shoebox {meta_s['label']} full")
        ax[1].plot(t[:int(0.05*FS)], rir_s[:int(0.05*FS)], lw=0.7)
        ax[1].set_title("0-50ms (DIRECT + discrete EARLY REFLECTIONS)")
        for a in ax: a.set_xlabel("ms"); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "shoebox_rir.png", dpi=120); plt.close(fig)
        print(f"    → plot: rir_out/shoebox_rir.png")
    except Exception as e:
        print(f"    shoebox FAILED: {e}")

    # ---- 3. MIT ----
    print("\n[3] MIT measured RIR (testing):")
    import os
    mit_candidates = ["data/MIT_RIR", "data/mit_rir", "data/IR_Survey"]
    found = [d for d in mit_candidates if os.path.isdir(d)]
    if found:
        print(f"    found at {found[0]}")
    else:
        print(f"    NOT on server. {MIT_DOWNLOAD}")

    # ---- pairing demo ----
    print("\n[pairing] room_clap = clean ⊛ RIR (verified flip conv):")
    import soundfile as sf, json, csv
    split = json.load(open("data/real_claps/split.json"))
    meta_rows = list(csv.DictReader(open("data/real_claps/metadata.csv")))
    # grab one clean test clap
    r = next(rr for rr in meta_rows
             if int(rr['participant']) in split['test'] and rr['tier'] == 'clean')
    p, blk, clp = int(r['participant']), int(r['block_idx']), int(r['clap_idx'])
    fp = f"data/real_claps/participant{p:02d}/block{blk:02d}/clap{clp:02d}.wav"
    clean, sr = sf.read(fp, always_2d=True)
    clean = clean[:882, 3]   # ch3, 20ms
    room = convolve_clap_rir(clean, rir_ex, out_len=882)
    print(f"    clean fs={sr}  clean peak={np.max(np.abs(clean)):.3f}  "
          f"room peak={np.max(np.abs(room)):.3f}  shapes {clean.shape}/{room.shape}")
    print(f"    train participants: {split['train']}  test: {split['test']}")
    print("\nPHASE 0 done — review rir_out/ plots before model build.")
