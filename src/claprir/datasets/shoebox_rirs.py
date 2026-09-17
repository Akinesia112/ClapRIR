#!/usr/bin/env python3
"""
supervised_baseline.py — BUDDy-style conditional diffusion baseline.

Estimates the CLEAN CLAP from a ROOM-CLAP (clean ⊛ RIR) via conditional
flow-matching. The reverberant observation is concatenated as a 2nd input
channel (BUDDy conditional-baseline structure), NOT cross-attention.

I/O (all at 4096 samples, 44.1 kHz):
  condition  = room_clap, delay-trimmed, first 4096 (93 ms)
  target     = clean clap, 882 real samples zero-padded to 4096
  model in   = [noisy_target(4096), room_clap(4096)]  (2 ch)
  model out  = velocity for the target (1 ch)

Key design (from window-plot analysis):
  - TRAINING RIRs MUST include early reflections — RT60-synthetic alone (no
    early reflections) would be OOD on shoebox/MIT. Train mixes RT60 + randomized
    shoebox rooms; test uses DISJOINT shoebox rooms + MIT.
  - two-fold split: participant-level (claps) AND room-level (RIRs).

Backbone: reuses model.py's DilatedResBlock/time-embed; num_blocks=11 so
RF=4117 ≥ 4096 (the §1 receptive-field rule). (a)-weight partial init: blocks
0-9 + out_proj + time-embed transfer from checkpoints_real_a; block 10 fresh;
in_proj ch0 copied from (a), ch1 (room_clap) fresh.

flow-matching (NOT EDM), NO CFG — one change at a time (EDM/CFG are separate
later experiments).
"""

import os, json, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import soundfile as sf
from claprir.models.spectrogram_network import DilatedResBlock, SinusoidalTimeEmbedding, TimeMLP, receptive_field
from claprir.rir.signals import make_rt60_rir_random, convolve_clap_rir, FS

LENGTH = 4096          # conditional window (93 ms)
CLAP_LEN = 882         # real clap support (20 ms); rest zero-padded
META = "data/real_claps/metadata.csv"
SPLIT = json.load(open("data/real_claps/split.json"))


# ── conditional backbone (2ch in, 1ch out, num_blocks=11) ─────────────────────

class ConditionalWaveNet(nn.Module):
    """2-channel input [noisy_target, room_clap] → 1-channel velocity."""
    def __init__(self, channels=64, num_blocks=11, t_embed_dim=128, t_hidden_dim=256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(t_embed_dim)
        self.time_mlp = TimeMLP(t_embed_dim, t_hidden_dim)
        self.in_proj  = nn.Conv1d(2, channels, kernel_size=1)        # 2ch in
        self.dilations = [2 ** i for i in range(num_blocks)]
        self.receptive_field = receptive_field(self.dilations)
        self.blocks = nn.ModuleList(
            [DilatedResBlock(channels, t_hidden_dim, dilation=d) for d in self.dilations])
        self.out_norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.out_proj = nn.Conv1d(channels, 1, kernel_size=1)        # 1ch out
        print(f"ConditionalWaveNet: channels={channels}, num_blocks={num_blocks}, "
              f"RF={self.receptive_field}")

    def forward(self, x_noisy, cond, t):
        # x_noisy: (B,1,L) noisy target; cond: (B,1,L) room_clap
        h = self.in_proj(torch.cat([x_noisy, cond], dim=1))
        t_ctx = self.time_mlp(self.time_emb(t))
        for blk in self.blocks:
            h = blk(h, t_ctx)
        return self.out_proj(F.silu(self.out_norm(h)))

    def init_from_prior(self, ckpt_path):
        """Partial init from prior (a): blocks 0-9 + out_proj + time-embed; in_proj ch0."""
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['velocity_model_state']
        own = self.state_dict()
        copied, skipped = [], []
        for k, v in sd.items():
            if k == 'in_proj.weight':
                # (a) (C,1,1) → copy into our (C,2,1) channel 0; channel 1 fresh
                own['in_proj.weight'][:, 0:1, :] = v
                copied.append(k + ' (→ch0)')
            elif k in own and own[k].shape == v.shape:
                own[k] = v
                copied.append(k)
            else:
                skipped.append(k)
        self.load_state_dict(own)
        n_blk_params = sum(1 for c in copied if c.startswith('blocks.'))
        print(f"  init_from_prior: copied {len(copied)} tensors "
              f"({n_blk_params} block params), skipped {len(skipped)} "
              f"(block 10 + in_proj ch1 stay fresh)")


# ── compressed-STFT magnitude loss (BUDDy l2_comp_stft, c=0.667) ──────────────

def comp_stft_loss(est, target, c=0.667, n_fft=510, hop=128):
    """L2 on compressed STFT magnitude: || |STFT|^c (est) - |STFT|^c (target) ||."""
    est, target = est.squeeze(1), target.squeeze(1)
    win = torch.hann_window(n_fft, device=est.device)
    E = torch.stft(est,    n_fft, hop, window=win, return_complex=True)
    T = torch.stft(target, n_fft, hop, window=win, return_complex=True)
    Em = (E.abs() + 1e-8) ** c
    Tm = (T.abs() + 1e-8) ** c
    return F.mse_loss(Em, Tm)


# ── flow-matching wrapper ─────────────────────────────────────────────────────

class CondFlow(nn.Module):
    def __init__(self, net): super().__init__(); self.net = net

    def fm_loss(self, target, cond, stft_weight=1.0):
        B = target.shape[0]
        t = torch.rand(B, device=target.device)
        eps = torch.randn_like(target)
        x = t[:, None, None] * target + (1 - t[:, None, None]) * eps
        v = self.net(x, cond, t)
        fm = F.mse_loss(v, target - eps)
        # also a light compressed-STFT term on the implied x_hat0 = x + (1-t)v
        x_hat0 = x + (1 - t[:, None, None]) * v
        stft = comp_stft_loss(x_hat0, target)
        return fm + stft_weight * stft, fm.item(), stft.item()

    @torch.no_grad()
    def sample(self, cond, n_steps=50):
        B, _, L = cond.shape
        x = torch.randn(B, 1, L, device=cond.device)
        h = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((B,), k * h, device=cond.device)
            v = self.net(x, cond, t)
            x = x + h * v
        return x


# ── RIR training/test sets ────────────────────────────────────────────────────

def build_shoebox_random(n, rng, fs=FS):
    """Randomized shoebox rooms (varied size/positions/absorption) → RIRs w/ early refl."""
    import pyroomacoustics as pra
    rirs = []
    for _ in range(n):
        L = rng.uniform(3, 10); Wd = rng.uniform(3, 8); H = rng.uniform(2.4, 4)
        rt60 = rng.uniform(0.2, 0.7)
        try:
            e_abs, max_o = pra.inverse_sabine(rt60, [L, Wd, H])
            room = pra.ShoeBox([L, Wd, H], fs=fs, materials=pra.Material(e_abs),
                               max_order=min(max_o, 8))
            src = [rng.uniform(0.5, L-0.5), rng.uniform(0.5, Wd-0.5), rng.uniform(1, H-0.5)]
            mic = [rng.uniform(0.5, L-0.5), rng.uniform(0.5, Wd-0.5), rng.uniform(1, H-0.5)]
            room.add_source(src); room.add_microphone(np.array(mic).reshape(3, 1))
            room.compute_rir()
            h = room.rir[0][0].astype(np.float32)
            h = h / (np.max(np.abs(h)) + 1e-8)
            rirs.append(h)
        except Exception:
            continue
    return rirs


def delay_trim(rir, frac=0.5):
    pk = np.max(np.abs(rir))
    onset = int(np.argmax(np.abs(rir) >= frac * pk))
    return rir[onset:]


class PairedClapDataset(Dataset):
    """(room_clap[4096], clean_clap[4096 zero-padded]) pairs. RIR drawn per-item."""
    def __init__(self, participants, rir_bank, rng_seed=0, size=None):
        self.rng = np.random.RandomState(rng_seed)
        self.rir_bank = rir_bank          # list of (precomputed) RIRs, or 'rt60' marker
        # gather clean claps
        self.claps = []
        for r in csv.DictReader(open(META)):
            if int(r['participant']) in participants and r['tier'] == 'clean' \
               and int(r['primary_channel']) == 3:
                p, b, c = int(r['participant']), int(r['block_idx']), int(r['clap_idx'])
                self.claps.append(f"data/real_claps/participant{p:02d}/block{b:02d}/clap{c:02d}.wav")
        self.size = size or len(self.claps)

    def __len__(self): return self.size

    def _draw_rir(self):
        if self.rir_bank == 'rt60':
            rir, _ = make_rt60_rir_random(rng=self.rng)
            return rir
        return self.rir_bank[self.rng.randint(len(self.rir_bank))]

    def __getitem__(self, idx):
        fp = self.claps[idx % len(self.claps)]
        clean, _ = sf.read(fp, always_2d=True)
        clean = clean[:CLAP_LEN, 3].astype(np.float32)
        pk = np.max(np.abs(clean)) + 1e-8
        clean = clean / pk
        rir = delay_trim(self._draw_rir())
        room = convolve_clap_rir(clean, rir, out_len=LENGTH)
        room = room / (np.max(np.abs(room)) + 1e-8)
        # target: clean zero-padded to LENGTH
        target = np.zeros(LENGTH, dtype=np.float32); target[:CLAP_LEN] = clean
        return (torch.from_numpy(target).unsqueeze(0),
                torch.from_numpy(room.astype(np.float32)).unsqueeze(0))


def env_corr(a, b):
    def henv(x):
        x = np.asarray(x, np.float64); N = len(x)
        Xf = np.fft.fft(x); h = np.zeros(N)
        if N % 2 == 0: h[0]=h[N//2]=1; h[1:N//2]=2
        else: h[0]=1; h[1:(N+1)//2]=2
        return np.abs(np.fft.ifft(Xf*h))
    ea, eb = henv(a), henv(b)
    if ea.std() < 1e-7 or eb.std() < 1e-7: return float('nan')
    return float(np.corrcoef(ea, eb)[0, 1])


# ── STEP 0.5: single fixed room sanity ────────────────────────────────────────

def step05_single_room(device, epochs=40):
    """Train on ONE fixed shoebox room; confirm the model can recover the clap.
    If env_corr stays low here, the architecture/conditioning is wrong."""
    print("\n=== STEP 0.5: single fixed-room well-posedness sanity ===")
    rng = np.random.RandomState(7)
    fixed_rir = build_shoebox_random(1, rng)[0]
    print(f"  fixed room RIR len={len(fixed_rir)}")

    class FixedRoomDS(PairedClapDataset):
        def _draw_rir(self): return fixed_rir
    ds = FixedRoomDS(SPLIT['train'], rir_bank='fixed', rng_seed=1)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, drop_last=True)

    net = ConditionalWaveNet().to(device)
    net.init_from_prior('checkpoints_real_a/velocity_model_final.pt')
    flow = CondFlow(net).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4)

    net.train()
    for ep in range(1, epochs + 1):
        for target, cond in dl:
            target, cond = target.to(device), cond.to(device)
            loss, fm, st = flow.fm_loss(target, cond)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if ep % 10 == 0 or ep == epochs:
            print(f"  ep{ep}: loss={loss.item():.4f} (fm={fm:.4f} stft={st:.4f})")

    # eval on a few TEST claps with the SAME fixed room
    test_ds = FixedRoomDS(SPLIT['test'], rir_bank='fixed', rng_seed=99)
    net.eval()
    ecs = []
    for i in range(8):
        target, cond = test_ds[i]
        est = flow.sample(cond.unsqueeze(0).to(device))[0, 0, :CLAP_LEN].cpu().numpy()
        tru = target[0, :CLAP_LEN].numpy()
        ecs.append(env_corr(tru, est))
    ecs = [e for e in ecs if not np.isnan(e)]
    mean_ec = float(np.mean(ecs)) if ecs else float('nan')
    print(f"  STEP 0.5 fixed-room env_corr(est,true) = {mean_ec:.4f} "
          f"({len(ecs)}/8 valid)  per={[f'{e:.2f}' for e in ecs]}")
    print(f"  → {'PASS: architecture can learn the easy case' if mean_ec > 0.5 else 'FAIL: arch/conditioning wrong — stop before scaling'}")
    return mean_ec


# ── RIR banks (two-fold split: train rooms disjoint from test rooms) ──────────

def build_shoebox_bank(n, seed, fs=FS):
    """A bank of n randomized shoebox rooms. seed range fixes which rooms."""
    rng = np.random.RandomState(seed)
    return build_shoebox_random(n, rng, fs)


def load_mit_test_rirs(fs=FS, n=8):
    """The 8 MIT RIRs spanning RT60 (test only)."""
    import scipy.signal as sps
    mit_dir = Path("data/MIT_RIR/Audio")
    def rt60(x, sr):
        edc = np.cumsum(x[::-1]**2)[::-1]
        db = 10*np.log10(edc/(edc[0]+1e-20)+1e-20)
        idx = np.where(db <= -60)[0]
        return idx[0]/sr if len(idx) else len(x)/sr
    items = []
    for wav in sorted(mit_dir.glob("*.wav")):
        x, sr = sf.read(str(wav))
        if x.ndim > 1: x = x[:, 0]
        items.append((wav.stem, x, sr, rt60(x, sr)))
    items.sort(key=lambda r: r[3])
    idxs = np.linspace(0, len(items)-1, n).astype(int)
    out = []
    for i in idxs:
        name, x, sr, _ = items[i]
        if sr != fs: x = sps.resample_poly(x, fs, sr)
        out.append((x/np.max(np.abs(x))).astype(np.float32))
    return out


# ── caching: precompute fixed (room_clap, target) pairs to disk ───────────────

CACHE_DIR = Path("data/real_claps/cache_supervised")

def build_cache(K=4, n_shoebox_train=60, seed=2000):
    """
    For each training clap, draw K RIRs (50% RT60-synthetic, 50% train-shoebox),
    convolve once, cache room_clap[4096] + target[4096]. Train-shoebox rooms
    (seed 2000+) are DISJOINT from test rooms (seed 5000+).
    """
    import time
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Building cache (K={K} RIRs/clap) ===")
    t0 = time.time()
    print("  generating train shoebox bank...")
    shoe_bank = build_shoebox_bank(n_shoebox_train, seed)
    print(f"  train shoebox rooms: {len(shoe_bank)}")

    rng = np.random.RandomState(seed)
    clap_files = []
    for r in csv.DictReader(open(META)):
        if int(r['participant']) in SPLIT['train'] and r['tier'] == 'clean' \
           and int(r['primary_channel']) == 3:
            p, b, c = int(r['participant']), int(r['block_idx']), int(r['clap_idx'])
            clap_files.append(f"data/real_claps/participant{p:02d}/block{b:02d}/clap{c:02d}.wav")

    n_pairs = len(clap_files) * K
    rooms = np.zeros((n_pairs, LENGTH), dtype=np.float32)
    targets = np.zeros((n_pairs, LENGTH), dtype=np.float32)
    idx = 0
    for cf in clap_files:
        clean, _ = sf.read(cf, always_2d=True)
        clean = clean[:CLAP_LEN, 3].astype(np.float32)
        clean = clean / (np.max(np.abs(clean)) + 1e-8)
        tgt = np.zeros(LENGTH, dtype=np.float32); tgt[:CLAP_LEN] = clean
        for _ in range(K):
            # TRAINING is shoebox-only — RT60-diffuse was an unrealistic distractor
            # that HURT generalization (ablation: shoebox-only 0.950/0.897 vs
            # mixed 0.920/0.892 on held-out shoebox / MIT). RT60-diffuse is kept
            # as a TEST-only "hardest case" in evaluate(), not training data.
            rir = delay_trim(shoe_bank[rng.randint(len(shoe_bank))])
            room = convolve_clap_rir(clean, rir, out_len=LENGTH)
            room = room / (np.max(np.abs(room)) + 1e-8)
            rooms[idx] = room; targets[idx] = tgt; idx += 1
    np.save(CACHE_DIR / "rooms.npy", rooms)
    np.save(CACHE_DIR / "targets.npy", targets)
    dt = time.time() - t0
    sz = (rooms.nbytes + targets.nbytes) / 1e6
    print(f"  cached {n_pairs} pairs  ({sz:.0f} MB)  in {dt:.0f}s → {CACHE_DIR}")


class CachedPairDataset(Dataset):
    def __init__(self):
        self.rooms = np.load(CACHE_DIR / "rooms.npy", mmap_mode='r')
        self.targets = np.load(CACHE_DIR / "targets.npy", mmap_mode='r')
    def __len__(self): return len(self.rooms)
    def __getitem__(self, i):
        return (torch.from_numpy(self.targets[i].copy()).unsqueeze(0),
                torch.from_numpy(self.rooms[i].copy()).unsqueeze(0))


# ── full training ─────────────────────────────────────────────────────────────

def train_full(device, epochs=120, ckpt_dir="checkpoints_supervised"):
    import time
    os.makedirs(ckpt_dir, exist_ok=True)
    ds = CachedPairDataset()
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    print(f"\n=== FULL training: {len(ds)} cached pairs, {epochs} epochs ===")

    net = ConditionalWaveNet().to(device)
    net.init_from_prior('checkpoints_real_a/velocity_model_final.pt')
    flow = CondFlow(net).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4)

    log = open(os.path.join(ckpt_dir, "loss_log.csv"), "w", buffering=1)
    log.write("epoch,loss,fm,stft\n")
    net.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        last = (0, 0, 0)
        for target, cond in dl:
            target, cond = target.to(device), cond.to(device)
            loss, fm, st = flow.fm_loss(target, cond)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            last = (loss.item(), fm, st)
        log.write(f"{ep},{last[0]:.6f},{last[1]:.6f},{last[2]:.6f}\n")
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"  ep{ep}: loss={last[0]:.4f} (fm={last[1]:.4f} stft={last[2]:.4f})  "
                  f"{time.time()-t0:.0f}s/ep")
            torch.save({"net_state": net.state_dict(), "epoch": ep},
                       os.path.join(ckpt_dir, "model_final.pt"))
    log.close()
    print(f"  saved → {ckpt_dir}/model_final.pt")
    return flow


# ── 3-RIR-type evaluation ─────────────────────────────────────────────────────

def comp_stft_dist(a, b, c=0.667, n_fft=510, hop=128):
    a = torch.from_numpy(a).float().unsqueeze(0)
    b = torch.from_numpy(b).float().unsqueeze(0)
    win = torch.hann_window(n_fft)
    A = (torch.stft(a, n_fft, hop, window=win, return_complex=True).abs()+1e-8)**c
    B = (torch.stft(b, n_fft, hop, window=win, return_complex=True).abs()+1e-8)**c
    return float((A-B).pow(2).mean().sqrt())


def evaluate(device, ckpt_dir="checkpoints_supervised"):
    print("\n=== EVAL: held-out claps × 3 RIR types ===")
    net = ConditionalWaveNet().to(device)
    net.load_state_dict(torch.load(os.path.join(ckpt_dir, "model_final.pt"),
                                   map_location=device, weights_only=False)["net_state"])
    flow = CondFlow(net).to(device); net.eval()

    # test claps
    test_claps = []
    for r in csv.DictReader(open(META)):
        if int(r['participant']) in SPLIT['test'] and r['tier'] == 'clean' \
           and int(r['primary_channel']) == 3:
            p, b, c = int(r['participant']), int(r['block_idx']), int(r['clap_idx'])
            test_claps.append(f"data/real_claps/participant{p:02d}/block{b:02d}/clap{c:02d}.wav")
    test_claps = test_claps[:24]   # subset for speed

    # RIR types: held-out shoebox (seed 5000, disjoint from train 2000) / MIT / RT60
    shoe_test = build_shoebox_bank(8, seed=5000)
    mit_test = load_mit_test_rirs()
    rt60_test = [make_rt60_rir_random(rng=np.random.RandomState(7000+i))[0] for i in range(8)]

    rng = np.random.RandomState(123)
    results = {}
    overlay = {}
    for rir_type, bank in [("shoebox-heldout", shoe_test), ("MIT-real", mit_test),
                          ("RT60-synth", rt60_test)]:
        ecs, dists = [], []
        for j, cf in enumerate(test_claps):
            clean, _ = sf.read(cf, always_2d=True)
            clean = clean[:CLAP_LEN, 3].astype(np.float32)
            clean = clean / (np.max(np.abs(clean)) + 1e-8)
            # RT60-diffuse starts at t=0 (synthetic, no propagation delay) → no
            # delay-trim; shoebox/MIT have a real direct-path delay → trim.
            rir = bank[j % len(bank)] if rir_type == "RT60-synth" else delay_trim(bank[j % len(bank)])
            room = convolve_clap_rir(clean, rir, out_len=LENGTH)
            room = room / (np.max(np.abs(room)) + 1e-8)
            cond = torch.from_numpy(room).float().view(1, 1, -1).to(device)
            est = flow.sample(cond)[0, 0, :CLAP_LEN].cpu().numpy()
            ec = env_corr(clean, est)
            if not np.isnan(ec):
                ecs.append(ec)
                dists.append(comp_stft_dist(clean, est))
            if j < 3:
                overlay.setdefault(rir_type, []).append((clean, est, room[:CLAP_LEN]))
        results[rir_type] = (float(np.mean(ecs)), float(np.mean(dists)), len(ecs))
        print(f"  {rir_type:16s}: env_corr={np.mean(ecs):.4f}  "
              f"comp-stft-dist={np.mean(dists):.4f}  (n={len(ecs)})")

    # overlay figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
    t_ms = np.arange(CLAP_LEN) / FS * 1000
    for r, (rt, exs) in enumerate(overlay.items()):
        for c in range(3):
            clean, est, _ = exs[c]
            axes[r, c].plot(t_ms, clean, 'k', lw=1.2, alpha=0.6, label='true')
            axes[r, c].plot(t_ms, est, 'tab:orange', lw=0.9, label='est')
            axes[r, c].grid(alpha=0.3)
            if c == 0: axes[r, c].set_ylabel(rt, fontsize=9)
            if r == 0: axes[r, c].set_title(f"ex {c+1}", fontsize=9)
        axes[r, 0].legend(fontsize=7)
    for ax in axes[-1]: ax.set_xlabel("ms")
    fig.suptitle("Supervised baseline: estimated vs true clap, by test RIR type")
    fig.tight_layout(); fig.savefig("rir_out/supervised_eval.png", dpi=120); plt.close(fig)
    print("  → rir_out/supervised_eval.png")
    print("\n  TABLE: RIR type | env_corr | comp-stft-dist")
    for rt, (ec, d, n) in results.items():
        print(f"    {rt:16s}  {ec:.4f}  {d:.4f}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', default='0.5',
                    choices=['0.5', 'cache', 'train', 'eval', 'full'])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.step == '0.5':
        step05_single_room(device)
    elif args.step == 'cache':
        build_cache()
    elif args.step == 'train':
        train_full(device)
    elif args.step == 'eval':
        evaluate(device)
    elif args.step == 'full':
        if not (CACHE_DIR / "rooms.npy").exists():
            build_cache()
        train_full(device)
        evaluate(device)
