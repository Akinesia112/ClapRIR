#!/usr/bin/env python3
"""The Anechoic Clap Dataset figure: what the excitations actually look like.

Nils asked to "show some examples ... then we can also explain why unregularized
deconvolution fails by pointing to the notches". The first half is what this
figure does; the second half needs care, because our own D0 measurement shows
unregularised inversion does NOT fail in the noiseless synthetic benchmark --
it beats Tikhonov on 4 of 5 sources. What the notches support is the
conditioning statement:

    Y = X H + N   =>   H_0 = Y / X = H + N / X

so wherever |X(f)| is small the error term is amplified by 1/|X(f)|. The inset
plots exactly that gain against the Tikhonov gain |X| / (|X|^2 + lambda), which
is bounded. No claim is made about the noiseless benchmark.

Notches are marked with the LOCAL-envelope definition from clap_spectrum_audit
-- a dip relative to a half-octave smoothed reference -- not the global-peak
definition, which that audit rejected because a clap rolls off ~25 dB by 10 kHz
and a global threshold marks the whole HF shoulder.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from claprir.analysis.clap_spectrum import (BAND_HZ, N_FFT,
                                                         null_mask, octave_smooth)

SAMPLE_RATE = 44_100
CLAP_SUPPORT = 882          # the training support
THRESHOLD_DB = 20.          # the audit's tau=20 local-reference threshold
METADATA = Path("data/real_claps/metadata.csv")
SPLIT = Path("data/real_claps/split.json")


def load_clap(row: dict, base: Path) -> np.ndarray:
    participant = int(row["participant"])
    folder = base / ("_review" if participant in (3, 6) else "")
    path = (folder / f"participant{participant:02d}"
            / f"block{int(row['block_idx']):02d}"
            / f"clap{int(row['clap_idx']):02d}.wav")
    data, _ = sf.read(path, always_2d=True)
    channel = data[:, int(row["primary_channel"])]
    peak = int(np.argmax(np.abs(channel)))
    start = max(0, peak - CLAP_SUPPORT // 8)
    clip = channel[start:start + CLAP_SUPPORT]
    if len(clip) < CLAP_SUPPORT:
        clip = np.pad(clip, (0, CLAP_SUPPORT - len(clip)))
    return clip / (np.abs(clip).max() + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--out", type=Path,
                        default=Path("publication/figures/fig_clap_dataset.png"))
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(args.metadata.open()))
    train = set(json.loads(SPLIT.read_text())["train"])
    # Quality-screened training claps only: the same filter the model sees.
    pool = [r for r in rows if r["tier"] == "clean"
            and int(r["primary_channel"]) == 3
            and int(r["participant"]) in train]
    if not pool:
        raise SystemExit("no quality-screened training claps matched")

    # One clap from each of N different participants, so "representative" means
    # spread across the corpus rather than five claps by one person.
    chosen, seen = [], set()
    for row in pool:
        participant = int(row["participant"])
        if participant not in seen:
            seen.add(participant)
            chosen.append(row)
        if len(chosen) == args.n:
            break

    base = args.metadata.parent
    frequency = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    time = np.arange(CLAP_SUPPORT) / SAMPLE_RATE * 1000
    inband = (frequency >= BAND_HZ[0]) & (frequency <= BAND_HZ[1])

    # Three bands, not an inset: the first attempt put the gain plot inside a
    # spectrum panel, where it overlapped the very notches it was explaining.
    fig = plt.figure(figsize=(2.7 * len(chosen), 6.2))
    grid = fig.add_gridspec(3, len(chosen), height_ratios=[1, 1.4, 1.25],
                            hspace=.42, wspace=.12,
                            left=.055, right=.995, top=.955, bottom=.075)
    notch_counts, notch_freqs = [], []
    for column, row in enumerate(chosen):
        clip = load_clap(row, base)
        spectrum = np.abs(np.fft.rfft(clip, N_FFT))
        local = octave_smooth(spectrum, frequency, 1 / 2)
        mask, _ = null_mask(clip, "rect", THRESHOLD_DB, frequency, reference="local")
        notch_counts.append(int(mask.sum()))
        notch_freqs.append(frequency[mask])

        top = fig.add_subplot(grid[0, column])
        top.plot(time, clip, lw=.5, color="#1f77b4")
        top.set(xlim=(0, time[-1]), ylim=(-1.05, 1.05), xlabel="ms")
        top.set_title(f"participant {int(row['participant']):02d}", fontsize=9)
        top.set_ylabel("amplitude" if column == 0 else "")
        if column: top.set_yticklabels([])

        bottom = fig.add_subplot(grid[1, column])
        db = 20 * np.log10(spectrum / (spectrum.max() + 1e-20) + 1e-20)
        local_db = 20 * np.log10(local / (spectrum.max() + 1e-20) + 1e-20)
        bottom.semilogx(frequency[inband], db[inband], lw=.45, color="#1f77b4")
        bottom.semilogx(frequency[inband], local_db[inband], lw=1.1, color="#7f7f7f",
                        linestyle=(0, (4, 2)), label="half-octave envelope")
        marked = mask & inband
        bottom.plot(frequency[marked], db[marked], linestyle="none", marker="v",
                    ms=3.4, color="black", label=f"deep notch ($n$={mask.sum()})")
        bottom.set(xlim=BAND_HZ, ylim=(-70, 3), xlabel="Hz")
        bottom.set_ylabel("dB re peak" if column == 0 else "")
        if column: bottom.set_yticklabels([])
        bottom.legend(fontsize=6, loc="lower left")

        gain = fig.add_subplot(grid[2, column])
        magnitude = spectrum / (spectrum.max() + 1e-20)
        gain.loglog(frequency[inband], (1.0 / (magnitude + 1e-12))[inband],
                    lw=.5, color="#c44e52", label=r"$\lambda=0$:  $1/|X|$")
        for lam, colour, dash in ((1e-3, "#1f77b4", None),
                                  (1e-2, "black", (0, (4, 2)))):
            style = {"linestyle": dash} if dash else {}
            gain.loglog(frequency[inband],
                        (magnitude / (magnitude ** 2 + lam))[inband],
                        lw=.9, color=colour, label=rf"$\lambda={lam:g}$", **style)
        gain.set(xlim=BAND_HZ, ylim=(1e-1, 1e6), xlabel="Hz")
        gain.set_ylabel("inverse-filter gain" if column == 0 else "")
        if column: gain.set_yticklabels([])
        gain.legend(fontsize=5.5, loc="upper left", handlelength=1.4)

        for axis in (top, bottom, gain):
            axis.grid(alpha=.3, lw=.4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"wrote {args.out} and {args.out.with_suffix('.pdf')}")
    print(f"notches per shown clap (local reference, tau={THRESHOLD_DB:g} dB): "
          f"{notch_counts}")
    for row, freqs in zip(chosen, notch_freqs):
        below = int((freqs < 8000).sum())
        print(f"  P{int(row['participant']):02d}: {len(freqs)} notches, "
              f"{below} below 8 kHz, {len(freqs)-below} above")


if __name__ == "__main__":
    main()
