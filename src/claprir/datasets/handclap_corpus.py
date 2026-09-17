#!/usr/bin/env python3
"""Real repeated claps with a matched sweep-measured reference RIR (Spheres).

The meeting asked for "one or two rooms, actually record about ten claps, see
whether the joint/average output at least produces a plausible RIR", and said a
matching reference RIR would be better still. The Spheres session in
``data/spheres`` has both, and they are matched at the *source position*:

* chapters ``Session 1 - Sweeps - Mic Position - <P>`` gave the reference RIRs
  in ``RIRs/session_1/npy/source_<P>.npy``, shape ``[23 mics, 32768]``;
* chapters ``Session 1 - Claps - Mic Position - <P>`` are human claps performed
  at the same position <P>, recorded by the same 23 microphones.

So for a given (position, mic) the claps and the reference RIR share a source
location, a receiver and a room. That is the ``h_i = h + delta h_i`` case the
meeting warned about -- unlike the synthetic shards, the K claps here really are
different excitations at slightly different hand positions.

Two mismatches are recorded rather than papered over:

* the claps are hand claps at roughly the position where the sweep loudspeaker
  stood, not at the identical point, so the reference is a same-position
  reference, not a same-transducer one;
* the reference comes from an ESS measurement whose linear part is taken at the
  *latest* peak above -6 dB, ten samples of lead-in included
  (``spheres_RIRs.py``); the claps are aligned by this repository's own
  ``delay_trim`` rule instead. Both are re-aligned here by the same rule so the
  comparison does not inherit two different onset conventions.

Alignment and normalisation deliberately mirror
``multiroom_generalization.manifest``, because that is the distribution the
model was trained on: 44.1 kHz, onset at the first sample reaching 50 % of the
peak, 250 ms, peak-normalised.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf

SAMPLE_RATE = 44_100
SOURCE_RATE = 48_000
HORIZON_MS = 250
LENGTH = round(SAMPLE_RATE * HORIZON_MS / 1000)
#: Minimum claps a position must yield to be usable at K=5.
MIN_CLAPS = 5
DEFAULT_ROOT = Path("data/spheres/TheSpheresDataset-ClapsAndSweeps")
CACHE = Path("data/real_clap_multiclap/spheres_session1.npz")
MIC_REGEX = r"_T\s*\d+\s*(.*)$"


def chapters(audio: Path) -> list[dict]:
    result = subprocess.run(
        ["ffprobe", "-show_chapters", "-print_format", "json", "-v", "error", str(audio)],
        capture_output=True, text=True, check=True)
    return json.loads(result.stdout).get("chapters", [])


def mic_files(root: Path) -> list[Path]:
    """Sorted exactly as ``spheres_RIRs.py`` sorted them -- this IS the mic order.

    The reference arrays carry no channel names, only a first axis of length 23.
    Their order is whatever ``sorted(glob('*.wav'))`` produced when they were
    written, so reproducing that sort is the only way to know which row belongs
    to which microphone. Getting it wrong would silently compare a clap at one
    microphone against a reference at another.
    """
    files = sorted((root / "Claps_Sweeps_Solos_Multitrack").glob("*.wav"))
    if len(files) != 23:
        raise RuntimeError(f"expected 23 multitrack files, found {len(files)}")
    return files


def mic_name(path: Path) -> str:
    match = re.search(MIC_REGEX, path.stem)
    return match.group(1).strip().replace(" ", "_") if match else path.stem


#: The clap chapters and the sweep-derived RIR files name the same physical
#: position differently -- the clap chapters use the section plural and carry
#: take numbers, the RIR files use the singular stem. Left unmapped, 75 of 129
#: (position, mic) sets are discarded for "no matching sweep reference" purely
#: on spelling. Multiple takes at one position map to the same reference,
#: which is correct: the sweep was measured once per position.
POSITION_ALIASES = {
    "Vln1": "Vln_1", "Vln2": "Vln_2",
    "Flutes": "Flute", "Oboes": "Oboe", "Clarinets": "Clarinet",
    "Bassoons": "Bassoon", "Bass_Drum_and_Cymbals": "Bass_Drum",
}
#: Positions with claps but no sweep at the same position: the claps were
#: performed AT a microphone position, where no loudspeaker ever stood. They
#: cannot be scored against a matched reference and are reported as skipped.
UNMATCHED_POSITIONS = ("Main_C", "Main_R", "Main_L", "Main_Pair_L", "Main_Pair_R")


def canonical_position(title_tail: str) -> tuple[str, str]:
    """``"Vcl take 2"`` -> ``("Vcl", "take_2")``; applies the alias table."""
    cleaned = title_tail.strip().replace(" ", "_")
    take = ""
    match = re.search(r"_?take_?(\d+)$", cleaned, re.IGNORECASE)
    if match:
        take = f"take_{match.group(1)}"
        cleaned = cleaned[:match.start()]
    return POSITION_ALIASES.get(cleaned, cleaned), take


def resample(signal: np.ndarray) -> np.ndarray:
    return sps.resample_poly(signal, SAMPLE_RATE, SOURCE_RATE).astype(np.float32)


def align_and_crop(signal: np.ndarray, fraction: float = .5) -> np.ndarray:
    """``manifest.fit_rir``'s convention: onset at 50 % of peak, 250 ms, peak 1."""
    peak = np.max(np.abs(signal))
    if peak <= 0:
        return np.zeros(LENGTH, np.float32)
    onset = int(np.argmax(np.abs(signal) >= fraction * peak))
    out = np.zeros(LENGTH, np.float32)
    trimmed = signal[onset:onset + LENGTH]
    out[:len(trimmed)] = trimmed
    return (out / (np.max(np.abs(out)) + 1e-8)).astype(np.float32)


#: Minimum spacing between accepted claps. It is the 250 ms model horizon plus
#: a 50 ms margin: at exactly 250 ms the next clap's direct sound lands on the
#: last sample of the previous clap's window, and a window holding two claps is
#: not an observation of one clap. Detected at 250 ms spacing, one such pair
#: showed a waveform correlation of 0.96 against its neighbour, against a
#: median of 0.30 for genuinely distinct claps.
MIN_SPACING_S = .30
#: A clap is only an observation of ``x_k * h`` if the previous clap's tail has
#: died away first. This session's reference RIRs measure T30 = 0.51 s, so a
#: clap arriving less than half a second after its predecessor sits on a tail
#: that is under 30 dB down, and the window is a mixture of two excitations.
#: Most of the session claps at 0.5-0.7 s spacing and passes; a handful of
#: takes clap at 0.31 s and are dropped, which is why the retained count is
#: reported per (mic, position) rather than assumed.
MIN_PRE_GAP_S = .50


def clap_onsets(segment: np.ndarray, rate: int, max_claps: int = 12) -> list[int]:
    """Peak-pick clap events: >= 6 % of the segment peak, >= MIN_SPACING_S apart."""
    envelope = np.abs(segment)
    width = max(1, int(.001 * rate))
    smoothed = np.convolve(envelope, np.ones(width) / width, "same")
    peaks, _ = sps.find_peaks(smoothed, height=.06 * smoothed.max(),
                              distance=int(MIN_SPACING_S * rate))
    order = np.argsort(smoothed[peaks])[::-1][:max_claps]
    return sorted(int(peaks[i]) for i in order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=CACHE)
    parser.add_argument("--mics", nargs="+", default=["Main_L", "Main_R", "Main_C"],
                        help="microphone names to extract; default is the main triple")
    parser.add_argument("--max-claps", type=int, default=12)
    args = parser.parse_args()

    files = mic_files(args.root)
    names = [mic_name(p) for p in files]
    index = {name: i for i, name in enumerate(names)}
    missing = [m for m in args.mics if m not in index]
    if missing:
        raise SystemExit(f"unknown mics {missing}; available: {names}")

    rir_dir = args.root / "RIRs/session_1/npy"
    rows, observations, references = [], [], []
    for mic in args.mics:
        audio = files[index[mic]]
        clap_chapters = [c for c in chapters(audio)
                         if "session 1 - claps" in
                         (c.get("tags", {}).get("title", "") or "").lower()]
        # Takes at one position are pooled: the sweep was measured once per
        # position, so every take shares the same reference and they are simply
        # more repetitions of the same measurement.
        by_position: dict[str, list[np.ndarray]] = {}
        detail: dict[str, list[dict]] = {}
        for chapter in clap_chapters:
            title = chapter["tags"]["title"]
            position, take = canonical_position(title.split("-")[-1])
            reference_path = rir_dir / f"source_{position}.npy"
            if not reference_path.exists():
                rows.append({"mic": mic, "position": position, "take": take,
                             "chapter": title, "n_claps": 0,
                             "skipped": "claps at a microphone position, no sweep there"
                             if position in UNMATCHED_POSITIONS
                             else "no matching sweep reference"})
                continue
            segment, rate = sf.read(audio, start=int(chapter["start"]),
                                    stop=int(chapter["end"]), dtype="float32")
            onsets = clap_onsets(segment, rate)
            gaps = [np.inf] + [(b - a) / rate for a, b in zip(onsets, onsets[1:])]
            accepted = [o for o, gap in zip(onsets, gaps) if gap >= MIN_PRE_GAP_S]
            windows = []
            for onset in accepted:
                start = max(0, onset - int(.005 * rate))
                window = segment[start:start + int(.4 * rate)]
                if len(window) < int(.3 * rate):
                    continue
                windows.append(align_and_crop(resample(window)))
            by_position.setdefault(position, []).extend(windows)
            detail.setdefault(position, []).append(
                {"take": take, "chapter": title, "detected": len(onsets),
                 "kept_after_pre_gap": len(windows)})

        for position, windows in by_position.items():
            takes = detail[position]
            if len(windows) < MIN_CLAPS:
                rows.append({"mic": mic, "position": position, "takes": takes,
                             "n_claps": len(windows),
                             "skipped": f"fewer than {MIN_CLAPS} claps survive "
                                        f"the {MIN_PRE_GAP_S:.2f}s pre-gap rule"})
                continue
            padded = np.zeros((args.max_claps, LENGTH), np.float32)
            kept = np.stack(windows)[:args.max_claps]
            padded[:len(kept)] = kept
            observations.append(padded)
            references.append(align_and_crop(
                resample(np.load(rir_dir / f"source_{position}.npy")[index[mic]])))
            rows.append({"mic": mic, "position": position, "takes": takes,
                         "n_claps": len(kept), "skipped": ""})
            print(f"{mic:<10} {position:<20} {len(kept):>2} claps "
                  f"({len(takes)} take(s), {sum(t['detected'] for t in takes)} detected)",
                  flush=True)

    if not observations:
        raise SystemExit("no (position, mic) pair yielded a usable clap set")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = [r for r in rows if not r["skipped"]]
    np.savez_compressed(
        args.out,
        observation=np.stack(observations), rir=np.stack(references),
        mic=np.array([r["mic"] for r in kept]),
        position=np.array([r["position"] for r in kept]),
        n_claps=np.array([r["n_claps"] for r in kept]),
    )
    manifest = args.out.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({
        "source": str(args.root), "sample_rate": SAMPLE_RATE, "horizon_ms": HORIZON_MS,
        "mic_order": names, "requested_mics": args.mics, "rows": rows,
        "alignment": "onset at first sample >= 50% of peak, 250 ms, peak-normalised",
        "min_clap_spacing_s": MIN_SPACING_S, "min_pre_gap_s": MIN_PRE_GAP_S,
        "kept": len(kept), "skipped": len(rows) - len(kept),
    }, indent=2) + "\n")
    print(f"wrote {args.out} with {len(kept)} (position, mic) sets; "
          f"{len(rows) - len(kept)} skipped -> {manifest}")


if __name__ == "__main__":
    main()
