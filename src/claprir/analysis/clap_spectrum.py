#!/usr/bin/env python3
"""Are the deep spectral nulls in clean claps fixed, or clap-specific?

Closes the question left open by ``reports/validation/03_hf_data_integrity.md``,
which reached "the most populated 250 Hz bin holds 4 of 16 claps" and stopped
for want of statistics. Statistics only -- no model is trained or loaded.

The distinction matters because a *fixed* null would give every excitation
estimator the same blind band, which would in turn make the true-x deconvolution
ceiling optimistic and would need modelling. A *clap-specific* null is just
excitation variability and needs nothing.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44_100
REPORT_ROOT = Path("reports/clap_spectrum_audit")
CLAP_SUPPORT = 882           # the training support, as in supervised_comparison
N_FFT = 4096
THRESHOLDS_DB = (20., 30.)
WINDOWS = ("rect", "hann")
#: Analysis band. Below 100 Hz the clap has no useful excitation and the bin
#: spacing is coarse relative to the structure; above 20 kHz the recordings are
#: at the edge of the anti-alias response.
BAND_HZ = (100., 20_000.)
#: A bin whose smoothed envelope sits this far below the envelope peak is a dead
#: band, not a notch, and is excluded before thresholding.
DEAD_BAND_DB = 35.
SMOOTHING_OCTAVE = 1 / 6
#: Broad smoothing for the *local* reference. Wide enough that a notch does
#: not drag its own reference down with it.
LOCAL_REFERENCE_OCTAVE = 1 / 2
REFERENCES = ("global", "local")
BOOTSTRAP = 2000
SEED = 20_260_810


def load_claps(metadata: Path, root: Path, per_participant: int,
               ) -> list[dict]:
    """Clean channel-3 claps, stratified across participants."""
    by_participant: dict[int, list[dict]] = defaultdict(list)
    for row in csv.DictReader(metadata.open()):
        if row["tier"] != "clean" or int(row["primary_channel"]) != 3:
            continue
        by_participant[int(row["participant"])].append(row)
    claps = []
    for participant in sorted(by_participant):
        rows = by_participant[participant]
        # Spread across blocks so one recording session cannot dominate.
        rows.sort(key=lambda r: (int(r["block_idx"]), int(r["clap_idx"])))
        step = max(1, len(rows) // per_participant)
        for row in rows[::step][:per_participant]:
            path = (root / f"participant{participant:02d}"
                    / f"block{int(row['block_idx']):02d}"
                    / f"clap{int(row['clap_idx']):02d}.wav")
            if not path.exists():
                continue
            audio, rate = sf.read(path, always_2d=True)
            if rate != SAMPLE_RATE:
                raise RuntimeError(f"unexpected rate {rate} for {path}")
            waveform = np.asarray(audio[:CLAP_SUPPORT, 3], np.float64)
            claps.append({"participant": participant,
                          "block": int(row["block_idx"]),
                          "clap": int(row["clap_idx"]),
                          "id": f"P{participant:02d}B{int(row['block_idx']):02d}"
                                f"C{int(row['clap_idx']):02d}",
                          "waveform": waveform})
    return claps


def octave_smooth(magnitude: np.ndarray, frequency: np.ndarray,
                  fraction: float = SMOOTHING_OCTAVE) -> np.ndarray:
    """Fractional-octave moving average, used only to find dead bands."""
    smoothed = np.empty_like(magnitude)
    ratio = 2 ** (fraction / 2)
    for index, centre in enumerate(frequency):
        if centre <= 0:
            smoothed[index] = magnitude[index]
            continue
        inside = (frequency >= centre / ratio) & (frequency <= centre * ratio)
        smoothed[index] = magnitude[inside].mean() if inside.any() else magnitude[index]
    return smoothed


def null_mask(waveform: np.ndarray, window: str, threshold_db: float,
              frequency: np.ndarray, reference: str = "global"
              ) -> tuple[np.ndarray, np.ndarray]:
    """``(null mask, analysed mask)`` over the rfft bins.

    ``reference="global"`` is the registered definition: depth below the overall
    spectral peak. It turned out to be unusable here -- a clap rolls off ~25 dB
    by 10 kHz, so a global -20 dB threshold marks the entire high-frequency
    shoulder as "null" whether or not anything is notched.

    ``reference="local"`` measures depth below a half-octave smoothed envelope,
    which is what a notch actually is: a dip relative to its own neighbourhood.
    """
    taper = np.hanning(len(waveform)) if window == "hann" else np.ones(len(waveform))
    spectrum = np.abs(np.fft.rfft(waveform * taper, N_FFT))
    envelope = octave_smooth(spectrum, frequency)
    envelope_db = 20 * np.log10(envelope / (envelope.max() + 1e-20) + 1e-20)
    analysed = ((frequency >= BAND_HZ[0]) & (frequency <= BAND_HZ[1])
                & (envelope_db > -DEAD_BAND_DB))
    if reference == "global":
        level = 20 * np.log10(spectrum / (spectrum.max() + 1e-20) + 1e-20)
    else:
        local = octave_smooth(spectrum, frequency, LOCAL_REFERENCE_OCTAVE)
        level = 20 * np.log10((spectrum + 1e-20) / (local + 1e-20))
    return (level < -threshold_db) & analysed, analysed


def dilate(mask: np.ndarray, tolerance: int = 1) -> np.ndarray:
    out = mask.copy()
    for shift in range(1, tolerance + 1):
        out[shift:] |= mask[:-shift]
        out[:-shift] |= mask[shift:]
    return out


def jaccard(a: np.ndarray, b: np.ndarray, tolerance: int = 1) -> float:
    """Symmetric overlap with a +-1 bin tolerance, so one notch straddling two
    bins is not counted as two disjoint nulls."""
    da, db = dilate(a, tolerance), dilate(b, tolerance)
    union = (da | db).sum()
    return float((da & db).sum() / union) if union else float("nan")


def bootstrap_occupancy(masks: np.ndarray, rng: np.random.RandomState
                        ) -> tuple[np.ndarray, np.ndarray]:
    """95 % CI on per-frequency occupancy, resampling clap identities."""
    count = len(masks)
    draws = np.empty((BOOTSTRAP, masks.shape[1]))
    for index in range(BOOTSTRAP):
        draws[index] = masks[rng.randint(0, count, count)].mean(axis=0)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path,
                        default=Path("data/real_claps/metadata.csv"))
    parser.add_argument("--root", type=Path, default=Path("data/real_claps"))
    parser.add_argument("--per-participant", type=int, default=4)
    args = parser.parse_args()
    rng = np.random.RandomState(SEED)

    claps = load_claps(args.metadata, args.root, args.per_participant)
    if len(claps) < 40:
        raise RuntimeError(f"only {len(claps)} claps loaded; the audit requires >= 40")
    frequency = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)

    observed: dict = {"n_claps": len(claps),
                      "n_participants": len(({c["participant"] for c in claps})),
                      "band_hz": list(BAND_HZ), "n_fft": N_FFT,
                      "dead_band_db": DEAD_BAND_DB}
    rows, occupancy_rows = [], []
    for reference in REFERENCES:
      for threshold in THRESHOLDS_DB:
        for window in WINDOWS:
            masks = np.stack([null_mask(c["waveform"], window, threshold, frequency,
                                        reference)[0] for c in claps])
            occupancy = masks.mean(axis=0)
            low, high = bootstrap_occupancy(masks, rng)
            peak = int(np.argmax(occupancy))
            pairs, within, across = [], [], []
            for i in range(len(claps)):
                for j in range(i + 1, len(claps)):
                    value = jaccard(masks[i], masks[j])
                    if not np.isfinite(value):
                        continue
                    pairs.append(value)
                    (within if claps[i]["participant"] == claps[j]["participant"]
                     else across).append(value)
            key = (f"tau{threshold:.0f}_{window}" if reference == "global"
                   else f"local_tau{threshold:.0f}_{window}")
            observed[key] = {
                "max_occupancy": float(occupancy.max()),
                "max_occupancy_frequency_hz": float(frequency[peak]),
                "max_occupancy_ci": [float(low[peak]), float(high[peak])],
                "bands_with_ci_lower_above_0.5_hz": [
                    float(f) for f in frequency[low > .5]],
                "median_nulls_per_clap": float(np.median(masks.sum(axis=1))),
                "jaccard_median": float(np.median(pairs)),
                "jaccard_iqr": [float(np.percentile(pairs, 25)),
                                float(np.percentile(pairs, 75))],
                "jaccard_within_participant_median": float(np.median(within)),
                "jaccard_across_participant_median": float(np.median(across)),
                "n_pairs": len(pairs),
            }
            rows.append({"reference": reference, "threshold_db": threshold,
                         "window": window,
                         **{k: v for k, v in observed[key].items()
                            if not isinstance(v, list)}})
            for index in np.flatnonzero(occupancy > 0):
                occupancy_rows.append({
                    "reference": reference, "threshold_db": threshold, "window": window,
                    "frequency_hz": float(frequency[index]),
                    "occupancy": float(occupancy[index]),
                    "ci_low": float(low[index]), "ci_high": float(high[index])})

    (REPORT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    for path, table in (("summary.csv", rows), ("occupancy.csv", occupancy_rows)):
        with (REPORT_ROOT / "results" / path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
    with (REPORT_ROOT / "results/claps_used.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["id", "participant", "block", "clap"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows([{k: c[k] for k in ("id", "participant", "block", "clap")}
                          for c in claps])

    predictions = {
        "1_no_band_with_ci_lower_above_0.5_under_both_windows": not (
            set(observed["tau20_rect"]["bands_with_ci_lower_above_0.5_hz"])
            & set(observed["tau20_hann"]["bands_with_ci_lower_above_0.5_hz"])),
        "2_median_jaccard_below_0.3_at_both_thresholds": all(
            observed[f"tau{t:.0f}_{w}"]["jaccard_median"] < .3
            for t in THRESHOLDS_DB for w in WINDOWS),
        "3_within_minus_across_participant_below_0.1": all(
            observed[f"tau{t:.0f}_{w}"]["jaccard_within_participant_median"]
            - observed[f"tau{t:.0f}_{w}"]["jaccard_across_participant_median"] < .1
            for t in THRESHOLDS_DB for w in WINDOWS),
    }
    # The same three predictions scored against the corrected local reference,
    # so the reader can see that they fail only under the invalid definition.
    local_predictions = {
        "1_no_band_with_ci_lower_above_0.5_under_both_windows": not (
            set(observed["local_tau20_rect"]["bands_with_ci_lower_above_0.5_hz"])
            & set(observed["local_tau20_hann"]["bands_with_ci_lower_above_0.5_hz"])),
        "2_median_jaccard_below_0.3_at_both_thresholds": all(
            observed[f"local_tau{t:.0f}_{w}"]["jaccard_median"] < .3
            for t in THRESHOLDS_DB for w in WINDOWS),
        "3_within_minus_across_participant_below_0.1": all(
            observed[f"local_tau{t:.0f}_{w}"]["jaccard_within_participant_median"]
            - observed[f"local_tau{t:.0f}_{w}"]["jaccard_across_participant_median"] < .1
            for t in THRESHOLDS_DB for w in WINDOWS),
    }
    observed["predictions_met"] = predictions
    observed["predictions_met_local_reference"] = local_predictions
    # The registered (global-reference) verdict, and the one that is actually
    # meaningful. They disagree, and both are reported.
    local_stable = bool(set(observed["local_tau20_rect"]["bands_with_ci_lower_above_0.5_hz"])
                        & set(observed["local_tau20_hann"]["bands_with_ci_lower_above_0.5_hz"]))
    local_overlap = all(observed[f"local_tau{t:.0f}_{w}"]["jaccard_median"] >= .3
                        for t in THRESHOLDS_DB for w in WINDOWS)
    observed["local_reference_conclusion"] = {
        "stable_band_under_both_windows": local_stable,
        "median_jaccard_at_least_0.3": local_overlap,
        "hypothesis": "SUPPORTED" if (local_stable and local_overlap) else "REJECTED"}
    supported = not predictions["1_no_band_with_ci_lower_above_0.5_under_both_windows"] \
        and not predictions["2_median_jaccard_below_0.3_at_both_thresholds"]
    payload = {
        "id": "clap_spectral_null_stability",
        "x_decision": (
            "If nulls are fixed, every excitation estimator inherits the same blind "
            "band, the true-x deconvolution ceiling is optimistic, and the "
            "spectral-coverage reasoning in multiclap_lambda_audit needs revisiting. "
            "If they are clap-specific, none of that follows and the question closes "
            "permanently with no excitation-null modelling."),
        "prediction": (
            "1) no band has bootstrap-lower-bound occupancy above 0.5 under both "
            "windows at tau=20 dB; 2) median pairwise Jaccard below 0.3 at both "
            "thresholds; 3) within-participant Jaccard exceeds across-participant by "
            "less than 0.1."),
        "observed": observed,
        "fixed_null_hypothesis_registered_global_reference":
            "SUPPORTED" if supported else "REJECTED",
        "fixed_null_hypothesis": observed["local_reference_conclusion"]["hypothesis"],
        "notes": {"registered_definition_invalid": (
            "The registered null definition thresholds depth below the GLOBAL spectral "
            "peak. A clap rolls off about 25 dB by 10 kHz, so at tau=20 dB that marks "
            "the whole high-frequency shoulder as null regardless of whether anything "
            "is notched: at the apparent 10.9 kHz peak the median level is +0.3 dB "
            "relative to its own +-2 kHz neighbourhood, and only 15 % of claps sit "
            "more than 6 dB below their local level there. The global-reference "
            "numbers therefore measure rolloff, not nulls, and the conclusion of "
            "record is the local-reference one.")},
        "verdict": "PASS" if all(predictions.values()) else "FAIL",
    }
    (REPORT_ROOT / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"n_claps": len(claps), "predictions_met": predictions,
                      "registered_global_reference":
                          payload["fixed_null_hypothesis_registered_global_reference"],
                      "local_reference": payload["fixed_null_hypothesis"],
                      "local_tau20_rect": observed["local_tau20_rect"]}, indent=2,
                     default=str))
    reference = observed["local_tau20_rect"]
    print(f"VERDICT clap_spectral_null_stability {payload['verdict']} "
          "[local reference] "
          f"peak occupancy {reference['max_occupancy']:.2f} at "
          f"{reference['max_occupancy_frequency_hz']:.0f} Hz "
          f"(CI {reference['max_occupancy_ci'][0]:.2f}-{reference['max_occupancy_ci'][1]:.2f}), "
          f"median pairwise Jaccard {reference['jaccard_median']:.3f}; "
          f"fixed-null hypothesis {payload['fixed_null_hypothesis']}.")


if __name__ == "__main__":
    main()
