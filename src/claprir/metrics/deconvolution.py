"""Shared waveform, spectrum, decay, band, and forward-consistency metrics."""
from __future__ import annotations

import numpy as np
import scipy.signal as sps
import torch

from claprir.models.excitation_estimator import metrics as core_metrics


def waveform_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, np.float32)
    estimate = np.asarray(estimate, np.float32)
    result = core_metrics(torch.from_numpy(reference)[None, None],
                          torch.from_numpy(estimate)[None, None])
    er = np.abs(sps.hilbert(reference))
    ee = np.abs(sps.hilbert(estimate))
    result["env_corr"] = float(np.corrcoef(er, ee)[0, 1])
    result["post_target_rms_db"] = float(20 * np.log10(
        np.sqrt(np.mean(estimate[882:] ** 2)) + 1e-12))
    return result


def schroeder_edc(signal: np.ndarray) -> np.ndarray:
    """Backward-integrated energy decay curve, normalised to 0 dB at t=0."""
    curve = np.cumsum(np.asarray(signal, np.float64)[::-1] ** 2)[::-1]
    return 10 * np.log10(np.maximum(curve / (curve[0] + 1e-20), 1e-8))


def edc_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((schroeder_edc(reference)
                                  - schroeder_edc(estimate)) ** 2)))


def band_errors(reference: np.ndarray, estimate: np.ndarray,
                sample_rate: int = 44100) -> dict[str, float]:
    length = len(reference)
    frequency = np.fft.rfftfreq(length, 1 / sample_rate)
    r = np.abs(np.fft.rfft(reference))
    e = np.abs(np.fft.rfft(estimate))
    rows = {}
    for name, low, high in (("lf", 20, 500), ("mf", 500, 4000),
                            ("hf", 4000, sample_rate / 2)):
        use = (frequency >= low) & (frequency < high)
        delta = 20 * np.log10(e[use] + 1e-7) - 20 * np.log10(r[use] + 1e-7)
        rows[f"{name}_signed_db"] = float(np.mean(delta))
        rows[f"{name}_abs_db"] = float(np.mean(np.abs(delta)))
    return rows


def _decay_curve(x: np.ndarray) -> np.ndarray:
    curve = np.cumsum(np.asarray(x, np.float64)[::-1] ** 2)[::-1]
    return 10 * np.log10(np.maximum(curve / (curve[0] + 1e-20), 1e-12))


def t30_seconds(x: np.ndarray, sample_rate: int = 44100) -> float:
    """Reverberation time from the -5 dB to -35 dB slope of the decay curve.

    Returns NaN when the curve never reaches -35 dB inside the horizon, which is
    common for a 250 ms window; callers must treat NaN as "not measurable here"
    rather than dropping it silently.
    """
    curve = _decay_curve(x)
    try:
        start = int(np.flatnonzero(curve <= -5)[0])
        stop = int(np.flatnonzero(curve <= -35)[0])
    except IndexError:
        return float("nan")
    if stop <= start:
        return float("nan")
    time = np.arange(start, stop) / sample_rate
    slope = np.polyfit(time, curve[start:stop], 1)[0]
    return float(-60 / slope) if slope < 0 else float("nan")


#: Fit intervals tried in order, as (upper dB, lower dB, decibels spanned).
DECAY_INTERVALS = ((-5., -35., 30.), (-5., -25., 20.), (-5., -15., 10.))


def decay_estimate(x: np.ndarray, sample_rate: int = 44100) -> dict[str, float]:
    """Reverberation time with the diagnostics needed to know if it is usable.

    ``t30_seconds`` returns NaN whenever the curve does not reach -35 dB, which
    conflates two very different cases: a decay too long to be seen inside the
    horizon, and a curve too noisy to fit. Bucketing on that NaN is worse than
    useless -- a censored RIR is evidence of a *long* decay, so mapping it to the
    short bucket biases the stratification backwards.

    This falls back to progressively shallower fit intervals (T30, then T20,
    then T10) and reports which one was used, how well it fit, and whether the
    result is censored by the horizon. A caller that wants only uncensored,
    well-fit values can filter on ``censored`` and ``r_squared``.
    """
    curve = _decay_curve(x)
    horizon = len(curve) / sample_rate
    # Backward integration drives the curve to -inf at the last sample, so the
    # raw curve ALWAYS crosses -35 dB and censoring would never fire. Measured:
    # a true 3.0 s decay "reaches" -35 dB inside a 250 ms window and fits to
    # 0.48 s purely from that terminal plunge. The last tenth is therefore not
    # eligible for a crossing, which is what makes `censored` mean anything.
    usable = curve[:max(8, int(.9 * len(curve)))]
    out = {"reverberation_time_s": float("nan"), "r_squared": float("nan"),
           "fit_upper_db": float("nan"), "fit_lower_db": float("nan"),
           "decibels_spanned": float("nan"), "censored": 1., "extrapolated": 1.,
           "horizon_s": float(horizon)}
    for upper, lower, span in DECAY_INTERVALS:
        head = np.flatnonzero(usable <= upper)
        tail = np.flatnonzero(usable <= lower)
        if not head.size or not tail.size:
            continue
        start, stop = int(head[0]), int(tail[0])
        if stop - start < 8:
            continue
        time = np.arange(start, stop) / sample_rate
        segment = curve[start:stop]
        slope, intercept = np.polyfit(time, segment, 1)
        if slope >= 0:
            continue
        residual = segment - (slope * time + intercept)
        variance = float(np.var(segment))
        out.update({
            "reverberation_time_s": float(-60 / slope),
            "r_squared": float(1 - np.var(residual) / variance) if variance > 0
                         else float("nan"),
            "fit_upper_db": upper, "fit_lower_db": lower,
            "decibels_spanned": span,
            # Censored means the horizon, not the fit, limited what could be
            # seen: the curve never reached the deepest interval.
            "censored": 0.,
            # Extrapolated means the 60 dB figure was projected from a shallower
            # span than 30 dB, so it rests on the decay staying linear.
            "extrapolated": float(span < DECAY_INTERVALS[0][2])})
        return out
    # Nothing fit inside the usable region: the decay outlasts the horizon.
    # That is a lower bound on the reverberation time, not a missing value, and
    # emphatically not evidence of a short decay.
    floor = float(usable[-1])
    if floor < -1:
        out["reverberation_time_s"] = float(horizon * -60 / floor)
    else:
        out["reverberation_time_s"] = float(horizon)
    out["decibels_spanned"] = abs(floor)
    return out


def c50_db(x: np.ndarray, sample_rate: int = 44100) -> float:
    """Clarity: energy in the first 50 ms against everything after it."""
    split = min(len(x), round(.05 * sample_rate))
    early = float(np.sum(np.asarray(x, np.float64)[:split] ** 2))
    late = float(np.sum(np.asarray(x, np.float64)[split:] ** 2))
    return 10 * np.log10((early + 1e-20) / (late + 1e-20))


def direct_path_diagnostics(reference: np.ndarray, estimate: np.ndarray,
                            sample_rate: int = 44100,
                            window_ms: float = 1.) -> dict[str, float]:
    """Descriptive quantities for the direct arrival, plus the sanctioned C50/T30.

    Peak amplitude ratio and peak delay are reported as *descriptions* of the
    first millisecond, not as scores: averaging independent predictions was seen
    to suppress the direct arrival, and an aggregate early NRMSE over 70 ms
    cannot show that. C50 and T30 are the acoustic errors of record.
    """
    window = max(1, round(window_ms * sample_rate / 1000))
    reference_peak = float(np.max(np.abs(reference[:window])) + 1e-12)
    estimate_peak = float(np.max(np.abs(estimate[:window])))
    t30_reference, t30_estimate = (t30_seconds(reference, sample_rate),
                                   t30_seconds(estimate, sample_rate))
    return {
        "direct_peak_ratio": estimate_peak / reference_peak,
        "direct_peak_delay_samples": float(
            np.argmax(np.abs(estimate[:window])) - np.argmax(np.abs(reference[:window]))),
        "c50_error_db": abs(c50_db(estimate, sample_rate) - c50_db(reference, sample_rate)),
        "t30_error_s": (abs(t30_estimate - t30_reference)
                        if np.isfinite(t30_reference) and np.isfinite(t30_estimate)
                        else float("nan")),
    }


#: What each metric name MEANS, because two of them mean different things
#: depending on who wrote the column.
#:
#: ``lower_is_better``  a magnitude. A larger value is worse, so a positive
#:                      paired delta is a deterioration and the generic
#:                      "delta > 0 = worse" rule is correct.
#: ``signed_bias``      an estimate-minus-reference difference. The sign says
#:                      over- or under-estimate, NOT better or worse. Scoring
#:                      accuracy on it is a bug: score ``abs(...)`` and report
#:                      the signed value separately as a calibration diagnostic.
#:
#: The collision this exists to stop: ``rir_metrics`` below emits
#: ``c50_error_db`` and ``t30_error_s`` already wrapped in ``abs``, while
#: ``experiments/evaluate_arm.py`` and
#: ``experiments/real_clap_multiclap/run.py`` emit the SAME column names signed.
#: A consumer that does not know which produced its CSV cannot score them
#: correctly, and one did not: ``sampler_repair_verdict.py`` first read
#: evaluate_arm's signed C50 as a magnitude and reported a penalty that was a
#: calibration shift toward the target. Check the producer before scoring.
METRIC_SEMANTICS = {
    # magnitudes -- generic sign rule is safe
    "nrmse": "lower_is_better",
    "nrmse_full": "lower_is_better",
    "nrmse_0_10ms": "lower_is_better",
    "nrmse_10_70ms": "lower_is_better",
    "nrmse_70_endms": "lower_is_better",
    "edc_rmse_db": "lower_is_better",
    "late_energy_ratio_error": "lower_is_better",
    "roundtrip_nrmse": "lower_is_better",
    # produced BOTH ways -- resolve against the producer, never assume
    "c50_error_db": "depends_on_producer",
    "t30_error_s": "depends_on_producer",
    # signed by construction wherever they appear
    "onset_error_ms": "signed_bias",
    "direct_peak_delay_samples": "signed_bias",
    # a ratio whose ideal is 1, not 0: neither rule above applies unaltered
    "direct_peak_ratio": "ideal_is_one",
}
#: The producers that emit the ambiguous names signed rather than absolute.
SIGNED_PRODUCERS = ("experiments/evaluate_arm.py",
                    "clapgen/experiments/real_clap_multiclap/run.py")


def accuracy(value: float, semantics: str) -> float:
    """Score one metric value so that lower is always better.

    ``signed_bias`` becomes its magnitude. ``ideal_is_one`` becomes distance
    from one. ``depends_on_producer`` raises rather than guessing -- a verdict
    that cannot say which producer wrote its column must not score it.
    """
    if semantics == "lower_is_better":
        return float(value)
    if semantics == "signed_bias":
        return abs(float(value))
    if semantics == "ideal_is_one":
        return abs(float(value) - 1.)
    raise ValueError(
        f"cannot score a metric with semantics {semantics!r} without knowing its "
        f"producer; resolve it to 'lower_is_better' or 'signed_bias' first")


def roundtrip_metrics(clean: np.ndarray, reference_h: np.ndarray,
                      estimate_h: np.ndarray, length: int | None = None) -> dict[str, float]:
    length = len(reference_h) if length is None else length
    target = sps.fftconvolve(clean[:882], reference_h)[:length]
    estimate = sps.fftconvolve(clean[:882], estimate_h)[:length]
    corr = float(np.corrcoef(target, estimate)[0, 1])
    nrmse = float(np.sqrt(np.mean((estimate - target) ** 2)) /
                  (np.sqrt(np.mean(target ** 2)) + 1e-8))
    spectrum = float(np.sqrt(np.mean((20 * np.log10(np.abs(np.fft.rfft(estimate)) + 1e-7)
                                      - 20 * np.log10(np.abs(np.fft.rfft(target)) + 1e-7)) ** 2)))
    return {"roundtrip_corr": corr, "roundtrip_nrmse": nrmse,
            "roundtrip_lsd_db": spectrum}


def rir_metrics(reference: np.ndarray, estimate: np.ndarray,
                clean: np.ndarray) -> dict[str, float]:
    result = waveform_metrics(reference, estimate)
    result.update(band_errors(reference, estimate))
    result["edc_rmse_db"] = edc_rmse(reference, estimate)
    result.update(roundtrip_metrics(clean, reference, estimate))
    return result


def regularized_deconvolution(observation: np.ndarray, clean: np.ndarray,
                              length: int = 4096,
                              relative_regularization: float = 1e-2) -> np.ndarray:
    n_fft = 1
    while n_fft < 2 * length:
        n_fft *= 2
    y = np.fft.rfft(observation, n_fft)
    x = np.fft.rfft(clean, n_fft)
    power = np.abs(x) ** 2
    h = np.fft.irfft(y * np.conj(x) /
                     (power + relative_regularization * np.max(power) + 1e-20), n_fft)[:length]
    h = np.asarray(h, np.float32)
    return h / (np.max(np.abs(h)) + 1e-8)
