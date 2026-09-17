#!/usr/bin/env python3
"""Acoustic diagnosis suite for RIR estimates, added 2026-08-28.

The programme's validation rules forbid inventing metrics *during validation*.
This is not that: these quantities were requested specifically because the
open question is no longer "is flow worse" but "were we measuring the right
thing". Waveform NRMSE is sample-wise and can call a spectrally sensible
estimate bad because its taps are misaligned. Every metric here is a published,
standard room-acoustic quantity, and each one is introduced with the failure it
is meant to be able to see.

    stft_magnitude_errors   Is the spectrum right even when the samples are not?
    echo_density_profile    Do sparse early reflections become a dense late
                            field, and when? (Abel-Huang)
    mixing_time_ms          The time that profile first reaches Gaussian density.
    edt_seconds             Is the *initial* decay slope right? EDP is
                            deliberately insensitive to decay rate and level, so
                            it has to be paired with an energy/decay metric.
    early_structure_errors  Direct arrival and first significant reflection,
                            timing and amplitude.

C50 already exists in `metrics.py` and is reused rather than reimplemented; it
is scored as an ABSOLUTE error here (see `reports/metric_semantics_audit`).

Nothing in this file is a loss. None of it is added to any training objective.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

SAMPLE_RATE = 44_100
#: Abel-Huang sliding window. The original work gives 20-30 ms as the workable
#: range: shorter windows hold too few reflections and the profile jumps on
#: nothing. 24 ms is the middle of that range and is fixed here rather than
#: swept, so the profile is not tuned to the arms it scores.
ECHO_WINDOW_MS = 24.
#: Fraction of a Gaussian's samples lying beyond one standard deviation. The
#: profile is normalised by it so that Gaussian noise reads exactly 1.0.
GAUSSIAN_FRACTION = float(erfc(1 / np.sqrt(2)))          # 0.31731...
#: STFT for the magnitude errors, matching the project's existing spectral term
#: (`compressed_stft_loss`) so the two are comparable: same n_fft, same hop.
N_FFT = 510
HOP = N_FFT // 4


# -- spectrum ----------------------------------------------------------------

def _stft_magnitude(signal: np.ndarray, n_fft: int = N_FFT,
                    hop: int = HOP) -> np.ndarray:
    signal = np.asarray(signal, np.float64)
    n_fft = min(n_fft, len(signal))
    n_fft -= n_fft % 2
    window = np.hanning(n_fft + 1)[:n_fft]
    frames = 1 + max(0, (len(signal) - n_fft)) // hop
    out = np.empty((n_fft // 2 + 1, max(1, frames)))
    for index in range(max(1, frames)):
        chunk = signal[index * hop:index * hop + n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        out[:, index] = np.abs(np.fft.rfft(chunk * window))
    return out


#: Floor for the log-magnitude term, in dB below the REFERENCE's peak bin.
#: Without it the term is meaningless here: these spectrograms contain exactly
#: zero bins, so log(0 + eps) is set by eps rather than by the signal, and the
#: metric ends up dominated by empty bins instead of by the response.
LOG_FLOOR_DB = -80.


def stft_magnitude_errors(reference: np.ndarray, estimate: np.ndarray,
                          n_fft: int = N_FFT, hop: int = HOP,
                          floor_db: float = LOG_FLOOR_DB) -> dict[str, float]:
    """Linear and log magnitude MSE, plus a scale-free spectral convergence.

    The linear term is what was asked for. The log term is reported beside it
    because a linear magnitude MSE is dominated by the loudest bins -- the direct
    arrival's broadband energy -- and can look fine while the decay tail's
    spectrum is wrong by tens of dB.

    Neither is a loss. Both are evaluation only.
    """
    a = _stft_magnitude(estimate, n_fft, hop)
    b = _stft_magnitude(reference, n_fft, hop)
    width = min(a.shape[1], b.shape[1])
    a, b = a[:, :width], b[:, :width]
    floor = float(b.max()) * 10 ** (floor_db / 20)
    return {
        "stft_mag_mse": float(np.mean((a - b) ** 2)),
        "stft_logmag_mse": float(np.mean((np.log10(np.maximum(a, floor))
                                          - np.log10(np.maximum(b, floor))) ** 2)),
        # Scale-free: ||A - B||_F / ||B||_F, so a global gain error does not
        # dominate the comparison between arms.
        "spectral_convergence": float(np.linalg.norm(a - b)
                                      / (np.linalg.norm(b) + 1e-20)),
    }


# -- echo density (Abel-Huang) -----------------------------------------------

def echo_density_profile(signal: np.ndarray, sample_rate: int = SAMPLE_RATE,
                         window_ms: float = ECHO_WINDOW_MS
                         ) -> tuple[np.ndarray, np.ndarray]:
    """``(times in seconds, eta)`` -- the normalised echo density profile.

    At each position the profile is the Hann-weighted fraction of taps inside the
    window whose magnitude exceeds the window's own weighted standard deviation,
    divided by the fraction a Gaussian would give. So eta ~ 1 means the local tap
    distribution is as dense as Gaussian noise, which is what "the late field has
    formed" means; eta << 1 means sparse, isolated reflections.

    The threshold is *local*, which is why the profile is insensitive to overall
    level and to decay rate, and sensitive to diffusion. The profile also ignores
    how LOUD the peaks are -- it only asks whether each tap clears the local
    threshold -- which is why it has to be paired with an energy measure (C50)
    and an early-decay measure (EDT) rather than read alone.

    Verified against the reference implementation
    ``pyFDN/auxiliary/acoustics.py:echo_density`` (Abel & Huang 2006): same
    sum-normalised Hann window, same weighted-RMS threshold, same
    ``erfc(1/sqrt(2))`` normalisation. The zero-padding here is exactly
    equivalent to that implementation's truncated-window slicing, because a
    padded zero contributes nothing to the weighted sum of squares and never
    exceeds the threshold.

    Two deliberate departures, both documented rather than silent:

    * it evaluates at **every** sample, where the reference evaluates every 500
      and interpolates. Denser is strictly more accurate and the cost is 0.2 s;
    * it returns NaN when the profile never crosses, where the reference returns
      0.0 ms. Returning 0 would report "mixed instantly" for a response that
      never mixed at all, which is the opposite of what happened.
    """
    signal = np.asarray(signal, np.float64)
    width = max(8, int(round(window_ms * sample_rate / 1000)))
    width -= width % 2
    # Symmetric Hann, matching the reference implementation. A periodic Hann
    # (`np.hanning(width + 1)[:width]`) is the usual choice for STFT analysis
    # and was used here first; it disagrees with the reference by ~1e-2 in eta,
    # which is small but is a difference in the metric rather than in the data.
    window = np.hanning(width)
    window = window / window.sum()
    half = width // 2

    padded = np.pad(signal, (half, half), mode="constant")
    # Vectorised over positions. The naive loop is O(N * width) in Python and
    # takes seconds per call, which is prohibitive when every arm on every
    # record needs one. The strided view is a read-only window stack, so the
    # arithmetic below is identical to the loop it replaces.
    view = np.lib.stride_tricks.sliding_window_view(padded, width)[:len(signal)]
    # Weighted standard deviation inside each window: the local threshold.
    sigma = np.sqrt(view ** 2 @ window)
    with np.errstate(invalid="ignore"):
        exceeds = np.abs(view) > sigma[:, None]
    eta = (exceeds @ window) / GAUSSIAN_FRACTION
    eta[sigma <= 0] = 0.
    return np.arange(len(signal)) / sample_rate, eta


def mixing_time_ms(signal: np.ndarray, sample_rate: int = SAMPLE_RATE,
                   window_ms: float = ECHO_WINDOW_MS) -> float:
    """First time the echo density profile reaches 1, in milliseconds.

    NaN when the profile never reaches 1 inside the horizon -- for a 250 ms
    response that is a real possibility (4 of 68 test targets here) and must be
    reported as unmeasurable rather than silently clipped to the horizon, and
    certainly not as 0 ms.

    Strict ``>``, matching the reference implementation.
    """
    times, eta = echo_density_profile(signal, sample_rate, window_ms)
    crossings = np.flatnonzero(eta > 1.)
    if crossings.size == 0:
        return float("nan")
    return float(times[crossings[0]] * 1000)


def echo_density_profile_rmse(reference: np.ndarray, estimate: np.ndarray,
                              sample_rate: int = SAMPLE_RATE,
                              window_ms: float = ECHO_WINDOW_MS) -> float:
    _, a = echo_density_profile(estimate, sample_rate, window_ms)
    _, b = echo_density_profile(reference, sample_rate, window_ms)
    width = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:width] - b[:width]) ** 2)))


# -- early decay -------------------------------------------------------------

def edt_seconds(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Early decay time: the 0 to -10 dB slope extrapolated to -60 dB.

    EDT rather than T30 because the question is whether the *initial* decay is
    right, and because a 250 ms horizon frequently never reaches -35 dB, which
    makes T30 unmeasurable on exactly the responses of interest.
    """
    curve = np.cumsum(np.asarray(signal, np.float64)[::-1] ** 2)[::-1]
    curve = 10 * np.log10(np.maximum(curve / (curve[0] + 1e-20), 1e-12))
    try:
        stop = int(np.flatnonzero(curve <= -10)[0])
    except IndexError:
        return float("nan")
    if stop < 4:
        return float("nan")
    time = np.arange(stop) / sample_rate
    slope = np.polyfit(time, curve[:stop], 1)[0]
    return float(-60 / slope) if slope < 0 else float("nan")


# -- early structure ---------------------------------------------------------

def _first_reflection(signal: np.ndarray, direct: int, sample_rate: int,
                      guard_ms: float, relative_threshold: float
                      ) -> int | None:
    """Index of the first tap after the direct arrival that counts as a reflection.

    A reflection has to clear two bars: be at least ``relative_threshold`` of the
    direct amplitude, and sit at least ``guard_ms`` after it so the direct
    arrival's own ringing is not counted as its first reflection.
    """
    guard = max(1, int(round(guard_ms * sample_rate / 1000)))
    start = direct + guard
    if start >= len(signal):
        return None
    magnitude = np.abs(signal[start:])
    threshold = relative_threshold * abs(signal[direct])
    above = np.flatnonzero(magnitude >= threshold)
    if above.size == 0:
        return None
    # The first local maximum within the region that clears the threshold.
    index = int(above[0])
    while (index + 1 < len(magnitude)
           and magnitude[index + 1] >= magnitude[index]):
        index += 1
    return start + index


def early_structure_errors(reference: np.ndarray, estimate: np.ndarray,
                           sample_rate: int = SAMPLE_RATE,
                           search_ms: float = 1.,
                           guard_ms: float = .5,
                           relative_threshold: float = .1) -> dict[str, float]:
    """Direct arrival and first reflection: timing and amplitude error.

    **Both events are located on the TARGET**, and the estimate is only searched
    in a small neighbourhood around them. Letting the estimate nominate its own
    "first reflection" would let a hallucinated peak be scored as a correct one,
    which is the specific failure this metric exists to catch.
    """
    reference = np.asarray(reference, np.float64)
    estimate = np.asarray(estimate, np.float64)
    search = max(1, int(round(search_ms * sample_rate / 1000)))

    def matched(anchor: int) -> tuple[float, float, float]:
        """``(timing error ms, |estimate peak|, |reference peak|)``.

        Both signals are peaked the SAME way inside the same window. Taking the
        reference's amplitude at the anchor while taking the estimate's at its
        local argmax would make a perfect reconstruction score non-zero whenever
        the anchor is a local maximum but not the window maximum -- which is
        common for a first reflection sitting on the direct arrival's ringing.
        """
        low = max(0, anchor - search)
        high = min(len(estimate), anchor + search + 1)
        if low >= high:
            return float("nan"), float("nan"), float("nan")
        reference_peak = low + int(np.argmax(np.abs(reference[low:high])))
        estimate_peak = low + int(np.argmax(np.abs(estimate[low:high])))
        return ((estimate_peak - reference_peak) / sample_rate * 1000,
                abs(float(estimate[estimate_peak])),
                abs(float(reference[reference_peak])))

    direct = int(np.argmax(np.abs(reference)))
    delay_direct, amplitude_direct, reference_direct = matched(direct)
    out = {
        "direct_time_error_ms": abs(delay_direct),
        "direct_amplitude_error": abs(amplitude_direct - reference_direct),
        "direct_amplitude_ratio": amplitude_direct / (reference_direct + 1e-20),
    }
    reflection = _first_reflection(reference, direct, sample_rate, guard_ms,
                                   relative_threshold)
    if reflection is None:
        out.update({"reflection1_time_error_ms": float("nan"),
                    "reflection1_amplitude_error": float("nan"),
                    "reflection1_amplitude_ratio": float("nan"),
                    "reflection1_delay_ms": float("nan")})
        return out
    delay_reflection, amplitude_reflection, reference_reflection = matched(reflection)
    out.update({
        "reflection1_time_error_ms": abs(delay_reflection),
        "reflection1_amplitude_error": abs(amplitude_reflection
                                           - reference_reflection),
        "reflection1_amplitude_ratio": (amplitude_reflection
                                        / (reference_reflection + 1e-20)),
        "reflection1_delay_ms": (reflection - direct) / sample_rate * 1000,
    })
    return out


# -- the suite ---------------------------------------------------------------

def diagnosis_metrics(reference: np.ndarray, estimate: np.ndarray,
                      sample_rate: int = SAMPLE_RATE) -> dict[str, float]:
    """Every diagnosis quantity for one (target, estimate) pair.

    C50 and EDT are reported as ABSOLUTE errors. The signed values are reported
    beside them as calibration diagnostics and take no part in any verdict --
    see `reports/metric_semantics_audit` for why that distinction is enforced.
    """
    from claprir.metrics.deconvolution import c50_db

    c50_reference, c50_estimate = (c50_db(reference, sample_rate),
                                   c50_db(estimate, sample_rate))
    edt_reference, edt_estimate = (edt_seconds(reference, sample_rate),
                                   edt_seconds(estimate, sample_rate))
    mix_reference = mixing_time_ms(reference, sample_rate)
    mix_estimate = mixing_time_ms(estimate, sample_rate)
    out = {
        **stft_magnitude_errors(reference, estimate),
        "echo_density_rmse": echo_density_profile_rmse(reference, estimate,
                                                       sample_rate),
        "mixing_time_abs_error_ms": abs(mix_estimate - mix_reference),
        "mixing_time_ms_reference": mix_reference,
        "mixing_time_ms_estimate": mix_estimate,
        "abs_c50_error_db": abs(c50_estimate - c50_reference),
        "signed_c50_error_db": c50_estimate - c50_reference,
        "abs_edt_error_s": abs(edt_estimate - edt_reference),
        "signed_edt_error_s": edt_estimate - edt_reference,
        "edt_s_reference": edt_reference,
        "edt_s_estimate": edt_estimate,
        **early_structure_errors(reference, estimate, sample_rate),
    }
    return out


#: Orientation of every quantity above, in the vocabulary of
#: `metrics.METRIC_SEMANTICS`, so no consumer has to guess.
DIAGNOSIS_SEMANTICS = {
    "stft_mag_mse": "lower_is_better",
    "stft_logmag_mse": "lower_is_better",
    "spectral_convergence": "lower_is_better",
    "echo_density_rmse": "lower_is_better",
    "mixing_time_abs_error_ms": "lower_is_better",
    "abs_c50_error_db": "lower_is_better",
    "abs_edt_error_s": "lower_is_better",
    "direct_time_error_ms": "lower_is_better",
    "direct_amplitude_error": "lower_is_better",
    "reflection1_time_error_ms": "lower_is_better",
    "reflection1_amplitude_error": "lower_is_better",
    "signed_c50_error_db": "signed_bias",
    "signed_edt_error_s": "signed_bias",
    "direct_amplitude_ratio": "ideal_is_one",
    "reflection1_amplitude_ratio": "ideal_is_one",
}
