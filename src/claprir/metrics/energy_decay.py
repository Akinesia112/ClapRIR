"""EDCLoss from gdalsanto/similarity-metrics-for-rirs, ported for this repo.

Upstream: metrics.py::EDCLoss at commit 21bacf17. A third-octave filterbank
energy-decay comparison, normalised by the reference curve's own variance, as
opposed to this repo's broadband dB RMSE (``lundeby.edc_truncated``).

Two upstream defects are handled explicitly rather than reproduced:

1. ``FilterBank.forward`` raises for ``backend='scipy'``. The body is
   ``if scipy: out=...`` / ``if torch: out=...`` / ``else: raise``, so the
   scipy branch computes ``out`` and then falls into the ``else`` of the
   SECOND ``if`` and raises. Only ``backend='torch'`` returns. That torch path
   is FFT multiplication against ``sosfreqz(..., nfft, fs=48000)``, whose
   ``nfft`` must equal ``len(rfft(x))`` or the multiply is a shape error, and
   whose hard-coded 48 kHz is marked ``# TEST`` upstream. We therefore use the
   ``_forward_scipy`` body (cascaded ``sosfilt``), which is what that branch was
   written to do and is sample-rate correct.
2. Band selection. ``fmin=60`` is meant to start the bank at 63 Hz, but
   ``if fmin > f: index[0] = i+1; break`` breaks on the FIRST nominal frequency
   below fmin (16 Hz), giving ``index[0]=1`` and starting the bank at 20 Hz.
   At 44.1 kHz a 5th-order bandpass at [14.1, 28.3] Hz is numerically degenerate.
   ``band_selection`` exposes both: "upstream" reproduces the 20 Hz start,
   "intended" starts at the first band at or above ``fmin``.

Everything else follows upstream exactly: discard the last 0.5 % of the signal
BEFORE filtering, backward-integrate with the 1/N factor, convert to dB with a
1e-32 floor, subtract each curve's own t=0 level per band, and return
``MSE(pred - level_pred, true - level_true) / mean((true - level_true)**2)``.
The result is a dimensionless ratio, NOT decibels.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as sps

NOMINAL_THIRD_OCTAVE = (16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200,
                        250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
                        2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500,
                        16000, 20000, 25000, 32000)


def center_frequencies(fmin: float = 60.0, fmax: float = 15000.0,
                       sample_rate: int = 44100,
                       band_selection: str = "intended") -> list[float]:
    freqs = list(NOMINAL_THIRD_OCTAVE)
    if band_selection == "upstream":
        lo = 0
        for i, f in enumerate(freqs):
            if fmin > f:
                lo = i + 1
                break
    elif band_selection == "intended":
        lo = next(i for i, f in enumerate(freqs) if f >= fmin)
    else:
        raise ValueError(band_selection)
    hi = len(freqs)
    for i, f in enumerate(freqs):
        if f > fmax:
            hi = i
            break
    selected = freqs[lo:hi]
    # A band whose upper edge reaches Nyquist cannot be realised as a bandpass.
    return [f for f in selected if f * np.sqrt(2) < sample_rate / 2]


def octave_filters(centers, sample_rate: int, order: int = 5) -> list[np.ndarray]:
    """Upstream ``_get_octave_filters``: Butterworth SOS per band."""
    out = []
    for f in centers:
        cutoff = f * np.array([1 / np.sqrt(2), np.sqrt(2)])
        out.append(sps.butter(N=order, Wn=cutoff, fs=sample_rate,
                              btype="bandpass", analog=False, output="sos"))
    return out


def _discard_last_n_percent(x: np.ndarray, n_percent: float) -> np.ndarray:
    last = int(np.round((1 - n_percent / 100) * x.shape[-1]))
    return x[..., :last]


def _backward_int(x: np.ndarray) -> np.ndarray:
    rev = x[..., ::-1]
    out = (1.0 / x.shape[-1]) * np.cumsum(rev ** 2, axis=-1)
    return out[..., ::-1]


class EDCLoss:
    """Callable form of upstream ``EDCLoss``; returns a dimensionless ratio."""

    def __init__(self, sample_rate: int = 44100, fraction: int = 3, order: int = 5,
                 fmin: float = 60.0, fmax: float = 15000.0,
                 band_selection: str = "intended", discard_percent: float = 0.5):
        if fraction != 3:
            raise NotImplementedError("only third-octave is ported")
        self.sample_rate = sample_rate
        self.discard_percent = discard_percent
        self.centers = center_frequencies(fmin, fmax, sample_rate, band_selection)
        self.sos = octave_filters(self.centers, sample_rate, order)

    def filterbank(self, x: np.ndarray) -> np.ndarray:
        """(time,) -> (bands, time), cascaded sosfilt per band."""
        return np.stack([sps.sosfilt(s, x, axis=-1) for s in self.sos], axis=-2)

    def curves_db(self, x: np.ndarray) -> np.ndarray:
        x = _discard_last_n_percent(np.asarray(x, np.float64), self.discard_percent)
        return 10 * np.log10(_backward_int(self.filterbank(x)) + 1e-32)

    def __call__(self, pred: np.ndarray, true: np.ndarray) -> float:
        p, t = self.curves_db(pred), self.curves_db(true)
        p = p - p[..., :1]
        t = t - t[..., :1]
        num = float(np.mean((p - t) ** 2))
        den = float(np.mean(t ** 2))
        return num / den
