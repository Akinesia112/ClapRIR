"""Lundeby (1995) noise-floor crossing point, and EDC truncated at it.

Why this exists: `schroeder_edc` backward-integrates the whole signal, so the
curve includes the measurement's own noise floor and flattens at a level that
says nothing about the room. Anything scored to the right of the crossing is a
comparison against noise.

IMPORTANT — RUN THIS ON THE FULL-LENGTH RIR, NOT A 250 ms CROP.
The method needs a stretch of noise-only signal after the crossing to estimate
the floor from. A 250 ms window does not provide one for typical rooms, and the
crossing is then unreliable in either direction. Use the full 1 s `rir` row from
the npz to obtain the crossing, then apply that sample index to whatever window
is being scored. `is_reliable()` reports whether the input was long enough.
"""
import numpy as np


def lundeby(x, sr=44100, max_iter=8, lo_off=5.0, hi_off=25.0):
    """Return ``(crossing_sample, noise_level_dB, slope_dB_per_s)``.

    ``lo_off``/``hi_off`` bound the late-decay regression relative to the noise
    level. Lundeby specifies a 10-20 dB dynamic range starting 5-10 dB above the
    noise; the default 5..25 is a 20 dB range starting 5 dB above it. Narrower
    bands are also in spec but are less robust on short or noisy inputs.
    """
    x = np.asarray(x, np.float64)
    e = x ** 2
    nz = np.nonzero(e)[0]
    if len(nz) == 0:
        return 0, -np.inf, 0.0
    last_nonzero = nz[-1] + 1
    if last_nonzero < len(e) - sr * 0.005:      # exact digital silence at the end
        return int(last_nonzero), -np.inf, 0.0
    peak = e.max()

    def smooth(win_ms):
        w = max(1, int(sr * win_ms / 1000.0))
        n = len(e) // w
        if n < 4:
            return None, None
        seg = e[:n * w].reshape(n, w).mean(1)
        t = (np.arange(n) + .5) * w             # in SAMPLES, so slopes are dB/sample
        return t, 10 * np.log10(np.maximum(seg / peak, 1e-30))

    def fit(t, L, lo, hi):
        m = (L <= hi) & (L >= lo)
        if m.sum() < 2:
            return None
        return np.polyfit(t[m], L[m], 1)

    # 1-2. initial smoothing, noise estimate from the last 10 %
    t, L = smooth(30)
    if t is None:
        return len(e), -np.inf, 0.0
    noise = 10 * np.log10(max(np.mean(e[int(.9 * len(e)):]) / peak, 1e-30))
    # 3-4. first regression from 0 dB down to noise+10 dB
    p = fit(t, L, noise + 10, 0.0)
    if p is None or p[0] >= 0:
        return len(e), float(noise), 0.0
    cross = (noise - p[1]) / p[0]

    for _ in range(max_iter):
        slope_db_s = abs(p[0] * sr)
        if slope_db_s < 1e-6:
            break
        # 5. window length giving ~5 intervals per 10 dB of decay.
        # p[0] is dB/SAMPLE, so the seconds per 10 dB is 10/|p[0]*sr|.
        win_ms = float(np.clip(1000.0 * 10.0 / slope_db_s / 5.0, 3.0, 100.0))
        t2, L2 = smooth(win_ms)
        if t2 is None:
            break
        # 6b. noise from a segment starting one 10 dB decay-time past the crossing
        start = int(cross + 10.0 / abs(p[0]))
        if start >= len(e) - int(sr * 0.01):
            start = int(len(e) * 0.9)
        noise_new = 10 * np.log10(max(np.mean(e[start:]) / peak, 1e-30))
        # 6c. late-decay regression
        p_new = fit(t2, L2, noise_new + lo_off, noise_new + hi_off)
        if p_new is None or p_new[0] >= 0:
            break
        cross_new = (noise_new - p_new[1]) / p_new[0]
        converged = abs(cross_new - cross) < sr * 0.001
        cross, noise, p = cross_new, noise_new, p_new
        if converged:
            break
    return int(np.clip(cross, 1, len(e))), float(noise), float(p[0] * sr)


def is_reliable(x, cross, slope_db_per_s, sr=44100, min_margin_db=10.0):
    """False when the input is too short for the crossing to be trusted.

    The floor is estimated from what lies after the crossing, so there must be
    enough of it: at least ``min_margin_db`` worth of further decay time. On a
    250 ms crop of a typical room this is False, which is why the crossing
    should come from the full-length RIR.
    """
    if not np.isfinite(slope_db_per_s) or abs(slope_db_per_s) < 1e-6:
        return bool(cross < len(x))          # exact-silence case is fine
    needed = abs(min_margin_db / slope_db_s) * sr if (slope_db_s := abs(slope_db_per_s)) else 0
    return bool(len(x) - cross >= needed)


def edc_truncated(x, cross):
    """Backward-integrated EDC over ``x[:cross]``, normalised to 0 dB at t=0.

    No Lundeby tail-energy compensation: measured on this data the term is
    0.00-0.35 % of the total energy (under 0.02 dB on the curve), and estimating
    it from the level at the crossing is itself noisy -- a single sample differs
    by ~7x from a 5 ms mean at the same point. Add it only if the truncation
    point sits high above the floor, and estimate the level from the fitted
    decay rather than from the waveform.
    """
    x = np.asarray(x, np.float64)[:cross]
    if len(x) == 0:
        return np.array([0.0])
    c = np.cumsum(x[::-1] ** 2)[::-1]
    return 10 * np.log10(np.maximum(c / (c[0] + 1e-30), 1e-30))
