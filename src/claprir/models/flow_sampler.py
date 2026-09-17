"""Inference sampler for ``HybridDirectRIRFlow`` (clapgen ``hybrid_flow``).

Drop-in replacement for ``HybridDirectRIRFlow.predict``. Weights are unchanged;
this only changes how the ODE is integrated.

Three differences from the shipped sampler, in order of how much they matter:

1. LOG-SNR TIME GRID. The training targets are peak-normalised, so their RMS is
   ~0.047 against a unit-variance source. Signal and noise are equal in ``x_t``
   only at ``t = 0.955``, so a t-uniform grid spends ~96% of its steps where the
   RIR is below the noise and cannot be resolved. Spacing steps uniformly in
   ``log SNR(t) = t*sigma_d/(1-t)`` puts them where the transport actually
   happens.

2. STOCHASTIC (CHURN) STEPS. The deterministic ODE is a bijection, so error
   accumulated early is transported faithfully to t=1, where the denoiser reads
   it as signal and commits to it. Re-noising each step returns the state to the
   model's own marginal instead. The construction is Karras et al. 2022 (EDM),
   adapted to the flow-matching interpolant: overshoot to a lower noise level
   ``r`` then add exactly enough fresh noise to land back on ``t_next``.

3. PEAK PROJECTION. Every training target satisfies ``max|h| == 1`` exactly
   (``manifest.py:fit_rir`` divides by the peak), so that is a hard property of
   the data manifold. Enforcing it on the denoised estimate each step keeps the
   trajectory on the manifold the model was trained on.

Convention note: clapgen uses ``t=0`` noise, ``t=1`` data, ``x_t = t*h +
(1-t)*eps``, ``v* = h - eps``. The EDM churn step is written in the opposite
convention, so it is called with ``s = 1-t`` and ``-v``.
"""
from __future__ import annotations

import numpy as np
import torch

#: Mean RMS of the 250 ms training targets over the shoebox/mit/but/ace mixture.
#: This is the number that sets where signal emerges from noise; recompute it if
#: the training mixture or the horizon changes.
SIGMA_D = 0.0466


def logsnr_grid(steps: int, snr_min: float = 1e-2, snr_max: float = 3e2,
                sigma_d: float = SIGMA_D) -> np.ndarray:
    """Time grid uniform in log SNR, with ``t=0`` and ``t=1`` pinned.

    ``SNR(t) = t*sigma_d/(1-t)``, inverted as ``t = snr/(sigma_d + snr)``.
    """
    snr = np.exp(np.linspace(np.log(snr_min), np.log(snr_max), steps - 1))
    return np.concatenate([[0.0], snr / (sigma_d + snr), [1.0]])


def _churn_step(x, v, s_cur, s_next, gamma, generator=None):
    """One Euler step in the ``s = 1-t`` convention, stochastic when gamma > 0.

    At gamma = 0 this reduces exactly to ``x + (s_next - s_cur) * v``.
    Variance-preserving: after the overshoot to ``r`` the noise coefficient is
    exactly ``s_next`` and the signal coefficient exactly ``1 - s_next``.
    """
    g, s_cur, s_next = float(gamma), float(s_cur), float(s_next)
    if g <= 0.0:
        return x + (s_next - s_cur) * v
    r = s_next / (1.0 + g - g * s_next)
    x = x + (r - s_cur) * v
    n = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return (x + r * (g * g + 2.0 * g) ** 0.5 * n) / (g * r + 1.0)


@torch.no_grad()
def sample(model, observations, *, steps: int = 20, gamma: float = 0.5,
           project_peak: bool = True, generator: torch.Generator | None = None,
           grid: np.ndarray | None = None) -> torch.Tensor:
    """``[B, K, T]`` observations -> ``[B, 1, T]`` estimate.

    ``model`` is a ``HybridDirectRIRFlow``; only ``model.velocity`` is used.
    Defaults are the recommended configuration.
    """
    ts = logsnr_grid(steps) if grid is None else np.asarray(grid, float)
    device = observations.device
    length = model.config.signal_length
    x = torch.randn((observations.shape[0], 1, length), device=device,
                    generator=generator)
    for i in range(len(ts) - 1):
        t = float(ts[i])
        v = model.velocity(x, observations,
                           torch.full((observations.shape[0],), t, device=device))
        if project_peak:
            # Denoised estimate implied by the current velocity, projected onto
            # the training manifold, then the velocity that reaches it.
            x1 = x + (1.0 - t) * v
            x1 = x1 / (x1.abs().amax(dim=-1, keepdim=True) + 1e-8)
            v = (x1 - x) / max(1.0 - t, 1e-6)
        x = _churn_step(x, -v, 1.0 - ts[i], 1.0 - ts[i + 1], gamma,
                        generator=generator)
    if project_peak:
        # The final grid step has 1-t ~ 1e-4, small enough that the guard above
        # can skip it -- and that is the step which fixes the output's peak.
        # Projecting the returned sample is algebraically identical:
        #   x_final = x_t + (1-t)*(x1hat - x_t)/(1-t) = x1hat
        x = x / (x.abs().amax(dim=-1, keepdim=True) + 1e-8)
    return x
