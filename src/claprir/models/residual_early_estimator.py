#!/usr/bin/env python3
"""Very-early residual correctors on top of a frozen regression estimator.

The design question this file exists to answer is narrow and was chosen because
the broader one is already settled. Full-RIR flow reproduces the direct arrival
well and destroys the decay; a 70 ms early-only flow gained nothing held out and
regressed 10-70 ms. Both asked flow to *generate* an impulse response. This asks
whether flow can *correct the residual a good regression model leaves behind*,
on the only support where flow has ever looked useful:

    r* = M_e (h - mu(y)),      M_e = 1 over [0, 10 ms)

The corrector is spliced back hard, so beyond 10 ms the output is bit-identical
to the frozen anchor. That removes the confound that made the earlier arms hard
to read: an arm can now only lose by failing to fix the early residual, never by
damaging something else.

R3 and F3 are deliberately the SAME network with the same parameter count. The
deterministic arm is not a smaller model with the time embedding removed -- it
carries the identical trunk and is handed a zero state and a constant time, so
the only substantive difference is what those inputs carry. Otherwise a win for
flow could be a win for capacity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from claprir.models.hybrid_rir_estimator import compressed_stft_loss
from claprir.models.time_unet import UNet1D

SAMPLE_RATE = 44_100
#: 10 ms. The support where flow has ever shown an advantage -- sparse,
#: high-dynamic-range direct and first reflections.
EARLY_SAMPLES = round(.010 * SAMPLE_RATE)          # 441
#: The trunk downsamples by 8, so the segment is padded to a multiple and cropped
#: back. Padding rather than moving the boundary keeps the mask exactly 10 ms.
STRIDE_PRODUCT = 8
PADDED = ((EARLY_SAMPLES + STRIDE_PRODUCT - 1) // STRIDE_PRODUCT) * STRIDE_PRODUCT  # 448
#: Direct-path window for the peak term.
DIRECT_SAMPLES = round(.001 * SAMPLE_RATE)


def early_mask(length: int, device) -> torch.Tensor:
    mask = torch.zeros(1, 1, length, device=device)
    mask[..., :EARLY_SAMPLES] = 1.
    return mask


class ResidualEarlyCorrector(nn.Module):
    """``mode="regression"`` -> R3, ``mode="flow"`` -> F3. Identical parameters.

    Conditioning is ``c = (y_early, anchor_early)`` in both arms. The flow arm
    additionally puts the interpolation state in the third channel and a real
    time in the embedding; the regression arm receives zeros and a constant, so
    the tensors have the same shape and the weights the same count.
    """

    def __init__(self, mode: str = "flow", channels: tuple[int, ...] = (32, 64, 64, 64),
                 strides: tuple[int, ...] = (2, 2, 2), flow_steps: int = 8):
        super().__init__()
        if mode not in ("flow", "regression"):
            raise ValueError("mode must be 'flow' or 'regression'")
        self.mode, self.flow_steps = mode, flow_steps
        self.net = UNet1D(in_channels=3, depth=3, emb_dim=32, channels=channels,
                          strides=strides, use_norm=True, time_conditional=True)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _crop(x: torch.Tensor) -> torch.Tensor:
        return x[..., :EARLY_SAMPLES]

    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, PADDED - x.shape[-1]))

    def _condition(self, observation: torch.Tensor, anchor: torch.Tensor,
                   state: torch.Tensor) -> torch.Tensor:
        return torch.cat((self._pad(observation), self._pad(anchor),
                          self._pad(state)), dim=1)

    def velocity(self, state: torch.Tensor, observation: torch.Tensor,
                 anchor: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self._crop(self.net(self._condition(observation, anchor, state), time))

    # -- training --------------------------------------------------------------

    def loss(self, target_early: torch.Tensor, observation_early: torch.Tensor,
             anchor_early: torch.Tensor, sigma: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Returns (flow-matching term, predicted residual, components).

        The predicted residual is returned so the caller can add endpoint losses
        on the actual integrated output rather than on the velocity field, which
        is where the earlier arms and the acoustic objective came apart.
        """
        batch = target_early.shape[0]
        residual = target_early - anchor_early
        if self.mode == "regression":
            zero = torch.zeros_like(residual)
            time = torch.zeros(batch, device=residual.device)
            predicted = self.velocity(zero, observation_early, anchor_early, time)
            mse = F.mse_loss(predicted, residual)
            # Tensors, not floats: converting here would force a GPU sync on
            # every update. Measured at 0.87 s/update with 20% utilisation --
            # the step was waiting on the queue to drain, not computing.
            return mse, predicted, {"residual_mse": mse.detach()}
        # Flow: the source is the anchor's own neighbourhood, not unit Gaussian.
        # Starting from noise would ask the field to rebuild structure the anchor
        # already has.
        scale = sigma if sigma is not None else residual.std(dim=-1, keepdim=True)
        noise = torch.randn_like(residual) * scale
        time = torch.rand(batch, device=residual.device)
        state = (1 - time[:, None, None]) * noise + time[:, None, None] * residual
        predicted_velocity = self.velocity(state, observation_early, anchor_early, time)
        flow = F.mse_loss(predicted_velocity, residual - noise)
        # Implied endpoint from the current state, for the endpoint losses.
        implied = state + (1 - time[:, None, None]) * predicted_velocity
        return flow, implied, {"flow_mse": flow.detach()}

    # -- inference -------------------------------------------------------------

    @torch.no_grad()
    def predict_residual(self, observation_early: torch.Tensor,
                         anchor_early: torch.Tensor,
                         generator: torch.Generator | None = None,
                         sigma: torch.Tensor | None = None) -> torch.Tensor:
        batch = anchor_early.shape[0]
        if self.mode == "regression":
            zero = torch.zeros_like(anchor_early)
            time = torch.zeros(batch, device=anchor_early.device)
            return self.velocity(zero, observation_early, anchor_early, time)
        scale = sigma if sigma is not None else anchor_early.std(dim=-1, keepdim=True)
        state = torch.randn(anchor_early.shape, device=anchor_early.device,
                            generator=generator) * scale
        # Fixed-step Euler with a preregistered step count. No best-of-N: the
        # primary comparison must not reward drawing until a good sample appears.
        for step in range(self.flow_steps):
            time = torch.full((batch,), step / self.flow_steps,
                              device=anchor_early.device)
            state = state + self.velocity(state, observation_early, anchor_early,
                                          time) / self.flow_steps
        return state


def soft_arrival(signal: torch.Tensor, beta: float = 50.) -> torch.Tensor:
    """Differentiable arrival time: a softmax-weighted index over |signal|."""
    magnitude = signal.abs().squeeze(1)
    index = torch.arange(magnitude.shape[-1], device=signal.device, dtype=magnitude.dtype)
    weight = torch.softmax(beta * magnitude / (magnitude.amax(dim=-1, keepdim=True) + 1e-8),
                           dim=-1)
    return (weight * index).sum(dim=-1)


def endpoint_losses(estimate_early: torch.Tensor, target_early: torch.Tensor,
                    weights: tuple[float, ...] = (1., .5, .25, .25)
                    ) -> tuple[torch.Tensor, dict[str, float]]:
    """Losses on the acoustic endpoint, not on the velocity field.

    The spectral term is the project's existing compressed-STFT rather than a new
    multi-resolution loss: the programme's rules forbid inventing losses during
    validation, and this one is already in the allowed set.
    """
    w_nrmse, w_peak, w_arrival, w_spectral = weights
    nrmse = (torch.linalg.vector_norm(estimate_early - target_early, dim=-1)
             / (torch.linalg.vector_norm(target_early, dim=-1) + 1e-8)).mean()
    peak_estimate = estimate_early[..., :DIRECT_SAMPLES].abs().amax(dim=-1)
    peak_target = target_early[..., :DIRECT_SAMPLES].abs().amax(dim=-1)
    peak = ((peak_estimate - peak_target).abs() / (peak_target + 1e-8)).mean()
    arrival = ((soft_arrival(estimate_early) - soft_arrival(target_early)).abs()
               / EARLY_SAMPLES).mean()
    spectral = compressed_stft_loss(estimate_early, target_early)
    total = w_nrmse * nrmse + w_peak * peak + w_arrival * arrival + w_spectral * spectral
    return total, {"early_nrmse": nrmse.detach(), "peak": peak.detach(),
                   "arrival": arrival.detach(), "spectral": spectral.detach()}


class SetResidualCorrector(nn.Module):
    """Set-conditioned very-early residual corrector for K repeated claps.

    The conditioning is **permutation-invariant** by construction:

        z = rho( (1/K) sum_i phi(y_i, h0_i) )

    and not a channel concatenation. Concatenation makes the estimate depend on
    the order the claps happened to be listed in, which is not a property of the
    room; the earlier joint multi-clap regression used it and it is the one part
    of that design worth not repeating.

    The corrector downstream of the pooling is the SAME UNet as the K=1 arms,
    and ``mode`` selects deterministic or flow exactly as before, so R3-K and
    E4-K differ from each other only in the corrector family -- and from R3/F3
    only in the conditioning. That is what keeps three separate questions
    separable: does residualisation help, does flow help, does set conditioning
    help.
    """

    def __init__(self, mode: str = "flow", channels: tuple[int, ...] = (32, 64, 64, 64),
                 strides: tuple[int, ...] = (2, 2, 2), flow_steps: int = 8,
                 encoder_width: int = 32):
        super().__init__()
        self.corrector = ResidualEarlyCorrector(mode=mode, channels=channels,
                                                strides=strides, flow_steps=flow_steps)
        # phi: per-clap encoder over (observation, anchor); rho: back to the two
        # channels the corrector already expects, so the trunk is untouched.
        self.phi = nn.Sequential(
            nn.Conv1d(2, encoder_width, 9, padding=4), nn.GELU(),
            nn.Conv1d(encoder_width, encoder_width, 9, padding=4), nn.GELU())
        self.rho = nn.Sequential(
            nn.Conv1d(encoder_width, encoder_width, 9, padding=4), nn.GELU(),
            nn.Conv1d(encoder_width, 2, 1))

    @property
    def mode(self) -> str:
        return self.corrector.mode

    def pool(self, observations: torch.Tensor, anchors: torch.Tensor,
             mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B,K,T]`` observations and anchors -> two conditioning channels.

        ``mask`` marks which slots are real, so a record evaluated at K=2 pools
        two claps and not two claps plus three zeros.
        """
        batch, k, length = observations.shape
        stacked = torch.stack((observations, anchors), dim=2).reshape(batch * k, 2, length)
        features = self.phi(stacked).reshape(batch, k, -1, length)
        weights = mask[:, :, None, None]
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)
        conditioning = self.rho(pooled)
        return conditioning[:, :1], conditioning[:, 1:]

    def loss(self, target_early, observations, anchors, mask):
        observation, anchor_feature = self.pool(observations, anchors, mask)
        # The residual is defined against the AVERAGED anchor, which is the
        # strong existing multi-clap baseline this has to beat.
        anchor_mean = ((anchors * mask[:, :, None]).sum(dim=1)
                       / mask.sum(dim=1).clamp(min=1e-8)[:, None])[:, None]
        primary, predicted, parts = self.corrector.loss(
            target_early, observation, anchor_feature)
        return primary, predicted, anchor_mean, parts

    @torch.no_grad()
    def predict_residual(self, observations, anchors, mask, generator=None):
        observation, anchor_feature = self.pool(observations, anchors, mask)
        return self.corrector.predict_residual(observation, anchor_feature,
                                               generator=generator)


class ConcatResidualCorrector(nn.Module):
    """Order-sensitive channel concatenation: the whiteboard's literal proposal.

    Kept as a named control rather than folded into the set model, because the
    two answer different questions. Concatenation asks whether exposing the
    network to several claps at once helps; pooling asks whether it helps in a
    way that respects the fact that claps have no canonical ordering. Only the
    aggregation differs -- per-clap encoder, corrector trunk, output support and
    losses are identical to ``SetResidualCorrector`` -- so a difference between
    them is attributable to the aggregation and not to capacity elsewhere.

    Note the property that motivates the comparison: unless it learns to ignore
    order from data, ``G(y1,y2,y3) != G(y3,y1,y2)``, while the room is the same.
    """

    def __init__(self, mode: str = "flow", k: int = 5,
                 channels: tuple[int, ...] = (32, 64, 64, 64),
                 strides: tuple[int, ...] = (2, 2, 2), flow_steps: int = 8,
                 encoder_width: int = 32):
        super().__init__()
        self.k = k
        self.corrector = ResidualEarlyCorrector(mode=mode, channels=channels,
                                                strides=strides, flow_steps=flow_steps)
        # 2K input channels -- the claps and their anchors stacked, in order --
        # against the set model's 2 channels shared across claps.
        self.phi = nn.Sequential(
            nn.Conv1d(2 * k, encoder_width, 9, padding=4), nn.GELU(),
            nn.Conv1d(encoder_width, encoder_width, 9, padding=4), nn.GELU())
        self.rho = nn.Sequential(
            nn.Conv1d(encoder_width, encoder_width, 9, padding=4), nn.GELU(),
            nn.Conv1d(encoder_width, 2, 1))

    @property
    def mode(self) -> str:
        return self.corrector.mode

    def pool(self, observations: torch.Tensor, anchors: torch.Tensor,
             mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.cat((observations, anchors), dim=1)      # [B, 2K, T]
        conditioning = self.rho(self.phi(stacked))
        return conditioning[:, :1], conditioning[:, 1:]

    def loss(self, target_early, observations, anchors, mask):
        observation, anchor_feature = self.pool(observations, anchors, mask)
        anchor_mean = ((anchors * mask[:, :, None]).sum(dim=1)
                       / mask.sum(dim=1).clamp(min=1e-8)[:, None])[:, None]
        primary, predicted, parts = self.corrector.loss(
            target_early, observation, anchor_feature)
        return primary, predicted, anchor_mean, parts

    @torch.no_grad()
    def predict_residual(self, observations, anchors, mask, generator=None):
        observation, anchor_feature = self.pool(observations, anchors, mask)
        return self.corrector.predict_residual(observation, anchor_feature,
                                               generator=generator)


class AnchoredResidualFlow(nn.Module):
    """F5: transport from the deterministic estimate, not from zero-mean noise.

    F3 asks a velocity field to build ``r* = h - h_E0`` out of ``sigma * eps``.
    R3 already produces a good centre for that residual, so F5 starts there:

        z_0 = r_R3 + sigma * eps        z_1 = r*

    and the field only has to model the spread of the residual around a point
    that is already close, rather than regenerate its structure. ``source="zero"``
    reproduces F3's geometry, which is what makes F3-fixed available as the
    control that separates this change from the scale repair below.

    **The scale is the same at training and inference.** ``ResidualEarlyCorrector``
    trains at ``sigma = residual.std`` and samples at ``sigma = anchor.std``,
    which is 3.3x wider at the median and 6.9x at the 90th percentile, so every
    published flow arm sampled from a distribution it never trained on. Here the
    scale is a registered buffer, estimated once from training residuals and used
    unchanged in both paths.

    ``whiten=True`` is F5-W: the residual is divided by a per-sample standard
    deviation profile before transport and multiplied back afterwards. The
    profile is a buffer, fitted on training data only. One transform, no sweep.
    """

    def __init__(self, source: str = "regression", whiten: bool = False,
                 channels: tuple[int, ...] = (32, 64, 64, 64),
                 strides: tuple[int, ...] = (2, 2, 2), flow_steps: int = 8,
                 sigma: float = .035):
        super().__init__()
        if source not in ("regression", "zero"):
            raise ValueError("source must be 'regression' or 'zero'")
        self.source, self.whiten, self.flow_steps = source, whiten, flow_steps
        self.net = UNet1D(in_channels=3, depth=3, emb_dim=32, channels=channels,
                          strides=strides, use_norm=True, time_conditional=True)
        # One scalar, fitted on training residuals, used in both directions.
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        # Whitening profile: mean and std per sample index. Identity until fitted.
        self.register_buffer("profile_mean", torch.zeros(EARLY_SAMPLES))
        self.register_buffer("profile_std", torch.ones(EARLY_SAMPLES))

    # -- fitting the buffers ---------------------------------------------------

    @torch.no_grad()
    def fit_statistics(self, residuals: torch.Tensor) -> None:
        """Fit sigma and the whitening profile on TRAINING residuals only.

        Called once before training. Fitting on anything the arm is later scored
        against would put test information into the source distribution.
        """
        flat = residuals.reshape(-1, residuals.shape[-1])
        self.sigma.fill_(float(flat.std()))
        self.profile_mean.copy_(flat.mean(dim=0))
        self.profile_std.copy_(flat.std(dim=0).clamp_min(1e-6))

    def _forward_transform(self, residual: torch.Tensor) -> torch.Tensor:
        if not self.whiten:
            return residual
        return (residual - self.profile_mean) / self.profile_std

    def _inverse_transform(self, whitened: torch.Tensor) -> torch.Tensor:
        if not self.whiten:
            return whitened
        return whitened * self.profile_std + self.profile_mean

    # -- helpers, shared with the matched deterministic arm --------------------

    @staticmethod
    def _crop(x: torch.Tensor) -> torch.Tensor:
        return x[..., :EARLY_SAMPLES]

    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, PADDED - x.shape[-1]))

    def velocity(self, state: torch.Tensor, observation: torch.Tensor,
                 anchor: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        conditioning = torch.cat((self._pad(observation), self._pad(anchor),
                                  self._pad(state)), dim=1)
        return self._crop(self.net(conditioning, time))

    def _centre(self, centre: torch.Tensor | None,
                like: torch.Tensor) -> torch.Tensor:
        """The source centre: R3's prediction for F5, zero for F3-fixed."""
        if self.source == "zero" or centre is None:
            return torch.zeros_like(like)
        return self._forward_transform(centre)

    # -- training --------------------------------------------------------------

    def loss(self, target_early: torch.Tensor, observation_early: torch.Tensor,
             anchor_early: torch.Tensor, centre: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """``centre`` is the frozen deterministic corrector's residual estimate."""
        residual = self._forward_transform(target_early - anchor_early)
        base = self._centre(centre, residual)
        noise = base + torch.randn_like(residual) * self.sigma
        time = torch.rand(target_early.shape[0], device=residual.device)
        state = (1 - time[:, None, None]) * noise + time[:, None, None] * residual
        predicted = self.velocity(state, observation_early, anchor_early, time)
        flow = F.mse_loss(predicted, residual - noise)
        implied = state + (1 - time[:, None, None]) * predicted
        # Endpoint losses act on the acoustic residual, so undo the whitening.
        return flow, self._inverse_transform(implied), {"flow_mse": flow.detach()}

    # -- inference -------------------------------------------------------------

    @torch.no_grad()
    def predict_residual(self, observation_early: torch.Tensor,
                         anchor_early: torch.Tensor,
                         generator: torch.Generator | None = None,
                         centre: torch.Tensor | None = None) -> torch.Tensor:
        batch = anchor_early.shape[0]
        template = torch.zeros(anchor_early.shape[0], anchor_early.shape[1],
                               EARLY_SAMPLES, device=anchor_early.device,
                               dtype=anchor_early.dtype)
        base = self._centre(centre, template)
        state = base + torch.randn(template.shape, device=template.device,
                                   generator=generator,
                                   dtype=template.dtype) * self.sigma
        for step in range(self.flow_steps):
            time = torch.full((batch,), step / self.flow_steps,
                              device=anchor_early.device)
            state = state + self.velocity(state, observation_early, anchor_early,
                                          time) / self.flow_steps
        return self._inverse_transform(state)
