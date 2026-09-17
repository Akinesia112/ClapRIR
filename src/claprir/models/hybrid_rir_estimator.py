"""Hybrid time-frequency / time-domain direct RIR estimator.

Architecture transferred from Moliner et al., "Ambisonics Encoding of Room
Impulse Responses using a Device-Agnostic Diffusion Model", Section III-B and
Fig. 1:

    input -> STFT -> 2-D NCSN++ -> iSTFT -> [early | late]
    early(input) , early(stage-1) -> time U-Net -> refined early
    output = concat(refined early, stage-1 late)

with the paper's fixed 70 ms mixing time.  Only the backbone is transferred.
The Ambisonics parts of that work -- Wigner-D rotation equivariance, HOA channel
structure, array transfer functions, the diffusion/DPS inference procedure --
are deliberately absent: this model is a *deterministic supervised regressor*
from one or more reverberant clap observations to a mono RIR, which is what the
2026-08-06 meeting asked for ("before we do any kind of generative model we do
the supervised one first").

Two implementation choices are not specified by the paper and are recorded here:

* STFT window/hop.  The paper does not report them.  Following the author's
  guidance we use the larger of the two configurations shipped with the
  reference NCSN++ implementation, ``n_fft=510`` / ``hop=128``.  The hop is
  ``n_fft // 4``, i.e. below the ``n_fft // 2`` ceiling he asked us to respect,
  and ``n_fft // 2 + 1 = 256`` frequency bins divide cleanly by the network's
  three downsampling stages.
* Input feature scaling.  In the paper the network input is scaled by the
  diffusion preconditioner ``c_in(tau)``, which a regression model does not
  have.  We instead compress the input spectrogram magnitude with the same 2/3
  exponent the paper uses for its compressed-spectrogram distance.  The network
  output stays a linear complex spectrogram, so the iSTFT is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from claprir.models.ncsnpp import NCSNpp
from claprir.models.time_unet import UNet1D

SAMPLE_RATE = 44_100
#: Mixing time of the paper, in milliseconds.
PAPER_MIXING_TIME_MS = 70.0
#: Magnitude-compression exponent of the paper's compressed spectrogram.
COMPRESSION_EXPONENT = 2 / 3


@dataclass(frozen=True)
class HybridConfig:
    signal_length: int = 11_025           # 250 ms at 44.1 kHz
    sample_rate: int = SAMPLE_RATE
    k_max: int = 1
    n_fft: int = 510
    #: "stft" is the frozen default that produced every verdict in the
    #: programme; "multires" additionally conditions the 2-D stage on a shorter
    #: and a longer window. Synthesis is through the primary resolution either
    #: way -- see `claprir.models.representations`.
    analysis: str = "stft"
    hop_length: int = 128
    #: 3072 samples = 69.66 ms, the multiple of the U-Net stride product closest
    #: to the paper's 70 ms mixing time.
    mixing_time_samples: int = 3072
    frame_multiple: int = 16
    nf: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 1
    unet_depth: int = 5
    unet_channels: tuple[int, ...] = (32, 64, 64, 64, 64, 64)
    unet_strides: tuple[int, ...] = (2, 2, 2, 2, 2)
    unet_emb_dim: int = 32
    unet_use_norm: bool = True
    #: Exponent applied to the input spectrogram magnitude before the 2-D stage,
    #: phase kept. 2/3 is the frozen default and produced every verdict in the
    #: programme. **1.0 disables the compression entirely**, which is the matched
    #: ablation: the representation changes and nothing else does. Set it per
    #: arm rather than editing COMPRESSION_EXPONENT, so the two can be trained
    #: side by side and a difference is attributable to the representation.
    input_compression: float = COMPRESSION_EXPONENT
    #: Weight on the multi-resolution STFT term (Step 4A). 0.0 is the frozen
    #: default and reproduces every arm trained before it, because the term is
    #: then not merely scaled to nothing -- it is not computed at all, so the
    #: objective is bit-identical rather than merely equal.
    spectral_weight: float = 0.0
    #: Exponent applied to BOTH magnitudes inside the auxiliary spectral term.
    #: 2/3 is the frozen default that every arm before this was trained with.
    #: 1.0 makes that term the plain STFT-magnitude MSE. Separate from
    #: ``input_compression`` on purpose: one warps what the model SEES, this
    #: warps what it is SCORED on, and until now the two could not be set
    #: independently -- so every plain-representation arm was still supervised
    #: through the 2/3 compression it had just been freed from.
    spectral_exponent: float = COMPRESSION_EXPONENT

    @property
    def mixing_time_ms(self) -> float:
        return 1000 * self.mixing_time_samples / self.sample_rate


def stft_frames(signal: torch.Tensor, n_fft: int, hop_length: int,
                window: torch.Tensor, frame_multiple: int = 16) -> torch.Tensor:
    """``[B, C, T]`` -> ``[B, C, F, frames]`` complex, frames padded to a multiple.

    The padding exists because the 2-D backbone downsamples the frame axis; it is
    cropped again in :func:`istft_frames`, never handed to ``torch.istft``.
    """
    batch, channels, _ = signal.shape
    spec = torch.stft(signal.reshape(batch * channels, -1), n_fft, hop_length,
                      window=window, center=True, return_complex=True)
    spec = spec.reshape(batch, channels, spec.shape[-2], spec.shape[-1])
    pad = -spec.shape[-1] % frame_multiple
    if pad:
        spec = F.pad(spec, (0, pad))
    return spec.to(torch.complex64)


def istft_frames(spec: torch.Tensor, n_fft: int, hop_length: int,
                 window: torch.Tensor, length: int) -> torch.Tensor:
    """Inverse of :func:`stft_frames`.

    Drops the padded frames first: they lie outside the signal, but ``torch.istft``
    still counts them in the window envelope it divides by, which silently
    attenuates the tail.
    """
    batch, channels = spec.shape[:2]
    spec = spec[..., :length // hop_length + 1]
    signal = torch.istft(spec.reshape(batch * channels, *spec.shape[-2:]), n_fft,
                         hop_length, window=window, center=True, length=length)
    return signal.reshape(batch, channels, length)


class HybridDirectRIR(nn.Module):
    """``[B, K, T]`` reverberant claps -> ``[B, 1, T]`` room impulse response."""

    def __init__(self, config: HybridConfig | None = None):
        super().__init__()
        self.config = config = config or HybridConfig()
        if config.mixing_time_samples >= config.signal_length:
            raise ValueError("mixing time must be shorter than the target horizon")
        # One complex plane per observation; for K > 1 an extra plane per slot
        # carries the presence mask, mirroring the padded-observation contract of
        # the WaveNet baseline this model is compared against.
        self.uses_mask = config.k_max > 1
        complex_planes = config.k_max * (2 if self.uses_mask else 1)
        if config.analysis == "multires":
            from claprir.models.representations import MultiResolutionSTFT
            self.multires = MultiResolutionSTFT(config.n_fft, config.hop_length,
                                                frame_multiple=config.frame_multiple)
            # Only the observation planes gain resolutions; the mask planes are
            # constant in frequency and would be identical copies.
            complex_planes += config.k_max * (self.multires.channels - 1)
        elif config.analysis != "stft":
            raise ValueError(f"unknown analysis front-end {config.analysis}")
        self.tf_stage = NCSNpp(
            input_channels=2 * complex_planes, spatial_channels=1,
            time_conditional=False, nf=config.nf, ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks, attn_resolutions=(0,),
            image_size=config.n_fft // 2 + 1, fir=False,
        )
        # The 2026-08-20 meeting's constraint on the multi-clap model: the SAME
        # K observations must condition both stages.  Until now the early
        # refiner saw only observations[:, :1], so a K-channel model would have
        # been multi-clap in the time-frequency stage and single-clap in the
        # time-domain stage.  At k_max == 1 this is in_channels=2 and the stack
        # below is bit-identical to the frozen K=1 arms, so every existing
        # checkpoint still loads.
        self.early_refiner = UNet1D(
            in_channels=config.k_max + (config.k_max if self.uses_mask else 0) + 1,
            depth=config.unet_depth, emb_dim=config.unet_emb_dim,
            channels=config.unet_channels, strides=config.unet_strides,
            use_norm=config.unet_use_norm,
        )
        self.register_buffer("window", torch.hann_window(config.n_fft), persistent=False)

    # -- differentiable STFT / iSTFT -------------------------------------------------

    def stft(self, signal: torch.Tensor) -> torch.Tensor:
        return stft_frames(signal, self.config.n_fft, self.config.hop_length,
                           self.window, self.config.frame_multiple)

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return istft_frames(spec, self.config.n_fft, self.config.hop_length,
                            self.window, length)

    # -- stages ----------------------------------------------------------------------

    def time_frequency_stage(self, observations: torch.Tensor,
                             mask: torch.Tensor | None) -> torch.Tensor:
        spec = (self.multires(observations) if self.config.analysis == "multires"
                else self.stft(observations))
        exponent = getattr(self.config, "input_compression", COMPRESSION_EXPONENT)
        # exponent == 1 leaves the magnitude alone, so the stage sees the plain
        # complex spectrogram. Kept as a multiply rather than a branch so the
        # two arms differ only in the value of one number.
        compressed = ((spec.abs() + 1e-8) ** exponent
                      * torch.exp(1j * spec.angle()))
        if self.uses_mask:
            if mask is None:
                raise ValueError("a presence mask is required when k_max > 1")
            planes = mask[:, :, None, None].expand(
                -1, -1, *compressed.shape[-2:]).to(compressed.dtype)
            compressed = torch.cat((compressed, planes), dim=1)
        estimate = self.tf_stage(compressed)
        return self.istft(estimate, self.config.signal_length)

    def forward(self, observations: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(spliced estimate, first-stage estimate)``, both ``[B, 1, T]``."""
        if observations.shape[1] != self.config.k_max:
            raise ValueError(f"expected {self.config.k_max} observation channels, "
                             f"got {observations.shape[1]}")
        edge = self.config.mixing_time_samples
        coarse = self.time_frequency_stage(observations, mask)
        # The paper stacks the early segment of the input with the early segment
        # of the first-stage output.  With several observations ALL of them are
        # stacked, not just the first: the time-domain stage is where the direct
        # sound lives, and it is exactly the stage the meeting required to see
        # every clap.  Padded slots carry a constant presence plane, the same
        # contract the time-frequency stage uses, because a zeroed observation
        # channel is otherwise indistinguishable from a silent clap.
        parts = [observations[:, :, :edge], coarse[:, :, :edge]]
        if self.uses_mask:
            parts.insert(1, mask[:, :, None].expand(-1, -1, edge).to(observations.dtype))
        stacked = torch.cat(parts, dim=1)
        early = coarse[:, :, :edge] + self.early_refiner(stacked)
        return torch.cat((early, coarse[:, :, edge:]), dim=-1), coarse


def compressed_stft_loss(estimate: torch.Tensor, target: torch.Tensor,
                         n_fft: int = 510, exponent: float = COMPRESSION_EXPONENT
                         ) -> torch.Tensor:
    """Compressed-magnitude spectrogram MSE (the project's standard spectral term)."""
    n_fft = min(n_fft, estimate.shape[-1])
    n_fft -= n_fft % 2
    window = torch.hann_window(n_fft, device=estimate.device)
    a = torch.stft(estimate.squeeze(1), n_fft, n_fft // 4, window=window, return_complex=True)
    b = torch.stft(target.squeeze(1), n_fft, n_fft // 4, window=window, return_complex=True)
    return F.mse_loss((a.abs() + 1e-8) ** exponent, (b.abs() + 1e-8) ** exponent)


def masked_waveform_loss(estimate: torch.Tensor, target: torch.Tensor,
                         validity: torch.Tensor) -> torch.Tensor:
    """Waveform MSE over the valid support only, averaged PER EXAMPLE.

    The deterministic counterpart of the flow arm's masked velocity term, and
    deliberately the same shape: sum the squared error over each record's real
    support, divide by that record's own mask, then average over the batch. The
    alternative -- one global mean over the batch's total mask -- weights each
    recording by how long it happens to be, so a 560 ms MIT record would carry
    0.56x the gradient of a full-length BUT one, which silently down-weights
    exactly the providers the mask exists to handle honestly.

    Zeroing the residual and dividing by the full window is not the same thing
    and is worse than either extreme: it still tells the model the absent tail
    is worth fitting, weighted by how much of the record is missing.
    """
    m = validity if validity.dim() == estimate.dim() else validity[:, None]
    square = (estimate - target) ** 2
    kept = m.sum(dim=(-2, -1)).clamp_min(1.0)
    return ((square * m).sum(dim=(-2, -1)) / kept).mean()


def masked_compressed_stft_loss(estimate: torch.Tensor, target: torch.Tensor,
                                mask: torch.Tensor, n_fft: int = 510,
                                exponent: float = COMPRESSION_EXPONENT
                                ) -> torch.Tensor:
    """``compressed_stft_loss`` restricted to frames whose window is real data.

    A separate function rather than a flag on the existing one: six arms call
    that and a change to its return shape would move all of them.

    The 1 s shards are a fixed buffer and most sources are shorter, so a frame
    whose analysis window reaches past a record's last real sample is scored
    against padding. Supervising it teaches the model that the room fell silent
    there, which is the defect ``validity_masking`` exists to prevent -- and the
    padded region is exactly where the measured energy deficit is largest, so an
    unmasked auxiliary would push in the direction of the disease.

    A frame is kept only when its whole window lies inside the valid support.
    ``torch.stft`` with ``center=True`` pads by ``n_fft // 2``, so frame ``tau``
    covers samples ``[tau*hop - n_fft//2, tau*hop + n_fft//2)`` and the last kept
    frame is ``floor((T_valid - n_fft//2) / hop)``. Binary, not fractional: a
    fractional weighting is harder to audit and buys nothing here.

    Averaged PER EXAMPLE and then over the batch, matching the primary term. The
    alternative -- summing over the batch and dividing by the total number of
    kept frames -- would weight each recording by how long it happens to be, so
    a 560 ms MIT record would carry 0.56x the auxiliary gradient of a
    full-length BUT one, silently down-weighting the providers the mask exists
    to handle honestly.
    """
    n_fft = min(n_fft, estimate.shape[-1])
    n_fft -= n_fft % 2
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=estimate.device)
    a = torch.stft(estimate.squeeze(1), n_fft, hop, window=window, return_complex=True)
    b = torch.stft(target.squeeze(1), n_fft, hop, window=window, return_complex=True)
    error = ((a.abs() + 1e-8) ** exponent - (b.abs() + 1e-8) ** exponent) ** 2
    # per-frequency mean first, so the frame mask multiplies one number per frame
    per_frame = error.mean(dim=-2)                                  # (B, frames)
    m = mask if mask.dim() == 2 else mask.squeeze(1)                # (B, samples)
    valid = m.sum(dim=-1)                                           # T_valid, in samples
    # A fully valid record has no missing data, so the only thing past its end is
    # torch.stft's own centre padding -- an artefact of the transform that every
    # arm including the historical one already had, not an absent recording. Its
    # limit is therefore the padded end, which keeps every frame and makes this
    # function reduce EXACTLY to compressed_stft_loss at an all-ones mask. A
    # record that really is short is limited at its last real sample.
    full = torch.as_tensor(float(m.shape[-1]), device=m.device, dtype=valid.dtype)
    limit = torch.where(valid >= full, full + n_fft // 2, valid)
    last = torch.floor((limit - n_fft // 2) / hop)
    taus = torch.arange(per_frame.shape[-1], device=estimate.device, dtype=last.dtype)
    keep = (taus[None, :] <= last[:, None]).to(per_frame.dtype)     # (B, frames)
    kept = keep.sum(dim=-1).clamp_min(1.0)
    return ((per_frame * keep).sum(dim=-1) / kept).mean()


#: Resolutions for the multi-resolution spectral term. Three, spanning roughly
#: 12 to 46 ms of window, so the term sees both the reflection structure the
#: short window resolves and the noise floor the long one averages. The shortest
#: matches the existing single-resolution term, so the new loss strictly extends
#: what was already there rather than replacing it at a different scale.
MRSTFT_FFTS = (510, 1022, 2046)
MRSTFT_EPS = 1e-7


def multires_stft_loss(estimate: torch.Tensor, target: torch.Tensor,
                       ffts: tuple[int, ...] = MRSTFT_FFTS,
                       eps: float = MRSTFT_EPS) -> torch.Tensor:
    """Spectral convergence plus log-magnitude L1, averaged over resolutions.

    The standard multi-resolution STFT loss, in its published form rather than a
    variant of this project's own. Two reasons the form matters here.

    A magnitude MSE would be dominated by the loud early bins, which are already
    fit; the defect under repair is a *quiet* broadband floor. Spectral
    convergence is relative to the target's own norm and the log term compresses
    the dynamic range, so both see quiet bins.

    A log-MSE instead of log-L1 is ill-conditioned on this data: measured on real
    training targets it reaches 51.2 against 0.64 for the existing spectral term,
    so adding it would replace the objective rather than extend it.

    ``eps`` floors the log. It also caps how hard this term can push an estimate
    toward an exactly silent target, which is a real limit on what it can fix and
    is stated in the Step 4A preregistration rather than discovered afterwards.
    """
    total = estimate.new_zeros(())
    for n_fft in ffts:
        size = min(n_fft, estimate.shape[-1])
        size -= size % 2
        window = torch.hann_window(size, device=estimate.device)
        a = torch.stft(estimate.squeeze(1), size, size // 4, window=window,
                       return_complex=True).abs()
        b = torch.stft(target.squeeze(1), size, size // 4, window=window,
                       return_complex=True).abs()
        convergence = torch.linalg.norm(b - a) / (torch.linalg.norm(b) + eps)
        log_magnitude = F.l1_loss(torch.log(a + eps), torch.log(b + eps))
        total = total + convergence + log_magnitude
    return total / len(ffts)


class HybridDirectRIRRegression(nn.Module):
    """Deterministic training wrapper: waveform MSE plus a compressed-STFT term."""

    def __init__(self, config: HybridConfig | None = None, stft_weight: float = .25):
        super().__init__()
        self.net = HybridDirectRIR(config)
        self.stft_weight = stft_weight

    @property
    def config(self) -> HybridConfig:
        return self.net.config

    def loss(self, target: torch.Tensor, observations: torch.Tensor,
             presence_mask: torch.Tensor | None = None, *,
             validity_mask: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, dict[str, float]]:
        estimate, _ = self.net(observations, presence_mask)
        # Both terms obey the temporal validity mask when one is given, for the
        # reason the flow arm's does: past a record's last real sample the target
        # is zero because the RECORDING ENDED, not because the room fell silent.
        # A masked primary beside an unmasked auxiliary would supervise the
        # padding through the second term, which is the same defect wearing a
        # different hat.
        if validity_mask is not None:
            waveform = masked_waveform_loss(estimate, target, validity_mask)
            spectral = masked_compressed_stft_loss(estimate, target, validity_mask)
        else:
            waveform = F.mse_loss(estimate, target)
            spectral = compressed_stft_loss(estimate, target)
        total = waveform + self.stft_weight * spectral
        return total, {"waveform": float(waveform.detach()),
                       "compressed_stft": float(spectral.detach())}

    @torch.no_grad()
    def predict(self, observations: torch.Tensor,
                presence_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(observations, presence_mask)[0]


class HybridWeightedRegression(nn.Module):
    """E0-W: the same estimator, trained with the residual arms' endpoint terms.

    The control the residual-flow experiment cannot do without. If a corrector
    on the first 10 ms improves that region, the first thing to rule out is that
    plain regression would improve it too given the same emphasis -- the earlier
    full-flow arm's exact-train result already makes sparse-peak weighting a live
    explanation. Architecture and budget are identical to the deterministic
    baseline; only the objective changes.
    """

    def __init__(self, config: HybridConfig | None = None, stft_weight: float = .25):
        super().__init__()
        self.net = HybridDirectRIR(config)
        self.stft_weight = stft_weight

    @property
    def config(self) -> HybridConfig:
        return self.net.config

    def loss(self, target: torch.Tensor, observations: torch.Tensor,
             presence_mask: torch.Tensor | None = None, *,
             validity_mask: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, dict[str, float]]:
        if validity_mask is not None:
            raise NotImplementedError(
                "HybridWeightedRegression has no masked objective; training it with "
                "--validity-masking would silently ignore the mask")
        from claprir.models.residual_early_estimator import EARLY_SAMPLES, endpoint_losses
        estimate, _ = self.net(observations, presence_mask)
        waveform = F.mse_loss(estimate, target)
        spectral = compressed_stft_loss(estimate, target)
        early, parts = endpoint_losses(estimate[..., :EARLY_SAMPLES],
                                       target[..., :EARLY_SAMPLES])
        total = waveform + self.stft_weight * spectral + early
        return total, {"waveform": float(waveform.detach()),
                       "compressed_stft": float(spectral.detach()), **parts}

    @torch.no_grad()
    def predict(self, observations: torch.Tensor,
                presence_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(observations, presence_mask)[0]


class WaveNetDirectRIRRegression(nn.Module):
    """The frozen ``DirectRIRWaveNet`` backbone under the same regression loss.

    This exists so the hybrid comparison varies the architecture and nothing
    else: identical data, horizon, objective, optimiser and budget.
    """

    def __init__(self, k_max: int = 1, channels: int = 16, num_blocks: int = 13,
                 stft_weight: float = .25):
        super().__init__()
        from claprir.models.direct_rir_estimator import DirectRIRWaveNet
        self.net = DirectRIRWaveNet(k_max=k_max, channels=channels, num_blocks=num_blocks)
        self.k_max = k_max
        self.stft_weight = stft_weight

    def _forward(self, observations: torch.Tensor,
                 mask: torch.Tensor | None) -> torch.Tensor:
        batch, _, length = observations.shape
        if mask is None:
            mask = torch.ones(batch, self.k_max, device=observations.device)
        state = torch.zeros(batch, 1, length, device=observations.device)
        time = torch.ones(batch, device=observations.device)
        return self.net(state, observations, mask, time)

    def loss(self, target: torch.Tensor, observations: torch.Tensor,
             presence_mask: torch.Tensor | None = None, *,
             validity_mask: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, dict[str, float]]:
        if validity_mask is not None:
            raise NotImplementedError(
                "WaveNetDirectRIRRegression has no masked objective; training it with "
                "--validity-masking would silently ignore the mask")
        estimate = self._forward(observations, presence_mask)
        waveform = F.mse_loss(estimate, target)
        spectral = compressed_stft_loss(estimate, target)
        total = waveform + self.stft_weight * spectral
        return total, {"waveform": float(waveform.detach()),
                       "compressed_stft": float(spectral.detach())}

    @torch.no_grad()
    def predict(self, observations: torch.Tensor,
                presence_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self._forward(observations, presence_mask)


class HybridEarlyFlow(nn.Module):
    """Conditional flow on the early segment, deterministic late branch frozen.

    The targeted answer to "the first few milliseconds are still the weakest
    part". Everything the regression hybrid already does well is left alone:

    * the time-frequency stage is loaded from a trained regression checkpoint and
      **frozen**, so the late branch is bit-identical to the deterministic model;
    * only the early refiner changes, from a deterministic residual predictor to
      a conditional flow over ``h[:mixing_time]``.

    The flow is conditioned on the early observation and on the frozen stage's
    own early estimate, so it starts from the same information the deterministic
    refiner had. Parameterisation follows the repository convention
    ``h_t = t*h_0 + (1-t)*eps`` with target velocity ``h_0 - eps``.
    """

    def __init__(self, config: HybridConfig | None = None, stft_weight: float = .25,
                 flow_steps: int = 20):
        super().__init__()
        self.config = config = config or HybridConfig()
        if config.k_max > 1:
            raise ValueError(
                f"{type(self).__name__} is a single-observation arm; the "
                "multi-clap question is answered by the regression model")
        self.backbone = HybridDirectRIR(config)
        # y_early, coarse_early, and the noisy state.
        self.refiner = UNet1D(
            in_channels=3, depth=config.unet_depth, emb_dim=config.unet_emb_dim,
            channels=config.unet_channels, strides=config.unet_strides,
            use_norm=config.unet_use_norm, time_conditional=True)
        self.stft_weight = stft_weight
        self.flow_steps = flow_steps

    def load_regression_backbone(self, state_dict: dict) -> None:
        """Take the time-frequency stage from a trained regression hybrid, freeze it.

        The deterministic refiner in that checkpoint is discarded -- it is the
        component being replaced.
        """
        prefix = "net."
        weights = {key[len(prefix):]: value for key, value in state_dict.items()
                   if key.startswith(prefix)}
        missing = self.backbone.load_state_dict(weights, strict=False)
        if any(k.startswith("tf_stage") for k in missing.missing_keys):
            raise RuntimeError(f"checkpoint has no time-frequency stage: {missing}")
        for parameter in self.backbone.tf_stage.parameters():
            parameter.requires_grad_(False)
        self.backbone.tf_stage.eval()

    @torch.no_grad()
    def coarse(self, observations: torch.Tensor,
               mask: torch.Tensor | None = None) -> torch.Tensor:
        self.backbone.tf_stage.eval()
        return self.backbone.time_frequency_stage(observations, mask)

    def _condition(self, observations: torch.Tensor, coarse: torch.Tensor,
                   state: torch.Tensor) -> torch.Tensor:
        edge = self.config.mixing_time_samples
        return torch.cat((observations[:, :1, :edge], coarse[:, :, :edge], state), dim=1)

    def loss(self, target: torch.Tensor, observations: torch.Tensor,
             presence_mask: torch.Tensor | None = None, *,
             validity_mask: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, dict[str, float]]:
        if validity_mask is not None:
            raise NotImplementedError(
                "HybridEarlyFlow has no masked objective; training it with "
                "--validity-masking would silently ignore the mask")
        edge = self.config.mixing_time_samples
        coarse = self.coarse(observations, presence_mask)
        start = target[..., :edge]
        noise = torch.randn_like(start)
        time = torch.rand(target.shape[0], device=target.device)
        state = time[:, None, None] * start + (1 - time[:, None, None]) * noise
        velocity = self.refiner(self._condition(observations, coarse, state), time)
        primary = F.mse_loss(velocity, start - noise)
        # Compressed-STFT on the implied endpoint, matching the regression
        # objective's structure so the arms differ only in the objective itself.
        implied = state + (1 - time[:, None, None]) * velocity
        spectral = compressed_stft_loss(implied, start)
        total = primary + self.stft_weight * spectral
        return total, {"velocity": float(primary.detach()),
                       "compressed_stft": float(spectral.detach())}

    @torch.no_grad()
    def predict(self, observations: torch.Tensor,
                presence_mask: torch.Tensor | None = None,
                steps: int | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        edge = self.config.mixing_time_samples
        steps = steps or self.flow_steps
        coarse = self.coarse(observations, presence_mask)
        shape = (observations.shape[0], 1, edge)
        state = torch.randn(shape, device=observations.device, generator=generator)
        step = 1 / steps
        for index in range(steps):
            time = torch.full((observations.shape[0],), index * step,
                              device=observations.device)
            state = state + step * self.refiner(
                self._condition(observations, coarse, state), time)
        return torch.cat((state, coarse[..., edge:]), dim=-1)


class HybridDirectRIRFlow(nn.Module):
    """The whole hybrid as one conditional flow -- the unrestricted control arm.

    Both stages predict velocity and both see the noisy state, so this varies the
    objective everywhere rather than only on the early segment. It exists so that
    an early-only win can be attributed to sparse early structure rather than to
    the flow objective in general.
    """

    def __init__(self, config: HybridConfig | None = None, stft_weight: float = .25,
                 flow_steps: int = 20):
        super().__init__()
        self.config = config = config or HybridConfig()
        if config.k_max > 1:
            raise ValueError(
                f"{type(self).__name__} is a single-observation arm; the "
                "multi-clap question is answered by the regression model")
        # One complex plane for the observation, one for the noisy state.
        self.tf_stage = NCSNpp(
            input_channels=4, spatial_channels=1, time_conditional=True,
            nf=config.nf, ch_mult=config.ch_mult, num_res_blocks=config.num_res_blocks,
            attn_resolutions=(0,), image_size=config.n_fft // 2 + 1, fir=False)
        self.refiner = UNet1D(
            in_channels=3, depth=config.unet_depth, emb_dim=config.unet_emb_dim,
            channels=config.unet_channels, strides=config.unet_strides,
            use_norm=config.unet_use_norm, time_conditional=True)
        self.helper = HybridDirectRIR(config)   # reused only for stft/istft
        self.stft_weight = stft_weight
        self.flow_steps = flow_steps

    def velocity(self, state: torch.Tensor, observations: torch.Tensor,
                 time: torch.Tensor) -> torch.Tensor:
        edge = self.config.mixing_time_samples
        spec = self.helper.stft(torch.cat((observations[:, :1], state), dim=1))
        # The exponent comes from the config, not from the module constant.
        # This is the flow arm's only compression site and it read the constant,
        # while HybridDirectRIR.time_frequency_stage read the config -- so
        # --input-compression moved the regression path and silently left the
        # flow path at 2/3. The representation ablation trains `hybrid_flow`,
        # which made both of its arms the same experiment.
        exponent = getattr(self.config, "input_compression", COMPRESSION_EXPONENT)
        compressed = ((spec.abs() + 1e-8) ** exponent
                      * torch.exp(1j * spec.angle()))
        coarse = self.helper.istft(self.tf_stage(compressed, time),
                                   self.config.signal_length)
        early = coarse[:, :, :edge] + self.refiner(
            torch.cat((observations[:, :1, :edge], coarse[:, :, :edge],
                       state[:, :, :edge]), dim=1), time)
        return torch.cat((early, coarse[:, :, edge:]), dim=-1)

    def loss(self, target: torch.Tensor, observations: torch.Tensor,
             presence_mask: torch.Tensor | None = None, *,
             validity_mask: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, dict[str, float]]:
        # This class refuses k_max > 1, so a multi-clap presence mask cannot be
        # meaningful here. It used to accept the validity mask in that same
        # positional slot, which is exactly the silent overload this signature
        # exists to end -- so being handed one is now an error rather than a
        # quietly different experiment.
        if presence_mask is not None:
            raise ValueError(
                f"{type(self).__name__} takes no presence mask; pass the "
                "temporal validity mask as validity_mask=")
        mask = validity_mask
        noise = torch.randn_like(target)
        time = torch.rand(target.shape[0], device=target.device)
        state = time[:, None, None] * target + (1 - time[:, None, None]) * noise
        velocity = self.velocity(state, observations, time)
        # `mask` here is a per-sample VALIDITY mask with the target's shape, not
        # the multi-clap presence mask the other arms take: this class refuses
        # k_max > 1, so it can never receive one of those. 1 where the target is
        # real, 0 past the end of the source recording.
        #
        # Averaged over the mask, not over the window. Zeroing the residual and
        # dividing by the full length would still tell the model that the absent
        # tail is worth fitting -- it would just weight the lesson by how much of
        # the record is missing, which is worse than either extreme.
        if mask is not None and mask.shape[-1] == target.shape[-1]:
            m = mask if mask.dim() == target.dim() else mask[:, None]
            square = (velocity - (target - noise)) ** 2
            # PER EXAMPLE, then average over the batch. Summing over the whole
            # batch and dividing by the total mask would weight each example by
            # how much of it survived, so a 560 ms MIT record would carry 0.56x
            # the gradient of a full-length BUT one purely because its recording
            # is shorter -- silently down-weighting exactly the providers the
            # mask exists to handle honestly.
            kept = m.sum(dim=(-2, -1)).clamp_min(1.0)
            primary = ((square * m).sum(dim=(-2, -1)) / kept).mean()
        else:
            primary = F.mse_loss(velocity, target - noise)
        total = primary
        parts = {"velocity": float(primary.detach())}
        # Both spectral terms score the endpoint the current velocity implies,
        # so it is computed once, and only when something reads it: at
        # stft_weight=0 with no mrstft the pure flow-matching arm computes the
        # identical graph a model without either term would, rather than an
        # equal one.
        if self.stft_weight or self.config.spectral_weight:
            implied = state + (1 - time[:, None, None]) * velocity
        # Guarded rather than multiplied by zero, the same way the term below is.
        if self.stft_weight:
            # Identical code path, identical n_fft/hop/window/eps -- only the
            # exponent moves. That is what makes this a one-variable change from
            # the arm it is compared against.
            #
            # When a validity mask is present the auxiliary obeys it too. It has
            # to: the primary term already refuses to be supervised on padding,
            # and an auxiliary that still is would reintroduce the same defect
            # through a second door -- worst of all at 1 s, where the padded
            # region is where the measured energy deficit is largest
            # (reports/flow_1s_pilot). At an all-ones mask the masked form
            # reduces to the unmasked one, which is a test, not a claim.
            exponent = getattr(self.config, "spectral_exponent",
                               COMPRESSION_EXPONENT)
            if mask is not None and mask.shape[-1] == target.shape[-1]:
                spectral = masked_compressed_stft_loss(
                    implied, target, mask, exponent=exponent)
            else:
                spectral = compressed_stft_loss(implied, target, exponent=exponent)
            total = total + self.stft_weight * spectral
            parts["compressed_stft"] = float(spectral.detach())
        # Step 4A. Guarded rather than multiplied by zero so an arm at the frozen
        # default computes the identical graph it always did.
        if self.config.spectral_weight:
            mrstft = multires_stft_loss(implied, target)
            total = total + self.config.spectral_weight * mrstft
            parts["mrstft"] = float(mrstft.detach())
        return total, parts

    @torch.no_grad()
    def predict(self, observations: torch.Tensor,
                presence_mask: torch.Tensor | None = None,
                steps: int | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        steps = steps or self.flow_steps
        shape = (observations.shape[0], 1, self.config.signal_length)
        state = torch.randn(shape, device=observations.device, generator=generator)
        step = 1 / steps
        for index in range(steps):
            time = torch.full((observations.shape[0],), index * step,
                              device=observations.device)
            state = state + step * self.velocity(state, observations, time)
        return state


class LateSetDecoder(nn.Module):
    """DeepSets-style aggregation over claps, acting only on the late segment.

    Permutation-invariant by construction: each clap is encoded independently,
    the codes are mean-pooled, and the decoder never sees clap order or count.
    That matters because the alternative -- concatenating observations on the
    channel axis -- fixes K at training time and makes the model sensitive to
    ordering, which is what the earlier channel-concat multi-clap attempts did.
    """

    def __init__(self, channels: int = 64, depth: int = 4, stride: int = 4,
                 in_channels: int = 2):
        super().__init__()
        encoder, width = [], in_channels
        for level in range(depth):
            out = channels * min(2 ** level, 4)
            encoder += [nn.Conv1d(width, out, 9, stride=stride, padding=4),
                        nn.GroupNorm(min(8, out), out), nn.SiLU()]
            width = out
        self.encoder = nn.Sequential(*encoder)
        decoder = []
        for level in reversed(range(depth)):
            out = channels * min(2 ** level, 4) if level else channels
            decoder += [nn.ConvTranspose1d(width, out, 9, stride=stride,
                                           padding=4, output_padding=stride - 1),
                        nn.GroupNorm(min(8, out), out), nn.SiLU()]
            width = out
        self.decoder = nn.Sequential(*decoder)
        # Zero-initialised head, so the model starts as an exact identity on the
        # aggregated late estimate, so training starts from the established
        # baseline rather than from an arbitrary output. That is a safety
        # initialisation, NOT a guarantee: F_theta0 == L0 says nothing about
        # F_theta after training, and the optimiser is free to make it worse.
        # Any claim that this arm beats L0 must come from the gate, not here.
        self.head = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, observations: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        """``[B, K, T]`` late observations and coarse estimates -> ``[B, 1, T]``."""
        batch, claps, length = observations.shape
        stacked = torch.stack((observations, coarse), dim=2).reshape(
            batch * claps, 2, length)
        codes = self.encoder(stacked)
        pooled = codes.reshape(batch, claps, *codes.shape[1:]).mean(dim=1)
        decoded = self.decoder(pooled)
        residual = self.head(decoded)[..., :length]
        return coarse.mean(dim=1, keepdim=True) + residual


class HybridSetLate(nn.Module):
    """Frozen per-clap early mean + a set-conditioned late branch.

    The named residual from ``multiclap_direct``: independent predictions agree
    early (r = 0.79) and disagree late (r = 0.31), so averaging cancels the tail.
    Averaging is kept where it works and replaced only where it does not.

    Early output is the plain mean of the frozen model's per-clap early
    estimates; late output comes from a decoder that sees all claps jointly, so
    it can share decay structure instead of cancelling it.
    """

    def __init__(self, config: HybridConfig | None = None, stft_weight: float = .25,
                 channels: int = 64):
        super().__init__()
        self.config = config = config or HybridConfig()
        self.backbone = HybridDirectRIR(config)
        self.late = LateSetDecoder(channels=channels)
        self.stft_weight = stft_weight

    def load_regression_backbone(self, state_dict: dict) -> None:
        prefix = "net."
        weights = {k[len(prefix):]: v for k, v in state_dict.items()
                   if k.startswith(prefix)}
        missing = self.backbone.load_state_dict(weights, strict=False)
        if any(k.startswith("tf_stage") for k in missing.missing_keys):
            raise RuntimeError(f"checkpoint has no time-frequency stage: {missing}")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    @torch.no_grad()
    def per_clap(self, observations: torch.Tensor) -> torch.Tensor:
        """Frozen single-clap estimates, one per clap: ``[B, K, T]``."""
        self.backbone.eval()
        batch, claps, length = observations.shape
        flat = observations.reshape(batch * claps, 1, length)
        estimate, _ = self.backbone(flat)
        return estimate.reshape(batch, claps, length)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        edge = self.config.mixing_time_samples
        estimates = self.per_clap(observations)
        early = estimates[:, :, :edge].mean(dim=1, keepdim=True)
        late = self.late(observations[:, :, edge:], estimates[:, :, edge:])
        return torch.cat((early, late), dim=-1)

    def loss(self, target: torch.Tensor, observations: torch.Tensor,
             presence_mask: torch.Tensor | None = None, *,
             validity_mask: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, dict[str, float]]:
        if validity_mask is not None:
            raise NotImplementedError(
                "HybridSetLate has no masked objective; training it with "
                "--validity-masking would silently ignore the mask")
        edge = self.config.mixing_time_samples
        estimate = self.forward(observations)
        # Supervise the late branch only; the early half is a frozen average and
        # carries no gradient, so including it would just add a constant.
        waveform = F.mse_loss(estimate[..., edge:], target[..., edge:])
        spectral = compressed_stft_loss(estimate[..., edge:], target[..., edge:])
        # Energy term aimed at the actual defect: averaging loses late energy, and
        # a plain MSE is minimised by losing even more of it.
        predicted = torch.mean(estimate[..., edge:] ** 2, dim=-1)
        reference = torch.mean(target[..., edge:] ** 2, dim=-1)
        energy = F.mse_loss(torch.log(predicted + 1e-10), torch.log(reference + 1e-10))
        total = waveform + self.stft_weight * spectral + .1 * energy
        return total, {"waveform": float(waveform.detach()),
                       "compressed_stft": float(spectral.detach()),
                       "late_energy": float(energy.detach())}

    @torch.no_grad()
    def predict(self, observations: torch.Tensor,
                presence_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(observations)
