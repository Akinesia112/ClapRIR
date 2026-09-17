"""Wave 3: alternative time-frequency front-ends for the 2-D stage.

The hybrid analyses with one STFT (510-point, 128 hop) chosen once and never
compared against. That is a fixed 5.8 ms window at every frequency, which is a
compromise the signal does not actually justify: the direct arrival needs time
resolution finer than a window that long, while the modal structure of the late
tail needs frequency resolution finer than 86 Hz. A single linear-frequency
resolution cannot have both.

Two front-ends here, both **analysis-side only**:

``MultiResolutionSTFT``
    Analyses at several window lengths, resamples each to the primary grid, and
    stacks them as extra input channels.

``CQTAnalysis``
    A constant-Q filter bank -- geometrically spaced centre frequencies, constant
    fractional bandwidth -- giving fine frequency resolution at low frequencies
    where room modes are, and fine time resolution at high frequencies where the
    transient is.

**Synthesis always goes back through the verified iSTFT of the primary
resolution.** This is a deliberate restriction. An invertible NSGT would let the
network output in the new domain, but the iSTFT in this repository is the piece
that already carries a regression test (a frame-padding bug in it silently
attenuated the tail and invalidated a whole set of runs). Adding a second,
unverified synthesis path at the same time as a new analysis path would make an
unexplained result impossible to attribute. The extra resolutions therefore
condition the network; they do not reconstruct it.

Nothing here is on by default, so every frozen verdict in the programme stands.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 44_100


class MultiResolutionSTFT(nn.Module):
    """Several STFT resolutions on one common grid, as stacked channels.

    The primary resolution is the hybrid's own, so channel 0 of the output is
    bit-for-bit what the single-resolution model already sees and the ablation
    is strictly additive.
    """

    def __init__(self, n_fft: int = 510, hop_length: int = 128,
                 extra_n_fft: tuple[int, ...] = (254, 1022),
                 frame_multiple: int = 16):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.extra_n_fft = tuple(extra_n_fft)
        self.frame_multiple = frame_multiple
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        for size in self.extra_n_fft:
            self.register_buffer(f"window_{size}", torch.hann_window(size),
                                 persistent=False)

    @property
    def channels(self) -> int:
        return 1 + len(self.extra_n_fft)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """``[B, C, T]`` -> ``[B, C*(1+len(extra)), F, frames]`` complex."""
        from claprir.models.hybrid_rir_estimator import stft_frames
        primary = stft_frames(signal, self.n_fft, self.hop_length, self.window,
                              self.frame_multiple)
        planes = [primary]
        for size in self.extra_n_fft:
            # Same hop, so the frame axis already matches up to edge effects;
            # only the frequency axis has to be brought onto the primary grid.
            other = stft_frames(signal, size, self.hop_length,
                                getattr(self, f"window_{size}"), self.frame_multiple)
            planes.append(_resample_grid(other, primary.shape[-2:]))
        return torch.cat(planes, dim=1)


def _resample_grid(spec: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resample a complex spectrogram onto ``shape``.

    Real and imaginary parts are interpolated separately. Interpolating the
    complex pair directly would be wrong for a different reason than it looks:
    it is magnitude that survives, while phase -- which is what wraps -- is
    destroyed by any averaging. These planes are conditioning input, never
    inverted, so the loss is acceptable; it would not be for a synthesis path.
    """
    if tuple(spec.shape[-2:]) == tuple(shape):
        return spec
    batch, channels = spec.shape[:2]
    stacked = torch.stack((spec.real, spec.imag), dim=2).reshape(
        batch * channels * 2, 1, *spec.shape[-2:])
    resized = F.interpolate(stacked, size=shape, mode="bilinear", align_corners=False)
    resized = resized.reshape(batch, channels, 2, *shape)
    return torch.complex(resized[:, :, 0], resized[:, :, 1])


class CQTAnalysis(nn.Module):
    """Constant-Q analysis by frequency-domain filtering. Magnitude and phase.

    Implemented as a bank of Hann-shaped bandpass windows applied to the DFT of
    the whole signal, one output frame per hop by inverse-transforming each band
    and decimating. This is the naive route rather than a fast NSGT: at 250 ms
    and 44.1 kHz it costs one FFT plus one IFFT per band, which is negligible
    next to the network, and it keeps the implementation short enough to check
    against a chirp.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, f_min: float = 31.25,
                 bins_per_octave: int = 24, n_bins: int = 216,
                 hop_length: int = 128):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        # 216 bins at 24/octave from 31.25 Hz reaches 16 kHz. The first attempt
        # stopped at 192 bins / 7.8 kHz, which left real RIR content unanalysed.
        frequencies = f_min * 2 ** (np.arange(n_bins) / bins_per_octave)
        if frequencies[-1] >= sample_rate / 2:
            raise ValueError(
                f"top CQT bin {frequencies[-1]:.0f} Hz is at or above Nyquist; "
                f"reduce n_bins or bins_per_octave")
        # Constant Q: bandwidth proportional to centre frequency, so every band
        # spans the same number of octaves and the time/frequency trade-off
        # slides with frequency instead of being fixed.
        self.quality = 1 / (2 ** (1 / bins_per_octave) - 2 ** (-1 / bins_per_octave))
        self.register_buffer("frequencies",
                             torch.as_tensor(frequencies, dtype=torch.float32))

    def kernel(self, size: int, device) -> torch.Tensor:
        """``[n_bins, size//2+1]`` real filter bank on the rFFT grid."""
        grid = torch.fft.rfftfreq(size, 1 / self.sample_rate).to(device)
        centre = self.frequencies.to(device)[:, None]
        bandwidth = centre / self.quality
        offset = (grid[None] - centre) / bandwidth
        # Hann-shaped passband, zero outside, so the bank is smooth and compact.
        return torch.where(offset.abs() < 1, .5 * (1 + torch.cos(np.pi * offset)),
                           torch.zeros_like(offset))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """``[B, C, T]`` -> ``[B, C, n_bins, frames]`` complex."""
        batch, channels, length = signal.shape
        size = int(2 ** np.ceil(np.log2(length)))
        spectrum = torch.fft.rfft(signal.reshape(batch * channels, -1), size)
        bands = spectrum[:, None] * self.kernel(size, signal.device)[None]
        # Analytic (one-sided) reconstruction: the bank already discards negative
        # frequencies, so the inverse transform is complex and its magnitude is
        # the band envelope.
        analytic = torch.fft.ifft(
            F.pad(bands, (0, size - bands.shape[-1])), size)[..., :length]
        decimated = analytic[..., ::self.hop_length]
        return decimated.reshape(batch, channels, self.n_bins, -1)

    def magnitude(self, signal: torch.Tensor) -> torch.Tensor:
        return self(signal).abs()


def cqt_loss(estimate: torch.Tensor, target: torch.Tensor, analysis: CQTAnalysis,
             compression: float = .3) -> torch.Tensor:
    """Compressed constant-Q magnitude distance.

    Off by default everywhere. The programme rule is that no new loss is invented
    during validation, and every frozen verdict was produced without this one.
    """
    predicted = analysis.magnitude(estimate).clamp_min(1e-8) ** compression
    reference = analysis.magnitude(target).clamp_min(1e-8) ** compression
    return F.l1_loss(predicted, reference)
