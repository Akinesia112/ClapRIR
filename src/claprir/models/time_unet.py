"""Time-domain 1-D U-Net used as the early-reflection refiner.

Adapted from https://github.com/01tot10/neural-tape-modeling
(``code/networks/unet_1d.py``, authors Eloi Moliner and Otto Mikkonen), which is
the network the hybrid architecture of Moliner et al., "Ambisonics Encoding of
Room Impulse Responses using a Device-Agnostic Diffusion Model" (Sec. III-B,
Fig. 1) refers to as the *time U-Net*.

Deviations from the upstream file, all of them mechanical:

* configuration is passed as keyword arguments instead of an OmegaConf ``args``
  namespace, and the module no longer takes a ``device`` argument;
* the input may carry several channels (here: the early segment of the
  observation stacked with the early segment of the time-frequency estimate),
  so the input pyramid is projected from ``in_channels`` rather than from one;
* the diffusion noise level is optional.  With ``time_conditional=False`` the
  FiLM path is driven by one learned embedding vector, which keeps the block
  structure identical to upstream while making the network a deterministic
  regressor.  With ``time_conditional=True`` it is driven by the upstream random
  Fourier-feature embedding of the flow time, restoring the original behaviour;
* ``forward`` takes and returns ``[B, C, T]`` instead of ``[B, T]``.

The upstream output pyramid is built from zero-initialised 1x1 convolutions, so
a freshly constructed ``UNet1D`` outputs all zeros.  The caller relies on this:
the refiner is added as a residual on top of the first-stage estimate, hence at
initialisation the hybrid model reproduces its time-frequency stage exactly.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class UNet1D(nn.Module):
    def __init__(self, in_channels: int = 2, depth: int = 5, emb_dim: int = 32,
                 channels: tuple[int, ...] = (32, 64, 64, 64, 64, 64),
                 strides: tuple[int, ...] = (2, 2, 2, 2, 2),
                 use_norm: bool = True, num_dilations: int = 8,
                 time_conditional: bool = False):
        super().__init__()
        if len(channels) != depth + 1:
            raise ValueError(f"channels must hold depth+1={depth + 1} entries")
        if len(strides) < depth:
            raise ValueError(f"strides must hold at least depth={depth} entries")
        self.depth = depth
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        Ns, Ss = channels, strides

        # Upstream conditions every ResnetBlock on a noise-level embedding.  A
        # regression model has no noise level, so one learned vector plays that
        # role and the FiLM layers stay untouched.  A flow model does have one,
        # and then the upstream RFF embedding is used instead.
        self.time_conditional = time_conditional
        if time_conditional:
            self.embedding = RFF_MLP_Block(emb_dim)
        else:
            self.embedding = nn.Parameter(torch.zeros(1, emb_dim))

        self.init_conv = nn.Conv1d(in_channels, Ns[0], 5, padding="same",
                                   padding_mode="zeros", bias=False)
        self.downs = nn.ModuleList()
        self.middle = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(depth):
            dim_in = Ns[i] if i == 0 else Ns[i - 1]
            dim_out = Ns[i]
            if i < depth - 1:
                self.downs.append(nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, use_norm, emb_dim=emb_dim,
                                num_dils=num_dilations, bias=False),
                    Downsample(Ss[i]),
                    CombinerDown("sum", in_channels, dim_out, bias=False),
                ]))
            else:  # no downsampling in the last encoder level
                self.downs.append(nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, use_norm, emb_dim=emb_dim,
                                num_dils=num_dilations, bias=False),
                ]))
        self.middle.append(nn.ModuleList([
            ResnetBlock(Ns[depth], Ns[depth], use_norm, emb_dim=emb_dim,
                        num_dils=num_dilations, bias=False),
        ]))
        for i in range(depth - 1, -1, -1):
            dim_in = Ns[i] * 2
            dim_out = Ns[i] if i == 0 else Ns[i - 1]
            if i > 0:
                self.ups.append(nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, use_norm=use_norm, emb_dim=emb_dim,
                                num_dils=num_dilations, bias=False),
                    Upsample(Ss[i]),
                    CombinerUp("sum", 1, dim_out, bias=False),
                ]))
            else:  # no upsampling in the last decoder level
                self.ups.append(nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, use_norm=use_norm, emb_dim=emb_dim,
                                num_dils=num_dilations, bias=False),
                ]))
        self.cropconcat = CropConcatBlock()

    def forward(self, inputs: torch.Tensor,
                time: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {inputs.shape[1]}")
        if self.time_conditional:
            if time is None:
                raise ValueError("a time-conditional UNet1D requires the flow time")
            emb = self.embedding(time.reshape(-1, 1))
        else:
            emb = self.embedding.expand(inputs.shape[0], -1)
        pyr = inputs
        x = self.init_conv(inputs)

        hs = []
        for i, modules in enumerate(self.downs):
            if i < self.depth - 1:
                resnet, downsample, combiner = modules
                x = resnet(x, emb)
                hs.append(x)
                x = downsample(x)
                pyr = downsample(pyr)
                x = combiner(pyr, x)
            else:
                (resnet,) = modules
                x = resnet(x, emb)
                hs.append(x)

        for modules in self.middle:
            (resnet,) = modules
            x = resnet(x, emb)

        # Upstream reuses the combiner of the previous decoder level for the
        # final level; the channel counts agree, so the behaviour is kept.
        pyr, combiner = None, None
        for i, modules in enumerate(self.ups):
            j = self.depth - i - 1
            if j > 0:
                resnet, upsample, combiner = modules
                x = self.cropconcat(x, hs.pop())
                x = resnet(x, emb)
                pyr = combiner(pyr, x)
                x = upsample(x)
                pyr = upsample(pyr)
            else:
                (resnet,) = modules
                x = self.cropconcat(x, hs.pop())
                x = resnet(x, emb)
                pyr = combiner(pyr, x)
        if pyr.shape[-1] != inputs.shape[-1]:
            raise RuntimeError(
                f"output length {pyr.shape[-1]} != input length {inputs.shape[-1]}; "
                "the early-segment length must be divisible by the product of the strides")
        return pyr


class RFF_MLP_Block(nn.Module):
    """Random Fourier-feature embedding of the flow time (upstream, verbatim)."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.RFF_freq = nn.Parameter(16 * torch.randn([1, 16]), requires_grad=False)
        self.MLP = nn.ModuleList([nn.Linear(32, emb_dim), nn.Linear(emb_dim, emb_dim)])

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        table = 2 * np.pi * sigma * self.RFF_freq
        x = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, use_norm=False, groups=8, emb_dim=32,
                 num_dils=8, bias=True):
        super().__init__()
        self.bias = bias
        self.use_norm = use_norm
        self.num_layers = num_dils
        self.film = Film(dim, emb_dim, bias=bias)
        self.res_conv = (nn.Conv1d(dim, dim_out, 1, padding_mode="zeros", bias=bias)
                         if dim != dim_out else nn.Identity())
        if use_norm:
            self.gnorm = nn.GroupNorm(groups, dim)
        self.first_conv = nn.Sequential(nn.GELU(), nn.Conv1d(dim, dim_out, 1, bias=bias))
        self.H = nn.ModuleList(
            GatedResidualLayer(dim_out, 5, 2 ** i, bias=bias) for i in range(num_dils))

    def forward(self, x, emb):
        gamma, beta = self.film(emb)
        if self.use_norm:
            x = self.gnorm(x)
        x = x * gamma + beta if self.bias else x * gamma
        y = self.first_conv(x)
        for h in self.H:
            y = h(y)
        return (y + self.res_conv(x)) / 2 ** .5


class Film(nn.Module):
    def __init__(self, output_dim, emb_dim, bias=True):
        super().__init__()
        self.bias = bias
        self.output_layer = nn.Linear(emb_dim, (2 if bias else 1) * output_dim)

    def forward(self, encoding):
        encoding = self.output_layer(encoding).unsqueeze(-1)
        if self.bias:
            gamma, beta = torch.chunk(encoding, 2, dim=1)
            return gamma, beta
        return encoding, None


class GatedResidualLayer(nn.Module):
    def __init__(self, dim, kernel_size, dilation, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, dilation=dilation,
                              stride=1, padding="same", padding_mode="zeros", bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        return (x + self.conv(self.act(x))) / 2 ** .5


class Upsample(nn.Module):
    def __init__(self, S):
        super().__init__()
        # 2**12 is an arbitrary base rate; only the ratio matters for the latents.
        N = 2 ** 12
        self.resample = torchaudio.transforms.Resample(N, N * S)

    def forward(self, x):
        return self.resample(x)


class Downsample(nn.Module):
    def __init__(self, S):
        super().__init__()
        N = 2 ** 12
        self.resample = torchaudio.transforms.Resample(N, N // S)

    def forward(self, x):
        return self.resample(x)


class CombinerUp(nn.Module):
    def __init__(self, mode, Npyr, Nx, bias=True):
        super().__init__()
        if mode != "sum":
            raise NotImplementedError(mode)
        self.conv1x1 = nn.Conv1d(Nx, Npyr, 1, bias=bias)
        nn.init.constant_(self.conv1x1.weight, 0)

    def forward(self, pyr, x):
        x = self.conv1x1(x)
        if pyr is None:
            return x
        return (pyr[..., :x.shape[-1]] + x) / 2 ** .5


class CombinerDown(nn.Module):
    def __init__(self, mode, Nin, Nout, bias=True):
        super().__init__()
        if mode != "sum":
            raise NotImplementedError(mode)
        self.conv1x1 = nn.Conv1d(Nin, Nout, 1, bias=bias)

    def forward(self, pyr, x):
        return (self.conv1x1(pyr) + x) / 2 ** .5


class CropConcatBlock(nn.Module):
    def forward(self, down_layer, x):
        offset = (down_layer.shape[2] - x.shape[2]) // 2
        return torch.cat((down_layer[:, :, offset:x.shape[2] + offset], x), 1)
