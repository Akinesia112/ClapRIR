"""Direct conditional flow matching for one or more clap observations -> RIR."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from claprir.models.spectrogram_network import DilatedResBlock, SinusoidalTimeEmbedding, TimeMLP, receptive_field


class DirectRIRWaveNet(nn.Module):
    def __init__(self, k_max: int = 5, channels: int = 16, num_blocks: int = 13,
                 t_embed_dim: int = 64, t_hidden_dim: int = 128):
        super().__init__()
        self.k_max = k_max
        self.time_emb = SinusoidalTimeEmbedding(t_embed_dim)
        self.time_mlp = TimeMLP(t_embed_dim, t_hidden_dim)
        # noisy target + K observations + K broadcast mask channels
        self.in_proj = nn.Conv1d(1 + 2 * k_max, channels, 1)
        self.dilations = [2 ** i for i in range(num_blocks)]
        self.receptive_field = receptive_field(self.dilations)
        self.blocks = nn.ModuleList(
            DilatedResBlock(channels, t_hidden_dim, dilation=d) for d in self.dilations)
        self.out_norm = nn.GroupNorm(min(8, channels), channels)
        self.out_proj = nn.Conv1d(channels, 1, 1)

    def forward(self, noisy_target: torch.Tensor, observations: torch.Tensor,
                mask: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if observations.shape[1] != self.k_max:
            raise ValueError(f"expected {self.k_max} padded observations")
        mask_channels = mask[:, :, None].expand_as(observations)
        hidden = self.in_proj(torch.cat((noisy_target, observations, mask_channels), 1))
        context = self.time_mlp(self.time_emb(time))
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.out_proj(F.silu(self.out_norm(hidden)))


class DirectRIRFlow(nn.Module):
    def __init__(self, net: DirectRIRWaveNet):
        super().__init__(); self.net = net

    def loss(self, target: torch.Tensor, observations: torch.Tensor,
             mask: torch.Tensor, objective: str = "plain_flow",
             return_components: bool = False):
        batch = target.shape[0]
        time = torch.rand(batch, device=target.device)
        noise = torch.randn_like(target)
        state = time[:, None, None] * target + (1 - time[:, None, None]) * noise
        velocity = self.net(state, observations, mask, time)
        error = (velocity - (target - noise)) ** 2
        if objective == "plain_flow":
            total = error.mean()
            components = {"weighted_velocity": float(total.detach()), "compressed_stft": 0.0}
            return (total, components) if return_components else total
        if objective != "energy_weighted_flow":
            raise ValueError(f"unknown flow objective: {objective}")
        relative_amplitude = target.abs() / (target.abs().amax(dim=-1, keepdim=True) + 1e-8)
        weight = 1 + 15 * torch.sqrt(relative_amplitude)
        velocity_loss = (error * weight).sum() / weight.sum()
        prediction = state + (1 - time[:, None, None]) * velocity
        n_fft = min(510, target.shape[-1]); n_fft -= n_fft % 2
        window = torch.hann_window(n_fft, device=target.device)
        target_stft = torch.stft(target[:, 0], n_fft, n_fft // 4, window=window, return_complex=True)
        prediction_stft = torch.stft(prediction[:, 0], n_fft, n_fft // 4, window=window, return_complex=True)
        spectral_loss = F.mse_loss((prediction_stft.abs() + 1e-8) ** .667,
                                   (target_stft.abs() + 1e-8) ** .667)
        total = velocity_loss + .1 * spectral_loss
        components = {"weighted_velocity": float(velocity_loss.detach()),
                      "compressed_stft": float(spectral_loss.detach())}
        return (total, components) if return_components else total

    @torch.no_grad()
    def sample(self, observations: torch.Tensor, mask: torch.Tensor, steps: int = 20,
               initial_noise: torch.Tensor | None = None) -> torch.Tensor:
        batch, _, length = observations.shape
        state = (torch.randn(batch, 1, length, device=observations.device)
                 if initial_noise is None else initial_noise.clone())
        step = 1 / steps
        for index in range(steps):
            time = torch.full((batch,), index * step, device=state.device)
            state += step * self.net(state, observations, mask, time)
        return state


def pad_observations(items: list[torch.Tensor], k_max: int,
                     randomize: bool = False, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.stack(items)
    if randomize and len(items) > 1:
        values = values[torch.randperm(len(items), generator=generator)]
    if len(items) > k_max:
        raise ValueError("more observations than k_max")
    length = values.shape[-1]
    padded = torch.zeros(k_max, length, dtype=values.dtype)
    padded[:len(items)] = values
    mask = torch.zeros(k_max, dtype=values.dtype); mask[:len(items)] = 1
    return padded, mask
