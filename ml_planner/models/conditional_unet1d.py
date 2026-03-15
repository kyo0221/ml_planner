import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -scale)
        emb = x.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int, n_groups: int):
        super().__init__()
        padding = kernel_size // 2
        groups = min(n_groups, out_channels)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.cond_proj(cond).unsqueeze(-1)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.residual(x)


class Downsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUNet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (64, 128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.down_dims = tuple(down_dims)

        cond_dim = global_cond_dim + diffusion_step_embed_dim
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
            nn.SiLU(),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
        )

        self.input_proj = nn.Conv1d(input_dim, self.down_dims[0], kernel_size=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx, dim in enumerate(self.down_dims):
            self.down_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim, dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim, dim, cond_dim, kernel_size, n_groups),
            ]))
            if idx < len(self.down_dims) - 1:
                self.downsamples.append(Downsample1D(dim, self.down_dims[idx + 1]))

        mid_dim = self.down_dims[-1]
        self.mid_blocks = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for idx in range(len(self.down_dims) - 1, 0, -1):
            in_dim = self.down_dims[idx]
            out_dim = self.down_dims[idx - 1]
            self.upsamples.append(Upsample1D(in_dim, out_dim))
            self.up_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(out_dim * 2, out_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(out_dim, out_dim, cond_dim, kernel_size, n_groups),
            ]))

        self.output_norm = nn.GroupNorm(min(n_groups, self.down_dims[0]), self.down_dims[0])
        self.output_proj = nn.Conv1d(self.down_dims[0], input_dim, kernel_size=1)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        x = sample.transpose(1, 2)
        x = self.input_proj(x)

        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.ndim == 1 and timestep.shape[0] == 1 and global_cond.shape[0] > 1:
            timestep = timestep.expand(global_cond.shape[0])

        time_emb = self.time_encoder(timestep)
        cond = torch.cat((time_emb, global_cond), dim=1)

        skips = []
        for idx, blocks in enumerate(self.down_blocks):
            x = blocks[0](x, cond)
            x = blocks[1](x, cond)
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)

        x = self.mid_blocks[0](x, cond)
        x = self.mid_blocks[1](x, cond)

        for idx, blocks in enumerate(self.up_blocks):
            x = self.upsamples[idx](x)
            skip = skips[-(idx + 2)]
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
            x = torch.cat((x, skip), dim=1)
            x = blocks[0](x, cond)
            x = blocks[1](x, cond)

        x = self.output_norm(x)
        x = F.silu(x)
        x = self.output_proj(x)
        return x.transpose(1, 2)
