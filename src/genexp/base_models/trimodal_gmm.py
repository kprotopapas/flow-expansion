"""1D trimodal GMM base model: large central mode at 0, smaller modes at ±1."""

import math
from typing import Any, Optional

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import nn

from diffusiongym.registry import base_model_registry
from diffusiongym.schedulers import OptimalTransportScheduler, Scheduler
from diffusiongym.types import DDTensor
from diffusiongym.utils import append_dims, train_base_model

from diffusiongym.base_models.base import BaseModel


@base_model_registry.register("1d/trimodal_gmm")
class TrimodalGMMBaseModel(BaseModel[DDTensor]):
    """1D flow matching model trained on a trimodal GMM.

    Mixture: large mode N(0, 0.4) with weight 0.6, two smaller modes
    N(±1, 0.2) each with weight 0.2.
    """

    output_type = "velocity"

    def __init__(
        self,
        device: Optional[torch.device] = None,
        scheduler: Optional[Scheduler] = None,
        train_steps: int = 8_000,
    ):
        super().__init__(device)

        if device is None:
            device = torch.device("cpu")
        self.device = device

        if scheduler is None:
            scheduler = OptimalTransportScheduler()
        self._scheduler = scheduler

        # Trimodal GMM: weight 0.6 at 0, weight 0.2 each at ±1
        p1 = dist.MixtureSameFamily(
            dist.Categorical(torch.tensor([0.6, 0.2, 0.2])),
            dist.Normal(
                torch.tensor([0.0, 1.0, -1.0]),
                torch.tensor([0.4, 0.2, 0.2]),
            ),
        )
        data = [DDTensor(p1.sample((4096, 1)).to(device))]

        self.net = MLP(1, 1).to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        train_base_model(self, opt, data, steps=train_steps, batch_size=512, pbar=True)

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[DDTensor, dict[str, Any]]:
        return DDTensor(torch.randn(n, 1, device=self.device)), kwargs

    def forward(self, x: DDTensor, t: torch.Tensor, **kwargs: Any) -> DDTensor:
        return DDTensor(self.net(x.data, t))


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_dim: int = 128,
        cond_dim: int = 256,
        depth: int = 3,
        width: int = 256,
        window_size: float = 1000.0,
        t_mult: float = 1000.0,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim, cond_dim, window_size, t_mult)
        blocks = [Block(in_dim, width, cond_dim)]
        for _ in range(depth - 1):
            blocks.append(Block(width, width, cond_dim))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(width, out_dim, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = self.time_embed(t)
        x_ = x.flatten(start_dim=t.ndim)
        for block in self.blocks:
            x_ = block(x_, cond)
        return self.head(x_).reshape(*x.shape[:-1], -1)


class Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.film = FiLM(out_dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return F.silu(self.film(self.norm(self.linear(x)), cond))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        hidden_dim: int = 256,
        window_size: float = 1000.0,
        t_mult: float = 1000.0,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.t_mult = t_mult
        half = dim // 2
        freqs = torch.exp(-math.log(window_size) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float() * self.t_mult
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(-2)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        result: torch.Tensor = self.mlp(emb)
        return result


class FiLM(nn.Module):
    def __init__(self, n_channels: int, cond_dim: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * n_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)
        gamma = append_dims(gamma, x.ndim)
        beta = append_dims(beta, x.ndim)
        return x * (1 + gamma) + beta
