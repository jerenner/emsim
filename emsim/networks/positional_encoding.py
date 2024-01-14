from typing import Optional

import torch
from torch import Tensor, nn


class PositionalEncodingFourier(nn.Module):
    # https://arxiv.org/pdf/2106.02795.pdf
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        normalize: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        assert out_dim % 2 == 0
        self.out_dim = out_dim
        self.normalize = normalize

        self.weight = nn.Parameter(
            torch.zeros([in_dim, out_dim // 2], dtype=torch.float)
        )
        self.reset_parameters()
        self._scaling = self.out_dim ** (-0.5)

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, positions: Tensor, input_spatial_range: Optional[Tensor] = None):
        if self.normalize:
            assert input_spatial_range is not None
            positions = positions / input_spatial_range

        positions *= 2 * torch.pi
        proj = positions @ self.weight
        out = torch.cat([proj.sin(), proj.cos()], dim=-1)
        out *= self._scaling
        return out
