from typing import Optional

import torch
from torch import Tensor, nn


class AbsolutePositionalEncodingFourier(nn.Module):
    # https://arxiv.org/pdf/2106.02795.pdf
    def __init__(
        self,
        in_features: int,
        out_features: int,
        normalize: bool = True,
    ):
        super().__init__()
        self.in_dim = in_features
        assert out_features % 2 == 0
        self.out_dim = out_features
        self.normalize = normalize

        self.weight = nn.Parameter(
            torch.zeros([in_features, out_features // 2], dtype=torch.float)
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


class RelativePositionalEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        features: int,
        max_distance: int,
    ):
        super().__init__()
        self.dim = dim
        self.features = features
        self.max_distance = max_distance

        param_shape = [max_distance] * dim + [features]
        self.weight = nn.Parameter(torch.zeros(param_shape), dtype=torch.float)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: Tensor):
        assert x.ndim == self.dim + 1
        raise NotImplementedError
