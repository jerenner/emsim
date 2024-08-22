from typing import Optional
from contextlib import ExitStack

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RoPEEncoding2D(nn.Module):
    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        assert head_dim % 2 == 0
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.init_freq_params(n_heads, head_dim)
        self.reset_parameters()

    def forward(self, xq: Tensor, xk: Tensor, positions: Tensor):
        assert xq.shape[-1] == xk.shape[-1] == self.head_dim
        assert xq.shape[-2] == xk.shape[-2] == self.n_heads
        assert positions.shape[-1] == 2
        assert xq.shape == xk.shape
        assert all([i in (j, 1) for i, j in zip(positions.shape[:-1], xq.shape[:-1])])

        freqs_y, freqs_x = self.get_freqs()

        rot_y = freqs_y * positions[..., 0].unsqueeze(-1)
        rot_x = freqs_x * positions[..., 1].unsqueeze(-1)
        rot = torch.polar(torch.ones_like(rot_x), rot_x + rot_y)

        xq_ = torch.view_as_complex(xq.view(*xq.shape[:-1], self.head_dim // 2, 2))
        xk_ = torch.view_as_complex(xk.view(*xk.shape[:-1], self.head_dim // 2, 2))
        xq_out = torch.view_as_real(xq_ * rot).flatten(-2)
        xk_out = torch.view_as_real(xk_ * rot).flatten(-2)
        return xq_out, xk_out


class RoPEEncoding2DWithBaseFreq(RoPEEncoding2D):
    def init_freq_params(self, n_heads, *args):
        self.base_theta_y = nn.Parameter(torch.empty(n_heads))
        self.base_theta_x = nn.Parameter(torch.empty(n_heads))

    def get_freqs(self):
        freqs_y = 1.0 / (
            self.base_theta_y.unsqueeze(-1)
            ** (
                torch.arange(0, self.head_dim, 2, device=self.base_theta_y.device)[
                    : (self.head_dim // 2)
                ].float()
                / self.head_dim
            )
        )
        freqs_x = 1.0 / (
            self.base_theta_x.unsqueeze(-1)
            ** (
                torch.arange(0, self.head_dim, 2, device=self.base_theta_x.device)[
                    : (self.head_dim // 2)
                ].float()
                / self.head_dim
            )
        )
        return freqs_y, freqs_x

    def reset_parameters(self):
        nn.init.trunc_normal_(self.base_theta_y, 10000.0, 1000.0, a=1e-5, b=1e5)
        nn.init.trunc_normal_(self.base_theta_x, 10000.0, 1000.0, a=1e-5, b=1e5)


class RoPEEncoding2DFullLearnableFreqs(RoPEEncoding2D):
    def init_freq_params(self, n_heads, head_dim):
        self.theta_y = nn.Parameter(torch.empty(n_heads, head_dim // 2))
        self.theta_x = nn.Parameter(torch.empty(n_heads, head_dim // 2))

    def get_freqs(self):
        return self.theta_y, self.theta_x

    def reset_parameters(self):
        mag = 1.0 / (
            10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        theta_y_scaler = torch.randn(self.n_heads, 1).exp()
        theta_x_scaler = torch.randn(self.n_heads, 1).exp()
        with torch.no_grad():
            self.theta_y.copy_(mag * theta_y_scaler)
            self.theta_x.copy_(mag * theta_x_scaler)


class RoPEEncodingND(nn.Module):
    def __init__(self, n_dims: int, n_heads: int, head_dim: int):
        super().__init__()
        assert head_dim % 2 == 0
        self.n_dims = n_dims
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.init_freq_params()
        self.reset_paramters()

    def forward(self, xq: Tensor, xk: Tensor, positions: Tensor):
        assert xq.shape[-1] == xk.shape[-1] == self.head_dim
        assert xq.shape[-2] == xk.shape[-2] == self.n_heads
        assert positions.shape[-1] == self.n_dims
        assert xq.shape == xk.shape
        assert all([i in (j, i) for i, j in zip(positions.shape[:-1], xq.shape[:-1])])

        freqs = self.get_freqs()

        raise NotImplementedError


class FourierEncoding(nn.Module):
    def __init__(self, in_features, out_features, dtype=torch.float):
        super().__init__()
        self.in_dim = in_features
        assert out_features % 2 == 0
        self.out_dim = out_features

        self.weight = nn.Linear(in_features, out_features // 2, bias=False, dtype=dtype)
        self._scaling = 1000

    def reset_parameters(self):
        self.weight.reset_parameters()

    def forward(self, positions: Tensor):
        positions = positions * 2 * torch.pi
        proj = self.weight(positions)
        out = torch.cat([proj.sin(), proj.cos()], -1)
        out = out * self._scaling
        return out


class PixelPositionalEncoding(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.encoding = FourierEncoding(2, out_features)

    def forward(self, pixel_indices: Tensor, image_size: Tensor):
        positions = pixel_indices / image_size
        out = self.encoding(positions)
        return out


class SubpixelPositionalEncoding(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.encoding = FourierEncoding(2, out_features)

    def forward(self, subpixel_positions: Tensor):
        out = self.encoding(subpixel_positions)
        return out


# class PixelSubpixelAdditiveEncoding(nn.Module):
#     def __init__(self, out_features):
#         super().__init__()
#         self.pixel_encoding = PixelPositionalEncoding()
#         self.subpixel_encoding = SubpixelPositionalEncoding()

#     def forward(self, pixel_indices: Tensor, image_size: Tensor, subpixel_positions: Tensor):
#         positions =


class RelativePositionalEncodingTableInterpolate2D(nn.Module):
    def __init__(
        self,
        features: int,
        table_rows: int,
        table_columns: int,
    ):
        super().__init__()
        self.features = features
        self.table_rows = table_rows
        self.table_columns = table_columns

        param_shape = [features, table_rows, table_columns]
        self.rpe_table = nn.Parameter(torch.zeros(param_shape, dtype=torch.float))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.rpe_table)

    def forward(self, query_subpixel_positions: Tensor, key_relative_indices: Tensor):
        assert query_subpixel_positions.ndim == 2
        assert key_relative_indices.ndim == 3
        n_queries = query_subpixel_positions.shape[0]
        qk_fractional_offsets = (
            query_subpixel_positions.view(n_queries, 1, 1, 2)
            - 0.5
            + key_relative_indices.unsqueeze(0)
        )
        qk_fractional_offsets /= qk_fractional_offsets.new_tensor(
            [self.table_rows / 2, self.table_columns / 2]
        )
        with ExitStack() as stack:
            # https://github.com/pytorch/pytorch/issues/88380
            if qk_fractional_offsets.shape[0] >= 65536:
                stack.enter_context(torch.backends.cudnn.flags(enabled=False))
            out = F.grid_sample(
                self.rpe_table.unsqueeze(0).expand(n_queries, -1, -1, -1),
                qk_fractional_offsets,
                "bilinear",
                align_corners=True,
            )
        return out
