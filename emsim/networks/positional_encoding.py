from typing import Optional
from contextlib import ExitStack

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FourierEncoding(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_dim = in_features
        assert out_features % 2 == 0
        self.out_dim = out_features

        self.weight = nn.Linear(in_features, out_features // 2, bias=False)
        self._scaling = self.out_dim ** (-0.5)

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
