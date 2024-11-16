from typing import Optional

import spconv.pytorch as spconv
import torch
from timm.layers import make_divisible
from torch import nn, Tensor

from .blocks import SparseResnetV2Stage
from ...utils.sparse_utils import torch_sparse_to_spconv

class SparseResnetV2(nn.Module):
    def __init__(
        self,
        layers: list[int],
        channels: list[int] = [32, 64, 128, 256],
        in_chans: int = 1,
        stem_channels: int = 16,
        output_stride: Optional[int] = None,
        act_layer: nn.Module = nn.ReLU,
        norm_layer: nn.Module = nn.BatchNorm1d,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.feature_info = []
        # stem_chs = make_divisible(stem_chs)
        self.stem = spconv.SubMConv2d(in_chans, stem_channels, 7)
        self.feature_info.append(dict(num_chs=stem_channels, reduction=1, module="stem"))
        if output_stride is None:
            output_stride = 2 ** len(layers)

        prev_chs = stem_channels
        curr_stride = 1
        dilation = 1
        block_dprs = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(layers)).split(tuple(layers))
        ]
        self.stages = nn.ModuleList()
        for stage_index, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            # out_chs = make_divisible(c)
            out_chs = c
            # stride = 1 if stage_index == 0 else 2
            stride = 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            stage = SparseResnetV2Stage(
                stage_index,
                prev_chs,
                out_chs,
                stride=stride,
                dilation=dilation,
                depth=d,
                in_reduction=curr_stride,
                out_reduction=curr_stride * stride,
                block_dpr=bdpr,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            prev_chs = out_chs
            curr_stride *= stride
            self.feature_info += [
                dict(
                    num_chs=prev_chs,
                    reduction=curr_stride,
                    module=f"stages.{stage_index}",
                )
            ]
            self.stages.append(stage)

        indice_keys = [stage.blocks[0].indice_key_3x3 for stage in self.stages]
        self.downsample_indice_keys = [key for key in indice_keys if "down" in key]
        self.num_features = prev_chs

    def forward(self, x):
        if isinstance(x, Tensor):
            assert x.is_sparse
            x = torch_sparse_to_spconv(x)
        out = []
        x = self.stem(x)
        out.append(x)
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out
