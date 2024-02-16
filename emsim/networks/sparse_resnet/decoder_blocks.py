from typing import Optional

import numpy as np
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp
import torch
from timm.layers import get_padding, DropPath
from torch import nn


class InverseSparseBottleneckV2(spconv.SparseModule):
    def __init__(
        self,
        stage_index: int,
        block_index: int,
        in_chs: int,
        out_chs: int,
        in_reduction: int,
        out_reduction: Optional[int] = None,
        encoder_skip_chs: Optional[int] = None,
        bottle_ratio: float = 0.25,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None,
        drop_path_rate: float = 0.0,

    ):
        super().__init__()
        self.stage_index = stage_index
        self.block_index = block_index
        self.in_reduction = in_reduction
        self.out_reduction = out_reduction or self.in_reduction
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        out_chs = out_chs or in_chs
        # mid_chs = make_divisible(out_chs * bottle_ratio)
        mid_chs = int(out_chs * bottle_ratio)

        if block_index == 0:
            if out_reduction < in_reduction:
                assert encoder_skip_chs is not None
                self.upsample = spconv.SparseInverseConv2d(
                    in_chs, in_chs, 3, indice_key=f"down_3x3_{out_reduction}to{in_reduction}"
                )
            else:
                self.upsample = None
            conv1_in_chs = (
                in_chs + encoder_skip_chs if encoder_skip_chs is not None else in_chs
            )
        else:
            assert encoder_skip_chs is None
            self.upsample = None
            conv1_in_chs = in_chs

        if conv1_in_chs != out_chs:
            assert block_index == 0
            self.shortcut = spconv.SubMConv2d(
                conv1_in_chs,
                out_chs,
                1,
                indice_key=f"1x1_subm_{out_reduction}",
            )
        else:
            self.shortcut = None

        norm1 = norm_layer(conv1_in_chs)
        act1 = act_layer()
        self.preact = spconv.SparseSequential(norm1, act1)
        self.conv1 = spconv.SubMConv2d(
            conv1_in_chs,
            mid_chs,
            1,
            indice_key=f"1x1_subm_{out_reduction}",
        )

        self.norm_relu_conv_2 = spconv.SparseSequential(
            norm_layer(mid_chs),
            act_layer(),
            spconv.SubMConv2d(
                mid_chs,
                mid_chs,
                kernel_size=3,
                padding=get_padding(3, 1, 1),
                indice_key=f"3x3_{out_reduction}",
            ),
        )

        self.norm_relu_conv_3 = spconv.SparseSequential(
            norm_layer(mid_chs),
            act_layer(),
            spconv.SubMConv2d(
                mid_chs, out_chs, 1, indice_key=f"1x1_subm_{out_reduction}"
            ),
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(
        self,
        x: spconv.SparseConvTensor,
        x_skip: Optional[spconv.SparseConvTensor] = None,
    ):
        if self.upsample is not None:
            x = self.upsample(x)

        if x_skip is not None:
            assert x.spatial_shape == x_skip.spatial_shape
            assert x.indices.shape == x_skip.indices.shape
            x = x.replace_feature(torch.cat([x.features, x_skip.features], -1))

        x_preact = self.preact(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(x_preact)
        else:
            shortcut = x

        x = self.conv1(x_preact)
        x = self.norm_relu_conv_2(x)
        x = self.norm_relu_conv_3(x)
        x = self.drop_path(x)
        # if x.indices.shape != shortcut.indices.shape:
        #     out = Fsp.sparse_add(x, shortcut)
        # else:
        #     out = x + shortcut
        assert x.indices.shape == shortcut.indices.shape
        out = x + shortcut
        return out


class SparseInverseResnetV2Stage(spconv.SparseModule):
    def __init__(
        self,
        stage_index: int,
        in_chs: int,
        out_chs: int,
        depth: int,
        in_reduction: int,
        out_reduction: int,
        encoder_skip_chs: Optional[int] = None,
        bottle_ratio: float = 0.25,
        block_dpr: Optional[list[float]] = None,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.stage_index = stage_index
        prev_chs = in_chs
        self.blocks = nn.ModuleList()
        for block_index in range(depth):
            drop_path_rate = block_dpr[block_index] if block_dpr else 0.0
            encoder_skip_chs = encoder_skip_chs if block_index == 0 else None
            self.blocks.add_module(
                str(block_index),
                InverseSparseBottleneckV2(
                    stage_index,
                    block_index,
                    prev_chs,
                    out_chs,
                    in_reduction=in_reduction,
                    out_reduction=out_reduction,
                    encoder_skip_chs=encoder_skip_chs,
                    bottle_ratio=bottle_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop_path_rate=drop_path_rate,
                ),
            )
            prev_chs = out_chs

    def forward(self, x: spconv.SparseConvTensor, x_skip: spconv.SparseConvTensor):
        first = self.blocks[0]
        remainder = self.blocks[1:]
        x = first(x, x_skip)
        for block in remainder:
            x = block(x)
        return x
