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
        encoder_chs: Optional[int] = None,
        bottle_ratio: float = 0.25,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        out_chs = out_chs or in_chs
        # mid_chs = make_divisible(out_chs * bottle_ratio)
        mid_chs = int(out_chs * bottle_ratio)

        if in_chs != out_chs:
            assert encoder_chs is not None
            assert block_index == 0
            self.upsample_shortcut = spconv.SparseInverseConv2d(
                in_chs,
                out_chs,
                1,
                indice_key=f"down_1x1_{stage_index}",
            )
        else:
            self.upsample_shortcut = None

        norm1 = norm_layer(in_chs)
        act1 = act_layer()
        self.preact = spconv.SparseSequential(norm1, act1)
        self.conv1 = spconv.SubMConv2d(
            in_chs,
            mid_chs,
            1,
            indice_key=f"1x1_subm_{stage_index}",
        )

        if block_index == 0:
            conv2 = spconv.SparseInverseConv2d(
                mid_chs, mid_chs, 3, indice_key=f"down_3x3_{stage_index}"
            )
        else:
            conv2 = spconv.SubMConv2d(
                mid_chs,
                mid_chs,
                kernel_size=3,
                padding=get_padding(3, 1, 1),
                indice_key=f"3x3_{stage_index}",
            )
        self.norm_relu_conv_2 = spconv.SparseSequential(
            norm_layer(mid_chs),
            act_layer,
            conv2,
        )

        if encoder_chs is not None:
            mid_chs += encoder_chs
        self.norm_relu_conv_3 = spconv.SparseSequential(
            norm_layer(mid_chs),
            act_layer(),
            spconv.SubMConv2d(
                mid_chs, out_chs, 1, indice_key=f"1x1_subm_{stage_index}"
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
        x_preact = self.preact(x)
        if self.upsample_shortcut is not None:
            shortcut = self.upsample_shortcut(x_preact)
        else:
            shortcut = x

        x = self.conv1(x_preact)
        x = self.norm_relu_conv2(x)

        if x_skip is not None:
            assert x.spatial_shape == x_skip.spatial_shape
            assert x.indices.shape == x_skip.indices.shape
            x = x.replace_feature(torch.cat([x.features, x_skip.features], -1))

        x = self.norm_relu_conv3(x)
        x = self.drop_path(x)
        if x.indices.shape != shortcut.indices.shape:
            out = Fsp.sparse_add(x, shortcut)
        else:
            out = x + shortcut
        return out


class SparseInverseResnetV2Stage(spconv.SparseModule):
    def __init__(
        self,
        stage_index: int,
        in_chs: int,
        encoder_skip_chs: int,
        out_chs: int,
        stride: int,
        dilation: int,
        depth: int,
        bottle_ratio: float = 0.25,
        block_dpr: Optional[list[float]] = None,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        prev_chs = in_chs
        self.blocks = spconv.SparseSequential()
        for block_index in range(depth):
            drop_path_rate = block_dpr[block_index] if block_dpr else 0.0
            stride = stride if block_index == 0 else 1
            encoder_skip_chs = encoder_skip_chs if block_index == 0 else 1
            self.blocks.add_module(
                str(block_index),
                InverseSparseBottleneckV2(
                    stage_index,
                    block_index,
                    prev_chs,
                    out_chs,
                    bottle_ratio=bottle_ratio,
                    stride=stride,
                    dilation=dilation,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop_path_rate=drop_path_rate,
                ),
            )
            prev_chs = out_chs

    def forward(self, x: spconv.SparseConvTensor, x_skip: spconv.SparseConvTensor):
        x = self.blocks(x, x_skip)
        return x
