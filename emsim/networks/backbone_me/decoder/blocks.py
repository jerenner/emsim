from functools import partial
from typing import Optional

import numpy as np
import torch
from timm.layers import get_padding, DropPath
from torch import nn, Tensor
import MinkowskiEngine as ME


@torch.compiler.disable
class MinkowskiInverseSparseBottleneckV2(nn.Module):
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
        bias: bool = True,
        dimension: int = 2,
        act_layer: nn.Module = ME.MinkowskiReLU,
        norm_layer: nn.Module = ME.MinkowskiBatchNorm,
    ):
        super().__init__()
        self.stage_index = stage_index
        self.block_index = block_index
        self.in_reduction = in_reduction
        self.out_reduction = out_reduction or self.in_reduction
        if act_layer == ME.MinkowskiReLU:
            act_layer = partial(act_layer, inplace=True)
        out_chs = out_chs or in_chs
        # mid_chs = make_divisible(out_chs * bottle_ratio)
        mid_chs = int(out_chs * bottle_ratio)

        # encoder skip upsample block
        if block_index == 0:
            if out_reduction < in_reduction:
                assert encoder_skip_chs is not None
                self.upsample = ME.MinkowskiConvolutionTranspose(
                    in_chs,
                    in_chs,
                    kernel_size=3,
                    bias=bias,
                    stride=in_reduction // out_reduction,
                    dimension=dimension,
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

        # shortcut
        if conv1_in_chs != out_chs:
            assert block_index == 0
            self.shortcut = ME.MinkowskiConvolution(
                conv1_in_chs,
                out_chs,
                kernel_size=1,
                bias=bias,
                dimension=dimension,
            )
        else:
            self.shortcut = nn.Identity()

        # main path
        self.preact = nn.Sequential(
            norm_layer(conv1_in_chs),
            act_layer(),
        )
        self.conv1 = ME.MinkowskiConvolution(
            conv1_in_chs,
            mid_chs,
            kernel_size=1,
            bias=bias,
            dimension=dimension,
        )

        self.norm_act_conv_2 = nn.Sequential(
            norm_layer(mid_chs),
            act_layer(),
            ME.MinkowskiConvolution(
                mid_chs,
                mid_chs,
                kernel_size=3,
                bias=bias,
                dimension=dimension,
            ),
        )

        self.norm_act_conv_3 = nn.Sequential(
            norm_layer(mid_chs),
            act_layer(),
            ME.MinkowskiConvolution(
                mid_chs,
                out_chs,
                kernel_size=1,
                bias=bias,
                dimension=dimension,
            ),
        )

    def forward(
        self,
        x: ME.SparseTensor,
        x_skip: Optional[ME.SparseTensor] = None,
    ):
        if self.upsample is not None:
            x = self.upsample(x)

        if x_skip is not None:
            x = ME.cat(x, x_skip)

        x_preact = self.preact(x)
        shortcut = self.shortcut(x_preact)

        x = self.conv1(x_preact)
        x = self.norm_act_conv_2(x)
        x = self.norm_act_conv_3(x)
        assert torch.equal(x.coordinates, shortcut.coordinates)
        out = x + shortcut
        return out


class MinkowskiSparseInverseResnetV2Stage(nn.Module):
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
        bias: bool = True,
        dimension: int = 2,
        act_layer: nn.Module = ME.MinkowskiReLU,
        norm_layer: nn.Module = ME.MinkowskiBatchNorm,
    ):
        super().__init__()
        self.stage_index = stage_index
        prev_chs = in_chs
        self.blocks = nn.ModuleList()
        for block_index in range(depth):
            encoder_skip_chs = encoder_skip_chs if block_index == 0 else None
            self.blocks.add_module(
                str(block_index),
                MinkowskiInverseSparseBottleneckV2(
                    stage_index,
                    block_index,
                    prev_chs,
                    out_chs,
                    in_reduction=in_reduction,
                    out_reduction=out_reduction,
                    encoder_skip_chs=encoder_skip_chs,
                    bottle_ratio=bottle_ratio,
                    bias=bias,
                    dimension=dimension,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ),
            )
            prev_chs = out_chs

    def forward(self, x: ME.SparseTensor, x_skip: ME.SparseTensor):
        first = self.blocks[0]
        remainder = self.blocks[1:]
        x = first(x, x_skip)
        for block in remainder:
            x = block(x)
        return x
