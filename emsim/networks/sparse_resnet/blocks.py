import math
from typing import Optional

import numpy as np
import spconv.pytorch as spconv
import torch
from spconv.pytorch import functional as Fsp
from timm.layers import DropPath, get_padding, make_divisible
from torch import nn


class SparseBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        conv_type: str,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        base_width: int = 64,
        reduce_first: int = 1,
        dilation: int = 1,
        first_dilation: int = 1,
        act_layer: Optional[nn.Module] = spconv.SparseReLU,
        norm_layer: Optional[nn.Module] = spconv.SparseBatchNorm,
    ):
        assert ("sub" in conv_type) ^ ("sparse" in conv_type)
        if "sub" in conv_type:
            conv_layer = spconv.SubMConv2d
        elif "sparse" in conv_type:
            conv_layer = spconv.SparseConv2d
        else:
            raise ValueError(
                f"Expected either `sub` or `sparse` as `conv_type`, got {conv_type}"
            )
        super().__init__()


class SparseBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        conv_type: str,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        base_width: int = 64,
        reduce_first: int = 1,
        dilation: int = 1,
        first_dilation: int = 1,
        act_layer: Optional[nn.Module] = spconv.SparseReLU,
        norm_layer: Optional[nn.Module] = spconv.SparseBatchNorm,
    ):
        raise NotImplementedError("Not tested")
        assert ("sub" in conv_type) ^ ("sparse" in conv_type)
        if "sub" in conv_type:
            conv_layer = spconv.SubMConv2d
        elif "sparse" in conv_type:
            conv_layer = spconv.SparseConv2d
        else:
            raise ValueError(
                f"Expected either `sub` or `sparse` as `conv_type`, got {conv_type}"
            )
        super().__init__()

        width = int(math.floor(planes * (base_width / 64)))
        first_planes = width // reduce_first
        outplanes = planes * self.expansion

        self.conv1 = conv_layer(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = conv_layer(
            first_planes,
            width,
            kernel_size=3,
            stride=stride,
            padding=first_dilation,
            dilation=first_dilation,
            bias=False,
        )
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)

        self.conv3 = conv_layer(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x = x + shortcut
        x = self.act3(x)

        return x


class SparseBottleneckV2(spconv.SparseModule):
    def __init__(
        self,
        stage_index: int,
        block_index: int,
        in_chs: int,
        out_chs: int = None,
        bottle_ratio: float = 0.25,
        stride: int = 1,
        dilation: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None,
        drop_path_rate: float = 0.0,
        in_reduction: Optional[int] = None,
        out_reduction: Optional[int] = None,
    ):
        super().__init__()
        self.stage_index = stage_index
        self.block_index = block_index
        self.in_reduction = in_reduction or 2 ** stage_index
        self.out_reduction = out_reduction or self.in_reduction
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        out_chs = out_chs or in_chs
        # mid_chs = make_divisible(out_chs * bottle_ratio)
        mid_chs = int(out_chs * bottle_ratio)

        if stride > 1:
            conv_layer = spconv.SparseConv2d
            indice_key_3x3 = f"down_3x3_{in_reduction}to{out_reduction}"
        else:
            conv_layer = spconv.SubMConv2d
            indice_key_3x3 = f"3x3_{in_reduction}"
        self.indice_key_3x3 = indice_key_3x3

        if in_chs != out_chs or np.prod(stride) > 1:
            assert block_index == 0
            self.downsample_shortcut = conv_layer(
                in_chs,
                out_chs,
                kernel_size=3,
                stride=stride,
                dilation=dilation,
                padding=get_padding(3, stride=stride, dilation=dilation),
                indice_key=indice_key_3x3,
            )
        else:
            assert block_index > 0
            self.downsample_shortcut = None

        norm1 = norm_layer(in_chs)
        act1 = act_layer()
        self.preact = spconv.SparseSequential(norm1, act1)
        self.conv1 = spconv.SubMConv2d(
            in_chs,
            mid_chs,
            1,
            indice_key=f"1x1_subm_{in_reduction}"
        )

        self.norm_relu_conv_2 = spconv.SparseSequential(
            norm_layer(mid_chs),
            act_layer(),
            conv_layer(
                mid_chs,
                mid_chs,
                kernel_size=3,
                stride=stride,
                dilation=dilation,
                padding=get_padding(3, stride=stride, dilation=dilation),
                indice_key=indice_key_3x3,
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

    def forward(self, x):
        x_preact = self.preact(x)

        if self.downsample_shortcut is not None:
            shortcut = self.downsample_shortcut(x_preact)
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


class SparseResnetV2Stage(spconv.SparseModule):
    def __init__(
        self,
        stage_index: int,
        in_chs: int,
        out_chs: int,
        stride: int,
        dilation: int,
        depth: int,
        in_reduction: int,
        out_reduction: int,
        bottle_ratio: float = 0.25,
        block_dpr: Optional[list[float]] = None,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.stage_index = stage_index
        self.in_reduction = in_reduction
        self.out_reduction = out_reduction
        # first_dilation = 1 if dilation in (1, 2) else 2
        prev_chs = in_chs
        self.blocks = spconv.SparseSequential()
        for block_index in range(depth):
            drop_path_rate = block_dpr[block_index] if block_dpr else 0.0
            stride = stride if block_index == 0 else 1
            self.blocks.add_module(
                str(block_index),
                SparseBottleneckV2(
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
                    in_reduction=in_reduction if block_index == 0 else out_reduction,
                    out_reduction=out_reduction,
                ),
            )
            prev_chs = out_chs
            # first_dilation = dilation

    def forward(self, x):
        x = self.blocks(x)
        return x
