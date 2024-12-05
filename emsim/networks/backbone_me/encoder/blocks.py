import torch
from torch import nn, Tensor
import MinkowskiEngine as ME

from typing import Optional
from functools import partial


@torch.compiler.disable
class MinkowskiSparseBottleneckV2(nn.Module):
    def __init__(
        self,
        stage_index: int,
        block_index: int,
        in_chs: int,
        out_chs: int = None,
        bottle_ratio: float = 0.25,
        stride: int = 1,
        dilation: int = 1,
        dimension: int = 2,
        act_layer: nn.Module = ME.MinkowskiReLU,
        norm_layer: nn.Module = ME.MinkowskiBatchNorm,
        in_reduction: Optional[int] = None,
        out_reduction: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.stage_index = stage_index
        self.block_index = block_index
        self.in_reduction = in_reduction or 2**stage_index
        self.out_reduction = out_reduction or self.in_reduction
        if act_layer == ME.MinkowskiReLU:
            act_layer = partial(act_layer, inplace=True)

        mid_chs = int(out_chs * bottle_ratio)

        if in_chs != out_chs or stride > 1:
            self.downsample_shortcut = ME.MinkowskiConvolution(
                in_chs,
                out_chs,
                kernel_size=3,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dimension=dimension,
            )
        else:
            self.downsample_shortcut = nn.Identity()

        self.preact = nn.Sequential(
            ME.MinkowskiBatchNorm(in_chs),
            act_layer(),
        )
        self.conv1 = ME.MinkowskiConvolution(
            in_chs, mid_chs, kernel_size=1, bias=bias, dimension=dimension
        )
        self.norm_act_conv_2 = nn.Sequential(
            norm_layer(mid_chs),
            act_layer(),
            ME.MinkowskiConvolution(
                mid_chs,
                mid_chs,
                kernel_size=3,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dimension=dimension,
            ),
        )
        self.norm_act_conv_3 = nn.Sequential(
            norm_layer(mid_chs),
            act_layer(),
            ME.MinkowskiConvolution(
                mid_chs, out_chs, kernel_size=1, bias=bias, dimension=dimension
            ),
        )

    def forward(self, x):
        x_preact = self.preact(x)

        shortcut = self.downsample_shortcut(x_preact)

        x = self.conv1(x_preact)
        x = self.norm_act_conv_2(x)
        x = self.norm_act_conv_3(x)
        out = x + shortcut
        return out


class MinkowskiSparseResnetV2Stage(nn.Module):
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
        dimension: int = 2,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.stage_index = stage_index
        self.in_reduction = in_reduction
        self.out_reduction = out_reduction
        # first_dilation = 1 if dilation in (1, 2) else 2
        prev_chs = in_chs
        self.blocks = nn.Sequential()
        for block_index in range(depth):
            stride = stride if block_index == 0 else 1
            self.blocks.add_module(
                str(block_index),
                MinkowskiSparseBottleneckV2(
                    stage_index,
                    block_index,
                    prev_chs,
                    out_chs,
                    bottle_ratio=bottle_ratio,
                    stride=stride,
                    dilation=dilation,
                    dimension=dimension,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    in_reduction=in_reduction if block_index == 0 else out_reduction,
                    out_reduction=out_reduction,
                    bias=bias,
                ),
            )
            prev_chs = out_chs
            # first_dilation = dilation

    def forward(self, x):
        x = self.blocks(x)
        return x


@torch.compiler.disable
class MinkowskiStem(nn.Module):
    def __init__(
        self,
        in_chans: int,
        stem_channels: int,
        kernel_size: int,
        bias: bool,
        dimension: int,
    ):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_chans,
            stem_channels,
            kernel_size=kernel_size,
            bias=bias,
            dimension=dimension,
        )

    def forward(self, x):
        return self.conv(x)
