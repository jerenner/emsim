import torch
from torch import nn, Tensor
import MinkowskiEngine as ME
from typing import Optional, Union

from .blocks import MinkowskiSparseResnetV2Stage
from ....utils.sparse_utils import torch_sparse_to_minkowski


@torch.compiler.disable
class MinkowskiSparseResnetV2(nn.Module):
    def __init__(
        self,
        layers: list[int],
        channels: list[int] = [32, 64, 128, 256],
        in_chans: int = 1,
        stem_channels: int = 16,
        output_stride: Optional[int] = None,
        bias: bool = True,
        dimension: int = 2,
        act_layer: nn.Module = ME.MinkowskiReLU,
        norm_layer: nn.Module = ME.MinkowskiBatchNorm,
    ):
        super().__init__()
        self.feature_info = []
        self.stem = ME.MinkowskiConvolution(
            in_chans,
            stem_channels,
            kernel_size=7,
            bias=bias,
            dimension=dimension,
        )
        self.feature_info.append(
            dict(num_chs=stem_channels, reduction=1, module="stem")
        )
        if output_stride is None:
            output_stride = 2 ** len(layers)

        prev_chs = stem_channels
        curr_stride = 1
        dilation = 1
        self.stages = nn.ModuleList()
        for stage_index, (d, c) in enumerate(zip(layers, channels)):
            # out_chs = make_divisible(c)
            out_chs = c
            # stride = 1 if stage_index == 0 else 2
            stride = 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            stage = MinkowskiSparseResnetV2Stage(
                stage_index,
                prev_chs,
                out_chs,
                stride=stride,
                dilation=dilation,
                depth=d,
                in_reduction=curr_stride,
                out_reduction=curr_stride * stride,
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

    def forward(self, x: Union[Tensor, ME.SparseTensor]):
        if isinstance(x, Tensor):
            assert x.is_sparse
            x = torch_sparse_to_minkowski(x)
        out = []
        x = self.stem(x)
        out.append(x)
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out
