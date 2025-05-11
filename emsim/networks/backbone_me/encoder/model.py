from typing import Optional, Union

import MinkowskiEngine as ME
from torch import Tensor, nn

from ....utils.sparse_utils.conversion import torch_sparse_to_minkowski
from .blocks import MinkowskiSparseResnetV2Stage, MinkowskiStem

from emsim.config.backbone import BackboneEncoderConfig


# @torch.compiler.disable
class MinkowskiSparseResnetV2(nn.Module):
    def __init__(self, config: BackboneEncoderConfig):
        super().__init__()
        self.feature_info = []
        self.stem = MinkowskiStem(
            config.in_channels,
            config.stem_channels,
            kernel_size=config.stem_kernel_size,
            bias=config.bias,
            dimension=config.dimension,
        )
        self.feature_info.append(
            dict(num_chs=config.stem_channels, reduction=1, module="stem")
        )

        prev_chs = config.stem_channels
        curr_stride = 1
        dilation = 1
        self.stages = nn.ModuleList()
        for stage_index, (d, c) in enumerate(zip(config.layers, config.channels)):
            # out_chs = make_divisible(c)
            out_chs = c
            # stride = 1 if stage_index == 0 else 2
            stride = 2
            if curr_stride >= config.output_stride:
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
                act_layer=config.act_layer,
                norm_layer=config.norm_layer,
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

    def reset_parameters(self):
        self.stem.reset_parameters()
        for stage in self.stages:
            stage.reset_parameters()
