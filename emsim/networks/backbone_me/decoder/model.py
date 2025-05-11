import torch
from torch import nn, Tensor
import MinkowskiEngine as ME

from functools import partial

from .blocks import (
    MinkowskiSparseInverseResnetV2Stage,
)

from emsim.config.backbone import BackboneDecoderConfig

# @torch.compiler.disable
class MinkowskiSparseUnetDecoder(nn.Module):
    def __init__(
        self,
        config: BackboneDecoderConfig,
        encoder_channels: list[int],
        encoder_strides: list[int] = None,
    ):
        assert len(config.layers) == len(config.channels)
        super().__init__()
        prev_chs = encoder_channels[0]
        encoder_skip_channels = encoder_channels[1:]
        if encoder_strides is None:
            encoder_strides = [2**i for i in range(len(encoder_channels))][::-1]

        self.feature_info = []

        self.stages = nn.ModuleList()
        for stage_index, (depth, c) in enumerate(
            zip(
                config.layers,
                config.channels,
            )
        ):
            out_chs = c
            skip_chs = (
                encoder_skip_channels[stage_index]
                if stage_index < len(encoder_skip_channels)
                else None
            )
            in_reduction = encoder_strides[stage_index]
            out_reduction = (
                encoder_strides[stage_index + 1]
                if stage_index + 1 < len(encoder_strides)
                else 1
            )
            stage = MinkowskiSparseInverseResnetV2Stage(
                stage_index,
                prev_chs,
                out_chs,
                depth=depth,
                in_reduction=in_reduction,
                out_reduction=out_reduction,
                encoder_skip_chs=skip_chs,
                bias=config.bias,
                dimension=config.dimension,
                act_layer=config.act_layer,
                norm_layer=config.norm_layer,
            )
            prev_chs = out_chs
            self.feature_info += [
                dict(num_chs=prev_chs, module=f"stages.{stage_index}")
            ]
            self.stages.append(stage)

    def forward(self, x: list[ME.SparseTensor]):
        skips = x[1:]
        x = x[0]
        outputs = []
        for i, stage in enumerate(self.stages):
            skip = skips[i] if i < len(skips) else None
            x = stage(x, skip)
            outputs.append(x)
        return outputs

    def reset_parameters(self):
        for stage in self.stages:
            stage.reset_parameters()
