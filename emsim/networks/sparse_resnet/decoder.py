from typing import Optional

import spconv.pytorch as spconv
import torch
from torch import nn

from .decoder_blocks import SparseInverseResnetV2Stage


class SparseUnetDecoder(spconv.SparseModule):
    def __init__(
        self,
        layers: list[int],
        encoder_channels: list[int],
        encoder_strides: list[int] = None,
        channels: list[int] = [256, 128, 64, 32],
        act_layer: nn.Module = nn.ReLU,
        norm_layer: nn.Module = nn.BatchNorm1d,
        drop_path_rate=0.0,
    ):
        assert len(layers) == len(channels)
        super().__init__()

        prev_chs = encoder_channels[0]
        encoder_skip_channels = encoder_channels[1:]
        if encoder_strides is None:
            encoder_strides = [2**i for i in range(len(encoder_channels))][::-1]

        self.feature_info = []

        block_dprs = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(layers)).split(tuple(layers))
        ]
        self.stages = nn.ModuleList()
        for stage_index, (depth, c, bdpr) in enumerate(
            zip(
                layers,
                channels,
                block_dprs,
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
            stage = SparseInverseResnetV2Stage(
                stage_index,
                prev_chs,
                out_chs,
                depth=depth,
                in_reduction=in_reduction,
                out_reduction=out_reduction,
                encoder_skip_chs=skip_chs,
                block_dpr=bdpr,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            prev_chs = out_chs
            self.feature_info += [
                dict(num_chs=prev_chs, module=f"stages.{stage_index}")
            ]
            self.stages.append(stage)

    def forward(self, x: list[spconv.SparseConvTensor]):
        skips = x[1:]
        x = x[0]
        outputs = []
        for i, stage in enumerate(self.stages):
            skip = skips[i] if i < len(skips) else None
            x = stage(x, skip)
            outputs.append(x)
        return outputs
