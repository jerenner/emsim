from typing import Optional

import spconv.pytorch as spconv
import torch
from torch import nn

from .decoder_blocks import SparseInverseResnetV2Stage


class SparseUnetDecoder(spconv.SparseModule):
    def __init__(
        self,
        layers: list[int],
        encoder_out_channels: int,
        encoder_skip_channels: list[int],
        channels: list[int] = [256, 128, 64, 32],
        act_layer: nn.Module = nn.ReLU,
        norm_layer: nn.Module = nn.BatchNorm1d,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.feature_info = []

        prev_chs = encoder_out_channels
        block_dprs = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)
        ]
        self.stages = nn.ModuleList()
        encoder_skip_indices = reversed(range(len(encoder_skip_channels)))
        for stage_index, (depth, c, skip_chs, skip_ind, bdpr) in enumerate(
            zip(
                layers,
                channels,
                encoder_skip_channels,
                encoder_skip_indices,
                block_dprs,
            )
        ):
            out_chs = c
            stage = SparseInverseResnetV2Stage(
                stage_index,
                prev_chs,
                out_chs,
                depth=depth,
                encoder_skip_stage=skip_ind,
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
        for i, stage in enumerate(self.stages):
            skip = skips[i] if i < len(skips) else None
            x = stage(x, skip)
        return x
