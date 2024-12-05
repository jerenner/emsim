import torch
from torch import nn, Tensor

import spconv.pytorch as spconv


class SpconvValueEncoder(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [spconv.SubMConv2d(in_chans, out_channels, 1) for in_chans in in_channels]
        )

    def forward(self, inputs: list[spconv.SparseConvTensor]):
        assert len(inputs) == len(self.layers)
        return [layer(x) for layer, x in zip(self.layers, inputs)]
