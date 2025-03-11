import torch
from torch import nn, Tensor

import MinkowskiEngine as ME


class ValueEncoder(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ME.MinkowskiConvolution(in_chans, out_channels, 1, dimension=2)
                for in_chans in in_channels
            ]
        )

    def forward(self, inputs: list[ME.SparseTensor]):
        assert len(inputs) == len(self.layers)
        return [layer(x) for layer, x in zip(self.layers, inputs)]

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
