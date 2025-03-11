from typing import Union

import MinkowskiEngine as ME
from torch import nn, Tensor

from .encoder.model import MinkowskiSparseResnetV2
from .decoder.model import MinkowskiSparseUnetDecoder
from ...utils.sparse_utils import torch_sparse_to_minkowski


class MinkowskiSparseResnetUnet(ME.MinkowskiNetwork):
    def __init__(
        self,
        in_channels: int = 1,
        encoder_layers: list[int] = [2, 2, 2, 2],
        decoder_layers: list[int] = [2, 2, 2, 2],
        encoder_channels: list[int] = [32, 64, 128, 256],
        decoder_channels: list[int] = [256, 128, 64, 32],
        stem_channels: int = 16,
        bias: bool = True,
        dimension: int = 2,
        act_layer: nn.Module = ME.MinkowskiReLU,
        norm_layer: nn.Module = ME.MinkowskiBatchNorm,
    ):
        super().__init__(dimension)

        self.encoder = MinkowskiSparseResnetV2(
            encoder_layers,
            encoder_channels,
            in_chans=in_channels,
            stem_channels=stem_channels,
            bias=bias,
            dimension=dimension,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        encoder_strides = []
        encoder_skip_channels = []
        for info in self.encoder.feature_info[::-1]:
            encoder_strides.append(info["reduction"])
            encoder_skip_channels.append(info["num_chs"])
            if encoder_strides[-1] == 1:
                break

        self.decoder = MinkowskiSparseUnetDecoder(
            decoder_layers,
            encoder_skip_channels,
            encoder_strides,
            channels=decoder_channels,
            bias=bias,
            dimension=dimension,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    def forward(self, x: Union[Tensor, ME.SparseTensor]):
        if isinstance(x, Tensor):
            assert x.is_sparse
            x = torch_sparse_to_minkowski(x)
        x = self.encoder(x)
        # x = [  # don't use the stem output
        #     x_i
        #     for x_i, f_i in zip(x, self.encoder.feature_info)
        #     if "stages" in f_i["module"]
        # ]
        x.reverse()
        x = self.decoder(x)
        return x

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    @property
    def feature_info(self):
        return self.decoder.feature_info
