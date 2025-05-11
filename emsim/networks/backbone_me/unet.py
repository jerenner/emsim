from typing import Union

import MinkowskiEngine as ME
from torch import Tensor, nn

from ...utils.sparse_utils.conversion import torch_sparse_to_minkowski
from .decoder.model import MinkowskiSparseUnetDecoder
from .encoder.model import MinkowskiSparseResnetV2

from emsim.config.backbone import BackboneConfig


class MinkowskiSparseResnetUnet(ME.MinkowskiNetwork):
    def __init__(self, config: BackboneConfig):
        super().__init__(config.dimension)

        self.encoder = MinkowskiSparseResnetV2(config.encoder)

        encoder_strides = []
        encoder_skip_channels = []
        for info in self.encoder.feature_info[::-1]:
            encoder_strides.append(info["reduction"])
            encoder_skip_channels.append(info["num_chs"])
            if encoder_strides[-1] == 1:
                break

        self.decoder = MinkowskiSparseUnetDecoder(
            config.decoder, encoder_skip_channels, encoder_strides
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
