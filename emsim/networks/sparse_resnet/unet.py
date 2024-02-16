import torch
from torch import nn

from .decoder import SparseUnetDecoder
from .model import SparseResnetV2


class SparseResnetUnet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        encoder_layers: list[int] = [2, 2, 2, 2],
        decoder_layers: list[int] = [2, 2, 2, 2],
        encoder_channels: list[int] = [32, 64, 128, 256],
        decoder_channels: list[int] = [256, 128, 64, 32],
        encoder_stem_channels: int = 16,
        act_layer: nn.Module = nn.ReLU,
        norm_layer: nn.Module = nn.BatchNorm1d,
        encoder_drop_path_rate: float = 0.0,
        decoder_drop_path_rate: float = 0.0,
    ):
        super().__init__()

        self.encoder = SparseResnetV2(
            encoder_layers,
            encoder_channels,
            in_chans=in_channels,
            stem_chs=encoder_stem_channels,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path_rate=encoder_drop_path_rate,
        )

        encoder_strides = [
            info["reduction"]
            for info in self.encoder.feature_info
            if "stages" in info["module"]
        ][::-1]

        self.decoder = SparseUnetDecoder(
            decoder_layers,
            encoder_channels[::-1],
            encoder_strides,
            decoder_channels,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path_rate=decoder_drop_path_rate,
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = [  # don't use the stem output
            x_i
            for x_i, f_i in zip(x, self.encoder.feature_info)
            if "stages" in f_i["module"]
        ]
        x.reverse()
        x = self.decoder(x)
        return x
