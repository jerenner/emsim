from torch import nn
from .model import SparseResnetV2
from .decoder import SparseUnetDecoder

class SparseResnetUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
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
