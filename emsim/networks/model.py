import torch
from torch import nn

import spconv.pytorch as spconv

from .sparse_resnet.unet import SparseResnetUnet
from .transformer.decoder import EMTransformerDecoder
from .transformer.model import EMTransformer
from .occupancy import OccupancyPredictor
from .mask_predictor import SparseMaskPredictor
from .value_encoder import ValueEncoder
from .salience import ElectronSalienceCriterion


class EMModel(nn.Module):
    def __init__(
        self,
        unet_encoder_layers: list[int] = [2, 2, 2, 2],
        unet_decoder_layers: list[int] = [2, 2, 2, 2],
        unet_encoder_channels: list[int] = [32, 64, 128, 256],
        unet_decoder_channels: list[int] = [256, 128, 64, 32],
        unet_stem_channels: int = 16,
        unet_act_layer: nn.Module = nn.ReLU,
        unet_norm_layer: nn.Module = nn.BatchNorm1d,
        unet_encoder_drop_path_rate: float = 0.0,
        unet_decoder_drop_path_rate: float = 0.0,
        pixel_max_occupancy: int = 5,
        transformer_d_model: int = 256,
        transformer_hidden_dim: int = 1024,
        transformer_n_heads: int = 8,
        transformer_n_deformable_points: int = 4,
        transformer_dropout: float = 0.1,
        transformer_activation_fn: str = "gelu",
        transformer_encoder_layers: int = 6,
        transformer_decoder_layers: int = 6,
        transformer_level_filter_ratio: tuple[float] = (0.25, 0.5, 1.0, 1.0),
        transformer_layer_filter_ratio: tuple[float] = (1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
        transformer_n_query_embeddings: int = 1000,
    ):
        super().__init__()

        self.backbone = SparseResnetUnet(
            encoder_layers=unet_encoder_layers,
            decoder_layers=unet_decoder_layers,
            encoder_channels=unet_encoder_channels,
            decoder_channels=unet_decoder_channels,
            encoder_stem_channels=unet_stem_channels,
            act_layer=unet_act_layer,
            norm_layer=unet_norm_layer,
            encoder_drop_path_rate=unet_encoder_drop_path_rate,
            decoder_drop_path_rate=unet_decoder_drop_path_rate,
        )
        self.channel_uniformizer = ValueEncoder(
            [info["num_chs"] for info in self.backbone.feature_info],
            transformer_d_model,
        )
        self.mask_predictor = SparseMaskPredictor(
            transformer_d_model, transformer_d_model
        )

        self.transformer = EMTransformer(
            d_model=transformer_d_model,
            n_heads=transformer_n_heads,
            dim_feedforward=transformer_hidden_dim,
            n_feature_levels=len(self.backbone.feature_info),
            n_deformable_points=transformer_n_deformable_points,
            backbone_indice_keys=self.backbone.downsample_indice_keys,
            dropout=transformer_dropout,
            activation_fn=transformer_activation_fn,
            n_encoder_layers=transformer_encoder_layers,
            n_decoder_layers=transformer_decoder_layers,
            level_filter_ratio=transformer_level_filter_ratio,
            layer_filter_ratio=transformer_layer_filter_ratio,
            n_query_embeddings=transformer_n_query_embeddings,
        )

        self.salience_criterion = ElectronSalienceCriterion()

        self.occupancy_predictor = OccupancyPredictor(
            self.backbone.decoder.feature_info[-1]["num_chs"], pixel_max_occupancy + 1
        )

    def forward(self, batch: dict):
        image = batch["image_sparsified"]
        features = self.backbone(image)
        features = self.channel_uniformizer(features)

        (
            output_logits,
            output_positions,
            encoder_logits,
            encoder_positions,
            score_dict,
        ) = self.transformer(features)

        output = {"pred_logits": output_logits, "pred_positions": output_positions}

        output["enc_outputs"] = {"pred_logits": encoder_logits, "pred_positions": encoder_positions}

        return output
