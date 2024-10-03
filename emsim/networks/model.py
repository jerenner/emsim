import torch
from torch import nn

import spconv.pytorch as spconv

from .sparse_resnet.unet import SparseResnetUnet
from .transformer.decoder import EMTransformerDecoder
from .transformer.model import EMTransformer
from .occupancy import OccupancyPredictor
from .salience_mask_predictor import SparseMaskPredictor
from .value_encoder import ValueEncoder
from .salience import ElectronSalienceCriterion
from .loss.matcher import HungarianMatcher
from .loss.criterion import EMCriterion
from .segmentation_map import sparse_binary_segmentation_map


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
        transformer_encoder_max_tokens: int = 10000,
        transformer_n_query_embeddings: int = 1000,
        matcher_cost_coef_class: float = 1.0,
        matcher_cost_coef_mask: float = 1.0,
        matcher_cost_coef_dice: float = 1.0,
        matcher_cost_coef_dist: float = 1.0,
        loss_coef_class: float = 1.0,
        loss_coef_mask_bce: float = 1.0,
        loss_coef_mask_dice: float = 1.0,
        loss_coef_incidence_nll: float = 1.0,
        loss_coef_incidence_mse: float = 1.0,
        loss_coef_total_energy: float = 1.0,
        loss_coef_occupancy: float = 1.0,
        loss_no_electron_weight: float = 0.1,
        aux_loss=True,
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

        self.criterion = EMCriterion(
            loss_coef_class=loss_coef_class,
            loss_coef_mask_bce=loss_coef_mask_bce,
            loss_coef_mask_dice=loss_coef_mask_dice,
            loss_coef_incidence_nll=loss_coef_incidence_nll,
            loss_coef_incidence_mse=loss_coef_incidence_mse,
            loss_coef_total_energy=loss_coef_total_energy,
            loss_coef_occupancy=loss_coef_occupancy,
            no_electron_weight=loss_no_electron_weight,
            matcher_cost_coef_class=matcher_cost_coef_class,
            matcher_cost_coef_mask=matcher_cost_coef_mask,
            matcher_cost_coef_dice=matcher_cost_coef_dice,
            matcher_cost_coef_dist=matcher_cost_coef_dist,
        )

        self.salience_criterion = ElectronSalienceCriterion()

        self.occupancy_predictor = OccupancyPredictor(
            self.backbone.decoder.feature_info[-1]["num_chs"], pixel_max_occupancy + 1
        )
        self.aux_loss = True

    def forward(self, batch: dict):
        image = batch["image_sparsified"]
        features = self.backbone(image)
        features = self.channel_uniformizer(features)

        (
            output_logits,
            output_positions,
            output_queries,
            segmentation_logits,
            query_batch_offsets,
            encoder_logits,
            encoder_positions,
            encoder_out,
            score_dict,
        ) = self.transformer(features)

        output = {
            "pred_logits": output_logits[-1],
            "pred_positions": output_positions[-1],
            "pred_segmentation_logits": segmentation_logits,
            "pred_binary_mask": sparse_binary_segmentation_map(segmentation_logits),
            "query_batch_offsets": query_batch_offsets,
        }

        output["aux_outputs"] = [
            {
                "pred_logits": logits,
                "pred_positions": positions,
                "query_batch_offsets": query_batch_offsets,
            }
            for logits, positions in zip(
                output_logits[:-1],
                output_positions[:-1],
            )
        ]

        output["output_queries"] = output_queries
        output["enc_outputs"] = {
            "pred_logits": encoder_logits,
            "pred_positions": encoder_positions,
        }
        output["encoder_out"] = encoder_out
        output["score_dict"] = score_dict

        # matched = self.matcher(output, batch)

        return output
