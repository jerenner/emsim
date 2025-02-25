from typing import Optional
import logging

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from ..utils.misc_utils import _get_layer
from ..utils.sparse_utils.minkowskiengine import _get_me_layer
from .loss.criterion import EMCriterion
from .loss.salience_criterion import ElectronSalienceCriterion
from .backbone_me.unet import MinkowskiSparseResnetUnet
from .transformer.model import EMTransformer
from .me_value_encoder import ValueEncoder
from .denoising_generator import DenoisingGenerator


_logger = logging.getLogger(__name__)


class EMModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        channel_uniformizer: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        salience_criterion: nn.Module,
        denoising_generator: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.channel_uniformizer = channel_uniformizer
        self.transformer = transformer
        self.criterion = criterion
        self.salience_criterion = salience_criterion
        self.denoising_generator = denoising_generator

        self.aux_loss = getattr(self.criterion, "aux_loss", False)

    def forward(self, batch: dict):
        image = batch["image_sparsified"]
        features = self.backbone(image)
        features = self.channel_uniformizer(features)

        assert (
            batch["image_size_pixels_rc"].unique(dim=0).shape[0] == 1
        ), "Expected all images to be the same size"

        if self.training and self.denoising_generator is not None:
            denoising_queries, noised_positions = self.denoising_generator(batch)
            denoising_batch_offsets = batch["electron_batch_offsets"]
        else:
            denoising_queries = noised_positions = denoising_batch_offsets = None

        (
            output_logits,
            output_positions,
            std_dev_cholesky,
            output_queries,
            segmentation_logits,
            query_batch_offsets,
            denoising_out,
            encoder_logits,
            encoder_positions,
            encoder_out,
            score_dict,
        ) = self.transformer(
            features,
            batch["image_size_pixels_rc"][0],
            denoising_queries,
            noised_positions,
            denoising_batch_offsets,
        )

        output = {
            "pred_logits": output_logits[-1],
            "pred_positions": output_positions[-1],
            "pred_std_dev_cholesky": std_dev_cholesky[-1],
            "pred_segmentation_logits": segmentation_logits[-1],
            # "pred_binary_mask": sparse_binary_segmentation_map(segmentation_logits),
            "query_batch_offsets": query_batch_offsets,
            "output_queries": output_queries[-1],
        }

        # output["output_queries"] = output_queries[-1]
        output["enc_outputs"] = {
            "pred_logits": encoder_logits,
            "pred_positions": encoder_positions,
        }
        output["encoder_out"] = encoder_out
        output["score_dict"] = score_dict

        if self.training:
            if self.aux_loss:
                output["aux_outputs"] = [
                    {
                        "pred_logits": logits,
                        "pred_positions": positions,
                        "pred_std_dev_cholesky": cholesky,
                        "query_batch_offsets": query_batch_offsets,
                        "pred_segmentation_logits": seg_logits,
                        "output_queries": queries,
                    }
                    for logits, positions, cholesky, seg_logits, queries in zip(
                        # for logits, positions, cholesky, seg_logits in zip(
                        output_logits[:-1],
                        output_positions[:-1],
                        std_dev_cholesky[:-1],
                        segmentation_logits[:-1],
                        output_queries[:-1],
                    )
                ]
            _logger.debug("Begin loss calculation")
            if denoising_out is not None:
                denoising_output = self.prep_denoising_dict(denoising_out)
                output["denoising_output"] = denoising_output
            else:
                denoising_output = None
            loss_dict, output = self.compute_loss(batch, output)
            return loss_dict, output

        return output

    def compute_loss(
        self,
        batch: dict[str, Tensor],
        output: dict[str, Tensor],
    ):
        loss_dict, matched_indices = self.criterion(output, batch)
        output.update(matched_indices)
        return loss_dict, output

    def prep_denoising_dict(self, denoising_out: dict[str, Tensor]):
        dn_batch_mask_dict = denoising_out["dn_batch_mask_dict"]
        flattened_batch_offsets = (
            dn_batch_mask_dict["electron_batch_offsets"]
            * dn_batch_mask_dict["n_denoising_groups"]
            * 2
        )
        denoising_output = {
            "pred_logits": denoising_out["logits"][-1].flatten(0, -2),
            "pred_positions": denoising_out["positions"][-1].flatten(0, -2),
            "pred_std_dev_cholesky": denoising_out["std"][-1].flatten(0, -3),
            "pred_segmentation_logits": denoising_out["segmentation_logits"][-1],
            "query_batch_offsets": flattened_batch_offsets,
        }
        denoising_output.update(
            {
                "dn_batch_mask_dict": dn_batch_mask_dict,
                "denoising_matched_indices": dn_batch_mask_dict[
                    "denoising_matched_indices"
                ],
            }
        )

        if self.aux_loss:
            denoising_output["aux_outputs"] = [
                {
                    "pred_logits": logits.flatten(0, -2),
                    "pred_positions": positions.flatten(0, -2),
                    "pred_std_dev_cholesky": cholesky.flatten(0, -3),
                    "pred_segmentation_logits": seg_logits,
                    "query_batch_offsets": flattened_batch_offsets,
                    "dn_batch_mask_dict": dn_batch_mask_dict,
                    # "output_queries": queries,
                }
                for logits, positions, cholesky, seg_logits in zip(
                    denoising_out["logits"][:-1],
                    denoising_out["positions"][:-1],
                    denoising_out["std"][:-1],
                    denoising_out["segmentation_logits"][:-1],
                    # output_queries[:-1],
                )
            ]

        return denoising_output

    @classmethod
    def from_config(cls, cfg: DictConfig):
        backbone = MinkowskiSparseResnetUnet(
            encoder_layers=cfg.unet.encoder.layers,
            decoder_layers=cfg.unet.decoder.layers,
            encoder_channels=cfg.unet.encoder.channels,
            decoder_channels=cfg.unet.decoder.channels,
            stem_channels=cfg.unet.stem_channels,
            act_layer=_get_me_layer(cfg.unet.act_layer),
            norm_layer=_get_me_layer(cfg.unet.norm_layer),
        )
        channel_uniformizer = ValueEncoder(
            [info["num_chs"] for info in backbone.feature_info], cfg.transformer.d_model
        )
        transformer = EMTransformer(
            d_model=cfg.transformer.d_model,
            n_heads=cfg.transformer.n_heads,
            dim_feedforward=cfg.transformer.dim_feedforward,
            n_feature_levels=len(backbone.feature_info),
            n_deformable_points=cfg.transformer.n_deformable_points,
            dropout=cfg.transformer.dropout,
            activation_fn=_get_layer(cfg.transformer.activation_fn),
            n_encoder_layers=cfg.transformer.encoder.layers,
            n_decoder_layers=cfg.transformer.decoder.layers,
            level_filter_ratio=cfg.transformer.level_filter_ratio,
            layer_filter_ratio=cfg.transformer.layer_filter_ratio,
            rope_base_theta=cfg.transformer.rope_base_theta,
            encoder_max_tokens=cfg.transformer.max_tokens,
            encoder_topk_sa=cfg.transformer.encoder.topk_sa,
            encoder_use_rope=cfg.transformer.encoder.use_rope,
            encoder_use_ms_deform_attn=cfg.transformer.encoder.use_ms_deform_attn,
            n_query_embeddings=cfg.transformer.query_embeddings,
            decoder_cross_attn_type=cfg.transformer.decoder.cross_attn_type,
            decoder_look_forward_twice=cfg.transformer.decoder.look_forward_twice,
            decoder_detach_updated_positions=cfg.transformer.decoder.detach_updated_positions,
            decoder_use_rope=cfg.transformer.decoder.use_rope,
            mask_main_queries_from_denoising=cfg.denoising.mask_main_queries_from_denoising,
        )
        criterion = EMCriterion(
            loss_coef_class=cfg.criterion.loss_coef_class,
            loss_coef_mask_bce=cfg.criterion.loss_coef_mask_bce,
            loss_coef_mask_dice=cfg.criterion.loss_coef_mask_dice,
            loss_coef_incidence_nll=cfg.criterion.loss_coef_incidence_nll,
            loss_coef_incidence_likelihood=cfg.criterion.loss_coef_incidence_likelihood,
            loss_coef_incidence_huber=cfg.criterion.loss_coef_incidence_huber,
            no_electron_weight=cfg.criterion.no_electron_weight,
            matcher_cost_coef_class=cfg.criterion.matcher.cost_coef_class,
            matcher_cost_coef_mask=cfg.criterion.matcher.cost_coef_mask,
            matcher_cost_coef_dice=cfg.criterion.matcher.cost_coef_dice,
            matcher_cost_coef_dist=cfg.criterion.matcher.cost_coef_dist,
            matcher_cost_coef_nll=cfg.criterion.matcher.cost_coef_nll,
            matcher_cost_coef_likelihood=cfg.criterion.matcher.cost_coef_likelihood,
            use_aux_loss=cfg.criterion.aux_loss.use_aux_loss,
            aux_loss_use_final_matches=cfg.criterion.aux_loss.use_final_matches,
            aux_loss_weight=cfg.criterion.aux_loss.aux_loss_weight,
            n_aux_losses=cfg.transformer.decoder.layers - 1,
            detach_likelihood_mean=cfg.criterion.detach_likelihood_mean,
            use_denoising_loss=cfg.denoising.use_denoising,
            denoising_loss_weight=cfg.denoising.denoising_loss_weight,
            detection_metric_distance_thresholds=cfg.criterion.detection_metric_distance_thresholds,
        )
        salience_criterion = ElectronSalienceCriterion(
            alpha=cfg.criterion.salience.alpha, gamma=cfg.criterion.salience.gamma
        )
        if cfg.denoising.use_denoising:
            denoising_generator = DenoisingGenerator(
                d_model=cfg.transformer.d_model,
                max_electrons_per_image=cfg.denoising.max_electrons_per_image,
                max_total_denoising_queries=cfg.denoising.max_total_denoising_queries,
                position_noise_variance=cfg.denoising.position_noise_variance,
                pos_neg_queries_share_embedding=cfg.denoising.pos_neg_queries_share_embedding,
            )
        else:
            denoising_generator = None
        return cls(
            backbone=backbone,
            channel_uniformizer=channel_uniformizer,
            transformer=transformer,
            criterion=criterion,
            salience_criterion=salience_criterion,
            denoising_generator=denoising_generator,
        )
