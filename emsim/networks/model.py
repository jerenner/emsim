import spconv.pytorch as spconv
import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from ..utils.misc_utils import _get_layer
from ..utils.sparse_utils import spconv_to_torch_sparse
from .loss.criterion import EMCriterion
from .loss.salience_criterion import ElectronSalienceCriterion
from .sparse_resnet.unet import SparseResnetUnet
from .transformer.model import EMTransformer
from .value_encoder import ValueEncoder


class EMModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        channel_uniformizer: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        salience_criterion: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.channel_uniformizer = channel_uniformizer
        self.transformer = transformer
        self.criterion = criterion
        self.salience_criterion = salience_criterion

        self.aux_loss = getattr(self.criterion, "aux_loss", False)

    def forward(self, batch: dict):
        image = batch["image_sparsified"]
        features = self.backbone(image)
        features = self.channel_uniformizer(features)

        (
            output_logits,
            output_positions,
            std_dev_cholesky,
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
            "pred_std_dev_cholesky": std_dev_cholesky[-1],
            "pred_segmentation_logits": segmentation_logits[-1],
            # "pred_binary_mask": sparse_binary_segmentation_map(segmentation_logits),
            "query_batch_offsets": query_batch_offsets,
        }

        output["output_queries"] = output_queries[-1]
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
                        output_logits[:-1],
                        output_positions[:-1],
                        std_dev_cholesky[:-1],
                        segmentation_logits[:-1],
                        output_queries[:-1],
                    )
                ]
            return self.compute_loss(batch, output)

        return output

    def compute_loss(self, batch: dict[str, Tensor], output: dict[dict, str]):
        loss_dict, matched_indices = self.criterion(output, batch)
        output.update(matched_indices)
        return loss_dict, output

    @classmethod
    def from_config(cls, cfg: DictConfig):
        backbone = SparseResnetUnet(
            encoder_layers=cfg.unet.encoder.layers,
            decoder_layers=cfg.unet.decoder.layers,
            encoder_channels=cfg.unet.encoder.channels,
            decoder_channels=cfg.unet.decoder.channels,
            stem_channels=cfg.unet.stem_channels,
            act_layer=_get_layer(cfg.unet.act_layer),
            norm_layer=_get_layer(cfg.unet.norm_layer),
            encoder_drop_path_rate=cfg.unet.encoder.drop_path_rate,
            decoder_drop_path_rate=cfg.unet.decoder.drop_path_rate,
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
            backbone_indice_keys=backbone.downsample_indice_keys,
            dropout=cfg.transformer.dropout,
            activation_fn=_get_layer(cfg.transformer.activation_fn),
            n_encoder_layers=cfg.transformer.encoder_layers,
            n_decoder_layers=cfg.transformer.decoder_layers,
            level_filter_ratio=cfg.transformer.level_filter_ratio,
            layer_filter_ratio=cfg.transformer.layer_filter_ratio,
            encoder_max_tokens=cfg.transformer.max_tokens,
            n_query_embeddings=cfg.transformer.query_embeddings,
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
            aux_loss=cfg.criterion.aux_loss,
            n_aux_losses=cfg.transformer.decoder_layers - 1,
        )
        salience_criterion = ElectronSalienceCriterion(
            alpha=cfg.criterion.salience.alpha, gamma=cfg.criterion.salience.gamma
        )
        return cls(
            backbone=backbone,
            channel_uniformizer=channel_uniformizer,
            transformer=transformer,
            criterion=criterion,
            salience_criterion=salience_criterion,
        )
