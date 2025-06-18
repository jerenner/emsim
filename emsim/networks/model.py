from typing import Optional, Union, Any
import logging

from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .loss.criterion import EMCriterion
from .backbone_me.unet import MinkowskiSparseResnetUnet
from .transformer.model import EMTransformer
from .me_value_encoder import ValueEncoder
from .denoising_generator import DenoisingGenerator

from ..config.model import EMModelConfig

_logger = logging.getLogger(__name__)


class EMModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        channel_uniformizer: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        denoising_generator: Optional[nn.Module] = None,
        include_aux_outputs: bool = False,
    ):
        super().__init__()

        self.backbone = backbone
        self.channel_uniformizer = channel_uniformizer
        self.transformer = transformer
        self.criterion = criterion
        self.denoising_generator = denoising_generator

        self.aux_loss = criterion.config.aux_loss.use_aux_loss
        self.include_aux_outputs = True if self.aux_loss else include_aux_outputs

        self.reset_parameters()

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
            nms_encoder_output,
            score_dict,
            encoder_out,
            topk_scores,
            topk_indices,
            topk_bijl_indices,
            backbone_features,
            backbone_features_pos_encoded,
        ) = self.transformer(
            features,
            batch["image_size_pixels_rc"],
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
        output["nms_enc_outputs"] = nms_encoder_output
        output["score_dict"] = score_dict
        output["backbone_features"] = backbone_features
        output["backbone_features_pos_encoded"] = backbone_features_pos_encoded
        output["encoder_out"] = nms_encoder_output
        output["topk"] = {
            "topk_scores": topk_scores,
            "topk_indices": topk_indices,
            "topk_bijl_indices": topk_bijl_indices,
        }

        if (self.training and self.aux_loss) or self.include_aux_outputs:
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
        if self.training:
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

    def prep_denoising_dict(self, denoising_out: dict[str, Any]):
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

    def reset_parameters(self):
        self.backbone.reset_parameters()
        self.channel_uniformizer.reset_parameters()
        self.transformer.reset_parameters()

    @classmethod
    def from_config(cls, cfg: Union[DictConfig, EMModelConfig]):
        config: Union[DictConfig, EMModelConfig]
        if not isinstance(cfg, EMModelConfig):
            config = OmegaConf.structured(cfg)
        else:
            config = cfg

        backbone = MinkowskiSparseResnetUnet(config.backbone)
        channel_uniformizer = ValueEncoder(
            [info["num_chs"] for info in backbone.feature_info],
            config.transformer.d_model,
        )
        transformer = EMTransformer(config.transformer)
        criterion = EMCriterion(config.criterion)
        if config.denoising.use_denoising:
            denoising_generator = DenoisingGenerator(config.denoising)
        else:
            denoising_generator = None
        return cls(
            backbone=backbone,
            channel_uniformizer=channel_uniformizer,
            transformer=transformer,
            criterion=criterion,
            denoising_generator=denoising_generator,
            include_aux_outputs=cfg.include_aux_outputs,
        )
