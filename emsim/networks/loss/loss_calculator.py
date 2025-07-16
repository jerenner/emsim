import re
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops.boxes import generalized_box_iou

from ...config.criterion import CriterionConfig
from ...utils.sparse_utils.batching import (
    batch_offsets_to_indices,
    split_batch_concatted_tensor,
)
from ...utils.sparse_utils.indexing.indexing import union_sparse_indices
from ...utils.sparse_utils.indexing.script_funcs import flatten_nd_indices
from ...utils.sparse_utils.shape_ops import sparse_flatten
from ...utils.sparse_utils.indexing.unique import unique_rows
from ...utils.sparse_utils.indexing.sparse_index_select import sparse_index_select
from .salience_criterion import ElectronSalienceCriterion
from .utils import sort_predicted_true_maps, sort_tensor, _restack_sparse_segmaps


class LossComponent(nn.Module):
    """Base class for loss calculators."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def apply_weight(self, loss_value: Tensor) -> Tensor:
        return self.weight * loss_value

    def weighted_loss(self, *args, **kwargs) -> Tensor:
        return self.apply_weight(self(*args, **kwargs))


class ClassificationLoss(LossComponent):
    def __init__(
        self,
        no_electron_weight: float,
        standardize_no_electron_weight: bool = False,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.no_electron_weight = no_electron_weight
        self.standardize_no_electron_weight = standardize_no_electron_weight

    def forward(
        self,
        predicted_dict: dict[str, Any],
        labels: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        pred_logits = predicted_dict["pred_logits"]

        return F.binary_cross_entropy_with_logits(pred_logits, labels.float(), weights)

    def make_label_and_weight_tensors(
        self, predicted_dict: dict[str, Any], matched_indices: list[Tensor]
    ) -> tuple[Tensor, Optional[Tensor]]:
        pred_logits = predicted_dict["pred_logits"]
        query_batch_offsets = predicted_dict["query_batch_offsets"]

        # Make label and weight tensors
        labels = torch.zeros_like(pred_logits, dtype=torch.long)
        # assign label 1 to matched queries
        for batch, indices in enumerate(matched_indices):
            labels[indices + query_batch_offsets[batch]] = 1

        if self.standardize_no_electron_weight:
            weights = torch.ones_like(pred_logits, dtype=torch.float)
            num_pos = labels.sum()
            weights[labels.logical_not()] = num_pos / (labels.numel() - num_pos)
        elif self.no_electron_weight != 1.0:
            weights = torch.ones_like(pred_logits, dtype=torch.float)
            weights[labels.logical_not()] = self.no_electron_weight
        else:
            weights = None

        return labels, weights


class MaskBCELoss(LossComponent):
    def forward(self, unioned_predicted: Tensor, unioned_true: Tensor) -> Tensor:
        assert torch.equal(unioned_predicted.indices(), unioned_true.indices())
        return F.binary_cross_entropy_with_logits(
            unioned_predicted.values(), unioned_true.values()
        )


class MaskDICELoss(LossComponent):
    def forward(
        self,
        predicted_dict: dict[str, Any],
        target_dict: dict[str, Any],
        matched_indices: list[Tensor],
    ) -> Tensor:
        pred_seg_logits: Tensor = predicted_dict["pred_segmentation_logits"]
        true_seg: Tensor = target_dict["segmentation_mask"]

        if pred_seg_logits.ndim == true_seg.ndim:
            # main queries (not denoising)
            predicted_segmentation, true_segmap = self.dice_inputs_main(
                predicted_dict, target_dict, matched_indices
            )
        else:
            assert pred_seg_logits.ndim == true_seg.ndim + 1
            # denoising
            predicted_segmentation, true_segmap = self.dice_inputs_denoising(
                predicted_dict, target_dict, matched_indices
            )

        losses = []
        num = 2 * predicted_segmentation * true_segmap
        den = predicted_segmentation + true_segmap
        for num_i, den_i in zip(num, den):
            num_sum = num_i.coalesce().values().sum()
            den_sum = den_i.coalesce().values().sum()
            losses.append(1 - (num_sum + 1) / (den_sum + 1))
        loss = torch.stack(losses).mean()
        return loss

    def dice_inputs_main(
        self,
        predicted_dict: dict[str, Any],
        target_dict: dict[str, Any],
        matched_indices: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        pred_seg_logits: Tensor = predicted_dict["pred_segmentation_logits"]
        true_seg: Tensor = target_dict["segmentation_mask"]

        split_query_logits: list[Tensor] = split_batch_concatted_tensor(
            predicted_dict["pred_logits"], predicted_dict["query_batch_offsets"]
        )
        pred_segmaps = []
        true_segmaps = []
        for b, (q_logits, pred, tgt, inds) in enumerate(
            zip(
                split_query_logits,
                pred_seg_logits.unbind(0),
                true_seg.unbind(0),
                matched_indices,
            )
        ):
            # get the subset of queries corresponding to matched queries, additional
            # false positives, and the background
            pos_logit_subset = (q_logits.squeeze(-1) > 0).nonzero().squeeze(-1)
            extra_pos_logits = pos_logit_subset[~torch.isin(pos_logit_subset, inds[0])]
            bg_index_b = inds[0].new_tensor([q_logits.shape[0]])
            seg_queries_b = torch.cat([inds[0], bg_index_b, extra_pos_logits])

            # predicted segmentation logits with [TP, bg, FP]
            pred_logits_b: Tensor = sparse_index_select(
                pred, -1, seg_queries_b, check_bounds=False
            )
            pred_segmaps.append(pred_logits_b)

            # maybe reorder the segmentation map to [TP, bg]
            true_bg_index_b: Tensor = target_dict["batch_size"][b]
            seg_objs_b = torch.cat([inds[1], true_bg_index_b.unsqueeze(-1)])

            # true segmentation map with [TP, bg]
            true_map_b: Tensor = sparse_index_select(
                tgt, -1, seg_objs_b, check_bounds=False
            )

            # remove background pixels for which no prediction was made so they are
            # not counted in the loss
            shape = torch.tensor(true_map_b.shape, device=true_map_b.device)
            pred_bg_pixels_flat: Tensor = flatten_nd_indices(
                torch.cat(
                    [
                        pred_logits_b.indices()[:-1],
                        true_bg_index_b.expand(1, pred_logits_b._nnz()),
                    ]
                ),
                shape,
            )[0].squeeze(0)
            true_inds_flat: Tensor = flatten_nd_indices(true_map_b.indices(), shape)[
                0
            ].squeeze(0)
            keep = torch.logical_and(
                torch.isin(
                    true_inds_flat, pred_bg_pixels_flat, assume_unique=True, invert=True
                ),
                true_map_b.indices()[-1] == true_bg_index_b,
            ).logical_not()
            true_map_b = torch.sparse_coo_tensor(
                true_map_b.indices()[:, keep],
                true_map_b.values()[keep],
                true_map_b.shape,
            )

            true_segmaps.append(true_map_b)

        max_obj = max(
            [pred.shape[-1] for pred in pred_segmaps]
            + [true.shape[-1] for true in true_segmaps]
        )

        pred_seg_logits = _restack_sparse_segmaps(pred_segmaps, max_obj)
        true_segmap = _restack_sparse_segmaps(true_segmaps, max_obj)

        predicted_segmentation: Tensor = torch.sparse.softmax(pred_seg_logits, -1)

        return predicted_segmentation, true_segmap

    def dice_inputs_denoising(
        self,
        predicted_dict: dict[str, Any],
        target_dict: dict[str, Any],
        matched_indices: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        pred_seg_logits: Tensor = predicted_dict["pred_segmentation_logits"]
        true_seg: Tensor = target_dict["segmentation_mask"]
        dn_info_dict = predicted_dict["dn_info_dict"]
        device = pred_seg_logits.device

        pred_segmentation: Tensor = torch.sparse.softmax(pred_seg_logits, -1)

        pred_segmaps = []
        true_segmaps = []

        n_dn_groups = pred_seg_logits.shape[-2]

        for b, (pred, tgt, inds) in enumerate(
            zip(pred_segmentation.unbind(0), true_seg.unbind(0), matched_indices)
        ):
            n_obj: int = dn_info_dict["n_objects_per_image"][b]

            pos_inds = torch.arange(n_obj, device=device) * 2
            bg_index = pos_inds.max() + 2
            inds_with_bg = torch.cat([pos_inds, bg_index[None]])

            pos_maps: Tensor = sparse_index_select(
                pred, -1, inds_with_bg, check_bounds=False
            )
            pos_maps_flat: Tensor = sparse_flatten(pos_maps, -2, -1)
            pred_segmaps.append(pos_maps_flat)

            true_inds_with_bg = torch.arange(
                (n_obj + 1) * n_dn_groups, device=device
            ) % (n_obj + 1)
            assert torch.equal(
                true_inds_with_bg.view(n_dn_groups, -1)[:, :-1].flatten(), inds[1]
            )

            true_selected: Tensor = sparse_index_select(
                tgt, -1, true_inds_with_bg, check_bounds=False
            )

            ## filter out unpredicted pixels like in the main queries so the
            # loss has a minimum at 0
            shape = torch.tensor(true_selected.shape, device=device)
            pred_indices_per_dn_group = pos_maps.indices()[:-1].clone()
            flat_bg_indices = torch.arange(n_dn_groups, device=device) * (n_obj * 2) + 1
            pred_indices_per_dn_group[-1] = flat_bg_indices[
                pred_indices_per_dn_group[-1]
            ]
            pred_bg_inds_flat: Tensor = flatten_nd_indices(
                pred_indices_per_dn_group, shape
            )[0].squeeze(0)
            true_inds_flat: Tensor = flatten_nd_indices(true_selected.indices(), shape)[
                0
            ].squeeze(0)
            keep = torch.logical_and(
                torch.isin(
                    true_inds_flat, pred_bg_inds_flat, assume_unique=True, invert=True
                ),
                torch.isin(true_selected.indices()[-1], flat_bg_indices),
            ).logical_not()
            true_selected = torch.sparse_coo_tensor(
                true_selected.indices()[:, keep],
                true_selected.values()[keep],
                true_selected.shape,
            )

            true_segmaps.append(true_selected)

        max_obj = max(
            [pred.shape[-1] for pred in pred_segmaps]
            + [true.shape[-1] for true in true_segmaps]
        )
        predicted_segmentation = _restack_sparse_segmaps(pred_segmaps, max_obj)
        true_segmentation = _restack_sparse_segmaps(true_segmaps, max_obj)

        return predicted_segmentation, true_segmentation


class DistanceNLLLoss(LossComponent):
    def __init__(self, detach_likelihood_mean: bool, weight: float = 1.0):
        super().__init__(weight)
        self.detach_likelihood_mean = detach_likelihood_mean

    def forward(
        self,
        pred_positions: Tensor,
        pred_std_cholesky: Tensor,
        true_positions: Tensor,
        **kwargs,
    ) -> Tensor:
        if self.detach_likelihood_mean:
            pred_positions = pred_positions.detach()

        loss: Tensor = -torch.distributions.MultivariateNormal(
            pred_positions, scale_tril=pred_std_cholesky
        ).log_prob(true_positions)
        return loss.mean()


class DistanceLikelihoodLoss(LossComponent):
    def __init__(self, detach_likelihood_mean: bool, weight: float = 1.0):
        super().__init__(weight)
        self.detach_likelihood_mean = detach_likelihood_mean

    def forward(
        self,
        pred_positions: Tensor,
        pred_std_cholesky: Tensor,
        true_positions: Tensor,
        **kwargs,
    ) -> Tensor:
        if self.detach_likelihood_mean:
            pred_positions = pred_positions.detach()

        loss: Tensor = torch.distributions.MultivariateNormal(
            pred_positions, scale_tril=pred_std_cholesky
        ).log_prob(true_positions)
        loss = 1 - loss.exp()
        return loss.mean()


class DistanceHuberLoss(LossComponent):
    def __init__(self, huber_delta: float, weight: float = 1.0):
        super().__init__(weight)
        self.huber_delta = huber_delta

    def forward(
        self, pred_positions: Tensor, true_positions: Tensor, **kwargs
    ) -> Tensor:
        return F.huber_loss(pred_positions, true_positions, delta=self.huber_delta)


class BoxL1Loss(LossComponent):
    def forward(
        self, pred_boxes_normalized_xyxy: Tensor, true_boxes_normalized_xyxy: Tensor
    ):
        return F.l1_loss(pred_boxes_normalized_xyxy, true_boxes_normalized_xyxy)


class BoxGIoULoss(LossComponent):
    def forward(
        self, pred_boxes_normalized_xyxy: Tensor, true_boxes_normalized_xyxy: Tensor
    ) -> Tensor:
        loss = 1 - torch.diagonal(
            generalized_box_iou(pred_boxes_normalized_xyxy, true_boxes_normalized_xyxy)
        )
        return loss.mean()


class SalienceLoss(LossComponent):
    def __init__(self, focal_alpha: float, focal_gamma: float, weight: float = 1.0):
        super().__init__(weight)
        self.criterion = ElectronSalienceCriterion(focal_alpha, focal_gamma)

    def forward(
        self, predicted_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> Tensor:
        predicted_foreground_masks, peak_normalized_images = self.criterion.prep_inputs(
            predicted_dict["score_dict"], target_dict
        )

        return self.criterion(predicted_foreground_masks, peak_normalized_images)


class LossCalculator(nn.Module):
    _pattern = re.compile(r"loss_(\w+)")

    def __init__(self, config: CriterionConfig):
        super().__init__()
        self.losses = nn.ModuleDict(
            {
                "classification": ClassificationLoss(
                    config.no_electron_weight,
                    config.standardize_no_electron_weight,
                    config.loss_weights.classification,
                ),
                "mask_bce": MaskBCELoss(config.loss_weights.mask_bce),
                "mask_dice": MaskDICELoss(config.loss_weights.mask_dice),
                "distance_nll": DistanceNLLLoss(
                    config.detach_likelihood_mean, config.loss_weights.distance_nll
                ),
                "distance_likelihood": DistanceLikelihoodLoss(
                    config.detach_likelihood_mean,
                    config.loss_weights.distance_likelihood,
                ),
                "distance_huber": DistanceHuberLoss(
                    config.huber_delta, config.loss_weights.distance_huber
                ),
                "salience": SalienceLoss(
                    config.salience.alpha,
                    config.salience.gamma,
                    config.loss_weights.salience,
                ),
            }
        )

        if config.predict_box:
            self.losses["box_l1"] = BoxL1Loss(config.loss_weights.box_l1)
            self.losses["box_giou"] = BoxGIoULoss(config.loss_weights.box_giou)

    def forward(
        self,
        predicted_dict: dict[str, Any],
        target_dict: dict[str, Any],
        matched_indices: list[Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Computes all losses and returns them as well as intermediate tensors to be
        used for metric calculation.
        """
        loss_dict = {}
        extras_dict = {}  # Intermediate values for later metric calculation

        # Classification loss
        classification_loss = self.losses["classification"]
        assert isinstance(classification_loss, ClassificationLoss)
        classification_labels, classification_weights = (
            classification_loss.make_label_and_weight_tensors(
                predicted_dict, matched_indices
            )
        )
        loss_dict["loss_classification"] = classification_loss(
            predicted_dict, classification_labels, classification_weights
        )
        # Save classification labels for later metric calculation
        extras_dict["classification_labels"] = classification_labels

        # Mask losses
        sorted_predicted_map, sorted_true_map = sort_predicted_true_maps(
            predicted_dict["pred_segmentation_logits"],
            target_dict["segmentation_mask"],
            matched_indices,
            (
                predicted_dict["dn_info_dict"]
                if "dn_info_dict" in predicted_dict
                else None
            ),
        )
        unioned_predicted, unioned_true = self._get_unioned_masks(
            sorted_predicted_map, sorted_true_map
        )
        loss_dict["loss_mask_bce"] = self.losses["mask_bce"](
            unioned_predicted, unioned_true
        )
        loss_dict["loss_mask_dice"] = self.losses["mask_dice"](
            predicted_dict, target_dict, matched_indices
        )
        # Save unioned masks for later metric calculation
        extras_dict["unioned_predicted_mask"] = unioned_predicted.detach()
        extras_dict["unioned_true_mask"] = unioned_true.detach()

        # Distance losses
        position_data = self._prep_position_data(
            predicted_dict, target_dict, matched_indices
        )
        for loss_name in ["distance_nll", "distance_likelihood", "distance_huber"]:
            loss_dict[f"loss_{loss_name}"] = self.losses[loss_name](**position_data)
        extras_dict["position_data"] = position_data

        # Box losses if applicable
        if "pred_boxes" in predicted_dict:
            box_data = self._prep_box_data(predicted_dict, target_dict, matched_indices)
            loss_dict["loss_box_l1"] = self.losses["box_l1"](**box_data)
            loss_dict["loss_box_giou"] = self.losses["box_giou"](**box_data)

        # Salience loss if applicable (if this is not an aux or denoising loss)
        if "score_dict" in predicted_dict:
            loss_dict["loss_salience"] = self.losses["salience"](
                predicted_dict, target_dict
            )

        return loss_dict, extras_dict

    def apply_loss_weights(self, loss_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Applies loss weights.

        Does not apply aux or denoising loss weights, which are handled by
        AuxLossHandler and DenoisingHandler.
        """
        out_dict = {}
        for loss_name, loss in loss_dict.items():
            loss_match = self._pattern.search(loss_name)
            assert loss_match is not None
            loss_module = self.losses[loss_match.group(1)]
            assert isinstance(loss_module, LossComponent)
            weighted_loss = loss_module.apply_weight(loss)
            out_dict[loss_name] = weighted_loss

        return out_dict

    def _get_unioned_masks(
        self, sorted_predicted_logits: Tensor, sorted_true_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        unioned_predicted, unioned_true = union_sparse_indices(
            sorted_predicted_logits, sorted_true_mask
        )
        assert torch.equal(unioned_predicted.indices(), unioned_true.indices())
        return unioned_predicted, unioned_true

    def _prep_position_data(
        self,
        predicted_dict: dict[str, Any],
        target_dict: dict[str, Any],
        matched_indices: list[Tensor],
    ) -> dict[str, Tensor]:
        position_data = {}
        pred = predicted_dict["pred_positions"]

        # position centers
        position_data["pred_positions"] = sort_tensor(
            pred,
            predicted_dict["query_batch_offsets"],
            [inds[0] for inds in matched_indices],
        )

        # position std dev
        position_data["pred_std_cholesky"] = sort_tensor(
            predicted_dict["pred_std_dev_cholesky"],
            predicted_dict["query_batch_offsets"],
            [inds[0] for inds in matched_indices],
        )

        # true positions
        position_data["true_positions"] = sort_tensor(
            target_dict["incidence_points_pixels_rc"].to(pred),
            target_dict["electron_batch_offsets"],
            [inds[1] for inds in matched_indices],
        )

        # center of mass estimates
        position_data["centroid_positions"] = sort_tensor(
            target_dict["centers_of_mass_rc"].to(pred),
            target_dict["electron_batch_offsets"],
            [inds[1] for inds in matched_indices],
        )

        # image geometry
        image_size = target_dict["image_size_pixels_rc"].to(pred)
        position_data["image_size_pixels_rc"] = image_size

        return position_data

    def _prep_box_data(
        self,
        predicted_dict: dict[str, Any],
        target_dict: dict[str, Any],
        matched_indices: list[Tensor],
    ):
        box_data = {}
        device = predicted_dict["pred_boxes"].device
        image_size = target_dict["image_size_pixels_rc"].flip(-1)
        batch_indices = batch_offsets_to_indices(predicted_dict["query_batch_offsets"])
        image_size_per_query = image_size[batch_indices]
        normalized_pred_boxes = predicted_dict["pred_boxes"] / torch.cat(
            [image_size_per_query, image_size_per_query], -1
        )

        box_data["pred_boxes_normalized_xyxy"] = sort_tensor(
            normalized_pred_boxes,
            predicted_dict["query_batch_offsets"],
            [inds[0] for inds in matched_indices],
        )
        box_data["true_boxes_normalized_xyxy"] = sort_tensor(
            target_dict["bounding_boxes_normalized_xyxy"].to(device),
            target_dict["electron_batch_offsets"],
            [inds[1] for inds in matched_indices],
        )
        return box_data
