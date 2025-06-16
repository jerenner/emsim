import logging

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.ops.boxes import generalized_box_iou

from emsim.config.criterion import MatcherConfig
from emsim.utils.sparse_utils.batching.batch_utils import split_batch_concatted_tensor
from emsim.utils.sparse_utils.shape_ops import (
    sparse_flatten_hw,
)

_logger = logging.getLogger(__name__)


class HungarianMatcher(nn.Module):
    def __init__(self, config: MatcherConfig):
        super().__init__()
        self.cost_coef_class = config.cost_coef_class
        self.cost_coef_mask = config.cost_coef_mask
        self.cost_coef_dice = config.cost_coef_dice
        self.cost_coef_dist = config.cost_coef_dist
        self.cost_coef_nll = config.cost_coef_nll
        self.cost_coef_likelihood = config.cost_coef_likelihood
        self.cost_coef_box_l1 = config.cost_coef_box_l1
        self.cost_coef_box_giou = config.cost_coef_box_giou

    @torch.no_grad()
    def forward(
        self, predicted_dict: dict[str, Tensor], target_dict: dict[str, Tensor]
    ) -> list[Tensor]:
        n_queries = torch.cat(
            [
                predicted_dict["query_batch_offsets"],
                predicted_dict["query_batch_offsets"].new_tensor(
                    [predicted_dict["pred_logits"].shape[-2]]
                ),
            ]
        ).diff()
        n_electrons = target_dict["batch_size"]
        image_sizes_xy = target_dict["image_size_pixels_rc"].flip(-1)
        segmap = target_dict["segmentation_mask"].to(
            predicted_dict["pred_segmentation_logits"].device
        )
        incidence_points = target_dict["normalized_incidence_points_xy"].to(
            predicted_dict["pred_positions"].device
        )
        if "pred_boxes" in predicted_dict:
            bounding_boxes_abs_xy = target_dict["bounding_boxes_pixels_xyxy"].to(
                predicted_dict["pred_boxes"].device
            ) / image_sizes_xy.repeat_interleave(n_electrons, 0)

        class_cost = get_class_cost(
            predicted_dict["pred_logits"], predicted_dict["query_batch_offsets"]
        )
        mask_cost = get_bce_cost(
            predicted_dict["pred_segmentation_logits"], segmap, n_queries, n_electrons
        )
        dice_cost = get_dice_cost(
            predicted_dict["pred_segmentation_logits"], segmap, n_queries, n_electrons
        )
        distance_cost = get_huber_distance_cost(
            predicted_dict["pred_positions"],
            predicted_dict["query_batch_offsets"],
            incidence_points,
            target_dict["electron_batch_offsets"],
        )
        nll_cost = get_nll_distance_cost(
            predicted_dict["pred_positions"],
            predicted_dict["pred_std_dev_cholesky"],
            predicted_dict["query_batch_offsets"],
            incidence_points,
            target_dict["electron_batch_offsets"],
            image_sizes_xy,
        )
        likelihood_cost = get_likelihood_distance_cost(
            predicted_dict["pred_positions"],
            predicted_dict["pred_std_dev_cholesky"],
            predicted_dict["query_batch_offsets"],
            incidence_points,
            target_dict["electron_batch_offsets"],
            image_sizes_xy,
        )
        if "pred_boxes" in predicted_dict:
            box_l1_cost = get_box_l1_cost(
                predicted_dict["pred_boxes"],
                predicted_dict["query_batch_offsets"],
                bounding_boxes_abs_xy,
                target_dict["electron_batch_offsets"],
            )
            box_giou_cost = get_box_giou_cost(
                predicted_dict["pred_boxes"],
                predicted_dict["query_batch_offsets"],
                bounding_boxes_abs_xy,
                target_dict["electron_batch_offsets"],
            )

        total_costs = [
            self.cost_coef_class * _class
            + self.cost_coef_mask * mask
            + self.cost_coef_dice * dice
            + self.cost_coef_dist * dist
            + self.cost_coef_nll * nll
            + self.cost_coef_likelihood * likelihood
            for _class, mask, dice, dist, nll, likelihood in zip(
                class_cost,
                mask_cost,
                dice_cost,
                distance_cost,
                nll_cost,
                likelihood_cost,
            )
        ]
        if "pred_boxes" in predicted_dict:
            total_costs = [
                cost
                + self.cost_coef_box_l1 * box_l1
                + self.cost_coef_box_giou * box_giou
                for cost, box_l1, box_giou in zip(
                    total_costs, box_l1_cost, box_giou_cost
                )
            ]

        if any([c.shape[0] < c.shape[1] for c in total_costs]):
            _logger.warning(
                f"Got fewer queries than electrons: {[c.shape for c in total_costs]}"
            )

        indices = [linear_sum_assignment(cost.cpu()) for cost in total_costs]

        return [
            torch.stack([torch.as_tensor(i), torch.as_tensor(j)]) for i, j in indices
        ]


@torch.jit.script
def get_class_cost(is_electron_logit: Tensor, batch_offsets: Tensor) -> list[Tensor]:
    loss = F.binary_cross_entropy_with_logits(
        is_electron_logit, torch.ones_like(is_electron_logit), reduction="none"
    )
    batch_losses = split_batch_concatted_tensor(loss, batch_offsets)
    return list(batch_losses)


@torch.jit.script
def get_bce_cost(
    mask_logits: Tensor, segmap: Tensor, n_queries: Tensor, n_electrons: Tensor
) -> list[Tensor]:
    logits_indices = mask_logits.indices()
    logits_values = mask_logits.values()
    pos_values = F.binary_cross_entropy_with_logits(
        logits_values, torch.ones_like(logits_values), reduction="none"
    )
    nonzero_pos_indices = pos_values.nonzero().squeeze(1)
    pos_tensor = sparse_flatten_hw(
        torch.sparse_coo_tensor(
            logits_indices[:, nonzero_pos_indices],
            pos_values[nonzero_pos_indices],
            mask_logits.shape,
        )
    )

    neg_values = F.binary_cross_entropy_with_logits(
        logits_values, torch.zeros_like(logits_values), reduction="none"
    )
    nonzero_neg_indices = neg_values.nonzero().squeeze(1)
    neg_tensor = sparse_flatten_hw(
        torch.sparse_coo_tensor(
            logits_indices[:, nonzero_neg_indices],
            neg_values[nonzero_neg_indices],
            mask_logits.shape,
        )
    )

    true_segmap_binarized = sparse_flatten_hw(
        torch.sparse_coo_tensor(
            segmap.indices(),
            segmap.values().to(torch.bool).to(segmap.dtype),
            segmap.shape,
            is_coalesced=segmap.is_coalesced(),
        )
    )

    out = []
    for pos, neg, targ, q, e in zip(
        pos_tensor, neg_tensor, true_segmap_binarized, n_queries, n_electrons
    ):
        # pos_loss = torch.sparse.mm(pos.T, targ).to_dense()
        # neg_loss = (
        #     torch.sparse.sum(neg, (0,)).to_dense().unsqueeze(-1)
        #     - torch.sparse.mm(neg.T, targ).to_dense()
        # )
        pixels_with_predictions = pos.sum(1).to_dense().nonzero().squeeze(1)
        pos_select = pos.index_select(0, pixels_with_predictions).to_dense()
        neg_select = neg.index_select(0, pixels_with_predictions).to_dense()
        targ_select = targ.index_select(0, pixels_with_predictions).to_dense()
        pos_loss = torch.mm(pos_select.T, targ_select)
        neg_loss = torch.sum(neg_select, (0,)).unsqueeze(-1) - torch.mm(
            neg_select.T, targ_select
        )

        nnz = targ._nnz()
        loss = (pos_loss + neg_loss) / nnz
        out.append(loss[:q, :e])

    return out


@torch.jit.script
def get_dice_cost(
    portion_logits: Tensor,
    true_portions: Tensor,
    n_queries: Tensor,
    n_electrons: Tensor,
) -> list[Tensor]:
    portions = torch.sparse.softmax(portion_logits, -1)

    portions_flat = sparse_flatten_hw(portions)
    true_portions_flat = sparse_flatten_hw(true_portions)

    out = []
    for pred, true, q, e in zip(
        portions_flat, true_portions_flat, n_queries, n_electrons
    ):
        # num = 2 * torch.sparse.mm(pred.T, true).to_dense()
        # den = torch.sparse.sum(pred, (0,)).to_dense().unsqueeze(-1) + torch.sparse.sum(
        #     true, (0,)
        # ).to_dense().unsqueeze(0)
        pixels_with_predictions = pred.sum(1).to_dense().nonzero().squeeze(1)
        pred_select = pred.index_select(0, pixels_with_predictions).to_dense()
        true_select = true.index_select(0, pixels_with_predictions).to_dense()
        num = 2 * torch.mm(pred_select.T, true_select)
        den = pred_select.sum(0).unsqueeze(-1) + torch.sparse.sum(
            true, (0,)
        ).to_dense().unsqueeze(0)

        loss = 1 - ((num + 1) / (den + 1))
        out.append(loss[:q, :e])

    return out


@torch.jit.ignore  # pyright: ignore[reportArgumentType]
def batch_nll_distance_loss(
    predicted_positions: Tensor,
    predicted_std_dev_cholesky: Tensor,
    true_positions: Tensor,
    image_size_xy: Tensor,
):
    distn = torch.distributions.MultivariateNormal(
        predicted_positions.unsqueeze(-2) * image_size_xy,
        scale_tril=predicted_std_dev_cholesky.unsqueeze(-3),
    )  # distribution not supported by torchscript
    nll = -distn.log_prob(true_positions * image_size_xy)
    return nll


@torch.jit.script
def get_nll_distance_cost(
    predicted_positions: Tensor,
    predicted_std_dev_cholesky: Tensor,
    query_batch_offsets: Tensor,
    true_positions: Tensor,
    electron_batch_offsets: Tensor,
    image_sizes_xy: Tensor,
):
    predicted_pos_per_image = split_batch_concatted_tensor(
        predicted_positions, query_batch_offsets
    )
    predicted_std_dev_per_image = split_batch_concatted_tensor(
        predicted_std_dev_cholesky, query_batch_offsets
    )
    true_pos_per_image = split_batch_concatted_tensor(
        true_positions, electron_batch_offsets
    )
    image_size_per_image = image_sizes_xy.unbind(0)

    return [
        batch_nll_distance_loss(pred, std, true, size)
        for pred, std, true, size in zip(
            predicted_pos_per_image,
            predicted_std_dev_per_image,
            true_pos_per_image,
            image_size_per_image,
        )
    ]


@torch.jit.ignore  # pyright: ignore[reportArgumentType]
def batch_likelihood_distance_loss(
    predicted_positions: Tensor,
    predicted_std_dev_cholesky: Tensor,
    true_positions: Tensor,
    image_size_xy: Tensor,
):
    distn = torch.distributions.MultivariateNormal(
        predicted_positions.unsqueeze(-2) * image_size_xy,
        scale_tril=predicted_std_dev_cholesky.unsqueeze(-3),
    )  # distribution not supported by torchscript
    likelihood = distn.log_prob(true_positions * image_size_xy).exp()
    return 1 - likelihood


@torch.jit.script
def get_likelihood_distance_cost(
    predicted_positions: Tensor,
    predicted_std_dev_cholesky: Tensor,
    query_batch_offsets: Tensor,
    true_positions: Tensor,
    electron_batch_offsets: Tensor,
    image_sizes_xy: Tensor,
):
    predicted_pos_per_image = split_batch_concatted_tensor(
        predicted_positions, query_batch_offsets
    )
    predicted_std_dev_per_image = split_batch_concatted_tensor(
        predicted_std_dev_cholesky, query_batch_offsets
    )
    true_pos_per_image = split_batch_concatted_tensor(
        true_positions, electron_batch_offsets
    )
    image_size_per_image = image_sizes_xy.unbind(0)

    return [
        batch_likelihood_distance_loss(pred, std, true, size)
        for pred, std, true, size in zip(
            predicted_pos_per_image,
            predicted_std_dev_per_image,
            true_pos_per_image,
            image_size_per_image,
        )
    ]


def batch_huber_loss(predicted, true):
    n_queries, n_electrons = predicted.shape[0], true.shape[0]
    predicted = predicted.unsqueeze(1).expand(-1, n_electrons, -1)
    true = true.unsqueeze(0).expand(n_queries, -1, -1)
    return F.huber_loss(predicted, true, reduction="none").mean(-1)


@torch.jit.script
def get_huber_distance_cost(
    predicted_positions: Tensor,
    predicted_batch_offsets: Tensor,
    true_positions: Tensor,
    true_batch_offsets: Tensor,
) -> list[Tensor]:
    predicted_batches = split_batch_concatted_tensor(
        predicted_positions, predicted_batch_offsets
    )
    true_batches = split_batch_concatted_tensor(true_positions, true_batch_offsets)

    return [
        batch_huber_loss(predicted, true)
        for predicted, true in zip(predicted_batches, true_batches)
    ]


@torch.jit.script
def get_box_l1_cost(
    predicted_boxes_xyxy: Tensor,
    predicted_batch_offsets: Tensor,
    true_boxes_xyxy: Tensor,
    true_batch_offsets: Tensor,
) -> list[Tensor]:
    predicted_batches = split_batch_concatted_tensor(
        predicted_boxes_xyxy, predicted_batch_offsets
    )
    true_batches = split_batch_concatted_tensor(true_boxes_xyxy, true_batch_offsets)

    return [
        torch.cdist(predicted, true, p=1.0)
        for predicted, true in zip(predicted_batches, true_batches)
    ]


@torch.jit.script
def get_box_giou_cost(
    predicted_boxes_xyxy: Tensor,
    predicted_batch_offsets: Tensor,
    true_boxes_xyxy: Tensor,
    true_batch_offsets: Tensor,
) -> list[Tensor]:
    predicted_batches = split_batch_concatted_tensor(
        predicted_boxes_xyxy, predicted_batch_offsets
    )
    true_batches = split_batch_concatted_tensor(true_boxes_xyxy, true_batch_offsets)

    return [
        -generalized_box_iou(predicted, true)
        for predicted, true in zip(predicted_batches, true_batches)
    ]


def batch_alltoall_energy_loss(predicted_energies: Tensor, true_energies: Tensor):
    n_queries, n_electrons = predicted_energies.shape[0], true_energies.shape[0]
    predicted_energies = predicted_energies.unsqueeze(-1).expand(-1, n_electrons)
    true_energies = true_energies.unsqueeze(0).expand(n_queries, -1)

    return F.huber_loss(predicted_energies, true_energies)
