from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .matcher import HungarianMatcher
from ...utils.sparse_utils import gather_from_sparse_tensor, union_sparse_indices
from ...utils.batching_utils import split_batch_concatted_tensor


class EMCriterion(nn.Module):
    def __init__(
        self,
        loss_coef_class: float = 1.0,
        loss_coef_mask_bce: float = 1.0,
        loss_coef_mask_dice: float = 1.0,
        loss_coef_incidence_nll: float = 1.0,
        loss_coef_incidence_huber: float = 1.0,
        loss_coef_salience: float = 1.0,
        no_electron_weight: float = 0.1,
        matcher_cost_coef_class: float = 1.0,
        matcher_cost_coef_mask: float = 1.0,
        matcher_cost_coef_dice: float = 1.0,
        matcher_cost_coef_dist: float = 1.0,
        matcher_cost_coef_nll: float = 1.0,
        aux_loss=True,
    ):
        super().__init__()
        self.loss_coef_class = loss_coef_class
        self.loss_coef_mask_bce = loss_coef_mask_bce
        self.loss_coef_mask_dice = loss_coef_mask_dice
        self.loss_coef_incidence_nll = loss_coef_incidence_nll
        self.loss_coef_incidence_huber = loss_coef_incidence_huber
        self.loss_coef_salience = loss_coef_salience
        self.no_electron_weight = no_electron_weight
        self.aux_loss = aux_loss

        self.matcher = HungarianMatcher(
            cost_coef_class=matcher_cost_coef_class,
            cost_coef_mask=matcher_cost_coef_mask,
            cost_coef_dice=matcher_cost_coef_dice,
            cost_coef_dist=matcher_cost_coef_dist,
            cost_coef_nll=matcher_cost_coef_nll,
        )

    def compute_losses(
        self,
        predicted_dict: dict[str, Tensor],
        target_dict: dict[str, Tensor],
        matched_indices: Optional[Tensor] = None,
    ):
        if matched_indices is None:
            matched_indices = [
                indices.to(device=predicted_dict["pred_logits"].device)
                for indices in self.matcher(predicted_dict, target_dict)
            ]

        class_loss, class_acc, electron_acc, no_electron_acc = get_class_loss(
            predicted_dict["pred_logits"],
            predicted_dict["query_batch_offsets"],
            matched_indices,
            self.no_electron_weight,
        )

        # reorder query and electron tensors in order of matched indices
        sorted_predicted_logits, sorted_true_segmentation = _sort_predicted_true_maps(
            predicted_dict["pred_segmentation_logits"],
            target_dict["segmentation_mask"],
            matched_indices,
        )

        bce_loss, binary_acc = get_mask_bce_loss(
            sorted_predicted_logits, sorted_true_segmentation
        )
        dice_loss = get_mask_dice_loss(
            sorted_predicted_logits, sorted_true_segmentation
        )

        pred_positions = _sort_tensor(
            predicted_dict["pred_positions"],
            predicted_dict["query_batch_offsets"],
            [inds[0] for inds in matched_indices],
        )
        pred_std_cholesky = _sort_tensor(
            predicted_dict["pred_std_dev_cholesky"],
            predicted_dict["query_batch_offsets"],
            [inds[0] for inds in matched_indices],
        )
        true_positions = _sort_tensor(
            target_dict["normalized_incidence_points_xy"].to(pred_positions),
            target_dict["electron_batch_offsets"],
            [inds[1] for inds in matched_indices],
        )

        distance_nll_loss = get_distance_nll_loss(
            pred_positions, pred_std_cholesky, true_positions
        )
        huber_loss = get_distance_huber_loss(pred_positions, true_positions)

        model_mse = _mse(pred_positions, true_positions)

        loss_dict = {
            "loss_class": class_loss,
            "loss_bce": bce_loss,
            "loss_dice": dice_loss,
            "loss_incidence_nll": distance_nll_loss,
            "loss_incidence_huber": huber_loss,
        }

        aux_dict = {
            "mse": model_mse.item(),
            "matched_indices": matched_indices,
            "class_acc": class_acc.detach().item(),
            "electron_acc": electron_acc.detach().item(),
            "no_electron_acc": no_electron_acc.detach().item(),
            "binary_segmentation_acc": binary_acc.detach().item(),
        }

        return loss_dict, aux_dict

    def forward(
        self,
        predicted_dict: dict[str, Tensor],
        target_dict: dict[str, Tensor],
    ):
        loss_dict, aux_dict = self.compute_losses(predicted_dict, target_dict)

        if "aux_outputs" in predicted_dict:
            for i, aux_output_dict in enumerate(predicted_dict["aux_outputs"]):
                aux_loss_dict, aux_aux_dict = self.compute_losses(
                    aux_output_dict, target_dict
                )
                loss_dict.update({k + f"_{i}": v for k, v in aux_loss_dict.items()})
                aux_dict.update({k + f"_{i}": v for k, v in aux_aux_dict.items()})

        com_mse = _mse(
            target_dict["normalized_centers_of_mass_xy"],
            target_dict["normalized_incidence_points_xy"],
        )

        aux_dict["mse_com"] = com_mse.item()

        return loss_dict, aux_dict

    def __reweight_loss(self, loss_name, loss_tensor):
        if "class" in loss_name:
            return loss_tensor * self.loss_coef_class
        elif "bce" in loss_name:
            return loss_tensor * self.loss_coef_mask_bce
        elif "dice" in loss_name:
            return loss_tensor * self.loss_coef_mask_dice
        elif "distance_nll" in loss_name:
            return loss_tensor * self.loss_coef_incidence_nll
        elif "distance_huber" in loss_name:
            return loss_tensor * self.loss_coef_incidence_huber
        elif "salience" in loss_name:
            return loss_tensor * self.loss_coef_salience
        else:
            raise ValueError(f"Unrecognized loss {loss_name}: {loss_tensor}")

    def reweight_losses(self, loss_dict: dict[str, Tensor]):
        return {k: self.__reweight_loss(k, v) for k, v in loss_dict.items()}


def get_class_loss(
    pred_logits: Tensor,
    query_batch_offsets: Tensor,
    matched_indices: list[Tensor],
    no_electron_weight: float,
) -> Tensor:
    labels = torch.zeros_like(pred_logits)
    weights = torch.ones_like(pred_logits)

    true_entries = torch.cat(
        [
            indices[0] + offset
            for indices, offset in zip(
                matched_indices,
                query_batch_offsets.to(matched_indices[0].device),
            )
        ]
    )
    labels[true_entries] = 1.0
    weights[labels.logical_not()] = no_electron_weight
    loss = F.binary_cross_entropy_with_logits(pred_logits, labels, weights)
    with torch.no_grad():
        correct = (pred_logits > 0) == labels
        acc = torch.count_nonzero(correct) / labels.numel()
        electron_acc = torch.count_nonzero(
            correct[labels.to(torch.bool)]
        ) / torch.count_nonzero(labels.to(torch.bool))
        no_electron_acc = torch.count_nonzero(
            correct[labels.to(torch.bool).logical_not()]
        ) / torch.count_nonzero(labels.to(torch.bool).logical_not())
        acc = torch.count_nonzero((pred_logits > 0) == labels) / labels.numel()
    return loss, acc, electron_acc, no_electron_acc


def _sort_predicted_true_maps(
    predicted_segmentation_logits: Tensor,
    true_segmentation_map: Tensor,
    matched_indices: Tensor,
):
    assert predicted_segmentation_logits.is_sparse
    assert true_segmentation_map.is_sparse

    reordered_predicted = []
    reordered_true = []
    max_elecs = max([indices.shape[1] for indices in matched_indices])
    for predicted_map, true_map, indices in zip(
        predicted_segmentation_logits.unbind(0),
        true_segmentation_map.unbind(0),
        matched_indices,
    ):

        predicted_map = predicted_map.coalesce()
        true_map = true_map.coalesce()

        pred_index_masks = [predicted_map.indices()[-1] == i for i in indices[0]]
        true_index_masks = [true_map.indices()[-1] == i for i in indices[1]]

        pred_indices = [
            torch.cat(
                [
                    predicted_map.indices()[:-1, mask],
                    predicted_map.indices().new_full([1, mask.count_nonzero()], i),
                ]
            )
            for i, mask in enumerate(pred_index_masks)
        ]
        pred_values = [predicted_map.values()[mask] for mask in pred_index_masks]

        true_indices = [
            torch.cat(
                [
                    true_map.indices()[:-1, mask],
                    true_map.indices().new_full([1, mask.count_nonzero()], i),
                ]
            )
            for i, mask in enumerate(true_index_masks)
        ]
        true_values = [true_map.values()[mask] for mask in true_index_masks]

        reordered_predicted.append(
            torch.sparse_coo_tensor(
                torch.cat(pred_indices, -1),
                torch.cat(pred_values, 0),
                (*predicted_map.shape[:-1], max_elecs),
            )
        )
        reordered_true.append(
            torch.sparse_coo_tensor(
                torch.cat(true_indices, -1),
                torch.cat(true_values, 0),
                (*true_map.shape[:-1], max_elecs),
            )
        )

        # pred_unbound = predicted_map.unbind(-1)
        # true_unbound = true_map.unbind(-1)
        # pred_unbound[0].register_hook(lambda _: print("pred_unbound"))
        # reordered_pred_i = torch.stack([torch.select(predicted_map, -1, i) for i in indices[0]], -1).coalesce()
        # reordered_true_i = torch.stack([torch.select(true_map, -1, i) for i in indices[1]], -1).coalesce()

        # reordered_pred_i = torch.index_select(predicted_map, -1, indices[0]).coalesce()
        # reordered_true_i = torch.index_select(true_map, -1, indices[1]).coalesce()

    #     reordered_pred_i.register_hook(lambda _: print("reordered_pred_i"))

    #     reordered_predicted.append(reordered_pred_i)
    #     reordered_true.append(reordered_true_i)

    # reordered_predicted = [
    #     torch.sparse_coo_tensor(
    #         pred.indices(),
    #         pred.values(),
    #         (*pred.shape[:-1], max_elecs),
    #     )
    #     for pred in reordered_predicted
    # ]
    # reordered_true = [
    #     torch.sparse_coo_tensor(
    #         true.indices(),
    #         true.values(),
    #         (*true.shape[:-1], max_elecs),
    #     )
    #     for true in reordered_true
    # ]

    reordered_predicted = torch.stack(reordered_predicted).coalesce()
    reordered_true = torch.stack(reordered_true).coalesce()

    # reordered_predicted.register_hook(lambda x: print("reordered_predicted"))

    return reordered_predicted, reordered_true


def get_mask_bce_loss(
    sorted_predicted_logits: Tensor,
    sorted_true: Tensor,
) -> Tensor:
    assert sorted_predicted_logits.shape == sorted_true.shape

    unioned_predicted, unioned_true = union_sparse_indices(
        sorted_predicted_logits, sorted_true
    )
    assert torch.equal(unioned_predicted.indices(), unioned_true.indices())

    loss = F.binary_cross_entropy_with_logits(
        unioned_predicted.values(), unioned_true.values()
    )
    with torch.no_grad():
        acc = (
            torch.count_nonzero(
                unioned_true.values() == (unioned_predicted.values() > 0.0)
            )
            / unioned_true._nnz()
        )

    return loss, acc


def get_mask_dice_loss(
    sorted_predicted_logits: Tensor,
    sorted_true: Tensor,
) -> Tensor:
    assert sorted_predicted_logits.shape == sorted_true.shape

    predicted_segmentation = torch.sparse.softmax(sorted_predicted_logits, -1)
    # predicted_segmentation.register_hook(lambda x: print("predicted_segmentation"))

    unioned_predicted, unioned_true = union_sparse_indices(
        predicted_segmentation, sorted_true
    )
    assert torch.equal(unioned_predicted.indices(), unioned_true.indices())

    num = 2 * unioned_predicted * unioned_true
    den = unioned_predicted + unioned_true
    losses = []
    for num_i, den_i in zip(num.unbind(), den.unbind()):
        num_sum = num_i.coalesce().values().sum()
        den_sum = den_i.coalesce().values().sum()
        losses.append(1 - (num_sum + 1) / (den_sum + 1))
    loss = torch.stack(losses).mean()
    # loss.register_hook(lambda x: print("dice loss"))
    return loss


def _sort_tensor(
    batch_concatted_tensor: Tensor, batch_offsets: Tensor, matched_indices: Tensor
):
    split = split_batch_concatted_tensor(batch_concatted_tensor, batch_offsets)
    reordered = [points[inds] for points, inds in zip(split, matched_indices)]
    return torch.cat(reordered)


def get_distance_nll_loss(
    pred_centers: Tensor,
    pred_std_cholesky: Tensor,
    true_incidence_points: Tensor,
) -> Tensor:
    loss = -torch.distributions.MultivariateNormal(
        pred_centers, scale_tril=pred_std_cholesky
    ).log_prob(true_incidence_points)
    # print(torch.isinf(loss).any())
    loss = loss.clamp(-1e7, 1e7)
    # loss = torch.clamp_max(loss, 1e7)
    return loss.mean()


def get_distance_huber_loss(
    pred_centers: Tensor, true_incidence_points: Tensor, huber_delta: float = 0.1
) -> Tensor:
    loss = F.huber_loss(pred_centers, true_incidence_points, delta=huber_delta)
    return loss


# def get_occupancy_loss(
#     predicted_dict: dict[str, Tensor],
#     target_dict: dict[str, Tensor],
#     no_electron_weight: float,
# ):
#     logits = predicted_dict["occupancy_logits"]
#     targets = target_dict["electron_count_map_1/1"]
#     gathered_targets = gather_from_sparse_tensor(targets, logits.indices().T)[0]
#     weights = gathered_targets.new_ones((logits.shape[-1],), dtype=torch.float)
#     # weights[0] = 1 - (torch.count_nonzero(gathered_targets) / gathered_targets.numel())
#     weights[0] = no_electron_weight
#     loss = F.cross_entropy(logits.values(), gathered_targets, weights)
#     acc = (
#         torch.count_nonzero(logits.values().argmax(-1) == gathered_targets)
#         / gathered_targets.numel()
#     )
#     return loss, acc


def _mse(pred_centers: Tensor, true_points: Tensor) -> Tensor:
    with torch.no_grad():
        return F.mse_loss(
            pred_centers,
            true_points,
        )
