import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .matcher import HungarianMatcher
from ...utils.sparse_utils import gather_from_sparse_tensor
from ...utils.window_utils import key_offset_grid


class Criterion(nn.Module):
    def __init__(
        self,
        loss_coef_class: float = 1.0,
        loss_coef_mask_bce: float = 1.0,
        loss_coef_mask_dice: float = 1.0,
        loss_coef_incidence_nll: float = 1.0,
        loss_coef_total_energy: float = 1.0,
        loss_coef_occupancy: float = 1.0,
        no_electron_weight: float = 0.1,
        matcher_cost_coef_class: float = 1.0,
        matcher_cost_coef_mask: float = 1.0,
        matcher_cost_coef_dice: float = 1.0,
        matcher_cost_coef_dist: float = 1.0,
        bce_window_size: int = 7,
    ):
        super().__init__()
        self.loss_coef_class = loss_coef_class
        self.loss_coef_mask_bce = loss_coef_mask_bce
        self.loss_coef_mask_dice = loss_coef_mask_dice
        self.loss_coef_incidence_nll = loss_coef_incidence_nll
        self.loss_coef_total_energy = loss_coef_total_energy
        self.loss_coef_occupancy = loss_coef_occupancy
        self.no_electron_weight = no_electron_weight

        self.matcher = HungarianMatcher(
            cost_coef_class=matcher_cost_coef_class,
            cost_coef_mask=matcher_cost_coef_mask,
            cost_coef_dice=matcher_cost_coef_dice,
            cost_coef_dist=matcher_cost_coef_dist,
        )

        self.bce_window_size = bce_window_size

    def forward(
        self,
        predicted_dict: dict[str, Tensor],
        target_dict: dict[str, Tensor],
    ):
        matched_indices = [
            indices.cuda() for indices in self.matcher(predicted_dict, target_dict)
        ]
        class_loss, class_acc, electron_acc, no_electron_acc = self.get_class_loss(
            predicted_dict, matched_indices
        )

        # reorder query and electron tensors in order of matched indices
        query_offsets = predicted_dict["batch_offsets"][1:].cpu()
        electron_offsets = target_dict["electron_batch_offsets"][1:].cpu()

        # reorder the electron indices in the segmentation maps to the order
        # specified by the matcher
        true_segmap = _reorder_segmentation_map(
            target_dict["segmentation_mask"],
            [matched[1] for matched in matched_indices],
        ).coalesce()
        binary_logit_segmap = _reorder_segmentation_map(
            predicted_dict["binary_mask_logits_sparse"],
            [matched[0] for matched in matched_indices],
        ).coalesce()
        portion_logit_segmap = _reorder_segmentation_map(
            predicted_dict["portion_logits_sparse"],
            [matched[0] for matched in matched_indices],
        ).coalesce()

        bce_loss = self.get_mask_bce_loss(
            true_segmap,
            binary_logit_segmap,
            matched_indices,
            target_dict["incidence_points_pixels_rc"],
            electron_offsets,
        )
        dice_loss = self.get_mask_dice_loss(true_segmap, portion_logit_segmap)
        distance_nll_loss = self.get_distance_nll_loss(
            target_dict["incidence_points_pixels_rc"],
            predicted_dict["positions"],
            predicted_dict["position_std_dev_cholesky"],
            matched_indices,
            electron_offsets,
            query_offsets,
        )
        mse_loss = self.get_distance_mse_loss(
            target_dict["incidence_points_pixels_rc"],
            predicted_dict["positions"],
            matched_indices,
            electron_offsets,
            query_offsets,
        )
        occupancy_loss, occupancy_acc = self.get_occupancy_loss(
            predicted_dict, target_dict
        )

        loss_weights = torch.stack(
            [
                torch.tensor(self.loss_coef_class, device=class_loss.device),
                torch.tensor(self.loss_coef_mask_bce, device=bce_loss.device),
                torch.tensor(self.loss_coef_mask_dice, device=dice_loss.device),
                torch.tensor(
                    self.loss_coef_incidence_nll, device=distance_nll_loss.device
                ),
                torch.tensor(self.loss_coef_occupancy, device=occupancy_loss.device),
            ]
        )
        loss_terms = torch.stack(
            [class_loss, bce_loss, dice_loss, distance_nll_loss, occupancy_loss]
        )

        loss = torch.dot(loss_weights, loss_terms)
        aux_outputs = {
            "class_loss": class_loss.detach().item(),
            "class_acc": class_acc.detach().item(),
            "electron_acc": electron_acc.detach().item(),
            "no_electron_acc": no_electron_acc.detach().item(),
            "bce_loss": bce_loss.detach().item(),
            "dice_loss": dice_loss.detach().item(),
            "incidence_nll_loss": distance_nll_loss.detach().item(),
            "incidence_mse_loss": mse_loss.detach().item(),
            "occupancy_loss": occupancy_loss.detach().item(),
            "occupancy_acc": occupancy_acc.detach().item(),
            "total_loss": loss.detach().item(),
            "matched_indices": matched_indices,
        }

        return loss, aux_outputs

    def get_class_loss(
        self,
        predicted_dict: dict[str, Tensor],
        matched_indices: list[Tensor],
    ) -> Tensor:
        predicted_logits = predicted_dict["is_electron_logit"]

        labels = torch.zeros_like(predicted_logits)
        weights = torch.ones_like(predicted_logits)

        true_entries = torch.cat(
            [
                indices[0] + offset
                for indices, offset in zip(
                    matched_indices,
                    predicted_dict["batch_offsets"].to(matched_indices[0].device),
                )
            ]
        )
        labels[true_entries] = 1.0
        weights[labels.logical_not()] = self.no_electron_weight
        loss = F.binary_cross_entropy_with_logits(predicted_logits, labels, weights)
        with torch.no_grad():
            correct = (predicted_logits > 0) == labels
            acc = torch.count_nonzero(correct) / labels.numel()
            electron_acc = torch.count_nonzero(
                correct[labels.to(torch.bool)]
            ) / torch.count_nonzero(labels.to(torch.bool))
            no_electron_acc = torch.count_nonzero(
                correct[labels.to(torch.bool).logical_not()]
            ) / torch.count_nonzero(labels.to(torch.bool).logical_not())
        acc = torch.count_nonzero((predicted_logits > 0) == labels) / labels.numel()
        return loss, acc, electron_acc, no_electron_acc

    def get_mask_bce_loss(
        self,
        reordered_true_segmap: Tensor,
        reordered_binary_logits: Tensor,
        matched_indices: Tensor,
        incidence_points_pixels_rc: Tensor,
        electron_offsets: Tensor,
    ) -> Tensor:
        assert reordered_true_segmap.is_sparse
        assert reordered_binary_logits.is_sparse
        # pixel indices of all electron incidence point centers
        incidence_points_split = torch.tensor_split(
            incidence_points_pixels_rc, electron_offsets
        )
        reordered_indices = [
            torch.cat(
                [
                    torch.full(
                        [matched.shape[1], 1],
                        b,
                        dtype=torch.long,
                        device=reordered_true_segmap.device,
                    ),
                    points[matched[1]].floor().long(),
                    torch.arange(
                        matched.shape[1],
                        dtype=torch.long,
                        device=reordered_true_segmap.device,
                    ).unsqueeze(-1),
                ],
                -1,
            )
            for b, (points, matched) in enumerate(
                zip(incidence_points_split, matched_indices)
            )
        ]
        reordered_indices = torch.cat(reordered_indices)

        window_offsets = torch.cat(
            [
                reordered_indices.new_zeros(
                    [self.bce_window_size, self.bce_window_size, 1]
                ),
                key_offset_grid(
                    self.bce_window_size, self.bce_window_size, reordered_indices.device
                ),
                reordered_indices.new_zeros(
                    [self.bce_window_size, self.bce_window_size, 1]
                ),
            ],
            -1,
        )
        bce_window_indices = (
            reordered_indices.unsqueeze(1).unsqueeze(1) + window_offsets
        )

        true_values, true_specified_mask = gather_from_sparse_tensor(
            reordered_true_segmap, bce_window_indices
        )
        predicted_logits, predicted_specified_mask = gather_from_sparse_tensor(
            reordered_binary_logits, bce_window_indices
        )
        # specified_mask = torch.logical_and(
        #     true_specified_mask, predicted_specified_mask
        # )

        return F.binary_cross_entropy_with_logits(
            predicted_logits,
            true_values.float(),
            # weight=predicted_specified_mask.float(),
        )

    def get_mask_dice_loss(
        self, reordered_true_segmap: Tensor, reordered_portion_logit_segmap: Tensor
    ) -> Tensor:
        assert reordered_true_segmap.is_sparse
        assert reordered_portion_logit_segmap.is_sparse

        portion_segmap = torch.sparse.softmax(reordered_portion_logit_segmap, -1)

        num = torch.sparse.sum(
            2 * reordered_true_segmap * portion_segmap, [1, 2, 3]
        ).to_dense()
        den = torch.sparse.sum(
            reordered_true_segmap + portion_segmap, [1, 2, 3]
        ).to_dense()
        loss = 1 - (num + 1) / (den + 1)
        return loss.mean()

    def get_distance_nll_loss(
        self,
        true_incidence_points_pixels_rc,
        predicted_centers,
        predicted_std_dev_cholesky,
        matched_indices,
        electron_offsets,
        query_offsets,
    ) -> Tensor:
        incidence_points_split = torch.tensor_split(
            true_incidence_points_pixels_rc, electron_offsets
        )
        predicted_centers_split = torch.tensor_split(predicted_centers, query_offsets)
        predicted_std_dev_split = torch.tensor_split(
            predicted_std_dev_cholesky, query_offsets
        )

        reordered_incidence_points = torch.cat(
            [
                points[matched[1]]
                for points, matched in zip(incidence_points_split, matched_indices)
            ]
        )
        reordered_centers = torch.cat(
            [
                centers[matched[0]]
                for centers, matched in zip(predicted_centers_split, matched_indices)
            ]
        )
        reordered_std_dev = torch.cat(
            [
                std_dev[matched[0]]
                for std_dev, matched in zip(predicted_std_dev_split, matched_indices)
            ]
        )

        loss = -torch.distributions.MultivariateNormal(
            reordered_centers, scale_tril=reordered_std_dev
        ).log_prob(reordered_incidence_points)
        # print(torch.isinf(loss).any())
        loss[loss.isinf()] = 1e7
        # loss = torch.clamp_max(loss, 1e7)
        return loss.mean()

    def get_distance_mse_loss(
        self,
        true_incidence_points_pixels_rc,
        predicted_centers,
        matched_indices,
        electron_offsets,
        query_offsets,
    ) -> Tensor:
        incidence_points_split = torch.tensor_split(
            true_incidence_points_pixels_rc, electron_offsets
        )
        predicted_centers_split = torch.tensor_split(predicted_centers, query_offsets)

        reordered_incidence_points = torch.cat(
            [
                points[matched[1]]
                for points, matched in zip(incidence_points_split, matched_indices)
            ]
        )
        reordered_centers = torch.cat(
            [
                centers[matched[0]]
                for centers, matched in zip(predicted_centers_split, matched_indices)
            ]
        )

        loss = F.mse_loss(reordered_centers, reordered_incidence_points)

        return loss

    def get_occupancy_loss(
        self, predicted_dict: dict[str, Tensor], target_dict: dict[str, Tensor]
    ):
        logits = predicted_dict["occupancy_logits"]
        targets = target_dict["electron_count_map_1/1"]
        gathered_targets = gather_from_sparse_tensor(targets, logits.indices().T)[0]
        weights = gathered_targets.new_ones((logits.shape[-1],), dtype=torch.float)
        # weights[0] = 1 - (torch.count_nonzero(gathered_targets) / gathered_targets.numel())
        weights[0] = self.no_electron_weight
        loss = F.cross_entropy(logits.values(), gathered_targets, weights)
        acc = (
            torch.count_nonzero(logits.values().argmax(-1) == gathered_targets)
            / gathered_targets.numel()
        )
        return loss, acc


def _safe_gaussian_log_prob(mean: Tensor, std_dev_cholesky: Tensor, point: Tensor):
    half_log_det = std_dev_cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    # M =


def _reorder_segmentation_map(batched_segmaps: Tensor, matched_indices: list[Tensor]):
    assert batched_segmaps.is_sparse
    assert batched_segmaps.shape[0] == len(matched_indices)
    assert all([matched.ndim == 1 for matched in matched_indices])

    new_segmaps = []
    for segmap, matched in zip(batched_segmaps.unbind(), matched_indices):
        segmap = segmap.coalesce()

        in_matched_mask = torch.isin(segmap.indices()[-1], matched)
        indices = segmap.indices()[:, in_matched_mask]
        values = segmap.values()[in_matched_mask]

        pixel_indices, old_order = torch.tensor_split(indices, [-1], 0)
        new_order = torch.nonzero(old_order.T == matched.unsqueeze(0))[:, 1]
        new_indices = torch.cat([pixel_indices, new_order.unsqueeze(0)], 0)
        new_shape = [*segmap.shape[:-1], new_indices[-1].max() + 1]
        new_segmaps.append(
            torch.sparse_coo_tensor(
                new_indices,
                values,
                new_shape,
                dtype=segmap.dtype,
                device=segmap.device,
                requires_grad=segmap.requires_grad,
            ).coalesce()
        )

        # removed since index_select backward doesn't work
        # new_segmaps.append(torch.index_select(segmap, -1, matched).coalesce())

    # pad and stack the new segmaps
    max_electrons = max([segmap.shape[-1] for segmap in new_segmaps])
    new_segmaps = [
        torch.sparse_coo_tensor(
            segmap.indices(), segmap.values(), segmap.shape[:-1] + (max_electrons,)
        )
        for segmap in new_segmaps
    ]
    return torch.stack(new_segmaps)
