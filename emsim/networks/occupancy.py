import spconv.pytorch as spconv
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import sigmoid_focal_loss

from ..utils.sparse_utils import spconv_to_torch_sparse, torch_sparse_to_spconv


class OccupancyPredictor(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_classes: int,
        kernel_size: int = 1,
        map_list_index: int = -1,
    ):
        super().__init__()
        self.predictor = spconv.SubMConv2d(in_features, out_classes, kernel_size)
        self.map_list_index = map_list_index

    def forward(self, x: Tensor):
        if isinstance(x, list):
            x = x[self.map_list_index]
        if isinstance(x, Tensor) and x.is_sparse:
            x = torch_sparse_to_spconv(x)
        x = self.predictor(x)
        return spconv_to_torch_sparse(x)


def occupancy_loss(
    predicted_occupancy_logits: Tensor,
    groundtruth_occupancy: Tensor,
    return_accuracy=False,
    focal_loss=False,
):
    assert predicted_occupancy_logits.is_sparse
    assert groundtruth_occupancy.is_sparse

    predicted_unioned, groundtruth_unioned = union_occupancy_indices(
        predicted_occupancy_logits, groundtruth_occupancy
    )

    if focal_loss:
        loss = sigmoid_focal_loss(
            predicted_unioned.values(),
            F.one_hot(
                groundtruth_unioned.values(), predicted_unioned.values().shape[-1]
            ).float(),
            reduction="mean",
        )
    else:
        # weight by inverse of frequency
        uniques, counts = torch.unique(groundtruth_unioned.values(), return_counts=True)
        weights = uniques.new_zeros(predicted_unioned.shape[-1], dtype=torch.float)
        weights[uniques] = counts.max() / counts
        loss = F.cross_entropy(
            predicted_unioned.values(), groundtruth_unioned.values(), weights
        )
    if return_accuracy:
        with torch.no_grad():
            logits = predicted_unioned.values()
            max_prob = logits.argmax(-1)
            n_correct = torch.sum(max_prob == groundtruth_unioned.values())
            acc = n_correct / logits.shape[0]

            zero_mask = groundtruth_unioned.values() == 0
            zeros_correct = torch.sum(max_prob[zero_mask] == 0)
            zeros_acc = zeros_correct / zero_mask.sum()
            nonzero_mask = groundtruth_unioned.values() != 0
            nonzeros_correct = torch.sum(
                max_prob[nonzero_mask] == groundtruth_unioned.values()[nonzero_mask]
            )
            nonzeros_acc = nonzeros_correct / nonzero_mask.sum()

            # probs = torch.sparse.softmax(predicted_unioned, -1)
            # cum_occupancy_probs = torch.cumsum(probs.values(), -1)
            # cum_over_p95 = cum_occupancy_probs > 0.95
            # temp = torch.arange(
            #     cum_over_p95.shape[-1], 0, -1, device=cum_over_p95.device
            # )
            # temp = temp * cum_over_p95
            # p95_cdf_electrons = torch.argmax(temp, 1)
            # p95_cdf_electrons

        return loss, acc, zeros_acc, nonzeros_acc
    return loss


def union_occupancy_indices(
    predicted_occupancy_logits: Tensor, groundtruth_occupancy: Tensor
):
    assert groundtruth_occupancy.is_sparse
    assert predicted_occupancy_logits.is_sparse

    if not groundtruth_occupancy.is_coalesced():
        groundtruth_occupancy = groundtruth_occupancy.coalesce()
    if not predicted_occupancy_logits.is_coalesced():
        predicted_occupancy_logits = predicted_occupancy_logits.coalesce()

    groundtruth_indices = groundtruth_occupancy.indices()
    groundtruth_values = groundtruth_occupancy.values()
    predicted_indices = predicted_occupancy_logits.indices()
    predicted_values = predicted_occupancy_logits.values()

    indices_gt_gt_predicted = torch.cat(
        [groundtruth_indices, groundtruth_indices, predicted_indices], -1
    )
    uniques, counts = torch.unique(indices_gt_gt_predicted, dim=-1, return_counts=True)
    predicted_exclusives = uniques[:, counts == 1]
    groundtruth_exclusives = uniques[:, counts == 2]

    groundtruth_unioned = torch.sparse_coo_tensor(
        torch.cat([groundtruth_indices, predicted_exclusives], -1),
        torch.cat(
            [
                groundtruth_values,
                groundtruth_values.new_zeros(
                    predicted_exclusives.shape[-1], dtype=torch.long
                ),
            ],
            0,
        ),
        size=groundtruth_occupancy.shape[:3],
        device=groundtruth_occupancy.device,
        dtype=torch.long,
    ).coalesce()

    stacked_zero_logits = predicted_values.new_zeros(
        [groundtruth_exclusives.shape[-1], predicted_values.shape[-1]], dtype=torch.long
    )
    stacked_zero_logits[:, 0] = 1
    predicted_unioned = torch.sparse_coo_tensor(
        torch.cat([predicted_indices, groundtruth_exclusives], -1),
        torch.cat([predicted_values, stacked_zero_logits], 0),
        size=predicted_occupancy_logits.shape,
        device=predicted_occupancy_logits.device,
    ).coalesce()

    return predicted_unioned, groundtruth_unioned
