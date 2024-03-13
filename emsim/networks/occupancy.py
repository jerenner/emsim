import spconv.pytorch as spconv
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils.sparse_utils import spconv_to_torch_sparse


class OccupancyPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.predictor = spconv.SubMConv2d(in_features, out_classes, 1)

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        x = self.predictor(x)
        return spconv_to_torch_sparse(x)


def occupancy_loss(
    predicted_occupancy_logits: Tensor,
    groundtruth_occupancy: Tensor,
    return_accuracy=False,
):
    assert predicted_occupancy_logits.is_sparse
    assert groundtruth_occupancy.is_sparse

    predicted_unioned, groundtruth_unioned = union_occupancy_indices(
        predicted_occupancy_logits, groundtruth_occupancy
    )

    loss = F.cross_entropy(predicted_unioned.values(), groundtruth_unioned.values())
    if return_accuracy:
        with torch.no_grad():
            logits = predicted_unioned.values()
            n_correct = torch.sum(logits.argmax(-1) == groundtruth_unioned.values())
            acc = n_correct / logits.shape[0]
        return loss, acc
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
