import torch
from torch import Tensor, nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import sparse

from emsim.utils.sparse_utils import (
    bhwn_to_nhw_iterator_over_batches_torch,
    torch_sparse_to_pydata_sparse,
)


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_coef_class: float = 1.0,
        cost_coef_mask: float = 1.0,
        cost_coef_dice: float = 1.0,
        cost_coef_dist: float = 1.0,
    ):
        super().__init__()
        self.cost_coef_class = cost_coef_class
        self.cost_coef_mask = cost_coef_mask
        self.cost_coef_dice = cost_coef_dice
        self.cost_coef_dist = cost_coef_dist

    @torch.no_grad()
    def forward(
        self, predicted_dict: dict[str, Tensor], target_dict: dict[str, Tensor]
    ):
        segmap = target_dict["segmentation_mask"].to(
            predicted_dict["binary_mask_logits"].device
        )
        incidence_points = target_dict["incidence_points_pixels_rc"].to(
            predicted_dict["positions"].device
        )
        class_cost = get_class_cost(
            predicted_dict["is_electron_logit"], predicted_dict["batch_offsets"]
        )
        mask_cost = get_bce_cost(predicted_dict["binary_mask_logits_sparse"], segmap)
        dice_cost = get_dice_cost(predicted_dict["portion_logits_sparse"], segmap)
        distance_cost = get_huber_distance_cost(
            predicted_dict["positions"],
            predicted_dict["batch_offsets"],
            incidence_points,
            target_dict["electron_batch_offsets"],
        )

        total_costs = [
            self.cost_coef_class * _class
            + self.cost_coef_mask * mask
            + self.cost_coef_dice * dice
            + self.cost_coef_dist * dist
            for _class, mask, dice, dist in zip(
                class_cost, mask_cost, dice_cost, distance_cost
            )
        ]

        if any([c.shape[0] < c.shape[1] for c in total_costs]):
            print(f"Got fewer queries than electrons: {[c.shape for c in total_costs]}")

        indices = [linear_sum_assignment(cost) for cost in total_costs]

        return [
            torch.stack([torch.as_tensor(i), torch.as_tensor(j)]) for i, j in indices
        ]


@torch.jit.script
def batch_class_cost(is_electron_logit: Tensor) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(
        is_electron_logit, torch.ones_like(is_electron_logit), reduction="none"
    )
    return loss.unsqueeze(-1).cpu()


@torch.jit.script
def get_class_cost(is_electron_logit: Tensor, batch_offsets: Tensor) -> list[Tensor]:
    batches = torch.tensor_split(is_electron_logit, batch_offsets[1:].cpu(), 0)
    return [batch_class_cost(batch) for batch in batches]


@torch.jit.ignore
def _pydata_sparse_bce(pos: Tensor, neg: Tensor, targets: Tensor) -> Tensor:
    pos_cpu = torch_sparse_to_pydata_sparse(pos)
    neg_cpu = torch_sparse_to_pydata_sparse(neg)
    target_cpu = torch_sparse_to_pydata_sparse(targets)

    pos_loss = sparse.einsum("qhw,ehw->qe", pos_cpu, target_cpu)
    neg_loss = neg_cpu.sum([1, 2])[..., None] - sparse.einsum(
        "qhw,ehw->qe", neg_cpu, target_cpu
    )

    loss = pos_loss.todense() + neg_loss.todense()
    return torch.tensor(loss / target_cpu.nnz)


def get_bce_cost(
    binary_mask_logits: Tensor,
    true_segmap: Tensor,
) -> list[Tensor]:
    logits_batches = bhwn_to_nhw_iterator_over_batches_torch(binary_mask_logits)
    segmap_batches = bhwn_to_nhw_iterator_over_batches_torch(true_segmap)

    return [
        batch_bce_cost(logits, segmap)
        for logits, segmap in zip(logits_batches, segmap_batches)
    ]


@torch.jit.script
def batch_bce_cost(mask_logits: Tensor, segmap: Tensor) -> Tensor:
    pos_tensor = torch.sparse_coo_tensor(
        mask_logits.indices(),
        F.binary_cross_entropy_with_logits(
            mask_logits.values(),
            torch.ones_like(mask_logits.values()),
            reduction="none",
        ),
        mask_logits.shape,
    ).cpu()
    neg_tensor = torch.sparse_coo_tensor(
        mask_logits.indices(),
        F.binary_cross_entropy_with_logits(
            mask_logits.values(),
            torch.zeros_like(mask_logits.values()),
            reduction="none",
        ),
        mask_logits.shape,
    ).cpu()
    true_segmap_binarized = torch.sparse_coo_tensor(
        segmap.indices(),
        segmap.values().to(torch.bool).float(),
        segmap.shape,
    ).cpu()
    return _pydata_sparse_bce(pos_tensor, neg_tensor, true_segmap_binarized)


@torch.jit.ignore
def _pydata_sparse_dice(predicted: Tensor, true: Tensor) -> Tensor:
    predicted = torch_sparse_to_pydata_sparse(predicted)
    true = torch_sparse_to_pydata_sparse(true)
    num = 2 * sparse.einsum("qhw,ehw->qe", predicted, true).todense()
    den = predicted.sum([1, 2])[..., None].todense() + true.sum([1, 2])[None].todense()
    return torch.as_tensor(1 - ((num + 1) / (den + 1)))


def get_dice_cost(portion_logits: Tensor, true_portions: Tensor) -> list[Tensor]:
    portions = torch.sparse_coo_tensor(
        portion_logits.indices(),
        portion_logits.values().sigmoid(),
        portion_logits.shape,
    )

    portions_batches = bhwn_to_nhw_iterator_over_batches_torch(portions)
    segmap_batches = bhwn_to_nhw_iterator_over_batches_torch(true_portions)

    return [
        batch_dice_cost(portions, segmap)
        for portions, segmap in zip(portions_batches, segmap_batches)
    ]


@torch.jit.script
def batch_dice_cost(portion_logits: Tensor, true_portions: Tensor) -> Tensor:
    portions = torch.sparse_coo_tensor(
        portion_logits.indices(),
        portion_logits.values().sigmoid(),
        portion_logits.shape,
    ).cpu()
    true_portions = true_portions.cpu()

    return _pydata_sparse_dice(portions, true_portions)


def batch_alltoall_distance_nll_loss(
    predicted_positions: Tensor,
    predicted_std_dev_cholesky: Tensor,
    true_positions: Tensor,
):
    distn = torch.distributions.MultivariateNormal(
        predicted_positions.unsqueeze(-2),
        scale_tril=predicted_std_dev_cholesky.unsqueeze(-3),
    )
    nll = -distn.log_prob(true_positions)
    return nll


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
    predicted_batches = torch.tensor_split(
        predicted_positions, predicted_batch_offsets[1:].cpu(), 0
    )
    true_batches = torch.tensor_split(true_positions, true_batch_offsets[1:].cpu(), 0)

    return [
        batch_huber_loss(predicted, true).cpu()
        for predicted, true in zip(predicted_batches, true_batches)
    ]


def batch_alltoall_energy_loss(predicted_energies: Tensor, true_energies: Tensor):
    n_queries, n_electrons = predicted_energies.shape[0], true_energies.shape[0]
    predicted_energies = predicted_energies.unsqueeze(-1).expand(-1, n_electrons)
    true_energies = true_energies.unsqueeze(0).expand(n_queries, -1)

    return F.huber_loss(predicted_energies, true_energies)
