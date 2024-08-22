import torch
from torch import nn, Tensor
from torchvision.ops.focal_loss import sigmoid_focal_loss
from emsim.utils.sparse_utils import union_sparse_indices


def sparse_tensor_sigmoid_focal_loss(
    predicted, target, alpha: float = 0.25, gamma: float = 2, reduction: str = "none"
):
    predicted_unioned, target_unioned = union_sparse_indices(predicted, target)
    return sigmoid_focal_loss(
        predicted_unioned, target_unioned, alpha=alpha, gamma=gamma, reduction=reduction
    )
