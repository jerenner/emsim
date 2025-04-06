import torch
from torch import Tensor

from emsim.utils.sparse_utils.indexing.script_funcs import flattened_indices


def sparse_squeeze_dense_dim(tensor: Tensor, dim: int) -> Tensor:
    """Squeezes a specified dense dim out of a sparse tensor"""
    assert tensor.is_sparse, "Tensor is not sparse"
    dense_dim = tensor.dense_dim()
    assert dense_dim > 0, "Tensor has no dense dim to squeeze"

    dim = dim if dim >= 0 else dense_dim + dim  # handle negative indexing
    assert (
        0 <= dim < dense_dim
    ), f"dim {dim} is out of range for dense dims [0, {dense_dim -1 }]"

    shape = list(tensor.shape)
    if shape[tensor.sparse_dim() + dim] != 1:
        return tensor  # unsqueezable

    tensor = tensor.coalesce()
    new_shape = list(tensor.shape)
    del new_shape[tensor.sparse_dim() + dim]

    return torch.sparse_coo_tensor(
        tensor.indices(),
        tensor.values().squeeze(dim),  # squeeze the dim in values
        tuple(new_shape),
        requires_grad=tensor.requires_grad,
        is_coalesced=tensor.is_coalesced(),
    ).coalesce()


def sparse_resize(tensor: Tensor, new_shape: list[int]) -> Tensor:
    """Copies the indices and values of `tensor` to a new sparse tensor
    of different shape and same number of dims"""
    assert tensor.is_sparse
    assert len(new_shape) == tensor.ndim
    assert all(new >= old for new, old in zip(new_shape, tensor.shape))
    return torch.sparse_coo_tensor(
        tensor.indices(), tensor.values(), new_shape, is_coalesced=tensor.is_coalesced()
    ).coalesce()


@torch.jit.script
def sparse_flatten_hw(tensor: Tensor) -> Tensor:
    """Flattens the middle 2 dimensions of a 4D tensor"""
    assert tensor.is_sparse
    assert tensor.ndim == 4
    tensor = tensor.coalesce()
    indices = tensor.indices()
    i = indices[1]
    j = indices[2]
    H = tensor.shape[1]
    W = tensor.shape[2]
    ij = (i * W + j).unsqueeze(0)
    new_shape = tensor.shape[:1] + (H * W,) + tensor.shape[3:]
    new_indices = torch.cat([indices[:1], ij, indices[3:]], 0).long()
    return torch.sparse_coo_tensor(new_indices, tensor.values(), new_shape).coalesce()


@torch.jit.script
def sparse_flatten(tensor: Tensor, start_axis: int, end_axis: int) -> Tensor:
    """Flattens any number of dimensions of an n-D sparse tensor"""
    assert tensor.is_sparse
    if start_axis < 0:
        start_axis = tensor.ndim + start_axis
    if end_axis < 0:
        end_axis = tensor.ndim + end_axis
    assert end_axis > start_axis
    assert start_axis >= 0
    assert end_axis <= tensor.ndim
    tensor = tensor.coalesce()

    new_indices, new_shape, _ = flattened_indices(tensor, start_axis, end_axis)
    new_shape: list[int] = new_shape.tolist()
    return torch.sparse_coo_tensor(
        new_indices,
        tensor.values(),
        new_shape,
        is_coalesced=tensor.is_coalesced(),  # indices still unique and in correct order
    )
