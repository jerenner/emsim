import torch
from torch import Tensor

from emsim.utils.sparse_utils.indexing.script_funcs import flatten_sparse_indices


def sparse_squeeze(tensor: Tensor, dim: int) -> Tensor:
    """Squeeze (remove) a dimension of size 1 from a COO sparse tensor.

    The dimension to squeeze may be either:

        - a sparse dimension (0, ..., tensor.sparse_dim() - 1)
        - a dense dimension (tensor.sparse_dim(), ..., tensor.ndim - 1)

    If the dimension is not of length 1, the tensor is returned unchanged, as with
    the squeeze(dim) method of dense tensors.

    Args:
        tensor (Tensor): Sparse COO Tensor.
        dim (int): Dimension to squeeze.

    Returns:
        Tensor: Input tensor with specified sparse dim squeezed out, if applicable.
    """
    if not tensor.is_sparse:
        raise ValueError("Received non-sparse tensor.")

    ndim = tensor.ndim
    dim = dim if dim >= 0 else ndim + dim  # handle negative indexing
    if not 0 <= dim < ndim:
        raise IndexError(
            f"dim {dim} is out of range for tensor with {ndim} dimensions."
        )

    shape = list(tensor.shape)
    if shape[dim] != 1:
        return tensor  # unsqueezable

    sparse_dims = tensor.sparse_dim()
    indices = tensor._indices()
    values = tensor._values()

    if dim < sparse_dims:  # Squeeze sparse dim
        new_indices = torch.cat((indices[:dim], indices[dim + 1 :]), 0)
        new_values = values
    else:  # Squeeze dense dim
        new_indices = indices
        new_values = values.squeeze(dim - sparse_dims + 1)

    new_shape = shape[:dim] + shape[dim + 1 :]

    return torch.sparse_coo_tensor(
        new_indices,
        new_values,
        tuple(new_shape),
        requires_grad=tensor.requires_grad,
        is_coalesced=tensor.is_coalesced(),
    )


def sparse_resize(tensor: Tensor, new_shape: list[int]) -> Tensor:
    """Copies the indices and values of `tensor` to a new sparse tensor
    of different shape and same number of dims"""
    if not tensor.is_sparse:
        raise ValueError("Received non-sparse tensor.")
    if not len(new_shape) == tensor.ndim:
        raise ValueError(
            f"New shape {new_shape} has different number of dims "
            f"than existing shape {tensor.shape}"
        )
    if not all(new >= old for new, old in zip(new_shape, tensor.shape)):
        raise ValueError(
            "New shape must be at least as large as existing shape in every dim, but "
            f"got new shape {new_shape} and existing shape {tensor.shape}"
        )
    return torch.sparse_coo_tensor(
        tensor._indices(), tensor._values(), new_shape, is_coalesced=tensor.is_coalesced()
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

    new_indices, new_shape, _ = flatten_sparse_indices(tensor, start_axis, end_axis)
    new_shape: list[int] = new_shape.tolist()
    return torch.sparse_coo_tensor(
        new_indices,
        tensor.values(),
        new_shape,
        is_coalesced=tensor.is_coalesced(),  # indices still unique and in correct order
    )
