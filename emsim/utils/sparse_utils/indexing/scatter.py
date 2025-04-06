from emsim.utils.sparse_utils.indexing.script_funcs import linearize_sparse_and_index_tensors


import torch
from torch import Tensor


def scatter_to_sparse_tensor(
    sparse_tensor: Tensor, index_tensor: Tensor, values: Tensor
) -> Tensor:
    """Batch updating of elements in a torch sparse tensor. Should be
    equivalent to sparse_tensor[index_tensor] = values. It works by flattening
    the sparse tensor's sparse dims and the index tensor to 1D (and converting
    n-d indices to raveled indices), then using index_put along the flattened
    sparse tensor.

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension ..., M; where ... are
        S leading sparse dimensions and M is the dense dimension
        index_tensor (Tensor): Long tensor of dimension ..., S; where ... are
        leading batch dimensions.
        values (Tensor): Tensor of dimension ..., M; where ... are leading
        batch dimensions and M is the dense dimension

    Returns:
        Tensor: sparse_tensor with the new values scattered into it
    """
    if index_tensor.is_nested:
        assert values.is_nested
        index_tensor = torch.cat(index_tensor.unbind())
        values = torch.cat(values.unbind())

    assert index_tensor.shape[:-1] == values.shape[:-1]
    assert sparse_tensor.dense_dim() == values.ndim - 1

    sparse_tensor_values = sparse_tensor.values()
    (
        sparse_tensor_indices_linearized,
        index_tensor_linearized,
    ) = linearize_sparse_and_index_tensors(sparse_tensor, index_tensor)

    index_search = torch.searchsorted(
        sparse_tensor_indices_linearized,
        index_tensor_linearized,
    ).clamp_max(sparse_tensor_indices_linearized.shape[0] - 1)

    is_specified_mask = (
        sparse_tensor_indices_linearized[index_search] == index_tensor_linearized
    )

    new_values = sparse_tensor_values.clone()
    new_values[index_search[is_specified_mask]] = values[is_specified_mask]
    new_values = torch.cat([new_values, values[~is_specified_mask]], 0)
    new_indices = torch.cat(
        [sparse_tensor.indices(), index_tensor[~is_specified_mask].T], -1
    )

    out = torch.sparse_coo_tensor(
        new_indices,
        new_values,
        sparse_tensor.shape,
        dtype=sparse_tensor.dtype,
        device=sparse_tensor.device,
    ).coalesce()
    return out
