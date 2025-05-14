from emsim.utils.sparse_utils.indexing.script_funcs import get_sparse_index_mapping


import torch
from torch import Tensor


def scatter_to_sparse_tensor(
    sparse_tensor: Tensor,
    index_tensor: Tensor,
    values: Tensor,
    check_all_specified: bool = False,
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
    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )

    all_specified = torch.all(is_specified_mask)

    if check_all_specified and not all_specified:
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    if all_specified:
        new_values = sparse_tensor_values.index_copy(
            0, index_search[is_specified_mask], values[is_specified_mask]
        )
        new_indices = sparse_tensor.indices()
    else:
        n_old = sparse_tensor_values.size(0)
        n_old_plus_new = n_old + (~is_specified_mask).sum()
        new_values: Tensor = sparse_tensor_values.new_empty(
            (n_old_plus_new,) + sparse_tensor_values.shape[1:]
        )
        new_values[:n_old] = sparse_tensor_values
        new_values.index_copy_(
            0, index_search[is_specified_mask], values[is_specified_mask]
        )
        new_values[n_old:] = values[~is_specified_mask]

        new_indices = torch.cat(
            [sparse_tensor.indices(), index_tensor[~is_specified_mask].T], -1
        )

    out = torch.sparse_coo_tensor(
        new_indices,
        new_values,
        sparse_tensor.shape,
        dtype=sparse_tensor.dtype,
        device=sparse_tensor.device,
        is_coalesced=all_specified.item(),
    ).coalesce()
    return out
