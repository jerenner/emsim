import torch
from torch import Tensor

from emsim.utils.sparse_utils.indexing.script_funcs import (
    _make_linear_offsets,
    get_sparse_index_mapping,
    unflatten_nd_indices,
)


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
        check_all_specified (bool): If True, this function will throw a ValueError
            if any of the indices specified in index_tensor are not already present
            in sparse_tensor. Default: False.

    Returns:
        Tensor: sparse_tensor with the new values scattered into it

    Notes:
        This function uses index_copy as the underlying mechanism to write new values,
            so duplicate indices in index_tensor will have the same result as other
            uses of index_copy, i.e., the result will depend on which copy occurs last.
            This imitates the behavior of scatter-like operations rather than the
            typical coalescing deduplication behavior of sparse tensors.
    """
    if index_tensor.is_nested:
        assert values.is_nested
        index_tensor = torch.cat(index_tensor.unbind())
        values = torch.cat(values.unbind())

    assert index_tensor.shape[:-1] == values.shape[:-1]
    assert sparse_tensor.dense_dim() == values.ndim - 1

    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = sparse_tensor.values()
    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor, sanitize_linear_index_tensor=False
    )

    all_specified = torch.all(is_specified_mask)

    if check_all_specified and not all_specified:
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    # In-place update of existing values
    if not sparse_tensor_values.requires_grad:
        updated_values = sparse_tensor_values.index_copy_(
            0, index_search[is_specified_mask], values[is_specified_mask]
        )
    else:
        updated_values = sparse_tensor_values.index_copy(
            0, index_search[is_specified_mask], values[is_specified_mask]
        )

    if all_specified:  # No new values to append: tensor is fully updated
        return torch.sparse_coo_tensor(
            sparse_tensor.indices(),
            updated_values,
            sparse_tensor.shape,
            dtype=sparse_tensor.dtype,
            device=sparse_tensor.device,
            is_coalesced=True,
        )

    # Need to append at least one new value: pre-sort the index tensor to save
    # the final coalesce operation

    # Pull out new values and indices to be added
    new_insert_pos: Tensor = index_search[~is_specified_mask]
    new_nd_indices = index_tensor[~is_specified_mask]
    new_values = values[~is_specified_mask]

    # Get sparse shape info for linearization
    sparse_dim = sparse_tensor.sparse_dim()
    sparse_sizes = torch.tensor(
        sparse_tensor.shape[:sparse_dim], device=sparse_tensor.device
    )

    # Obtain linearized versions of all indices for sorting
    old_indices_nd = sparse_tensor.indices()
    linear_offsets = _make_linear_offsets(sparse_sizes)
    new_indices_lin: Tensor = (new_nd_indices * linear_offsets).sum(-1)

    # Find duplicate linear indices
    unique_new_indices_lin, inverse = new_indices_lin.unique(
        sorted=True, return_inverse=True
    )

    # Use inverse of indices unique to write to new values tensor and tensor of
    # insertion positions
    deduped_new_values = new_values.new_empty(
        (unique_new_indices_lin.size(0),) + new_values.shape[1:]
    )
    deduped_insert_pos = new_insert_pos.new_empty(unique_new_indices_lin.size(0))
    # Deciding which duplicate value wins is offloaded to index_copy
    deduped_new_values.index_copy_(0, inverse, new_values)
    deduped_insert_pos.index_copy_(0, inverse, new_insert_pos)

    # Convert uniqueified flattened indices to n-D for inclusion in indices tensor
    unique_new_indices_nd = unflatten_nd_indices(
        unique_new_indices_lin.unsqueeze(0), sparse_sizes, linear_offsets
    )

    # Concatenate old and new indices/values and sort
    combined_indices, combined_values = _merge_sorted(
        old_indices_nd,
        unique_new_indices_nd,
        updated_values,
        deduped_new_values,
        deduped_insert_pos,
    )

    return torch.sparse_coo_tensor(
        combined_indices,
        combined_values,
        sparse_tensor.shape,
        dtype=sparse_tensor.dtype,
        device=sparse_tensor.device,
        is_coalesced=True,
    )


def _merge_sorted(
    old_nd: Tensor,
    new_nd: Tensor,
    old_values: Tensor,
    new_values: Tensor,
    insertion_positions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Merges two sorted sequences of sparse indices/values and return them in coalesced
    order.

    All input args must be on the same device.

    Args:
        old_nd (Tensor): [S x n_old] tensor of N-D indices
        new_nd (Tensor): [S x n_new] tensor of N-D indices
        old_values (Tensor): [n_old, ...] tensor of values
        new_values (Tensor): [n_new, ...] tensor of values
        insertion_positions (Tensor): [n_new] tensor of insertion positions in
            old_linear for each element in new_linear

    Returns:
        merged_nd: [S, n_old+n_new]
        merged_values: [n_old+n_new, ...]
    """
    device = old_nd.device
    n_old, n_new = old_nd.size(1), new_nd.size(1)
    n_total = n_old + n_new

    # determine final positions of new values
    # account for previous insertions to get final positions of new rows
    new_positions = insertion_positions + torch.arange(
        n_new, device=device, dtype=insertion_positions.dtype
    )

    # determine final positions of old values by counting how many new values are
    # inserted before each old value
    hist = torch.bincount(insertion_positions, minlength=n_old + 1)
    old_shift = torch.cumsum(hist[:-1], 0)
    old_positions = torch.arange(n_old, device=device) + old_shift

    # allocate output tensors
    merged_nd = old_nd.new_empty(old_nd.size(0), n_total)
    merged_values = old_values.new_empty((n_total,) + old_values.shape[1:])

    # insert values
    merged_nd[:, old_positions] = old_nd
    merged_nd[:, new_positions] = new_nd
    merged_values[old_positions] = old_values
    merged_values[new_positions] = new_values

    return merged_nd, merged_values
