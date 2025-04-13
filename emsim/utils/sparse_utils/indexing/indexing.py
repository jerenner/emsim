import torch
from torch import Tensor

from emsim.utils.sparse_utils.indexing.script_funcs import (
    gather_and_mask,
    get_sparse_index_mapping,
)


@torch.jit.script
def sparse_select(tensor: Tensor, axis: int, index: int) -> Tensor:
    """Selects a single subtensor from a sparse tensor along the specified axis.

    This function extracts a slice of a sparse tensor by selecting elements
    where the coordinate along the specified sparse axis matches the given index value.
    Unlike Pytorch's built-in indexing, this implementation properly handles
    gradient flow for sparse tensors.

    Args:
        tensor (Tensor): The input sparse tensor.
        axis (int): The dimension along which to select the subtensor. Negative axes
            are supported.
        index (int): The index to select along the specified axis. Negative indices
            are supported.

    Returns:
        Tensor: A new sparse tensor with one fewer dimension than the input tensor.
            The shape will be tensor.shape[:axis] + tensor.shape[axis+1:].

    Raises:
        ValueError: If the input tensor is not sparse, or if the axis or index are
            out of bounds.

    Example:
        >>> # Create a sparse tensor with dimensions (3, 4, 2), where the last
        >>> # dimension is a dense dimension
        >>> i = torch.tensor([[0, 1], [1, 3], [2, 2]].T)
        >>> v = torch.tensor([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
        >>> x = torch.sparse_coo_tensor(i, v, (3, 4, 2))
        >>>
        >>> # Select values along sparse dimension 0
        >>> slice_0 = sparse_select(x, 0, 1)  # Get elements where dim 0 == 1
        >>> # slice_0 has shape (4, 2)
        >>>
        >>> # Select values along dense dimension 2
        >>> slice_2 = sparse_select(x, 2, 0)  # Get elements where dim 2 == 0
        >>> # slice_2 has shape (3, 4)
    """
    if not tensor.is_sparse:
        raise ValueError("Input tensor is not sparse.")

    if not isinstance(index, int):
        raise ValueError(f"Expected integer index, got type {type(index)}")

    # Normalize negative axis
    orig_axis = axis
    if axis < 0:
        axis = tensor.ndim + axis

    # Validate axis
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(
            f"Axis {orig_axis} out of bounds for tensor with {tensor.ndim} dimensions"
        )

    # Normalize negative index
    orig_index = index
    if index < 0:
        index = tensor.size(axis) + index

    # check if index is in bounds
    if index < 0 or index >= tensor.size(axis):
        # weird string construction to work around torchscript not liking concatting
        # multiple f strings
        error_str = "Index " + str(orig_index) + " is out of bounds on axis "
        error_str += (
            str(orig_axis) + " for tensor with shape " + str(tensor.shape) + "."
        )
        raise ValueError(error_str)

    tensor = tensor.coalesce()

    sparse_dims = tensor.sparse_dim()
    if axis < sparse_dims:
        # Selection along a sparse dimension
        index_mask = tensor.indices()[axis] == index
        values = tensor.values()[index_mask]
        indices = torch.cat(
            [
                tensor.indices()[:axis, index_mask],
                tensor.indices()[axis + 1 :, index_mask],
            ]
        )
    else:
        # Selecting along a dense dimension
        # This means we just index the values tensor along the appropriate dim
        # and the sparse indices stay the same
        indices = tensor.indices()
        dense_axis = axis - sparse_dims + 1  # +1 because first dim of values is nnz
        values = tensor.values().select(dense_axis, index)

    return torch.sparse_coo_tensor(
        indices, values, tensor.shape[:axis] + tensor.shape[axis + 1 :]
    ).coalesce()


@torch.jit.script
def batch_sparse_index(
    sparse_tensor: Tensor, index_tensor: Tensor, check_all_specified: bool = False
) -> tuple[Tensor, Tensor]:
    """Batch selection of elements from a torch sparse tensor. The index tensor may
    have arbitrary batch dimensions.

    If dense_tensor = sparse_tensor.to_sparse(), the equivalent dense indexing
    operation would be dense_tensor[index_tensor.unbind(-1)]

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension
            [S0, S1, ..., Sn-1, D0, D1, ..., Dm-1], with n sparse dimensions and
            m dense dimensions.
        index_tensor (Tensor): Long tensor of dimension [B0, B1, ..., Bp-1, n]
            with optional p leading batch dimensions and final dimension corresponding
            to the sparse dimensions of the sparse tensor. Negative indices are not
            supported and will be considered unspecified.
        check_all_specified (bool): If True, this function will raise a
            ValueError if any of the indices in `index_tensor` are not specified
            in `sparse_tensor`. If False, selections at unspecified indices will be
            returned with padding values of 0. Defaults to False.

    Returns:
        Tensor: Tensor of dimension [B0, B1, ..., Bp-1, D0, D1, ..., Dm-1].
        Tensor: Boolean tensor of dimension [B0, B1, ..., Bp-1], where each element is
            True if the corresponding index is a specified (nonzero) element of the
            sparse tensor and False if not.

    Raises:
        ValueError: If `check_all_specified` is set to True and not all indices in
            `index_tensor` had associated values specified in `sparse_tensor`, or if
            `index_tensor` is a nested tensor (feature planned but not implemented yet)

    Implementation details:
        This function first linearizes the sparse_tensor indices and index_tensor
        entries, reducing the overall complexity from O(prod(B) * log(nnz) * n)
        to O(prod(B) * (n + log(nnz))) and making it suitable for high-dimensional
        sparse tensors.
    """
    if index_tensor.is_nested:
        raise ValueError("Nested index tensor not supported yet")

    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = sparse_tensor.values()
    dense_dim = sparse_tensor.dense_dim()

    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )
    if check_all_specified and not is_specified_mask.all():
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    selected: Tensor = gather_and_mask(
        sparse_tensor_values, index_search, is_specified_mask
    )

    out_shape = index_tensor.shape[:-1]
    assert is_specified_mask.shape == out_shape
    if dense_dim > 0:
        out_shape = out_shape + (sparse_tensor.shape[-dense_dim:])

    assert selected.shape == out_shape

    return selected, is_specified_mask


@torch.jit.script
def __sparse_or_dense(tensor: Tensor) -> str:
    return "sparse" if tensor.is_sparse else "dense"


@torch.jit.script
def union_sparse_indices(
    sparse_tensor_1: Tensor, sparse_tensor_2: Tensor
) -> tuple[Tensor, Tensor]:
    """Creates unified sparse tensors with the union of indices from both input tensors.

    This function takes two sparse tensors and returns versions of them that share the
    same set of indices (the union of indices from both inputs). For indices present in
    only one of the tensors, zeros are filled in for the corresponding values in the
    other tensor.

    This function is useful for ensuring a one-to-one correspondence between two
    sparse tensors' respective values() tensors, which in turn may be useful for
    elementwise value comparisons like loss functions.

    Args:
        sparse_tensor_1 (Tensor): First sparse tensor.
        sparse_tensor_2 (Tensor): Second sparse tensor with the same sparse and dense
            dimensions as sparse_tensor_1.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - tensor_1_unioned (Tensor): First tensor with indices expanded to include
                all indices from the second tensor (with zeros for missing values).
            - tensor_2_unioned (Tensor): Second tensor with indices expanded to include
                all indices from the first tensor (with zeros for missing values).

    Raises:
        ValueError: If either input is not a sparse tensor or if the sparse and
            dense dimensions don't match between tensors.

    Note:
        For very large sparse tensors, this operation may require significant memory
        for intermediate tensors.
    """
    if not sparse_tensor_1.is_sparse or not sparse_tensor_2.is_sparse:
        raise ValueError(
            "Expected two sparse tensors; got "
            f"{__sparse_or_dense(sparse_tensor_1)} and {__sparse_or_dense(sparse_tensor_2)}"
        )
    if sparse_tensor_1.shape != sparse_tensor_2.shape:
        raise ValueError(
            "Expected tensors to have same shapes; got "
            f"{sparse_tensor_1.shape} and {sparse_tensor_2.shape}"
        )
    if sparse_tensor_1.sparse_dim() != sparse_tensor_2.sparse_dim():
        raise ValueError(
            "Expected both sparse tensors to have equal numbers of sparse dims; got "
            f"{sparse_tensor_1.sparse_dim()} and {sparse_tensor_2.sparse_dim()}"
        )
    if sparse_tensor_1.dense_dim() != sparse_tensor_2.dense_dim():
        raise ValueError(
            "Expected both sparse tensors to have equal numbers of dense dims; got "
            f"{sparse_tensor_1.dense_dim()} and {sparse_tensor_2.dense_dim()}"
        )

    M = sparse_tensor_1.sparse_dim()
    K = sparse_tensor_1.dense_dim()

    sparse_tensor_1 = sparse_tensor_1.coalesce()
    sparse_tensor_2 = sparse_tensor_2.coalesce()

    indices_1, values_1 = sparse_tensor_1.indices(), sparse_tensor_1.values()
    indices_2, values_2 = sparse_tensor_2.indices(), sparse_tensor_2.values()

    # Need to find all indices that are unique to each sparse tensor
    # To do this, stack one of them twice and the other once
    indices_2_2_1 = torch.cat([indices_2, indices_2, indices_1], -1)
    uniques, counts = torch.unique(indices_2_2_1, dim=-1, return_counts=True)
    # Any that appear twice in the stacked indices are unique to tensor 2
    # and any that appear once are unique to tensor 1
    # (indices that appear 3x are shared already)
    indices_only_in_tensor_1 = uniques[:, counts == 1]
    indices_only_in_tensor_2 = uniques[:, counts == 2]

    # Figure out how many new indices will be added to each sparse tensor
    n_exclusives_1 = indices_only_in_tensor_1.size(-1)
    n_exclusives_2 = indices_only_in_tensor_2.size(-1)

    # Make zero-padding for new values tensors
    pad_zeros_1 = values_1.new_zeros(
        (n_exclusives_2,) + sparse_tensor_1.shape[M : M + K]
    )
    pad_zeros_2 = values_2.new_zeros(
        (n_exclusives_1,) + sparse_tensor_1.shape[M : M + K]
    )

    # Make the new tensors by stacking indices and values together
    tensor_1_unioned = torch.sparse_coo_tensor(
        torch.cat([indices_1, indices_only_in_tensor_2], -1),
        torch.cat([values_1, pad_zeros_1], 0),
        size=sparse_tensor_1.shape,
        device=sparse_tensor_1.device,
    ).coalesce()

    tensor_2_unioned = torch.sparse_coo_tensor(
        torch.cat([indices_2, indices_only_in_tensor_1], -1),
        torch.cat([values_2, pad_zeros_2], 0),
        size=sparse_tensor_2.shape,
        device=sparse_tensor_2.device,
    ).coalesce()

    if not torch.equal(tensor_1_unioned.indices(), tensor_2_unioned.indices()):
        raise RuntimeError("Internal error: unioned tensors have different indices")

    return tensor_1_unioned, tensor_2_unioned


@torch.jit.ignore
def __gather_nested_index(
    sparse_tensor: Tensor, index_tensor: Tensor, check_all_specified: bool = False
) -> tuple[Tensor, Tensor]:
    results = [
        batch_sparse_index(
            sparse_tensor,
            index_subtensor,
            check_all_specified=check_all_specified,
        )
        for index_subtensor in index_tensor.unbind()
    ]
    selected, is_specified_mask = zip(*results)

    selected = torch.nested.as_nested_tensor(selected)
    is_specified_mask = torch.nested.as_nested_tensor(is_specified_mask)
    return selected, is_specified_mask
