import torch
from torch import Tensor

from emsim.utils.sparse_utils.indexing.script_funcs import gather_and_mask
from emsim.utils.sparse_utils.indexing.script_funcs import get_sparse_index_mapping

from .script_funcs import sparse_index_select_inner


def sparse_select(tensor: Tensor, axis: int, index: int) -> Tensor:
    """Subselects a single subtensor from a sparse tensor with working backward."""
    assert tensor.is_sparse
    tensor = tensor.coalesce()
    index_mask = tensor.indices()[axis] == index
    values = tensor.values()[index_mask]
    indices = torch.cat(
        [tensor.indices()[:axis, index_mask], tensor.indices()[axis + 1 :, index_mask]]
    )
    return torch.sparse_coo_tensor(
        indices, values, tensor.shape[:axis] + tensor.shape[axis + 1 :]
    ).coalesce()


@torch.jit.script
def sparse_index_select(
    tensor: Tensor,
    axis: int,
    index: Tensor,
    check_bounds: bool = True,
) -> Tensor:
    """Selects values from a sparse tensor along a specified dimension.

    This function is equivalent to tensor.index_select(axis, index) but works
    correctly with the backward pass for sparse tensors. It returns a new sparse
    tensor containing only the values at the specified indices along the given axis.

    This function falls back to the built-in tensor.index_select(axis, index)
    when gradients are not required and the sparse tensor is on cpu. Benchmarking
    on an A100 (Pytorch 2.5, CUDA 12.1) seems to indicate the built-in version is
    alawys more memory efficient, and always faster on CPU. On CUDA, the built-in
    version is faster if we are selecting more than about 500 indices or if there
    are no matches at selected indices in the sparse tensor.
    In this function, we use the built-in implementation when we don't require
    gradients and are either on cpu or are on CUDA with "index" longer than 500
    elements.

    Note that the built-in tensor.index_select will trigger mysterious errors
    of the form "RuntimeError: CUDA error: device-side assert triggered" if it is
    given indices outside the bounds of a sparse tensor.
    Unlike the built-in tensor.index_select, this function validates that indices
    are within bounds (when check_bounds=True), making it a safer alternative even
    when gradient support isn't needed.

    Args:
        tensor (Tensor): The input sparse tensor from which to select values.
        axis (int): The dimension along which to select values. Can be negative
            to index from the end.
        index (Tensor): The indices of the values to select along the specified
            dimension. Must be a 1D tensor or scalar.
        check_bounds (bool, optional): Whether to check if indices are within bounds.
            Set to False if indices are guaranteed to be in-bounds to avoid a CPU sync
            on CUDA tensors. Benchmarking shows the bounds check leads to an overhead
            of about 5% on cpu and 10% on cuda. Defaults to True.

    Returns:
        Tensor: A new sparse tensor containing the selected values.

    Raises:
        ValueError:
            - If the input tensor is not sparse.
            - If the index tensor has invalid shape.
            - If the axis is out of bounds for tensor dimensions.
            - If check_bounds is True and the index tensor contains out-of-bounds
              indices.
    """
    if not tensor.is_sparse:
        raise ValueError("Input tensor must be sparse")

    # Validate index tensor shape
    if index.ndim > 1:
        raise ValueError(f"Index tensor must be 0D or 1D, got {index.ndim}D")
    elif index.ndim == 0:
        index = index.unsqueeze(0)

    # Normalize negative axis
    orig_axis = axis
    if axis < 0:
        axis = tensor.ndim + axis

    # Validate axis
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(
            f"Axis {orig_axis} out of bounds for tensor with {tensor.ndim} dimensions"
        )

    # Validate index bounds (optional)
    if check_bounds and index.numel() > 0:
        out_of_bounds = ((index < 0) | (index >= tensor.shape[axis])).any()
        if out_of_bounds:  # CPU sync happens here
            raise ValueError(
                f"Index tensor has entries out of bounds for axis {orig_axis} with size {tensor.shape[axis]}"
            )

    if not tensor.requires_grad and (
        tensor.is_cpu or (tensor.is_cuda and index.size(0) > 500)
        # breakpoint of 500 could be more finely profiled
    ):
        # Fall back to built-in implementation
        return tensor.index_select(axis, index.long()).coalesce()

    tensor = tensor.coalesce()

    tensor_indices = tensor.indices()
    tensor_values = tensor.values()

    new_indices, new_values = sparse_index_select_inner(
        tensor_indices, tensor_values, axis, index
    )

    new_shape = list(tensor.shape)
    new_shape[axis] = len(index)

    return torch.sparse_coo_tensor(new_indices, new_values, new_shape).coalesce()


@torch.jit.script
def batch_sparse_index(
    sparse_tensor: Tensor, index_tensor: Tensor, check_all_specified: bool = False
) -> tuple[Tensor, Tensor]:
    """Batch selection of elements from a torch sparse tensor. Should be
    equivalent to sparse_tensor[index_tensor].

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension ..., M; where ... are
        S leading sparse dimensions and M is the dense dimension.
        index_tensor (Tensor): Long tensor of dimension ..., S; where ... are
        leading batch dimensions. Negative indices are not supported and will
        be considered unspecified.
        check_all_specified (bool): If True, this function will raise a
        ValueError if any of the indices in `index_tensor` are not specified
        in `sparse_tensor`. If False, selections at unspecified indices will be
        returned with padding values of 0. Defaults to False.

    Returns:
        Tensor: Tensor of dimension ..., M; where the leading dimensions are
        the same as the batch dimensions from `index_tensor`.
        Tensor: Boolean tensor of dimension ...; where each element is True if
        the corresponding index is a specified (nonzero) element of the sparse
        tensor and False if not.
    """
    if index_tensor.is_nested:
        raise ValueError("Nested index tensor not supported")
        # return __gather_nested_index(sparse_tensor, index_tensor, check_all_specified)

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

    selected = gather_and_mask(sparse_tensor_values, index_search, is_specified_mask)

    out_shape = list(index_tensor.shape[:-1])
    if dense_dim > 0:
        out_shape.extend(sparse_tensor.shape[-dense_dim:])
    assert list(selected.shape) == out_shape
    assert list(is_specified_mask.shape) == out_shape[:-1]
    # selected = selected.view(out_shape)
    # is_specified_mask = is_specified_mask.view(out_shape[:-1])

    return selected, is_specified_mask


@torch.jit.script
def union_sparse_indices(
    sparse_tensor_1: Tensor, sparse_tensor_2: Tensor
) -> tuple[Tensor, Tensor]:
    assert sparse_tensor_1.is_sparse
    assert sparse_tensor_2.is_sparse
    assert sparse_tensor_1.sparse_dim() == sparse_tensor_2.sparse_dim()
    assert sparse_tensor_1.dense_dim() == sparse_tensor_2.dense_dim()

    M = sparse_tensor_1.sparse_dim()
    K = sparse_tensor_1.dense_dim()

    # if sparse_tensor_1.shape != sparse_tensor_2.shape:
    #     max_shape = max([tensor.shape for tensor in (sparse_tensor_1, sparse_tensor_2)])
    #     sparse_tensor_1 = sparse_tensor_1.sparse_resize_(max_shape, M, K)
    #     sparse_tensor_2 = sparse_tensor_2.sparse_resize_(max_shape, M, K)

    sparse_tensor_1 = sparse_tensor_1.coalesce()
    sparse_tensor_2 = sparse_tensor_2.coalesce()

    indices_1 = sparse_tensor_1.indices()
    values_1 = sparse_tensor_1.values()
    indices_2 = sparse_tensor_2.indices()
    values_2 = sparse_tensor_2.values()

    indices_2_2_1 = torch.cat([indices_2, indices_2, indices_1], -1)
    uniques, counts = torch.unique(indices_2_2_1, dim=-1, return_counts=True)
    tensor_1_exclusives = uniques[:, counts == 1]
    tensor_2_exclusives = uniques[:, counts == 2]

    zeros_2_shape = [tensor_1_exclusives.shape[-1]]
    zeros_2_shape.extend(sparse_tensor_2.shape[M : M + K])
    tensor_2_unioned = torch.sparse_coo_tensor(
        torch.cat([indices_2, tensor_1_exclusives], -1),
        torch.cat([values_2, values_2.new_zeros(zeros_2_shape)], 0),
        size=sparse_tensor_2.shape,
        device=sparse_tensor_2.device,
    ).coalesce()

    zeros_1_shape = [tensor_2_exclusives.shape[-1]]
    zeros_1_shape.extend(sparse_tensor_1.shape[M : M + K])
    tensor_1_unioned = torch.sparse_coo_tensor(
        torch.cat([indices_1, tensor_2_exclusives], -1),
        torch.cat([values_1, values_1.new_zeros(zeros_1_shape)], 0),
        size=sparse_tensor_1.shape,
        device=sparse_tensor_1.device,
    ).coalesce()

    assert torch.equal(tensor_1_unioned.indices(), tensor_2_unioned.indices())

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
