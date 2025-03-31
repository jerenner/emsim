import re
from typing import Optional, Union

import sparse
import torch
from torch import Tensor
import torch.nn.functional as F

from ..batching_utils import (
    batch_dim_to_leading_index,
    deconcat_add_batch_dim,
    remove_batch_dim_and_concat,
)


def torch_sparse_to_pydata_sparse(tensor: Tensor) -> sparse.SparseArray:
    assert tensor.is_sparse
    tensor = tensor.detach().cpu().coalesce()
    assert tensor.is_coalesced
    nonzero_values = tensor.values().nonzero(as_tuple=True)
    return sparse.COO(
        tensor.indices()[:, nonzero_values[0]],
        tensor.values()[nonzero_values],
        tensor.shape,
        has_duplicates=False,
    )


def pydata_sparse_to_torch_sparse(
    sparse_array: sparse.SparseArray, device: Optional[torch.device] = None
) -> Tensor:
    return torch.sparse_coo_tensor(
        indices=sparse_array.coords,
        values=sparse_array.data,
        size=sparse_array.shape,
        device=device,
    ).coalesce()


def sparse_select(tensor: Tensor, axis: int, index: int) -> Tensor:
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


# sometimes this error happens, link to potential workaround
# https://github.com/pytorch/pytorch/issues/69078#issuecomment-1087217720
@torch.jit.script
def _sparse_index_select_inner(
    tensor_indices: Tensor, tensor_values: Tensor, axis: int, index: Tensor
) -> tuple[Tensor, Tensor]:
    index_masks = tensor_indices[axis] == index.unsqueeze(1)
    match_count = index_masks.sum(1)
    # selected_items = torch.where(index_masks)[1]
    selected_items = index_masks.nonzero()[:, 1]
    new_values = tensor_values[selected_items]
    selected_indices = tensor_indices[:, selected_items]
    # new_values = tensor_values.expand_as(index_masks)[index_masks]
    # selected_indices = tensor_indices.unsqueeze(1).expand(-1, index_masks.shape[0], -1)[:, index_masks]

    leading_indices = selected_indices[:axis]
    axis_indices = torch.repeat_interleave(
        torch.arange(
            index_masks.shape[0],
            device=tensor_indices.device,
            dtype=tensor_indices.dtype,
        ),
        match_count,
    ).unsqueeze(0)
    trailing_indices = selected_indices[axis + 1 :]
    new_indices = torch.cat([leading_indices, axis_indices, trailing_indices], 0)

    return new_indices, new_values


@torch.jit.script
def sparse_index_select(tensor: Tensor, axis: int, index: Tensor) -> Tensor:
    if not tensor.requires_grad:
        return tensor.index_select(axis, index.long()).coalesce()
    assert tensor.is_sparse
    tensor = tensor.coalesce()
    assert index.ndim <= 1
    if axis < 0:
        axis = tensor.ndim + axis
    assert axis >= 0
    assert (
        index.max() <= tensor.shape[axis]
    ), "index tensor has entries out of bounds for axis"
    tensor_indices = tensor.indices()
    tensor_values = tensor.values()

    new_indices, new_values = _sparse_index_select_inner(
        tensor_indices, tensor_values, axis, index
    )
    new_shape = list(tensor.shape)
    new_shape[axis] = len(index)
    assert len(new_shape) == tensor.ndim
    return torch.sparse_coo_tensor(new_indices, new_values, new_shape).coalesce()


def sparse_squeeze_dense_dim(tensor: Tensor) -> Tensor:
    assert tensor.is_sparse
    assert tensor.dense_dim() > 0, "Tensor has no dense dim to squeeze"
    assert tensor.shape[-1] == 1, f"Tensor dense dim is non-singleton: {tensor.shape=}"
    tensor = tensor.coalesce()
    return torch.sparse_coo_tensor(
        tensor.indices(),
        tensor.values().squeeze(-1),
        tensor.shape[:-1],
        requires_grad=tensor.requires_grad,
        is_coalesced=tensor.is_coalesced(),
    ).coalesce()


def sparse_resize(tensor: Tensor, new_shape: list[int]) -> Tensor:
    assert tensor.is_sparse
    assert len(new_shape) == tensor.ndim
    assert all(new >= old for new, old in zip(new_shape, tensor.shape))
    return torch.sparse_coo_tensor(
        tensor.indices(), tensor.values(), new_shape, is_coalesced=tensor.is_coalesced()
    ).coalesce()


@torch.jit.script
def sparse_flatten_hw(tensor: Tensor) -> Tensor:
    assert tensor.is_sparse
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
def __flattened_indices(
    tensor: Tensor, start_axis: int, end_axis: int
) -> tuple[Tensor, Tensor, Tensor]:
    tensor_indices = tensor.indices()
    indices_to_flatten = tensor_indices[start_axis : end_axis + 1]

    shape = torch._shape_as_tensor(tensor).to(tensor.device)
    dim_sizes_1 = torch.cat(
        [
            shape[start_axis : end_axis + 1],
            torch.ones(1, device=tensor.device, dtype=torch.long),
        ]
    )

    # reverse cumprod
    dim_linear_offsets = dim_sizes_1.flip([0]).cumprod(0).flip([0])[1:]

    # compute strided 1D indices over the flattened dims
    flattened_indices = (indices_to_flatten * dim_linear_offsets.unsqueeze(-1)).sum(
        0, keepdim=True
    )

    new_shape = torch.cat(
        [shape[:start_axis], dim_sizes_1.prod(0, keepdim=True), shape[end_axis + 1 :]]
    )
    assert new_shape.size(0) == tensor.ndim - (end_axis - start_axis)

    new_indices = torch.cat(
        [tensor_indices[:start_axis], flattened_indices, tensor_indices[end_axis + 1 :]]
    )
    return new_indices, new_shape, dim_linear_offsets


@torch.jit.script
def sparse_flatten(tensor: Tensor, start_axis: int, end_axis: int) -> Tensor:
    assert tensor.is_sparse
    if start_axis < 0:
        start_axis = tensor.ndim + start_axis
    if end_axis < 0:
        end_axis = tensor.ndim + end_axis
    assert end_axis > start_axis
    assert start_axis >= 0
    assert end_axis <= tensor.ndim
    tensor = tensor.coalesce()

    new_indices, new_shape, _ = __flattened_indices(tensor, start_axis, end_axis)
    new_shape: list[int] = new_shape.tolist()
    return torch.sparse_coo_tensor(
        new_indices,
        tensor.values(),
        new_shape,
        is_coalesced=tensor.is_coalesced(),
    )


@torch.jit.script
def linearize_sparse_and_index_tensors(
    sparse_tensor: Tensor, index_tensor: Tensor
) -> tuple[Tensor, Tensor]:
    if index_tensor.shape[-1] != sparse_tensor.sparse_dim():
        if (
            sparse_tensor.sparse_dim() - 1 == index_tensor.shape[-1]
            and sparse_tensor.shape[-1] == 1
            and sparse_tensor.dense_dim() == 0
        ):
            sparse_tensor = sparse_tensor[..., 0].coalesce()
        else:
            # build error like this because of torchscript not liking f strings
            error_str = "Expected last dim of `index_tensor` to be the same as "
            error_str += "`sparse_tensor.sparse_dim()`, got "
            error_str += str(index_tensor.shape[-1])
            error_str += " and "
            error_str += str(sparse_tensor.sparse_dim())
            error_str += ", respectively."
            raise ValueError(error_str)

    if index_tensor.shape[-1] != sparse_tensor.sparse_dim():
        assert index_tensor.shape[-1] == sparse_tensor.sparse_dim() - 1
        index_tensor = batch_dim_to_leading_index(index_tensor)

    sparse_tensor_indices_linear, _, dim_linear_offsets = __flattened_indices(
        sparse_tensor, 0, sparse_tensor.sparse_dim() - 1
    )
    sparse_tensor_indices_linear.squeeze_(0)

    index_tensor_linearized = (index_tensor * dim_linear_offsets).sum(-1).view(-1)

    return (
        sparse_tensor_indices_linear,
        index_tensor_linearized,
    )


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


@torch.jit.script
def _get_sparse_index_mapping(
    sparse_tensor: Tensor, index_tensor: Tensor
) -> tuple[Tensor, Tensor]:
    """Finds the locations along a sparse tensor's values tensor for specified sparse indices"""
    sparse_dim = sparse_tensor.sparse_dim()
    sparse_tensor_shape = torch._shape_as_tensor(sparse_tensor).to(
        device=index_tensor.device
    )

    sparse_shape = sparse_tensor_shape[:sparse_dim]

    # Check for out of bounds indices (below 0 or outside tensor dim)
    out_of_bounds_indices = torch.any(index_tensor < 0, -1)
    out_of_bounds_indices.logical_or_(torch.any(index_tensor > sparse_shape, -1))

    # put dummy value of 0 in the OOB indices.
    # Maybe it'll make the linearization computations and searchsorted faster
    index_tensor = index_tensor.clone().masked_fill_(
        out_of_bounds_indices.unsqueeze(-1), 0
    )
    (
        sparse_tensor_indices_linearized,
        index_tensor_linearized,
    ) = linearize_sparse_and_index_tensors(sparse_tensor, index_tensor)

    index_search = torch.searchsorted(
        sparse_tensor_indices_linearized, index_tensor_linearized
    ).clamp_max(sparse_tensor_indices_linearized.shape[0] - 1)

    is_specified_mask: Tensor = (
        sparse_tensor_indices_linearized[index_search] == index_tensor_linearized
    )
    is_specified_mask.logical_and_(~out_of_bounds_indices.view(-1))

    return index_search, is_specified_mask


@torch.jit.script
def _gather_and_mask(values: Tensor, indices: Tensor, mask: Tensor):
    """Performs values[indices].masked_fill(mask, 0) efficiently"""
    assert values.ndim == 2
    assert indices.ndim == 1
    assert indices.shape == mask.shape

    # significantly faster than values[indices] for some reason
    selected = torch.gather(
        values, 0, indices.unsqueeze(-1).expand(-1, values.shape[-1])
    )

    selected.masked_fill_(mask.unsqueeze(-1), 0)
    return selected


@torch.jit.script
def batch_sparse_index(
    sparse_tensor: Tensor, index_tensor: Tensor, check_all_specified: bool = False
) -> tuple[Tensor, Tensor]:
    """Batch selection of elements from a torch sparse tensor. Should be
    equivalent to sparse_tensor[index_tensor]. It works by flattening the sparse
    tensor's sparse dims and the index tensor to 1D (and converting n-d indices
    to raveled indices), then using searchsorted along the flattened sparse
    tensor indices.

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

    index_search, is_specified_mask = _get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )
    if check_all_specified and not is_specified_mask.all():
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    selected = _gather_and_mask(sparse_tensor_values, index_search, ~is_specified_mask)

    out_shape = list(index_tensor.shape[:-1])
    if dense_dim > 0:
        out_shape.extend(sparse_tensor.shape[-dense_dim:])
    selected = selected.view(out_shape)
    is_specified_mask = is_specified_mask.view(out_shape[:-1])

    return selected, is_specified_mask


class GatherAndTransformFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        sparse_tensor_values: Tensor,
        index_search: Tensor,
        is_specified_mask: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Performs F.linear(sparse_tensor_values[index_search], weight, bias)
        with minimal memory use.
        """
        selected = _gather_and_mask(
            sparse_tensor_values, index_search, ~is_specified_mask
        )
        out = F.linear(selected, weight, bias)

        ctx.save_for_backward(sparse_tensor_values, weight, bias)
        ctx.index_search = index_search
        ctx.is_specified_mask = is_specified_mask

        return out

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor):
        sparse_tensor_values, weight, bias = ctx.saved_tensors
        index_search = ctx.index_search
        is_specified_mask = ctx.is_specified_mask

        grad_values = None
        grad_weight = None
        grad_bias = None

        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(0)

        if ctx.needs_input_grad[3]:
            selected = _gather_and_mask(
                sparse_tensor_values, index_search, ~is_specified_mask
            )
            grad_weight = torch.mm(grad_output.t(), selected)

        if ctx.needs_input_grad[0]:
            grad_selected = torch.mm(grad_output, weight)
            grad_selected.masked_fill_(~is_specified_mask.unsqueeze(-1))

            grad_values = torch.zeros_like(sparse_tensor_values)
            grad_values.index_add_(0, index_search, grad_selected)

        return grad_values, None, None, grad_weight, grad_bias


@torch.jit.script
def batch_sparse_index_linear(
    sparse_tensor: Tensor,
    index_tensor: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    check_all_specified: bool = False,
) -> tuple[Tensor, Tensor]:
    """Batch selection of elements from a torch sparse tensor followed by a.
    linear transformation. Should be equivalent to
    F.linear(sparse_tensor[index_tensor], weight, bias). It works by flattening
    the sparse tensor's sparse dims and the index tensor to 1D
    (and converting n-d indices to raveled indices), then using searchsorted
    along the flattened sparse tensor indices. Then, the retrieved values are
    linearly transformed according to the input weight and optional bias in
    a custom autograd function to avoid storing an extra tensor of the retrieved
    sparse values.

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension ..., M; where ... are
            S leading sparse dimensions and M is the dense dimension.
        index_tensor (Tensor): Long tensor of dimension ..., S; where ... are
            leading batch dimensions. Negative indices are not supported and will
            be considered unspecified.
        weight (Tensor): Weight matrix, of shape [out_dim, in_dim]
        bias (Optional[Tensor]): Optional bias vector, of shape [out_dim]
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

    index_search, is_specified_mask = _get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )
    if check_all_specified and not is_specified_mask.all():
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    # Call into custom grad function
    transformed = GatherAndTransformFunction.apply(
        sparse_tensor_values, index_search, is_specified_mask, weight, bias
    )

    out_shape = list(index_tensor.shape[:-1])
    out_shape.append(weight.size(0))
    transformed = transformed.view(out_shape)
    is_specified_mask = is_specified_mask.view(out_shape[:-1])

    return transformed, is_specified_mask


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


@torch.jit.script
def batch_offsets_from_sparse_tensor_indices(indices_tensor: Tensor) -> Tensor:
    """Gets the batch offsets from an index tensor where the first element of the
    first dimension is the batch index, e.g. the indices() tensor of a sparse
    torch.Tensor.

    Args:
        indices_tensor (torch.Tensor): A tensor of shape (M x nnz), where M is
        the number of dimensions of the underlying sparse tensor and nnz is the
        # number of nonzero elements in the sparse tensor. Assumes the sparse
        # tensor has been coalesce()d.

    Returns:
        torch.Tensor: A 1D tensor with elements corresponding the the first
        incidence of each unique element in the first position of the M axis,
        i.e., the batch offsets if the first element is the batch index.
    """
    assert not torch.is_floating_point(indices_tensor)
    batch_indices = indices_tensor[0]
    max_batch_index = batch_indices.max()
    matching_indices = batch_indices.unsqueeze(-1) == torch.arange(
        max_batch_index + 1, device=batch_indices.device, dtype=batch_indices.dtype
    )
    out = matching_indices.to(torch.uint8).argmax(0)
    return out


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


@torch.jit.script
def sparse_tensor_to_batched(sparse_tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    assert sparse_tensor.is_sparse
    batch_offsets = batch_offsets_from_sparse_tensor_indices(sparse_tensor.indices())
    batched_tensor, pad_mask = deconcat_add_batch_dim(
        sparse_tensor.values(), batch_offsets
    )
    batched_indices, pad_mask_2 = deconcat_add_batch_dim(
        sparse_tensor.indices().T, batch_offsets
    )
    assert torch.equal(pad_mask, pad_mask_2)
    return batched_tensor, batched_indices, pad_mask


@torch.jit.script
def batched_sparse_tensor_to_sparse(
    batched_values: Tensor,
    batched_indices: Tensor,
    pad_mask: Tensor,
    sparse_shape: list[int],
) -> Tensor:
    stacked_values, batch_offsets = remove_batch_dim_and_concat(
        batched_values, pad_mask
    )
    stacked_indices, batch_offsets_2 = remove_batch_dim_and_concat(
        batched_indices, pad_mask
    )
    assert torch.equal(batch_offsets, batch_offsets_2)
    return torch.sparse_coo_tensor(
        stacked_indices.T, stacked_values, sparse_shape
    ).coalesce()


@torch.jit.script
def multilevel_normalized_xy(sparse_tensor: Tensor, spatial_shapes: Tensor) -> Tensor:
    assert sparse_tensor.ndim == 5  # batch, i, j, level, feature
    assert spatial_shapes.ndim == 2  # i, j
    spatial_shapes_per_token = spatial_shapes[sparse_tensor.indices()[3]]
    normalized_shapes = (
        sparse_tensor.indices()[1:3].T / spatial_shapes_per_token
    ).flip(-1)
    return normalized_shapes


def __trim(subtensor: Tensor) -> Tensor:
    subtensor = subtensor.coalesce()
    indices, values = subtensor.indices(), subtensor.values()
    shape = subtensor.shape
    n_electrons = indices[0].max().item() + 1
    new_shape = (n_electrons, *shape[1:])
    return torch.sparse_coo_tensor(indices, values, new_shape).coalesce()


def bhwn_to_nhw_iterator_over_batches_torch(tensor: Tensor) -> list[Tensor]:
    assert tensor.is_sparse
    tensor = tensor.permute(0, 3, 1, 2).coalesce()

    return [__trim(t) for t in tensor.unbind()]


def bhwn_to_nhw_iterator_over_batches_pydata_sparse(array: sparse.SparseArray):
    assert isinstance(array, sparse.SparseArray)

    array = array.transpose([0, 3, 1, 2])

    def trim(subarray: sparse.SparseArray):
        max_electron_index = subarray.coords[0].max() + 1
        return subarray[:max_electron_index]

    return [trim(subarray) for subarray in array]


def flatten_multi_level_sparse_maps_to_nested(tensors: list[Tensor]):
    # tensors is a list of pytorch sparse tensors of dimension batch x height x width x channel
    assert all(isinstance(tensor, Tensor) for tensor in tensors)
    assert all(tensor.is_sparse for tensor in tensors)
    feature_dims = [tensor.shape[-1] for tensor in tensors]
    if len(set(feature_dims)) != 1:
        raise ValueError(
            f"Expected all feature maps to have same feature dimension, got {feature_dims}"
        )

    batch_size = [tensor.shape[0] for tensor in tensors]
    if len(set(batch_size)) != 1:
        raise ValueError(
            f"Expected all feature maps to have same batch size, got {batch_size}"
        )
    batch_size = batch_size[0]

    tensors_unbatched = [tensor.unbind() for tensor in tensors]
    flattened_feature_tensors = [[] for _ in range(batch_size)]
    flattened_level_indices = [[] for _ in range(batch_size)]
    flattened_pixel_indices = [[] for _ in range(batch_size)]
    for level_index, level_tensors in enumerate(tensors_unbatched):
        for b in range(batch_size):
            tensor = level_tensors[b].coalesce()
            flattened_feature_tensors[b].append(tensor.values())
            flattened_level_indices[b].append(
                torch.full(
                    [tensor.values().shape[0]], level_index, device=tensor.device
                )
            )
            flattened_pixel_indices[b].append(tensor.indices().T)

    flattened_feature_tensors = [
        torch.cat(values) for values in flattened_feature_tensors
    ]
    flattened_level_indices = [
        torch.cat(indices) for indices in flattened_level_indices
    ]
    flattened_pixel_indices = [
        torch.cat(indices) for indices in flattened_pixel_indices
    ]

    features = torch.nested.as_nested_tensor(flattened_feature_tensors)
    levels = torch.nested.as_nested_tensor(flattened_level_indices)
    indices = torch.nested.as_nested_tensor(flattened_pixel_indices)

    return features, levels, indices


def nested_flattened_tensors_to_sparse_tensors(
    features: Tensor,
    levels: Tensor,
    indices: Tensor,
    level_spatial_shapes: list[list[int]],
):
    all_levels = torch.unique(torch.cat([torch.unique(lev) for lev in levels]))
    batch_size = features.size(0)
    assert levels.size(0) == indices.size(0) == batch_size

    features_unbatched = features.unbind()
    levels_unbatched = levels.unbind()
    indices_unbatched = indices.unbind()

    values_per_level = [[] for _ in range(len(all_levels))]
    indices_per_level = [[] for _ in range(len(all_levels))]

    for level_index, level_vals, level_inds in zip(
        all_levels, values_per_level, indices_per_level
    ):
        for batch_index, (feat_b, level_b, index_b) in enumerate(
            zip(features_unbatched, levels_unbatched, indices_unbatched)
        ):
            in_level = level_b == level_index
            n_elements = torch.count_nonzero(in_level)
            level_vals.append(feat_b[in_level])
            level_inds.append(
                torch.cat(
                    [
                        index_b.new_full([1, n_elements], batch_index),
                        index_b[in_level].T,
                    ],
                    0,
                )
            )

    values_per_level = [torch.cat(values, 0) for values in values_per_level]
    indices_per_level = [torch.cat(indices, 1) for indices in indices_per_level]

    out = [
        torch.sparse_coo_tensor(
            level_inds, level_vals, [batch_size] + level_shape + [level_vals.shape[-1]]
        ).coalesce()
        for level_inds, level_vals, level_shape in zip(
            indices_per_level, values_per_level, level_spatial_shapes
        )
    ]
    return out


def unpack_sparse_tensors(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """
    Takes in a batch dict and converts packed sparse tensors (with separate
    indices and values tensors, and shape tuple) into sparse torch.Tensors

    Args:
        batch (dict[str, Tensor]): Input batch dict

    Returns:
        dict[str, Tensor]: Input batch dict with sparse tensors unpacked into
        sparse torch.Tensor format
    """
    prefixes_indices = [
        match[0]
        for match in [re.match(".+(?=_indices$)", key) for key in batch.keys()]
        if match is not None
    ]
    prefixes_values = [
        match[0]
        for match in [re.match(".+(?=_values$)", key) for key in batch.keys()]
        if match is not None
    ]
    prefixes_shape = [
        match[0]
        for match in [re.match(".+(?=_shape$)", key) for key in batch.keys()]
        if match is not None
    ]
    prefixes = list(set(prefixes_indices) & set(prefixes_values) & set(prefixes_shape))
    for prefix in prefixes:
        assert not batch[prefix + "_values"].requires_grad
        shape = batch[prefix + "_shape"]
        if isinstance(shape, Tensor):
            shape = shape.tolist()
        batch[prefix] = torch.sparse_coo_tensor(
            batch[prefix + "_indices"],
            batch[prefix + "_values"],
            shape,
            dtype=batch[prefix + "_values"].dtype,
            device=batch[prefix + "_values"].device,
        ).coalesce()
        del batch[prefix + "_indices"]
        del batch[prefix + "_values"]
        del batch[prefix + "_shape"]
    return batch
