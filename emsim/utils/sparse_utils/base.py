import re
from typing import Optional, Union

import numpy as np
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
    """
    Converts a sparse tensor's multidimensional indices to linearized indices
    by flattening dimensions from start_axis to end_axis.
    """
    tensor_indices = tensor.indices()
    indices_to_flatten = tensor_indices[start_axis : end_axis + 1]

    # convert shape to tensor since we will be doing math on it.
    # it needs to be on the same device as the sparse tensor rather than
    # staying on cpu because downstream tensors will be interacting with
    # the sparse tensor's indices tensor
    shape = torch._shape_as_tensor(tensor).to(tensor.device)

    # concatenate a 1 onto the end of the dimensions to be flattened since
    # the trailing dimension will have a stride of 1
    dim_sizes_1 = torch.cat(
        [
            shape[start_axis + 1 : end_axis + 1],
            torch.ones(1, device=tensor.device, dtype=torch.long),
        ]
    )

    # calculate linear offsets for each multidimensional axis's step
    # i.e., for dims [d0, d1, d2], the offsets would be [d1*d2, d2, 1].
    # we accomplish this with a reversed cumprod
    dim_linear_offsets = dim_sizes_1.flip([0]).cumprod(0).flip([0])

    # compute strided 1D indices over the flattened dims by summing each axis's
    # individual contribution
    flattened_indices = indices_to_flatten * dim_linear_offsets.unsqueeze(-1)
    flattened_indices = flattened_indices.sum(0, keepdim=True)

    # make new shape with the flattened axes stacked together
    new_shape = torch.cat(
        [shape[:start_axis], dim_sizes_1.prod(0, keepdim=True), shape[end_axis + 1 :]]
    )
    # this assertion shouldn't cause a cpu sync
    assert new_shape.size(0) == tensor.ndim - (end_axis - start_axis)

    # plug the flattened indices into the existing indices
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
        is_coalesced=tensor.is_coalesced(),  # indices still unique and in correct order
    )


@torch.jit.script
def linearize_sparse_and_index_tensors(
    sparse_tensor: Tensor, index_tensor: Tensor
) -> tuple[Tensor, Tensor]:
    """Converts multidimensional indices of a sparse tensor and a tensor of indices
    that we want to retrieve to a shared linearized (flattened) format suitable
    for fast lookup.

    Args:
        sparse_tensor (Tensor): torch.sparse_coo_tensor with indices to linearize.
        index_tensor (Tensor): Dense tensor with indices matching sparse_tensor's
            sparse dims. Can be of any dimension as long as the last dimension
            has length equal to the sparse tensor's sparse dimension.

    Raises:
        ValueError: If the index tensor has a different last dimension than the
            sparse tensor's sparse dim.

    Returns:
        sparse_tensor_indices_linear (Tensor): Linearized version of
            sparse_tensor.indices().
        index_tensor_linearized (Tensor): Linearized version of index_tensor
            with the last dimension squeezed out.
    """
    if index_tensor.shape[-1] != sparse_tensor.sparse_dim():
        if (
            sparse_tensor.sparse_dim() - 1 == index_tensor.shape[-1]
            and sparse_tensor.shape[-1] == 1
            and sparse_tensor.dense_dim() == 0
        ):
            # handle case where there's a length-1 trailing sparse dim and the
            # index tensor ignores it
            sparse_tensor = sparse_tensor[..., 0].coalesce()
        else:
            # build error str like this because of torchscript not liking f strings
            error_str = "Expected last dim of `index_tensor` to be the same as "
            error_str += "`sparse_tensor.sparse_dim()`, got "
            error_str += str(index_tensor.shape[-1])
            error_str += " and "
            error_str += str(sparse_tensor.sparse_dim())
            error_str += ", respectively."
            raise ValueError(error_str)

    sparse_tensor_indices_linear, _, dim_linear_offsets = __flattened_indices(
        sparse_tensor, 0, sparse_tensor.sparse_dim() - 1
    )
    sparse_tensor_indices_linear.squeeze_(0)

    # repeat the index flattening for the index tensor. The sparse tensor's indices
    # were already flattened in __flattened_indices
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
def get_sparse_index_mapping(
    sparse_tensor: Tensor, index_tensor: Tensor
) -> tuple[Tensor, Tensor]:
    """Finds the locations along a sparse tensor's values tensor for specified
    sparse indices. Also returns a mask indicating which indices have values
    actually present in the sparse tensor. It works by flattening the sparse
    tensor's sparse dims and the index tensor to 1D (and converting n-d indices
    to raveled indices), then using searchsorted along the flattened sparse
    tensor indices.

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension ..., M; where ... are
            S leading sparse dimensions and M is the dense dimension.
        index_tensor (Tensor): Long tensor of dimension ..., S; where ... are
            leading batch dimensions. Negative indices and indices outside the
            bounds of the sparse dimensions are not supported and will
            be considered unspecified, with the corresponding entry in
            is_specified_mask being set to False.

    Returns:
        index_search: Long tensor of dimension ... of the locations in
            sparse_tensor.values() corresponding to the indices in index_tensor.
            Elements where is_specified_mask is False are junk data and should
            not be used.
        is_specified_mask: Boolean tensor of dimension ... that is True for
            indices in index_tensor where values where actually specified in
            the sparse tensor and False for indices that were unspecified in
            the sparse tensor.
    """
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
    # without requiring a cpu sync to pull them out of the tensor.
    index_tensor = index_tensor.masked_fill(out_of_bounds_indices.unsqueeze(-1), 0)
    (
        sparse_tensor_indices_linearized,
        index_tensor_linearized,
    ) = linearize_sparse_and_index_tensors(sparse_tensor, index_tensor)

    # The dummy value of 0 should always return searched index of 0 since
    # the sparse_tensor_indices_linearized values are always nonnegative.
    # Should be faster to find than random search values.
    index_search = torch.searchsorted(
        sparse_tensor_indices_linearized, index_tensor_linearized
    )
    # guard against IndexError
    index_search.clamp_max_(sparse_tensor_indices_linearized.shape[0] - 1)

    # Check if the indices were specified by checking for an exact match at the
    # resultant searched indices
    is_specified_mask: Tensor = (
        sparse_tensor_indices_linearized[index_search] == index_tensor_linearized
    )
    is_specified_mask.logical_and_(~out_of_bounds_indices.view(-1))

    index_search = index_search.view(index_tensor.shape[:-1])
    is_specified_mask = is_specified_mask.view(index_tensor.shape[:-1])

    return index_search, is_specified_mask


@torch.jit.script
def _gather_and_mask(
    values: Tensor, indices: Tensor, mask: Tensor, mask_inplace: bool = True
) -> Tensor:
    """Performs values[indices].masked_fill(mask, 0) efficiently.
    Set mask_inplace=False if you need the selected values to be backproppable."""
    if values.ndim != 2:
        error_str = "Expected values to be 2D, got shape "
        error_str += str(values.shape)
        raise ValueError(error_str)
    if indices.shape != mask.shape:
        error_str = "Expected indices and mask to have same shape, got "
        error_str += str(indices.shape)
        error_str += " and "
        error_str += str(mask.shape)
        raise ValueError(error_str)

    indices_flat = indices.reshape(-1)
    mask_flat = mask.reshape(-1)

    # significantly faster than values[indices] for some reason
    selected = torch.gather(
        values, 0, indices_flat.unsqueeze(-1).expand(-1, values.size(-1))
    )

    if mask_inplace:
        selected.masked_fill_(mask_flat.unsqueeze(-1), 0)
    else:
        selected = selected.masked_fill(mask_flat.unsqueeze(-1), 0)

    new_shape = indices.shape + (values.shape[-1],)
    selected = selected.reshape(new_shape)
    return selected


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

    selected = _gather_and_mask(sparse_tensor_values, index_search, ~is_specified_mask)

    out_shape = list(index_tensor.shape[:-1])
    if dense_dim > 0:
        out_shape.extend(sparse_tensor.shape[-dense_dim:])
    assert list(selected.shape) == out_shape
    assert list(is_specified_mask.shape) == out_shape[:-1]
    # selected = selected.view(out_shape)
    # is_specified_mask = is_specified_mask.view(out_shape[:-1])

    return selected, is_specified_mask


class GatherAndLinearFunction(torch.autograd.Function):
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
        ctx.set_materialize_grads(False)

        selected = _gather_and_mask(
            sparse_tensor_values, index_search, ~is_specified_mask
        )
        out = F.linear(selected, weight, bias)

        ctx.save_for_backward(sparse_tensor_values, weight, bias)
        ctx.index_search = index_search
        ctx.is_specified_mask = is_specified_mask

        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Optional[Tensor], ...]:
        sparse_tensor_values, weight, bias = ctx.saved_tensors
        index_search = ctx.index_search
        is_specified_mask = ctx.is_specified_mask

        grad_values = None
        grad_weight = None
        grad_bias = None

        if grad_output is not None:
            if bias is not None and ctx.needs_input_grad[4]:
                grad_bias = grad_output.sum(0)

            if ctx.needs_input_grad[3]:
                selected = _gather_and_mask(
                    sparse_tensor_values, index_search, ~is_specified_mask
                )
                grad_weight = torch.mm(grad_output.t(), selected)

            if ctx.needs_input_grad[0]:
                grad_selected = torch.mm(grad_output, weight)
                grad_selected.masked_fill_(~is_specified_mask.unsqueeze(-1), 0)

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
    """Batch selection of elements from a torch sparse tensor followed by a
    linear transformation. Should be equivalent to
    F.linear(sparse_tensor[index_tensor], weight, bias). The values are
    retrieved using get_sparse_index_mapping. Then, the retrieved values are
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

    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )
    if check_all_specified and not is_specified_mask.all():
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    # Call into custom grad function
    transformed = GatherAndLinearFunction.apply(
        sparse_tensor_values, index_search, is_specified_mask, weight, bias
    )

    out_shape = list(index_tensor.shape[:-1])
    out_shape.append(weight.size(0))
    assert list(transformed.shape) == out_shape
    assert list(is_specified_mask.shape) == out_shape[:-1]
    # transformed = transformed.view(out_shape)
    # is_specified_mask = is_specified_mask.view(out_shape[:-1])

    return transformed, is_specified_mask


class GatherAndSubsetAttentionFunction(torch.autograd.Function):
    @staticmethod
    def _prep_qkv(
        query_tensor: Tensor,
        sparse_tensor_values: Tensor,
        index_search: Tensor,
        is_specified_mask: Tensor,
        Wk: Tensor,
        Wv: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        n_heads: int,
        head_dim: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Computes the key and value tensor and reshapes them and the query tensor

        Returns: (q, k, v, selected (pre-q/v features))
        """
        assert query_tensor.ndim == 2
        assert index_search.ndim == 2
        assert sparse_tensor_values.ndim == 2

        n_queries = query_tensor.size(0)
        n_keys_per_query = index_search.size(1)

        selected = _gather_and_mask(
            sparse_tensor_values, index_search, ~is_specified_mask
        )

        k = F.linear(selected, Wk, bias_k)
        v = F.linear(selected, Wv, bias_v)

        # split heads
        # (n_queries, embed_dim) -> (n_queries, n_heads, head_dim)
        k = k.view(n_queries, n_keys_per_query, n_heads, head_dim)
        v = v.view(n_queries, n_keys_per_query, n_heads, head_dim)
        q = query_tensor.view(n_queries, n_heads, head_dim)

        # Move n_head dim forward
        q = q.transpose(-2, -3).contiguous()  # (n_heads, n_queries, head_dim)

        # (n_heads, n_queries, n_keys_per_query, head_dim)
        k = k.permute(2, 0, 1, 3).contiguous()
        v = v.permute(2, 0, 1, 3).contiguous()

        return q, k, v, selected

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query_tensor: Tensor,
        n_heads: int,
        sparse_tensor_values: Tensor,
        index_search: Tensor,
        is_specified_mask: Tensor,
        Wk: Tensor,
        Wv: Tensor,
        bias_k: Optional[Tensor] = None,
        bias_v: Optional[Tensor] = None,
        scale_factor: float = None,  # scaling for attn, default 1/sqrt(d)
    ) -> Tensor:
        """Performs sparse neighborhood attention with minimal memory use"""
        ctx.set_materialize_grads(False)

        assert query_tensor.ndim == 2  # (n_queries, embed_dim)
        assert index_search.ndim == 2  # (n_queries, n_keys_per_query)
        assert query_tensor.size(0) == index_search.size(0)  # n_queries
        assert index_search.shape == is_specified_mask.shape
        assert Wk.ndim == 2
        assert Wv.ndim == 2
        assert Wk.size(1) == Wv.size(1) == sparse_tensor_values.size(-1)
        assert Wk.size(0) == query_tensor.size(-1)

        n_queries = query_tensor.size(0)
        embed_dim = query_tensor.size(1)
        n_keys_per_query = index_search.size(1)
        head_dim = embed_dim // n_heads

        # save shape info
        ctx.n_queries = n_queries
        ctx.embed_dim = embed_dim
        ctx.n_heads = n_heads
        ctx.head_dim = head_dim
        ctx.n_keys_per_query = n_keys_per_query

        # default scale factor
        if scale_factor is None:
            scale_factor = embed_dim ** (-1 / 2)
        ctx.scale_factor = scale_factor

        # save tensors
        ctx.save_for_backward(
            query_tensor,
            sparse_tensor_values,
            Wk,
            Wv,
            bias_k,
            bias_v,
        )
        ctx.index_search = index_search
        ctx.is_specified_mask = is_specified_mask

        # fmt: off
        q, k, v, _ = GatherAndSubsetAttentionFunction._prep_qkv(
            query_tensor, sparse_tensor_values, index_search, is_specified_mask,
            Wk, Wv, bias_k, bias_v, n_heads, head_dim,
        )

        attn_scores = torch.matmul(
            q.unsqueeze(-2) * scale_factor, # (n_heads, n_queries, 1, head_dim)
            k.transpose(-1, -2)             # (n_heads, n_queries, head_dim, n_keys_per_query)
        ).squeeze(-2)                       # (n_heads, n_queries, n_keys_per_query)

        attn_scores.masked_fill_(~is_specified_mask, -torch.inf)
        attn_weights = attn_scores.softmax(-1)
        # nans expected if all of the keys a query tried to attend to were unspecified
        attn_weights.nan_to_num_(0.0)

        output = torch.matmul(
            attn_weights.unsqueeze(-2), # (n_heads, n_queries, 1, n_keys_per_query)
            v,                          # (n_heads, n_queries, n_keys_per_query, head_dim)
        ).squeeze(-2)                   # (n_heads, n_queries, head_dim)
        # fmt: on

        output = output.transpose(-2, -3)  # (n_queries, n_heads, head_dim)
        output = output.reshape(n_queries, embed_dim)

        ctx.attn_weights = attn_weights

        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Optional[Tensor], ...]:
        """Implements the backward pass for sparse neighborhood attn"""

        # retrieve tensors
        query_tensor, sparse_tensor_values, Wk, Wv, bias_k, bias_v = ctx.saved_tensors
        index_search: Tensor = ctx.index_search
        is_specified_mask: Tensor = ctx.is_specified_mask

        attn_weights: Tensor = ctx.attn_weights

        # retrieve shape info
        n_queries: int = ctx.n_queries
        embed_dim: int = ctx.embed_dim
        n_heads: int = ctx.n_heads
        head_dim: int = ctx.head_dim
        n_keys_per_query: int = ctx.n_keys_per_query

        # retrieve scale factor
        scale_factor: float = ctx.scale_factor

        # account for which inputs need gradients
        needs_grad_query = ctx.needs_input_grad[0]
        needs_grad_sparse_values = ctx.needs_input_grad[2]
        needs_grad_Wk = ctx.needs_input_grad[5]
        needs_grad_Wv = ctx.needs_input_grad[6]
        needs_grad_bias_k = bias_k is not None and ctx.needs_input_grad[7]
        needs_grad_bias_v = bias_v is not None and ctx.needs_input_grad[8]

        # initialize grad vars
        grad_query = None
        grad_sparse_values = None
        grad_Wk = None
        grad_Wv = None
        grad_bias_k = None
        grad_bias_v = None

        if grad_output is None:
            return (
                grad_query,  # query_tensor
                None,  # n_heads
                grad_sparse_values,  # sparse_tensor_values
                None,  # index_search
                None,  # is_specified_mask
                grad_Wk,  # Wk
                grad_Wv,  # Wv
                grad_bias_k,  # bias_k
                grad_bias_v,  # bias_v
                None,  # scale_factor
            )

        # recompute q, k, v
        # fmt: off
        q, k, v, selected = GatherAndSubsetAttentionFunction._prep_qkv(
            query_tensor, sparse_tensor_values, index_search, is_specified_mask,
            Wk, Wv, bias_k, bias_v, n_heads, head_dim,
        )
        # fmt: on

        # split heads on the grad tensor
        grad_output = grad_output.reshape(n_queries, n_heads, head_dim)
        # (n_heads, n_queries, head_dim)
        grad_output = grad_output.transpose(-2, -3).contiguous()

        if (
            needs_grad_query
            or needs_grad_sparse_values
            or needs_grad_Wk
            or needs_grad_Wv
            or needs_grad_bias_k
            or needs_grad_bias_v
        ):
            # fmt: off
            grad_attn_weights = torch.matmul(
                grad_output.unsqueeze(-2), # (n_heads, n_queries, 1, head_dim)
                v.transpose(-1, -2),       # (n_heads, n_queries, head_dim, n_keys_per_query)
            ).squeeze(-2)                  # (n_heads, n_queries, n_keys_per_query)
            # fmt: on

            # softmax gradient: dL/dz = S * (dL/dS - sum_j(S_j * dL/dS_j))
            # where z = attn_scores, S = softmax(z), dL/dS = grad_attn_weights
            # and j indexes keys
            grad_attn_scores = attn_weights * (
                grad_attn_weights
                - (attn_weights * grad_attn_weights).sum(-1, keepdim=True)
            )

            grad_attn_scores.masked_fill_(~is_specified_mask, 0)
        del v  # big tensor we no longer need

        if needs_grad_query:
            # fmt: off
            grad_q = torch.matmul(
                grad_attn_scores.unsqueeze(-2),  # (n_heads, n_queries, 1, n_keys_per_query)
                k,                               # (n_heads, n_queries, n_keys_per_query, head_dim)
            ).squeeze(-2)                        # (n_heads, n_queries, head_dim

            grad_q *= scale_factor

            # Flip dims back and stack heads
            grad_q = grad_q.transpose(-2, -3)  # (n_queries, n_heads, head_dim)
            grad_query = grad_q.reshape(n_queries, embed_dim)
            # fmt: on
        del k

        if (
            needs_grad_sparse_values
            or needs_grad_Wk
            or needs_grad_Wv
            or needs_grad_bias_k
            or needs_grad_bias_v
        ):
            if needs_grad_sparse_values or needs_grad_Wk or needs_grad_bias_k:
                # fmt: off
                grad_k = torch.matmul(
                    grad_attn_scores.unsqueeze(-1), # (n_heads, n_queries, n_keys_per_query, 1)
                    q.unsqueeze(-2) * scale_factor, # (n_heads, n_queries, 1, head_dim)
                )                                   # (n_heads, n_queries, n_keys_per_query, head_dim)
                # fmt: on

                # (n_queries, n_keys_per_query, n_heads, head_dim)
                grad_k = grad_k.permute(1, 2, 0, 3)
                grad_k = grad_k.reshape(n_queries, n_keys_per_query, embed_dim)

                # Flatten for grad calcs
                grad_k_flat = grad_k.view(-1, embed_dim)
            del grad_attn_scores
            del q

            if needs_grad_sparse_values or needs_grad_Wv or needs_grad_bias_v:
                # fmt: off
                grad_v = torch.matmul(
                    attn_weights.unsqueeze(-1), # (n_heads, n_queries, n_keys_per_query, 1)
                    grad_output.unsqueeze(-2)   # (n_heads, n_queries, 1, head_dim)
                )                               # (n_heads, n_queries, n_keys_per_query, head_dim)
                # fmt: on

                # (n_queries, n_keys_per_query, n_heads, head_dim)
                grad_v = grad_v.permute(1, 2, 0, 3)
                grad_v = grad_v.reshape(n_queries, n_keys_per_query, embed_dim)

                # Flatten for grad calcs
                grad_v_flat = grad_v.view(-1, embed_dim)

            if needs_grad_Wk or needs_grad_Wv:
                selected_flat = selected.view(-1, embed_dim)

                if needs_grad_Wk:
                    grad_Wk = torch.mm(grad_k_flat.t(), selected_flat)
                if needs_grad_Wv:
                    grad_Wv = torch.mm(grad_v_flat.t(), selected_flat)
                del selected_flat
            del selected

            if needs_grad_bias_k:
                grad_bias_k = grad_k_flat.sum(0)

            if needs_grad_bias_v:
                grad_bias_v = grad_v_flat.sum(0)

            if needs_grad_sparse_values:
                # two matrix multiplies - faster if we batch them
                grad_k_v_stacked = torch.stack([grad_k_flat, grad_v_flat])
                W_stacked = torch.stack([Wk, Wv])
                grad_selected = torch.bmm(grad_k_v_stacked, W_stacked).sum(0)
                grad_selected = grad_selected.view(
                    n_queries, n_keys_per_query, embed_dim
                )

                # Zero out grads for masked selecteds
                grad_selected.masked_fill_(~is_specified_mask.unsqueeze(-1), 0)

                # Scatter grads back into the sparse values
                grad_sparse_values = torch.zeros_like(sparse_tensor_values)
                grad_sparse_values.index_add_(
                    0, index_search.view(-1), grad_selected.view(-1, embed_dim)
                )

        return (
            grad_query,  # query_tensor
            None,  # n_heads
            grad_sparse_values,  # sparse_tensor_values
            None,  # index_search
            None,  # is_specified_mask
            grad_Wk,  # Wk
            grad_Wv,  # Wv
            grad_bias_k,  # bias_k
            grad_bias_v,  # bias_v
            None,  # scale_factor
        )


@torch.jit.script
def batch_sparse_index_subset_attn(
    sparse_tensor: Tensor,
    index_tensor: Tensor,
    query_tensor: Tensor,
    n_heads: int,
    key_weight: Tensor,
    value_weight: Tensor,
    key_bias: Optional[Tensor] = None,
    value_bias: Optional[Tensor] = None,
    scale_factor: Optional[float] = None,
    check_all_specified: bool = False,
):
    if index_tensor.is_nested:
        raise ValueError("Nested index tensor not supported")
        # return __gather_nested_index(sparse_tensor, index_tensor, check_all_specified)

    if query_tensor.shape[:-1] != index_tensor.shape[:-2]:
        error_str = "Expected the first n-1 dims of query_tensor and the first"
        error_str += "n-2 dims of index_tensor to match, got "
        error_str += str(query_tensor.shape)
        error_str += " and "
        error_str += str(index_tensor.shape)
        raise ValueError(error_str)

    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = sparse_tensor.values()

    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )
    if check_all_specified and not is_specified_mask.all():
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    # Call into custom grad function
    attended = GatherAndSubsetAttentionFunction.apply(
        query_tensor,
        n_heads,
        sparse_tensor_values,
        index_search,
        is_specified_mask,
        key_weight,
        value_weight,
        key_bias,
        value_bias,
        scale_factor,
    )

    out_shape = list(query_tensor.shape[:-1])
    out_shape.append(value_weight.size(0))
    assert list(attended.shape) == out_shape
    assert is_specified_mask.shape == index_tensor.shape[:-1]
    # attended = attended.view(out_shape)
    # is_specified_mask = is_specified_mask.view(out_shape[:-1])

    return attended, is_specified_mask


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
