from typing import Optional

import torch
from torch import Tensor


@torch.jit.script
def _make_linear_offsets(dim_sizes: Tensor) -> Tensor:
    """Computes the strides/offsets to transform a multidimensional index into
    a flattened scalar index.

    Args:
        dim_sizes (Tensor): Tensor of dimension sizes, shape [N]

    Returns:
        Tensor: Linear offsets for each dimension, shape [N]. Last dimension
            has stride 1, first has prod(sizes[1:])
    """
    # append a trailing 1 so cumprod gives us the "next" stride
    dim_sizes_1 = dim_sizes.new_ones(dim_sizes.size(0) + 1)
    dim_sizes_1[:-1] = dim_sizes

    # calculate linear offsets for each multidimensional axis's step
    # i.e., for dims [d0, d1, d2], the offsets would be [d1*d2, d2, 1].
    # we accomplish this with a reversed cumprod
    reverse_cumprod = dim_sizes_1.flip([0]).cumprod(0).flip([0])

    # drop the first element (total number of indices)
    offsets = reverse_cumprod[1:]

    return offsets


@torch.jit.script
def flatten_nd_indices(indices: Tensor, sizes: Tensor) -> tuple[Tensor, Tensor]:
    """Flattens N-dimensional indices into 1-dimensional scalar indices.

    Args:
        indices (Tensor): Integer coordinate tensor of shape [N, B], where N
            is the number of dimensions to be flattened and B is the batch dimension.
        sizes (Tensor): Extents of every dimension, of shape [N]

    Returns:
        flat_indices (Tensor): Flattened indices tensor, of shape [1, B]
        offsets (Tensor): Strides that were used for flattening (needed for unflatten),
            of shape [N]
    """
    offsets = _make_linear_offsets(sizes)  # [N]
    flat_indices = (indices * offsets.unsqueeze(-1)).sum(0, keepdim=True)
    return flat_indices, offsets


@torch.jit.script
def unflatten_nd_indices(
    flat_indices: Tensor, dim_sizes: Tensor, offsets: Optional[Tensor] = None
) -> Tensor:
    """Reconstructs ('unflattens') N-D indices from 1D 'flattened' indices.

    Args:
        flat_indices (Tensor): Flat indices tensor of shape [1, B]
        dim_sizes (Tensor): Original sizes of every dimension, of shape [N]
        offsets (Optional[Tensor]): Offsets that were used for flattening, as returned
            by _make_linear_offsets or flatten_nd_indices. If None, it will be
            recalculated from `dim_sizes`

    Returns:
        Tensor: N-D indices tensor of shape [N, B]
    """
    if offsets is None:
        offsets = _make_linear_offsets(dim_sizes)
    assert offsets is not None
    N = dim_sizes.numel()
    B = flat_indices.size(-1)
    out = torch.empty(N, B, device=flat_indices.device, dtype=torch.long)

    # integer divide by stride
    torch.div(
        flat_indices.expand_as(out),
        offsets.unsqueeze(1),
        rounding_mode="floor",
        out=out,
    )

    # modulus by sizes
    torch.remainder(out, dim_sizes.unsqueeze(1), out=out)

    return out


@torch.jit.script
def flatten_sparse_indices(
    tensor: Tensor, start_axis: int, end_axis: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Flattens a sparse tensor's indices along specified dimensions.

    This function takes a sparse tensor and flattens its indices along a
    contiguous range of dimension. It returns the new indices, the
    corresponding new shape, and the linear offsets used in the flattening
    process.

    Args:
        tensor (Tensor): The input tensor. Its indices are expected to be in COO format.
        start_axis (int): Starting axis (inclusive) of the dimensions to flatten.
        end_axis (int): Ending axis (inclusive) of the dimensions to flatten.

    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - new_indices (Tensor): The flattened indices of shape (D, N), where D is
                the number of dimensions in the flattened tensor and N is the number
                of nonzero elements.
            - new_shape (Tensor): The new shape of the flattened tensor of shape (D,)
            - dim_linear_offsets (Tensor): The linear offsets used during flattening,
                of shape (K,), where K is the number of flattened dimensions.
    """
    tensor_indices = tensor.indices()  # sparse_dim x nnz (counterintuitive)
    indices_to_flatten = tensor_indices[start_axis : end_axis + 1]

    # convert shape to tensor since we will be doing math on it.
    # it needs to be on the same device as the sparse tensor rather than
    # staying on cpu because downstream tensors will be interacting with
    # the sparse tensor's indices tensor
    shape = torch._shape_as_tensor(tensor).to(tensor.device)

    sizes_to_flatten = shape[start_axis : end_axis + 1]

    flattened_indices, dim_linear_offsets = flatten_nd_indices(
        indices_to_flatten, sizes_to_flatten
    )

    # make new shape with the flattened axes stacked together
    new_shape = torch.cat(
        [
            shape[:start_axis],
            torch.prod(sizes_to_flatten, 0, keepdim=True),
            shape[end_axis + 1 :],
        ]
    )
    # this assertion shouldn't cause a cpu sync
    assert new_shape.size(0) == tensor.ndim - (end_axis - start_axis)

    # plug the flattened indices into the existing indices
    new_indices = torch.cat(
        [tensor_indices[:start_axis], flattened_indices, tensor_indices[end_axis + 1 :]]
    )
    return new_indices, new_shape, dim_linear_offsets


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
            raise ValueError(
                "Expected last dim of `index_tensor` to be the same as "
                "`sparse_tensor.sparse_dim()`, got "
                f"{str(index_tensor.shape[-1])} and {sparse_tensor.sparse_dim()}, "
                "respectively."
            )

    sparse_tensor_indices_linear, _, dim_linear_offsets = flatten_sparse_indices(
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


@torch.jit.script
def get_sparse_index_mapping(
    sparse_tensor: Tensor,
    index_tensor: Tensor,
    sanitize_linear_index_tensor: bool = True,
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
        sanitize_linear_index_tensor (bool): If False, then the output values at
            linear_index_tensor[~is_specified_mask] will be the "insertion position"
            that would keep the sparse tensor's indices ordered. This is useful if you
            want to insert values, but means that
            sparse_tensor.values()[linear_index_tensor] will be potentially unsafe if
            some of the "insertion position" values are out of bounds. If this arg is
            True, linear_index_tensor[~is_specified_mask] values will be set to 0.
            Defaults to True.

    Returns:
        linear_index_tensor: Long tensor of dimension ... of the locations in
            sparse_tensor.values() corresponding to the indices in index_tensor.
            Elements where is_specified_mask is False are handled according to the
            value of sanitize_linear_index_tensor.
        is_specified_mask: Boolean tensor of dimension ... that is True for
            indices in index_tensor where values where actually specified in
            the sparse tensor and False for indices that were unspecified in
            the sparse tensor.
    """
    sparse_dim = sparse_tensor.sparse_dim()
    sparse_nnz = sparse_tensor._nnz()
    sparse_tensor_shape = torch._shape_as_tensor(sparse_tensor).to(
        device=index_tensor.device
    )
    sparse_shape = sparse_tensor_shape[:sparse_dim]

    # check for empty sparse tensor
    if sparse_nnz == 0:
        linear_index_tensor = index_tensor.new_zeros(index_tensor.shape[:-1])
        is_specified_mask = index_tensor.new_zeros(
            index_tensor.shape[:-1], dtype=torch.bool
        )
        return linear_index_tensor, is_specified_mask

    # Check for out of bounds indices (below 0 or outside tensor dim)
    out_of_bounds_indices = torch.any(index_tensor < 0, -1)
    out_of_bounds_indices |= torch.any(index_tensor >= sparse_shape, -1)

    # put dummy value of 0 in the OOB indices.
    # Maybe it'll make the linearization computations and searchsorted faster:
    # a compromise between just giving searchsorted random indices to find vs
    # causing a cpu sync to call nonzeros to filter them out
    index_tensor = index_tensor.masked_fill(out_of_bounds_indices.unsqueeze(-1), 0)
    (
        sparse_tensor_indices_linearized,
        index_tensor_linearized,
    ) = linearize_sparse_and_index_tensors(sparse_tensor, index_tensor)

    # The dummy value of 0 should always return searched index of 0 since
    # the sparse_tensor_indices_linearized values are always nonnegative.
    # Should be faster to find than random search values.
    linear_index_tensor = torch.searchsorted(  # binary search
        sparse_tensor_indices_linearized, index_tensor_linearized
    )

    # linear_index_tensor is distinct from index_tensor_linearized in that
    # index_tensor_linearized has the flattened version of the index in the sparse
    # tensor, while linear_index_tensor has the corresponding index in the sparse
    # tensor's values() tensor

    # guard against IndexError
    if sanitize_linear_index_tensor:
        index_clamped = linear_index_tensor.clamp_max_(sparse_nnz - 1)
    else:
        index_clamped = linear_index_tensor.clamp_max(sparse_nnz - 1)

    # Check if the indices were specified by checking for an exact match at the
    # resultant searched indices
    is_specified_mask: Tensor = (
        sparse_tensor_indices_linearized[index_clamped] == index_tensor_linearized
    )
    is_specified_mask &= ~out_of_bounds_indices.view(-1)

    linear_index_tensor = linear_index_tensor.view(index_tensor.shape[:-1])
    is_specified_mask = is_specified_mask.view(index_tensor.shape[:-1])

    return linear_index_tensor, is_specified_mask


@torch.jit.script
def gather_mask_and_fill(
    values: Tensor, indices: Tensor, mask: Tensor, fill: Optional[Tensor] = None
) -> Tensor:
    """Efficiently gathers elements from an ND tensor, applies a mask, and fills masked
    positions.

    This function performs the equivalent of
    `out = values[indices]
    out[~mask] = fill.expand_as(out)[~mask]  # or 0
    `
    but uses torch.index_select for better performance. It retrieves values at the
    specified indices and fills positions where the mask is False with either zeros
    (default) or the provided fill values.

    Args:
        values (Tensor): Source tensor to gather from, may be 1D with shape (N)
            or n-D with shape (N, D0, D1, ...), where N is the number of elements
            and D are potentially multiple feature dimensions.
        indices (Tensor): Long tensor of indices into the first dimension of values.
            Can be of any shape.
        mask (Tensor): Boolean tensor with the same shape as indices. True indicates
            positions to keep, False indicates positions to zero out.
        fill (Optional[Tensor]): A tensor that must be broadcast-compatible with the
            final output shape. It is inserted at positions where `mask` is False.
            When None (default), a zero tensor is used.

    Returns:
        Tensor: The gathered and masked values with shape
            (*indices.shape, *values.shape[-1]). Contains values from the source tensor
            at the specified indices, with masked positions filled with zeros or from
            `fill`.

    Raises:
        ValueError: If indices and mask have different shapes.
    """
    input_values_1d = False
    if values.ndim == 1:
        input_values_1d = True
        values = values.unsqueeze(1)

    if indices.shape != mask.shape:
        raise ValueError(
            "Expected indices and mask to have same shape, got "
            f"{indices.shape} and {mask.shape}"
        )

    indices_flat = indices.reshape(-1)
    mask_flat = mask.reshape(-1)

    # figure out how much to broadcast
    value_dims = values.shape[1:]
    n_value_dims = values.ndim - 1

    new_shape = indices.shape
    if not input_values_1d:
        new_shape += value_dims

    # pre-mask the indices to guard against unsafe indices in the masked portion
    indices_flat = torch.where(mask_flat, indices_flat, torch.zeros_like(indices_flat))

    selected = values.index_select(0, indices_flat)

    # unsqueeze mask
    mask_flat = mask_flat.view((mask_flat.size(0),) + (1,) * n_value_dims)

    if fill is None:
        selected.masked_fill_(~mask_flat, 0)
    else:
        fill_broadcast = fill.expand(new_shape).reshape(selected.shape)
        selected = torch.where(mask_flat, selected, fill_broadcast)

    selected = selected.reshape(new_shape)
    return selected
