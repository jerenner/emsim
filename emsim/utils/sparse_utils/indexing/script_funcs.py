import torch
from torch import Tensor


# sometimes this fucntion errors, link to potential workaround
# https://github.com/pytorch/pytorch/issues/69078#issuecomment-1087217720
@torch.jit.script
def sparse_index_select_inner(
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
def flattened_indices(
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
            # build error str like this because of torchscript not liking f strings
            error_str = "Expected last dim of `index_tensor` to be the same as "
            error_str += "`sparse_tensor.sparse_dim()`, got "
            error_str += str(index_tensor.shape[-1])
            error_str += " and "
            error_str += str(sparse_tensor.sparse_dim())
            error_str += ", respectively."
            raise ValueError(error_str)

    sparse_tensor_indices_linear, _, dim_linear_offsets = flattened_indices(
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
def gather_and_mask(
    values: Tensor, indices: Tensor, mask: Tensor, mask_inplace: bool = True
) -> Tensor:
    """Efficiently gathers elements from a 2D tensor and applies a mask.

    This function performs the equivalent of `values[indices].masked_fill(~mask, 0)`
    but uses torch.gather for better performance. It retrieves values at the specified
    indices and zeros out values where the mask is False.

    Args:
        values (Tensor): Source tensor to gather from, must be 2D with shape (N, D)
            where N is the number of elements and D is the feature dimension.
        indices (Tensor): Long tensor of indices into the first dimension of values.
            Can be of any shape.
        mask (Tensor): Boolean tensor with the same shape as indices. True indicates
            positions to keep, False indicates positions to zero out.
        mask_inplace (bool, optional): If True, performs the masking operation
            in-place, which is more memory efficient but affects backpropagation. Set
            to False if gradient flow through the masked values is needed.
            Defaults to True.

    Returns:
        Tensor: The gathered and masked values with shape
            (*indices.shape, values.shape[-1]). Contains values from the source tensor
            at the specified indices, with masked positions filled with zeros.

    Raises:
        ValueError: If values is not 2D or if indices and mask have different shapes.
    """
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
        selected.masked_fill_(~mask_flat.unsqueeze(-1), 0)
    else:
        selected = selected.masked_fill(~mask_flat.unsqueeze(-1), 0)

    new_shape = indices.shape + (values.shape[-1],)
    selected = selected.reshape(new_shape)
    return selected
