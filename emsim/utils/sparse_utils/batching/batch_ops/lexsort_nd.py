from typing import Optional, NamedTuple

import torch
from torch import Tensor

from emsim.utils.sparse_utils.misc import prod


def _lexsort_nd_robust(tensor: Tensor, descending: bool) -> tuple[Tensor, Tensor]:
    """Iterative (true) lexicographic sort. Complexity: O(V * N log N)

    Input tensor shape: [sort_len, ..., vector_len], with ... as batch dims

    Returns:
        tuple[Tensor, Tensor]:
            - Sorted tensor
            - Sort indices
    """
    vector_len = tensor.size(-1)
    sort_len = tensor.size(0)

    # This tensor will hold the running sorted indices
    perm = (
        torch.arange(sort_len, device=tensor.device)
        .view([sort_len] + [1] * (tensor.ndim - 2))
        .expand(tensor.shape[:-1])
        .contiguous()
    )

    for i in range(vector_len - 1, -1, -1):  # last element to first
        component = tensor[..., i]
        sort_indices = component.sort(dim=0, descending=descending, stable=True)[1]

        tensor = tensor.gather(0, sort_indices.unsqueeze(-1).expand_as(tensor))

        perm = perm.gather(0, sort_indices)

    return tensor, perm


_MANTISSA_BITS = {  #
    torch.float16: 11,
    torch.bfloat16: 8,
    torch.float32: 24,
    torch.float64: 53,
}

_EPSILON = {dtype: torch.finfo(dtype).eps for dtype in _MANTISSA_BITS}


def _break_float_ties(
    tensor: Tensor,
    proj_sorted: Tensor,
    idx_sorted: Tensor,
    descending: bool,
) -> Tensor:
    """
    Efficiently resolve equal-key blocks produced by the projection
    used in the fast lexicographic sort.

    Args:
        tensor (Tensor): Original unsorted data, already permuted so that the sort
            dimension is first and the vector dimension is last.
            Shape: [sort_len, ..., vector_len]
        proj_sorted (Tensor): Sorted projection values from the initial `torch.sort`.
            Shape: [sort_len, ...].
        idx_sorted (Tensor): Indices returned by that same `torch.sort`.
            Shape: [sort_len, ...].
        descending (bool): True for descending sort.

    Returns:
        Tensor: `idx_sorted`, but with ties broken so that the overall order is
            truly lexicographic.

    Notes:
        - I went around this with the wrong approach and this version needs too high
        a tolerance to catch all the possible ties
    """
    sort_len = tensor.size(0)  # N
    vector_len = tensor.size(-1)  # V
    batch_dims = tensor.shape[1:-1]  # M
    n_batches = prod(batch_dims)

    # Flatten batch dims
    proj_sorted_2d = proj_sorted.reshape(sort_len, n_batches)  # (N, M)
    idx_sorted_2d = idx_sorted.reshape(sort_len, n_batches)  # (N, M)
    vecs_3d = tensor.reshape(sort_len, n_batches, vector_len)  # (N, M, V)

    # Step 1: detect which batches actually contain ties
    diffs = torch.diff(proj_sorted_2d, dim=0).abs()
    # eps = _EPSILON[tensor.dtype]  # epsilon based on original tensor dtype
    # rel_factor = torch.maximum(proj_sorted_2d[:-1].abs(), proj_sorted_2d[1:].abs())
    # step = _MANTISSA_BITS[tensor.dtype] - 1
    # w1 = torch.exp2(
    #     torch.tensor(step * (vector_len - 2), device=tensor.device)
    # ).clamp_min(0.0)
    # pair_tol = torch.maximum(eps * rel_factor, w1)  # (N-1, M)
    pair_tol = torch.zeros_like(diffs)  # minimal change

    has_ties = (diffs <= pair_tol).any(dim=0)  # (M,)
    batches_with_ties = has_ties.nonzero().squeeze(-1)  # 1-D tensor

    if batches_with_ties.numel() == 0:  # fast exit: nothing to fix
        return idx_sorted

    # Step 2: identify the groups of equal projected elements in each batch
    for batch in batches_with_ties:
        batch_vals = proj_sorted_2d[:, batch]  # (N,)
        batch_indices = idx_sorted_2d[:, batch]  # (N,)

        # Find the first element of each (potential) group of uniques
        new_group = torch.zeros_like(batch_vals, dtype=torch.bool)
        new_group[1:] = (batch_vals[1:] - batch_vals[:-1]).abs() > pair_tol[:, batch]

        # Monotonically nondecreasing group id
        group_id = new_group.cumsum(0)  # (N,)

        # Any id that occurs more than once corresponds to a tie
        counts = torch.bincount(group_id)
        tied_groups = (counts > 1).nonzero().squeeze(-1)

        if tied_groups.numel() == 0:
            continue

        # Step 3: re-sort the vectors inside each tied group with robust lexsort
        for g in tied_groups:
            tied_vectors = (group_id == g).nonzero().squeeze(-1)
            if tied_vectors.numel() <= 1:
                # For safety - shouldn't hit this
                continue

            # `rows` are positions *within the sorted column*,
            # `orig_rows` are original positions in `tensor_perm`
            orig_rows = batch_indices[tied_vectors]

            orig_block = vecs_3d[orig_rows, batch]  # (G, V)

            if (orig_block - orig_block[0]).abs().max() == 0.0:
                # all vectors are exactly identical – put in original row index order
                batch_indices[tied_vectors] = orig_rows.sort()[0]
            else:
                # robust lexicographic sort (from original order) of this small block
                orig_abs_indices, orig_relative_order = orig_rows.sort()
                _, new_order = _lexsort_nd_robust(
                    orig_block[orig_relative_order], descending=descending
                )
                batch_indices[tied_vectors] = orig_abs_indices[new_order]

        idx_sorted_2d[:, batch] = batch_indices  # commit the whole batch

    # reshape back to the original layout and return
    return idx_sorted_2d.view_as(idx_sorted)


def _lexsort_nd_float(
    tensor: Tensor,
    descending: bool = False,
    stable: bool = False,
) -> tuple[Tensor, Tensor]:
    """Lexicographically sorts floating-point tensors.

    For now, this function just falls back to the robust sort. In the future, it will
    use a multi-pass top-down segmented sort instead for greater efficiency.

    Args:
        tensor (Tensor): Floating-point tensor to be sorted. Must be permuted to have
            the sort dim as the first dim and the vector dim as the last dim, with
            batch dims in the middle.
        descending (bool): Whether the sort should be in descending order.
            Default: False.
        stable (bool): Whether the sort should be stable (ordering of equal elements
            kept). Currently ignored, and the sort is always stable.

    Returns:
        sort_indices (Tensor): Long tensor of shape `tensor.shape[:-1]` with sort
            indices for the input tensor. The sorted vectors are retrievable with
            `tensor.gather(0, sort_indices.unsqueeze(-1).expand_as(tensor))`
    """
    return _lexsort_nd_robust(tensor, descending=descending)


class LexsortIntOut(NamedTuple):
    sort_indices: Tensor
    sorted_inverse: Optional[Tensor] = None
    has_duplicates: Optional[Tensor] = None


def _compute_sorted_inverse(sorted_tensor: Tensor) -> tuple[Tensor, Tensor]:
    """Computes the output of sorted_tensor.unique(dim=-1, return_inverse=True) for an
    already-sorted tensor.

    Args:
        sorted_tensor (Tensor): Sorted tensor of shape [sort_dim, ..., vector_dim]

    Returns:
        sorted_inverse (Tensor): Long tensor of shape [sort_dim, ...] with nondecreasing
            integers along the first dim specifying which of the unique vectors are in
            each index of sorted_tensor
        has_duplicates (Tensor): Boolean tensor of shape [...] that is True if that
            batch of sorted vectors has any duplicate vectors
    """
    new_group = sorted_tensor.new_zeros(sorted_tensor.shape[:-1], dtype=torch.bool)
    new_group[1:] = (sorted_tensor[1:] != sorted_tensor[:-1]).any(-1)
    sorted_inverse = new_group.cumsum(0, dtype=torch.long)
    has_duplicates = new_group[1:].all(0).logical_not()
    return sorted_inverse, has_duplicates


def _lexsort_nd_int(
    tensor: Tensor, descending: bool, stable: bool, return_unique_inverse: bool = False
) -> LexsortIntOut:
    """Lexicographically sorts integer tensors of vectors by packing each vector into a
    64-bit scalar key.

    If the input values cannot be compressed to 64 bits due to the vector dimension
    being too large and/or the ranges of values being too large, this function falls
    back to the robust sort.

    Args:
        tensor (Tensor): Integer tensor to be sorted. Must be permuted to have the sort
            dim as the first dim and the vector dim as the last dim, with batch dims
            in the middle.
        descending (bool): Whether the sort should be in descending order. Default: False
        stable (bool): Whether the sort should be stable (ordering of equal elements kept).
            Default: False
        return_unique_inverse (bool): If True, this function will also return the
            second output of unique_consecutive(return_inverse=True, dim=0) on the
            sorted keys, i.e., a tensor of ascending long integers.

    Returns:
        LexsortIntOut: A namedtuple subclass containing:
            - sort_indices (Tensor): Long tensor of shape `tensor.shape[:-1]` with sort
                indices for the input tensor. The sorted vectors are retrievable with
                `tensor.gather(0, sort_indices.unsqueeze(-1).expand_as(tensor))`
            - sorted_inverse (Optional[Tensor]): Returned if return_unique_inverse is True.
            - has_duplicates (Optional[Tensor]): A tensor with the same shape as the batch
                dimensions that specifies whether each sequence has duplicates. Returned
                only if return_unique_inverse is True.
    """
    vector_len = tensor.size(-1)

    # 1. Componentwise min/max across the sort dimension
    # (1, ..., vector_len)
    component_min, component_max = tensor.aminmax(dim=0, keepdim=True)
    component_range = component_max.long() - component_min.long()

    if (component_range + 1 < 0).any():  # Integer overflow
        # attempt sorting with float
        # (will itself fall back to robust if it can't sort)
        sort_indices = _lexsort_nd_float(
            tensor.double(), descending=descending, stable=stable
        )[1]
        if not return_unique_inverse:
            return LexsortIntOut(sort_indices)

        sorted_tensor = tensor.gather(0, sort_indices.unsqueeze(-1).expand_as(tensor))
        sorted_inverse, has_duplicates = _compute_sorted_inverse(sorted_tensor)
        return LexsortIntOut(sort_indices, sorted_inverse, has_duplicates)

    # 2. Absolute largest range of values for each vector component across batches
    max_range = component_range.view(-1, vector_len).amax(dim=0)  # (vector_len,)

    # bits needed per component
    bits_tensor = (max_range + 1).log2().ceil().long()  # (vector_len,)
    cum_bits = bits_tensor.cumsum(0).clamp_min_(1)

    # 3. Greedily assign components to subkeys
    MAX_KEY_BITS = 63
    subkey_id = (cum_bits - 1) // MAX_KEY_BITS  # (vector_len,)
    n_keys = int(subkey_id[-1].item()) + 1
    subkey_arange = torch.arange(n_keys, device=subkey_id.device)
    subkey_mask = subkey_arange.unsqueeze(1) == subkey_id.unsqueeze(0)  # (K, V)

    # 4. Compute shifts per component
    bits_per_key = bits_tensor.unsqueeze(0) * subkey_mask  # (K, V)
    bits_key_rev_cumsum = bits_per_key.flip(1).cumsum(1).flip(1)
    shift_tensor = bits_key_rev_cumsum - bits_per_key  # (K, V)

    # 5. build 64-bit keys
    global_min = component_min.view(-1, vector_len).amin(0)  # (V)
    normalized = tensor.long() - global_min.long()  # (N, ..., V)
    keys = normalized.unsqueeze(-2) << shift_tensor  # # (N, ..., K, V)
    keys = keys.masked_fill_(~subkey_mask, 0).sum(-1)  # (N, ..., K)

    # Handle descending for unsigned integers
    if descending and tensor.dtype in (
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ):  # Future guard: as of now uints above 8 don't support sort
        keys = keys.bitwise_not()
        descending = False  # ascending of flipped bits

    # 6. sort on the keys
    if n_keys == 1:
        # Accomplish with a single sort
        sorted_keys, sort_indices = keys.sort(
            dim=0, descending=descending, stable=stable
        )
        sort_indices = sort_indices.squeeze(-1)
    else:
        sorted_keys, sort_indices = _lexsort_nd_robust(
            keys,
            descending=descending,
        )
    if return_unique_inverse:
        sorted_inverse, has_duplicates = _compute_sorted_inverse(sorted_keys)
        return LexsortIntOut(sort_indices, sorted_inverse, has_duplicates)
    return LexsortIntOut(sort_indices)


_INT_TYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


def _permute_dims(
    tensor: Tensor, vector_dim: int, sort_dim: int
) -> tuple[Tensor, list[int]]:
    perm = list(range(tensor.ndim))
    perm.remove(vector_dim)
    perm.remove(sort_dim)
    perm = [sort_dim] + perm + [vector_dim]

    tensor_permuted = tensor.permute(perm)
    return tensor_permuted, perm


# @torch.jit.script
def lexsort_nd(
    tensor: Tensor,
    vector_dim: int,
    sort_dim: int,
    descending: bool = False,
    stable: bool = False,
    force_robust: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorts a tensor of vectors in lexicographic order.

    Given a tensor of vectors, performs a sort that orders the vectors
    in lexicographic order. The vectors are defined along the `vector_dim` dimension,
    and sorted along the `sort_dim` dimension.
    If `force_robust` is False, then a fast lexicographic sort based on projecting the
    vectors to an order-preserving 1D basis is used if possible, falling back to a
    "robust" (true) multi-pass lexicographic sort if the input vectors cannot be
    losslessly compressed to 1D. If `force_robust` is True, the robust sort is always
    used.
    Both integer and floating-point tensors are supported.

    Args:
        tensor (Tensor): Tensor to be sorted.
        vector_dim (int): Index along which vectors are defined.
        sort_dim (int): Index along which to sort.
        descending (bool): If True, vectors are sorted in descending order. Default: False.
        stable (bool): If True, stable sort is always used (order of equivalent values is kept).
            If False, unstable sorts are used when possible.
        force_robust (bool): If True, always use the "true" iterative lexsort. This requires
            tensor.shape[vector_dim] sorts instead of 1 sort, but is more reproducible.

    Returns:
        tuple[Tensor, Tensor]:
            - Tensor: Sorted tensor.
            - Tensor: Sort indices.

    Notes:
        - The relationship between the sorted tensor and the sort indices is:
            sort_indices_exp = sort_indices.unsqueeze(vector_dim).expand_as(tensor)
            sorted_tensor = tensor.gather(sort_dim, sort_indices_exp).
    """
    # Normalize dims
    ndim = tensor.ndim
    vector_dim = vector_dim if vector_dim >= 0 else vector_dim + ndim
    sort_dim = sort_dim if sort_dim >= 0 else sort_dim + ndim

    # Input checks
    if vector_dim < 0 or vector_dim >= ndim:
        raise ValueError(
            f"Normalized key_dim {vector_dim} is out of bounds for tensor with {ndim} "
            "dimensions."
        )
    if sort_dim < 0 or sort_dim >= ndim:
        raise ValueError(
            f"Normalized sort_dim {sort_dim} is out of bounds for tensor with {ndim} "
            "dimensions."
        )
    if sort_dim == vector_dim:
        raise ValueError(
            f"Expected vector_dim and sort_dim to be different, but got both "
            f"= {sort_dim}"
        )
    if tensor.isnan().any():
        raise ValueError("Tensor has nan values.")
    if tensor.isinf().any():
        raise ValueError("Tensor has infinite values.")

    # Get vector length
    vector_len = tensor.shape[vector_dim]

    # Handle edge cases
    if tensor.numel() == 0:
        indices_shape = list(tensor.shape)
        indices_shape.pop(vector_dim)
        return tensor, torch.zeros(
            indices_shape, device=tensor.device, dtype=torch.long
        )
    if tensor.size(sort_dim) == 1:
        indices_shape = list(tensor.shape)
        indices_shape.pop(vector_dim)
        return tensor, torch.zeros(
            indices_shape, device=tensor.device, dtype=torch.long
        )
    if vector_len == 1:  # Just do regular sort
        tensor, sort_indices = torch.sort(
            tensor,
            dim=sort_dim,
            descending=descending,
            stable=stable,
        )
        sort_indices = sort_indices.squeeze(vector_dim)
        return tensor, sort_indices

    # Move vector_dim to last position for projection reduction
    # and sort_dim to first position for faster sorting
    tensor_permuted, perm = _permute_dims(tensor, vector_dim, sort_dim)
    tensor_permuted = tensor_permuted.contiguous()

    # Pick appropriate sorting subroutine
    if force_robust:
        sorted_tensor_permuted, indices = _lexsort_nd_robust(
            tensor_permuted, descending=descending
        )
    elif torch.is_floating_point(tensor_permuted):
        sorted_tensor_permuted, indices = _lexsort_nd_float(
            tensor_permuted, descending, stable
        )
    elif tensor_permuted.dtype in _INT_TYPES:
        indices = _lexsort_nd_int(tensor_permuted, descending, stable).sort_indices
        sorted_tensor_permuted = None
    else:
        raise ValueError(f"Unsupported tensor dtype {tensor.dtype}")

    # Gather from the original tensor using the sort indices
    indices_unsq = indices.unsqueeze(-1)  # add singleton dim at permuted vector dim
    if sorted_tensor_permuted is None:  # get sorted tensor if not returned already
        sorted_tensor_permuted = torch.gather(
            tensor_permuted, dim=0, index=indices_unsq.expand_as(tensor_permuted)
        )

    # Permute tensor and indices back to original dimension order
    inverse_perm = [0] * tensor.ndim
    for i, p in enumerate(perm):
        inverse_perm[p] = i
    sorted_tensor = sorted_tensor_permuted.permute(inverse_perm)

    sort_indices = indices_unsq.permute(inverse_perm).squeeze(vector_dim)

    return sorted_tensor, sort_indices
