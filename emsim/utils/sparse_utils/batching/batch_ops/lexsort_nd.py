from typing import Optional

import torch
from torch import Tensor

from emsim.utils.sparse_utils.misc import prod


def _lexsort_nd_robust(tensor: Tensor, descending: bool) -> Tensor:
    """Iterative (true) lexicographic sort. Complexity: O(V * N log N)

    Input tensor shape: [sort_len, ..., vector_len], with ... as batch dims
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

    return perm


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
    eps = _EPSILON[tensor.dtype]  # epsilon based on original tensor dtype
    diffs = torch.diff(proj_sorted_2d, dim=0).abs()
    rel_factor = torch.maximum(proj_sorted_2d[:-1].abs(), proj_sorted_2d[1:].abs())
    pair_tol = eps * rel_factor  # (N-1, M)
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

            org_block = vecs_3d[orig_rows, batch]  # (G, V)

            if (org_block - org_block[0]).abs().max() == 0.0:
                # all vectors are exactly identical – put in original row index order
                batch_indices[tied_vectors] = orig_rows.sort()[0]
            else:
                # robust lexicographic sort (from original order) of this small block
                orig_abs_indices, orig_relative_order = orig_rows.sort()
                new_order = _lexsort_nd_robust(
                    org_block[orig_relative_order], descending
                )
                batch_indices[tied_vectors] = orig_abs_indices[new_order]

        idx_sorted_2d[:, batch] = batch_indices  # commit the whole batch

    # reshape back to the original layout and return
    return idx_sorted_2d.view_as(idx_sorted)


def _safe_float_dtype(
    input_dtype: torch.dtype, vec_len: int, step: int = 2
) -> Optional[torch.dtype]:
    """Return the narrowest dtype that can represent the full weight vector."""
    exp_span = step * (vec_len - 1)
    if exp_span <= _MANTISSA_BITS[input_dtype]:
        return input_dtype
    if (
        input_dtype in (torch.float16, torch.bfloat16)
        and exp_span <= _MANTISSA_BITS[torch.float32]
    ):
        return torch.float32
    if exp_span <= _MANTISSA_BITS[torch.float64]:
        return torch.float64
    return None  # even fp64 is not enough


def _lexsort_nd_float(
    tensor: Tensor,
    descending: bool = False,
    stable: bool = False,
    skip_fixup: bool = False,
) -> Tensor:
    """Lexicographically sorts floating-point tensors by projecting each vector into
    a 1D basis defined by powers of 2.

    This function normalizes each vector input element to [0, 1], and projects the
    vectors into a 1D basis defined as w_i = 2 ^ (step * (V - 1) - step * i) for
    i \\in {0, ..., V}. The value of `step` is taken to be the mantissa bits of
    `tensor`'s dtype minus 1. The input vectors are potentially upcasted if V is too
    large to fit larger w_i's into the input tensor's dtype.
    After sorting the projected elements, this function "breaks ties" with a targeted
    fixup routine that stably sorts the original values whose projected versions were
    found to be equal, either due to rounding errors or the input vectors actually
    being equal. This fixup requires N_equal * V sorts of N_equal_elements, but in
    practice should be significantly faster than the initial sort.

    Args:
        tensor (Tensor): Floating-point tensor to be sorted. Must be permuted to have
            the sort dim as the first dim and the vector dim as the last dim, with
            batch dims in the middle.
        descending (bool): Whether the sort should be in descending order.
            Default: False.
        stable (bool): Whether the sort should be stable (ordering of equal elements
            kept). Note that the post-processing fixup robust sort already stably
            sorts all duplicate elements, so the only real effect of this flag being
            True is to make the initial projection sort slightly more expensive but
            ensure its run-to-run consistency. Default: False.
        skip_fixup (bool): If True, the final step of stably re-sorting values that are
            equal in the projection space is skipped. Setting this to True means that
            the output is not actually guaranteed to be sorted, so only do so if
            performance is more important than correctness in your application.
            Default: False.

    Returns:
        sort_indices (Tensor): Long tensor of shape `tensor.shape[:-1]` with sort indices
            for the input tensor. The sorted vectors are retrievable with
            `tensor.gather(0, sort_indices.unsqueeze(-1).expand_as(tensor))`
    """
    vector_len = tensor.size(-1)

    # upcast bfloat - too flaky
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()

    # Exponential step size for base-2 encoding. Could be lowered to squeeze more
    # vector dims into smaller float dtypes at the cost of decreased initial sort
    # precision
    exp_step = _MANTISSA_BITS[tensor.dtype] - 1

    dtype = _safe_float_dtype(tensor.dtype, vector_len, step=exp_step)
    if dtype is None:  # Vector too long to losslessly compress: use robust impl
        return _lexsort_nd_robust(tensor, descending)

    if dtype != tensor.dtype:
        exp_step = _MANTISSA_BITS[dtype] - 1

    # 1. Normalize
    tensor_conv = tensor.to(dtype)
    component_min, component_max = tensor_conv.aminmax(dim=0, keepdim=True)
    component_range = component_max - component_min
    eps = _EPSILON[dtype]
    tensor_normed = (tensor_conv - component_min) / component_range.clamp_min(eps)

    # 2. weight vector (declining powers of 2)
    start_exp = exp_step * (vector_len - 1)
    weights = torch.exp2(
        torch.arange(start_exp, -1, -exp_step, dtype=dtype, device=tensor.device)
    )  # shape (vector_len,)
    # weights = torch.logspace(start_exp, 0, vector_len, base=2, device=weights.device) # alternate method

    # 3. projection + sort
    proj = torch.matmul(tensor_normed, weights)
    proj_sorted, idx_sorted = torch.sort(
        proj, dim=0, descending=descending, stable=stable
    )

    # 4. break ties
    if not skip_fixup:
        idx_sorted = _break_float_ties(tensor, proj_sorted, idx_sorted, descending)
    return idx_sorted


@torch.jit.script
def _reduce_bitwise_or(tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    out_shape = list(tensor.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape.pop(dim)
    out = tensor.new_zeros(out_shape)
    for subtensor in tensor.unbind(dim):
        if keepdim:
            subtensor = subtensor.unsqueeze(dim)
        out |= subtensor
    return out


def _lexsort_nd_int(tensor: Tensor, descending: bool, stable: bool) -> Tensor:
    """Lexicographically sorts integer tensors of vectors by packing each vector into a
    64-bit scalar key.

    If the input values cannot be compressed to 64 bits due to the vector dimension
    being too large and/or the ranges of values being too large, this function falls
    back to the "true" multi-pass lexicographic sort.

    Args:
        tensor (Tensor): Integer tensor to be sorted. Must be permuted to have the sort
            dim as the first dim and the vector dim as the last dim, with batch dims
            in the middle.
        descending (bool): Whether the sort should be in descending order. Default: False
        stable (bool): Whether the sort should be stable (ordering of equal elements kept).
            Default: False

    Returns:
        sort_indices (Tensor): Long tensor of shape `tensor.shape[:-1]` with sort indices
            for the input tensor. The sorted vectors are retrievable with
            `tensor.gather(0, sort_indices.unsqueeze(-1).expand_as(tensor))`
    """
    vector_len = tensor.size(-1)

    # 1. Componentwise min/max across the sort dimension
    # (1, ..., vector_len)
    component_min, component_max = tensor.aminmax(dim=0, keepdim=True)
    component_range = component_max.long() - component_min.long()

    if (component_range < 0).any():  # Integer overflow
        # attempt sorting with float
        # (will itself fall back to robust if it can't sort)
        return _lexsort_nd_float(tensor.double(), descending=descending, stable=stable)
        # return _lexsort_nd_robust(tensor, descending)

    # 2. Absolute largest range of values for each vector component across batches
    max_range = component_range.view(-1, vector_len).max(dim=0)[0]  # (vector_len,)

    # bits needed per component

    # max_range_int: list[int] = max_range.tolist()
    # bits = [range_.bit_length() for range_ in max_range_int]
    # total_bits = sum(bits)
    # bits_tensor = torch.tensor(bits, device=tensor.device, dtype=torch.long)

    bits_tensor = (max_range + 1).log2().ceil().long()
    total_bits = bits_tensor.sum()

    if total_bits >= 63:
        # cannot fit in last 63 bits of long: fall back to robust multi-pass sort
        return _lexsort_nd_robust(tensor, descending)

    # 3. shift map (most-significant component first)
    shifts = torch.zeros_like(bits_tensor)
    shifts[:-1] = bits_tensor[1:].flip(0).cumsum(0).flip(0)
    # add leading broadcast dims
    shifts = shifts.view([1] * (tensor.ndim - 1) + [vector_len])

    # 4. build 64-bit key
    global_min = component_min.view(-1, vector_len).amin(0).view_as(shifts)
    normalized = tensor.long() - global_min.long()
    key = normalized << shifts
    key = key.sum(-1)

    # Handle descending for unsigned integers
    if descending and tensor.dtype in (
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ):  # Future guard: as of now uints above 8 don't support sort
        key = key.bitwise_not()
        descending = False  # ascending of flipped bits

    # 5. single sort on the key
    _, sort_indices = key.sort(dim=0, descending=descending, stable=stable)
    return sort_indices


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
    Both integer and floating-point tensors are supported, with different 1D-ization
    schemes used for each.

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
        - If the input tensor's dtype is bfloat16, the tensor is always upcasted to
            float32 before projection sort to avoid precision issues.
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

    if force_robust:
        indices = _lexsort_nd_robust(tensor_permuted, descending)
    elif torch.is_floating_point(tensor_permuted):
        indices = _lexsort_nd_float(tensor_permuted, descending, stable)
    elif tensor_permuted.dtype in _INT_TYPES:
        indices = _lexsort_nd_int(tensor_permuted, descending, stable)
    else:
        raise ValueError(f"Unsupported tensor dtype {tensor.dtype}")

    # Gather from the original tensor using the sort indices
    indices_unsq = indices.unsqueeze(-1)  # add singleton dim at permuted vector dim
    sorted_tensor_permuted = torch.gather(
        tensor_permuted, dim=0, index=indices_unsq.expand_as(tensor_permuted)
    )

    # Permute back to original dimension order
    inverse_perm = [0] * tensor.ndim
    for i, p in enumerate(perm):
        inverse_perm[p] = i
    sorted_tensor = sorted_tensor_permuted.permute(inverse_perm)

    sort_indices = indices_unsq.permute(inverse_perm).squeeze(vector_dim)

    return sorted_tensor, sort_indices
