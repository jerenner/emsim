from typing import NamedTuple, Optional, Union

import torch
from torch import Tensor

from ..batch_utils import batch_offsets_to_seq_lengths, seq_lengths_to_batch_offsets


class BatchTopK(NamedTuple):
    indices: Tensor
    offsets: Tensor
    values: Optional[Tensor] = None


def _topk_out(
    input_tensor: Tensor,
    k: int,
    dim: int,
    largest: bool,
    sorted: bool,
    out_values: Tensor,
    out_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Helper function to handle both Pytorch and Torchscript topk variants."""
    if torch.jit.is_scripting():  # type: ignore
        return torch.topk(
            input_tensor,
            k,
            dim,
            largest,
            sorted,
            values=out_values,  # type: ignore
            indices=out_indices,  # type: ignore
        )
    else:
        return torch.topk(
            input_tensor, k, dim, largest, sorted, out=(out_values, out_indices)
        )


@torch.jit.script
def _normalize_k(
    k: Union[Tensor, list[int], int], batch_size: int, device: torch.device
) -> Tensor:
    if isinstance(k, int):
        return torch.full((batch_size,), k, dtype=torch.long, device=device)
    if isinstance(k, list):
        return torch.tensor(k, dtype=torch.long, device=device)
    assert isinstance(k, Tensor)
    return k.to(device=device, dtype=torch.long)


def batch_topk(
    tensor: Tensor,
    batch_offsets: Tensor,
    k: Union[Tensor, list[int], int],
    dim: int = 0,
    largest: bool = True,
    sorted: bool = True,
    return_values: bool = False,
) -> BatchTopK:
    """
    Performs top-k operation on a batch-concatenated tensor with variable sequence lengths.

    This function handles both uniform-length sequences (where a more efficient batch
    operation is used) and variable-length sequences (where per-batch processing occurs).
    The function returns indices adjusted to the original tensor's indexing space
    along with offsets to identify each batch's results.

    Args:
        tensor (Tensor): A batch-concatenated tensor of shape (total_length, d1, d2, ...)
            where total_length is the sum of all sequence lengths.
        batch_offsets (Tensor): A 1D tensor of indices indicating where each sequence
            begins in the batch-concatenated tensor. Should be of shape (batch_size + 1,)
            with the last element being the total length.
        k (Union[Tensor, int]): Number of top elements to select. Can be an integer
            for the same k across all batches, or a tensor or list for different k per
            batch. Will be clamped to each sequence's length if k > sequence_length.
        dim (int, optional): Dimension along which to perform the top-k operation for
            each concatenated subsequence. Default: 0 (sequence dimension).
        largest (bool, optional): If True, returns the indices of the largest elements.
            If False, returns those of the smallest elements. Default: True.
        sorted (bool, optional): If True, returns the elements in sorted order.
            Default: True.
        return_values (bool, optional): If True, the output namedtuple will include the
            topk values in addition to the indices and offsets. Default: False.

    Returns:
        BatchTopK: A namedtuple containing:
            - indices (Tensor): 1-D long tensor of the top-k indices in the original
                concatenated tensor space.
            - offsets (Tensor): 1-D long tensor of offsets with shape (batch_size + 1)
                indicating where each batch's results begin in the topk_indices tensor.
                Like the input batch_offsets,
                BatchTopK.offsets[-1] == len(BatchTopK.indices).
            - values (Optional[Tensor]): 1-D tensor of the same length as `indices` with
                the actual topk values. Returned only if return_values if True, otherwise
                is None.

    Usage Notes:
        1.  Slice per sequence
            The results for the b-th input sequence are obtained with

                idx_b   = out.indices[out.offsets[b] : out.offsets[b + 1]]
                if out.values is not None:
                    val_b = out.values[out.offsets[b] : out.offsets[b + 1]]

            where `out = batch_topk(...)`.

        2.  Expected slice length
            Let
                seq_len_b = batch_offsets[b + 1] - batch_offsets[b]
                k_b       = min(k, seq_len_b)       # or k[b] if k is per-sample

            length(idx_b) is

                k_b * prod(tensor.size(i) for i not in {0, dim})      if dim == 0
                k_b * seq_len_b * prod(tensor.size(i) for i not in {0, dim})   otherwise

            This is exactly the number of elements returned by a
            regular torch.topk call on that subsequence.

        3.  Reshaping to the usual top-k shape
            After selecting `subseq_b = tensor[batch_offsets[b] : batch_offsets[b + 1]]`
            you can turn the flat index slice back into the layout produced by
            torch.topk:

                out_shape = list(subseq_b.shape)
                out_shape[dim] = k_b
                idx_b = idx_b.view(*out_shape)

            The same `out_shape` works for `val_b` when it is present.

        4.  Recovering the values when they were not returned
            If `return_values=False`, the values can still be gathered later.

            -  When ``dim != 0`` the indices are **local** to every subsequence,
               therefore you can gather directly:

                   val_b = torch.take_along_dim(subseq_b, idx_b, dim)

            -  When ``dim == 0`` the indices are expressed in the **global**
               (concatenated-tensor) space.  Either use them on the original
               concatenated tensor,

                   val_b = torch.take_along_dim(tensor, idx_b, dim)

               or convert them back to local coordinates first:

                   idx_b_local = idx_b - batch_offsets[b]
                   val_b       = torch.take_along_dim(subseq_b, idx_b_local, dim)
    """
    seq_lens = batch_offsets_to_seq_lengths(batch_offsets)
    assert isinstance(seq_lens, Tensor)
    batch_size = seq_lens.numel()

    # Normalize k
    k = _normalize_k(k, batch_size, tensor.device)
    assert isinstance(k, Tensor)
    if torch.any(k < 0):
        raise ValueError(f"Expected nonnegative value for `k`, got {k}")

    # Normalize dim if negative
    dim = dim if dim >= 0 else dim + tensor.ndim
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(
            "Normalized dimension "
            f"{dim} is out of bounds for tensor with {tensor.ndim} dimensions"
        )

    # Early return for empty tensor
    if batch_size == 0 or tensor.numel() == 0:
        topk_indices = torch.empty(0, device=tensor.device, dtype=torch.long)
        topk_offsets = torch.zeros(1, device=tensor.device, dtype=torch.long)
        topk_values = tensor[0:0].flatten() if return_values else None  # keep gradients
        return BatchTopK(topk_indices, topk_offsets, topk_values)

    # Find product of dims besides seq and topk dims
    prod_extra_dims = 1
    for i, s in enumerate(tensor.shape):
        if i not in (0, dim):
            prod_extra_dims *= s

    same_len = torch.all(seq_lens == seq_lens[0])
    same_k = torch.all(k == k[0])

    if same_len:  # sequences same length, batch for efficiency
        # Clamp k to seq length
        seq_len_int = int(seq_lens[0].item())
        if dim == 0:
            k_max = torch.min(k.amax(), seq_lens[0])
        else:
            k_max = k.amax().clamp_max(tensor.shape[dim])
        k = k.clamp_max(k_max)
        k_max_int = int(k_max.item())

        # Compute per-batch result size
        if dim == 0:
            out_sizes = k * prod_extra_dims
        else:
            out_sizes = k * seq_len_int * prod_extra_dims
        topk_offsets = seq_lengths_to_batch_offsets(out_sizes)
        assert isinstance(topk_offsets, Tensor)

        if k_max_int == 0:
            topk_indices = tensor.new_empty(0, dtype=torch.long)
            topk_values = tensor[0:0].flatten() if return_values else None
            return BatchTopK(topk_indices, topk_offsets, topk_values)

        # reshape to [bsz, seq_len, ...]
        batch_shape = (batch_size, seq_len_int) + tensor.shape[1:]
        topk_dim = dim + 1  # account for new leading batch dim

        values_all, indices_all = tensor.reshape(batch_shape).topk(
            k_max_int, topk_dim, largest=largest, sorted=sorted
        )

        # If topk is along sequence length, need to add offsets to indices
        # to globalize them
        if dim == 0:
            unsqueeze_shape = (batch_size,) + (1,) * (indices_all.ndim - 1)
            indices_all = indices_all + batch_offsets[:-1].view(unsqueeze_shape)

        if same_k:
            topk_indices = indices_all.flatten()
            topk_values = values_all.flatten() if return_values else None
            return BatchTopK(topk_indices, topk_offsets, topk_values)

        # not all same k: slice into the topk output for each batch
        total_len = int(topk_offsets[-1])
        topk_indices = tensor.new_empty(total_len, dtype=torch.long)
        topk_values = tensor.new_empty(total_len) if return_values else None

        for b in range(batch_size):
            k_b = int(k[b])
            if k_b == 0:
                continue

            # slice into the already-computed result
            start, end = int(topk_offsets[b]), int(topk_offsets[b + 1])
            topk_indices[start:end] = indices_all[b].narrow(dim, 0, k_b).flatten()
            if return_values:
                assert topk_values is not None
                topk_values[start:end] = values_all[b].narrow(dim, 0, k_b).flatten()

        return BatchTopK(topk_indices, topk_offsets, topk_values)

    ##########
    # Slow path
    # -- Sequences different length, run topk for each --

    if dim == 0:
        batch_seq_ks = seq_lens.clamp_max(k)
        batch_out_sizes = batch_seq_ks * prod_extra_dims
    else:
        dim_size = tensor.size(dim)
        batch_seq_ks = k.clamp_max(dim_size)
        batch_out_sizes = batch_seq_ks * seq_lens * prod_extra_dims
    topk_offsets = seq_lengths_to_batch_offsets(batch_out_sizes)
    assert isinstance(topk_offsets, Tensor)

    # Allocate result tensor(s)
    topk_indices = torch.empty(
        int(topk_offsets[-1].item()), dtype=torch.long, device=tensor.device
    )
    topk_values: Optional[Tensor] = None
    if return_values:
        topk_values = tensor.new_empty(int(topk_offsets[-1]))

    # Allocate a "scratch" buffer to hold the topk values outputs
    max_seq_len = int(seq_lens.max().item())
    max_k = int(batch_seq_ks.max().item()) if dim == 0 else int(k.max().item())

    scratch_shape = list(tensor.shape)
    scratch_shape[dim] = max_k
    if dim != 0:
        scratch_shape[0] = max_seq_len
    scratch_values = tensor.new_empty(scratch_shape)

    # per-batch topk
    for b, k_b in enumerate(batch_seq_ks):
        k_b = int(k_b)
        batch_start, batch_end = int(batch_offsets[b]), int(batch_offsets[b + 1])
        slice_start, slice_end = int(topk_offsets[b]), int(topk_offsets[b + 1])

        # slice of the big concatted sequence tensor that represents this subsequence
        subseq_b = tensor[batch_start:batch_end]

        # Set up the view into the output topk tensor
        subseq_shape = list(subseq_b.shape)
        subseq_shape[dim] = k_b
        topk_inds_subseq_view = topk_indices[slice_start:slice_end].view(subseq_shape)

        # Set up view into reusable scratch holder for topk values output
        if dim == 0:
            val_buffer = scratch_values[:k_b]
        else:
            val_buffer = scratch_values.narrow(0, 0, subseq_b.size(0))
            val_buffer = val_buffer.narrow(dim, 0, k_b)

        _topk_out(
            subseq_b.detach(),
            k_b,
            dim=dim,
            largest=largest,
            sorted=sorted,
            out_values=val_buffer,
            out_indices=topk_inds_subseq_view,
        )
        if return_values:
            # clone topk inds to save unmodified tensor for take_along_dim backward
            values = torch.take_along_dim(subseq_b, topk_inds_subseq_view.clone(), dim)
            assert topk_values is not None
            topk_values[slice_start:slice_end] = values.reshape(-1)
        if dim == 0:
            topk_inds_subseq_view.add_(batch_start)

    return BatchTopK(topk_indices, topk_offsets, topk_values)


@torch.jit.script
def unpack_batch_topk(
    result: BatchTopK,
    batch_offsets: Tensor,
    original_shape: list[int],
    dim: int = 0,
) -> tuple[list[Tensor], Optional[list[Tensor]]]:
    """
    Re-shape and localize the flattened `indices`/`values` inside `BatchTopK` object.

    Args:
        result (BatchTopK): The object returned by :pyfunc:`batch_topk`.
        batch_offsets (Tensor): Same offsets that were passed to `batch_topk`.
        original_shape ([list[int]): ``tensor.shape`` of the concatenated tensor that
            was given to `batch_topk`.
        dim (int): Dimension along which top-k was computed (same value that was given
            to `batch_topk`).

    Returns
    -------
    indices_per_batch, values_per_batch
        Lists containing one tensor per input sequence.  Each tensor has
        the exact shape returned by a direct call to
        ``torch.topk(subseq, …)`` for that sequence.

        ``values_per_batch`` is *None* when the call to
        :pyfunc:`batch_topk` was made with ``return_values=False``.
    """
    # Normalize possibly negative dim
    dim = dim if dim >= 0 else dim + len(original_shape)

    indices_per_batch: list[Tensor] = []
    values_per_batch: Optional[list[Tensor]] = [] if result.values is not None else None

    # Compute the product of the other dims to determine subsequence topk size
    prod_other_dims = 1
    for i, s in enumerate(original_shape):
        if i not in (0, dim):
            prod_other_dims *= s

    for b in range(batch_offsets.numel() - 1):
        # Sub-range of the concatenated tensor
        start, end = int(batch_offsets[b]), int(batch_offsets[b + 1])
        seq_len_b = end - start

        # Slice into the flattened top-k output
        slice_start, slice_end = int(result.offsets[b]), int(result.offsets[b + 1])
        idx_flat_global = result.indices[slice_start:slice_end]

        # Convert to local coordinates when top-k was along the sequence dim
        idx_flat_local = idx_flat_global - start if dim == 0 else idx_flat_global

        # Derive k_b from the number of elements
        if idx_flat_local.numel() == 0:
            k_b = 0
        elif dim == 0:
            k_b = idx_flat_local.numel() // prod_other_dims
        else:
            k_b = idx_flat_local.numel() // (seq_len_b * prod_other_dims)

        # Build the full output shape for this subsequence
        out_shape = list(original_shape)
        if dim == 0:
            out_shape[0] = k_b
        else:
            out_shape[0] = seq_len_b
            out_shape[dim] = k_b

        # Reshape and store
        indices_per_batch.append(idx_flat_local.view(out_shape))

        if result.values is not None:
            vals = result.values
            assert vals is not None
            vals_b = vals[slice_start:slice_end]
            assert values_per_batch is not None
            values_per_batch.append(vals_b.view(out_shape))  # type: ignore

    return indices_per_batch, values_per_batch
