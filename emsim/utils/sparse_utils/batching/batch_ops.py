from typing import Union

import torch
from torch import Tensor

from .batch_utils import batch_offsets_to_seq_lengths, seq_lengths_to_batch_offsets


@torch.jit.script
def batch_topk(
    tensor: Tensor,
    batch_offsets: Tensor,
    k: Union[Tensor, int],
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[Tensor, Tensor]:
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
            for the same k across all batches, or a tensor for different k per batch.
            Will be clamped to each sequence's length if k > sequence_length.
        dim (int, optional): Dimension along which to perform the top-k operation.
            Default: -1 (last dimension).
        largest (bool, optional): If True, returns the largest elements.
            If False, returns the smallest elements. Default: True.
        sorted (bool, optional): If True, returns the elements in sorted order.
            Default: True.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - topk_indices (Tensor): Indices of the top-k elements in the original
              concatenated tensor space.
            - topk_offsets (Tensor): Offsets indicating where each batch's results
              begin in the topk_indices tensor.
    """

    token_seq_lens: Tensor = batch_offsets_to_seq_lengths(batch_offsets)
    bsz = token_seq_lens.size(0)
    min_len, max_len = torch.aminmax(token_seq_lens)

    single_k = isinstance(k, int) or k.numel() == 1

    if min_len == max_len and single_k:  # sequences same length, batch for efficiency
        k = min_len.clamp_max(k)
        topk_offsets = seq_lengths_to_batch_offsets(
            torch.empty_like(token_seq_lens).copy_(k)
        )
        batch_shape = (bsz, int(min_len)) + tensor.shape[1:]
        _, topk_indices = tensor.reshape(batch_shape).topk(
            int(k), dim, largest=largest, sorted=sorted
        )
        topk_indices += batch_offsets[:-1].unsqueeze(1)
        topk_indices = topk_indices.flatten()
    else:  # sequences different length, run topk for each
        batch_ks = token_seq_lens.clamp_max(k)
        topk_offsets = seq_lengths_to_batch_offsets(batch_ks)
        topk_indices = torch.empty(
            topk_offsets[-1], dtype=torch.long, device=tensor.device
        )

        # holder for topk's first (values) output
        scratch_values = tensor.new_empty(int(batch_ks.max().item()))

        # per-batch topk
        for b, k_b in enumerate(batch_ks):
            k_b = int(k_b)
            batch_start, batch_end = batch_offsets[b : b + 2]
            slice_topk = slice(*topk_offsets[b : b + 2])

            torch.topk(
                tensor[batch_start:batch_end].detach(),
                k_b,
                dim=dim,
                largest=largest,
                sorted=sorted,
                out=(scratch_values[:k_b], topk_indices[slice_topk])
            )
            topk_indices[slice_topk] += batch_start
        del scratch_values

    return topk_indices, topk_offsets
