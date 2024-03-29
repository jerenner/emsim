from typing import Optional

import numpy as np
import torch
from torch import Tensor


def deconcat_add_batch_dim(tensor: Tensor, batch_offsets: Tensor):
    assert batch_offsets.ndim == 1
    if batch_offsets[-1] != len(tensor):
        batch_offsets = torch.cat(
            [batch_offsets, batch_offsets.new_tensor([len(tensor)])]
        )
    batchsize = len(batch_offsets) - 1
    seq_lens = batch_offsets[1:] - batch_offsets[:-1]
    max_len = max(seq_lens)
    feature_dim = tensor.shape[-1]

    out = tensor.new_zeros([batchsize, max_len, feature_dim])
    padding_mask = tensor.new_ones((batchsize, max_len), dtype=torch.bool)
    # for b, out_b, mask_b in zip(range(batchsize), out, padding_mask):
    for b in range(batchsize):
        start = batch_offsets[b]
        end = batch_offsets[b + 1]
        out[b, : end - start] = tensor[start:end]
        padding_mask[b, : end - start] = False

    return out, padding_mask


def remove_batch_dim_and_concat(tensor: Tensor, padding_mask: Optional[Tensor] = None):
    batch_size = tensor.shape[0]
    max_len = tensor.shape[1]
    if padding_mask is None:
        padding_mask = tensor.new_zeros(batch_size, max_len, dtype=torch.bool)
    assert padding_mask.ndim == 2
    nonpadded_batch_sizes = padding_mask.shape[-1] - padding_mask.sum(-1)
    batch_offsets = torch.cat(
        [nonpadded_batch_sizes.new_zeros([1]), nonpadded_batch_sizes.cumsum(-1)]
    )
    out = tensor.new_zeros(nonpadded_batch_sizes.sum().item(), *tensor.shape[2:])

    assert len(tensor) == len(batch_offsets[:-1])

    for b, (batch, nonpadded_size, batch_start_index, batch_end_index) in enumerate(
        zip(tensor, nonpadded_batch_sizes, batch_offsets[:-1], batch_offsets[1:])
    ):
        assert nonpadded_size == batch_end_index - batch_start_index
        out[batch_start_index:batch_end_index] = batch[:nonpadded_size]

    return out, batch_offsets[:-1]


def batch_dim_to_leading_index(tensor: Tensor):
    batch_size = tensor.shape[0]
    last_dim = tensor.shape[-1]
    other_dims = tensor.shape[1:-1]
    batch_index = torch.repeat_interleave(
        torch.arange(batch_size, device=tensor.device), np.prod(other_dims), 0
    )
    flattened = torch.concat([batch_index.unsqueeze(-1), tensor.view(-1, last_dim)], -1)
    return flattened.reshape(batch_size, *other_dims, last_dim + 1)
