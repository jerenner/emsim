import numpy as np
import torch
from torch import Tensor


def deconcat_add_batch_dim(tensor: Tensor, batch_offsets: Tensor):
    assert batch_offsets.ndim == 1
    if batch_offsets[-1] != len(tensor):
        batch_offsets = torch.cat([batch_offsets, batch_offsets.new_tensor([len(tensor)])])
    batchsize = len(batch_offsets) - 1
    seq_lens = batch_offsets[1:] - batch_offsets[:-1]
    max_len = max(seq_lens)
    feature_dim = tensor.shape[-1]

    out = tensor.new_zeros(
        [batchsize, max_len, feature_dim], requires_grad=tensor.requires_grad
    )
    for b, out_b in zip(range(batchsize), out):
        start = batch_offsets[b]
        end = batch_offsets[b + 1]
        out_b[: end - start] = tensor[start:end]

    return out


def remove_batch_dim_and_concat(tensor: Tensor):
    batch_size = tensor.shape[0]
    max_len = tensor.shape[1]
    batch_index = torch.repeat_interleave(
        torch.arange(0, batch_size, device=tensor.device), max_len, 0
    )
    concatted = torch.reshape(tensor, [batch_size * max_len, *tensor.shape[2:]])
    return concatted, batch_index


def batch_dim_to_leading_index(tensor: Tensor):
    batch_size = tensor.shape[0]
    last_dim = tensor.shape[-1]
    other_dims = tensor.shape[1:-1]
    batch_index = torch.repeat_interleave(
        torch.arange(batch_size, device=tensor.device), np.prod(other_dims), 0
    )
    flattened = torch.concat([batch_index.unsqueeze(-1), tensor.view(-1, last_dim)], -1)
    return flattened.reshape(batch_size, *other_dims, last_dim + 1)
