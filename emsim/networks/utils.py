import torch
from torch import Tensor


def deconcat_batchify(tensor: Tensor, batch_offsets: Tensor):
    assert batch_offsets.ndim == 1
    if batch_offsets[-1] != len(tensor):
        batch_offsets = torch.cat([batch_offsets, torch.as_tensor([len(tensor)])])
    batchsize = len(batch_offsets) - 1
    seq_lens = batch_offsets[1:] - batch_offsets[:-1]
    max_len = max(seq_lens)
    feature_dim = tensor.shape[-1]

    out = tensor.new_zeros([batchsize, max_len, feature_dim])
    for b, out_b in zip(range(batchsize), out):
        start = batch_offsets[b]
        end = batch_offsets[b+1]
        out_b[:end-start] = tensor[start:end]

    return out
