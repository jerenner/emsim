from typing import Optional

import torch
from torch import Tensor


def split_batch_concatted_tensor(tensor: Tensor, batch_offsets: Tensor):
    return torch.tensor_split(tensor, batch_offsets[1:].cpu())


# @torch.compiler.disable
@torch.jit.script
def deconcat_add_batch_dim(
    tensor: Tensor, batch_offsets: Tensor, pad_value: float = 0.0
) -> tuple[Tensor, Tensor]:
    """Converts concatenated sequences to batched and padded sequences.

    Args:
        tensor (Tensor): A tensor of shape (total_sequence_length, D1, D2, ..., Dn)
        batch_offsets (Tensor): A 1D tensor specifying where along the first dimension
            of `tensor` each sequence starts
        pad_value (float, optional): Pad value. Defaults to 0.0.

    Returns:
        out (Tensor): A tensor of shape (batch_size, max_sequence_length, D1, D2, ..., Dn)
        padding_mask (Tensor): A boolean tensor of shape (batch_size, max_sequence_length)
            that is True at locations where `out` is padding
    """
    if not tensor.ndim >= 2:
        raise ValueError(
            f"Expected tensor with at least 2 dimensions, got {tensor.ndim}"
        )
    if not batch_offsets.ndim == 1:
        raise ValueError(f"Expected batch_offsets to be 1D, got {batch_offsets.ndim}")

    # add the total length to the end of the batch offsets if needed
    if batch_offsets[-1] != tensor.shape[0]:
        assert batch_offsets[-1] < tensor.shape[0]
        batch_offsets = torch.cat(
            [
                batch_offsets,
                torch.tensor(
                    [len(tensor)],
                    dtype=batch_offsets.dtype,
                    device=batch_offsets.device,
                ),
            ]
        )
    seq_lens = batch_offsets[1:] - batch_offsets[:-1]
    batchsize = batch_offsets.shape[0] - 1
    max_len = int(torch.max(seq_lens))

    feature_dims = tensor.shape[1:]
    out_shape = torch.Size([batchsize, max_len] + list(feature_dims))

    # If all sequences are equal length can just return a view
    if torch.all(seq_lens == max_len):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        out = tensor.view(out_shape)
        padding_mask = torch.zeros(
            batchsize, max_len, device=tensor.device, dtype=torch.bool
        )
        return out, padding_mask

    out = tensor.new_full(out_shape, pad_value)
    padding_mask = torch.ones(
        torch.Size([batchsize, max_len]), device=tensor.device, dtype=torch.bool
    )

    # Fill the output tensor with slices from the input
    for b in range(batchsize):
        start = int(batch_offsets[b])
        end = int(batch_offsets[b + 1])
        seq_len = int(seq_lens[b])
        out[b, :seq_len] = tensor[start:end]
        padding_mask[b, :seq_len] = False

    return out, padding_mask


@torch.jit.script
def remove_batch_dim_and_concat(
    tensor: Tensor, padding_mask: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    """Converts batched and padded sequences to concatenated sequences.

    Args:
        tensor (Tensor): A tensor of shape (batch_size, max_seq_length, D1, D2, ..., Dn)
        padding_mask (Tensor, optional): Optional boolean tensor of shape
            (batch_size, max_seq_length) where True indicates padded positions

    Returns:
        out (Tensor): A tensor of shape (total_seq_length, D1, D2, ..., Dn)
        batch_offsets (Tensor): A 1D tensor indicating where each batch element starts
    """
    assert (
        tensor.ndim >= 3
    ), f"Expected tensor with at least 3 dimensions; got {tensor.ndim}"
    batch_size = tensor.shape[0]
    max_len = tensor.shape[1]
    feature_dims = tensor.shape[2:]

    if padding_mask is not None:
        if not padding_mask.ndim == 2:
            raise ValueError(f"Expected padding_mask to be 2D, got {padding_mask.ndim}")
        if not padding_mask.shape[0] == batch_size:
            raise ValueError("Batch size mismatch between tensor and padding_mask")
        if not padding_mask.shape[1] == max_len:
            raise ValueError("Sequence length mismatch between tensor and padding_mask")

    if padding_mask is None or not padding_mask.any():
        # All sequences are same length so can just return a view
        total_len = batch_size * max_len
        out_shape = torch.Size([total_len] + list(feature_dims))
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        out = tensor.view(out_shape)
        batch_offsets = torch.arange(0, total_len, max_len, device=tensor.device)

        return out, batch_offsets

    nonpadded_seq_lens = padding_mask.shape[-1] - padding_mask.sum(-1)
    batch_offsets = torch.cat(
        [nonpadded_seq_lens.new_zeros([1]), nonpadded_seq_lens.cumsum(-1)]
    )
    total_len = int(batch_offsets[-1])

    out_shape = torch.Size([total_len] + list(feature_dims))
    out = torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)

    for b in range(batch_size):
        seq_len = int(nonpadded_seq_lens[b])
        if seq_len == 0:
            continue

        batch_start_index = int(batch_offsets[b])
        batch_end_index = int(batch_offsets[b + 1])
        out[batch_start_index:batch_end_index] = tensor[b, :seq_len]

    return out, batch_offsets[:-1]


# @torch.compiler.disable
@torch.jit.script
def batch_dim_to_leading_index(tensor: Tensor) -> Tensor:
    batch_size = tensor.shape[0]
    last_dim = tensor.shape[-1]
    other_dims = torch._shape_as_tensor(tensor)[1:-1]
    batch_index = torch.repeat_interleave(
        torch.arange(batch_size, device=tensor.device), torch.prod(other_dims), 0
    )
    flattened = torch.concat([batch_index.unsqueeze(-1), tensor.view(-1, last_dim)], -1)
    new_shape = tensor.shape
    new_shape[-1] = last_dim + 1
    return flattened.reshape(new_shape)


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
