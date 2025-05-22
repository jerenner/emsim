from typing import Optional, Union
from itertools import accumulate
import operator

import torch
from torch import Tensor

from emsim.utils.sparse_utils.validation import validate_atleast_nd


@torch.jit.script
def split_batch_concatted_tensor(tensor: Tensor, batch_offsets: Tensor) -> list[Tensor]:
    """Split a batch-concatenated tensor based on batch offsets.

    Args:
        tensor (Tensor): Tensor to split
        batch_offsets (Tensor): Tensor containing offsets where each batch starts
            May or may not include 0 as the first element
            May or may not include len(tensor) as the last element

    Returns:
        List of tensors, one for each batch
    """
    if batch_offsets[0] == 0:
        batch_offsets = batch_offsets[1:]

    # cpu transfer required for tensor_split
    split_tensor = torch.tensor_split(tensor, batch_offsets.cpu())

    # Check for empty last tensor (if batch_offsets[-1] == len(tensor))
    if split_tensor[-1].size(0) == 0:
        split_tensor = split_tensor[:-1]

    return list(split_tensor)


@torch.jit.script
def normalize_batch_offsets(
    batch_offsets: Union[Tensor, list[int]], total_length: int
) -> Union[Tensor, list[int]]:
    """Ensures batch_offsets starts with 0 and ends with the expected total length."""

    if isinstance(batch_offsets, Tensor):
        if torch.is_floating_point(batch_offsets):
            raise ValueError(
                "Expected integer tensor for batch_offsets, but got dtype "
                f"{batch_offsets.dtype}"
            )

        prepend_zero = batch_offsets[0] != 0
        append_len = batch_offsets[-1] != total_length

        if not prepend_zero and not append_len:  # Already normalized
            return batch_offsets

        if append_len and batch_offsets[-1] > total_length:
            raise ValueError(
                f"Max value of batch_offsets ({batch_offsets[-1]}) is greater than "
                f"provided total_length ({total_length})"
            )

        new_offset_len = batch_offsets.size(0) + int(prepend_zero) + int(append_len)
        out = batch_offsets.new_zeros(new_offset_len)

        copy_start = 1 if prepend_zero else 0
        copy_end = -1 if append_len else new_offset_len
        out[copy_start:copy_end] = batch_offsets
        if append_len:
            out[-1] = total_length
        return out

    else:
        prepend_zero = batch_offsets[0] != 0
        append_len = batch_offsets[-1] != total_length

        if not prepend_zero and not append_len:  # Already normalized
            return batch_offsets

        if append_len and batch_offsets[-1] > total_length:
            raise ValueError(
                f"Max value of batch_offsets ({batch_offsets[-1]}) is greater than "
                f"provided total_length ({total_length})"
            )

        out = batch_offsets.copy()
        if prepend_zero:
            out.insert(0, 0)
        if append_len:
            out.append(total_length)
        return out


@torch.jit.script
def batch_offsets_to_seq_lengths(
    batch_offsets: Union[Tensor, list[int]],
) -> Union[Tensor, list[int]]:
    """Computes sequence lengths from batch offsets."""
    if isinstance(batch_offsets, Tensor):
        return batch_offsets[1:] - batch_offsets[:-1]
    else:
        if torch.jit.is_scripting():  # type: ignore
            # Loop for Torchscript compilation
            n_seqs = len(batch_offsets) - 1
            seq_lengths: list[int] = []
            if n_seqs <= 0:
                return seq_lengths
            for i in range(n_seqs):
                seq_lengths.append(batch_offsets[i + 1] - batch_offsets[i])
            return seq_lengths
        else:
            return [
                end - start for start, end in zip(batch_offsets[:-1], batch_offsets[1:])
            ]


@torch.jit.script
def seq_lengths_to_batch_offsets(
    seq_lengths: Union[Tensor, list[int]],
) -> Union[Tensor, list[int]]:
    """Computes batch offsets from sequence lengths."""
    if isinstance(seq_lengths, Tensor):
        batch_offsets_tensor = torch.zeros(
            seq_lengths.size(0) + 1, dtype=seq_lengths.dtype, device=seq_lengths.device
        )
        batch_offsets_tensor[1:] = torch.cumsum(seq_lengths, dim=0)
        return batch_offsets_tensor
    else:
        if torch.jit.is_scripting():  # type: ignore
            batch_offsets_list: list[int] = [0]
            running_sum = 0
            for length in seq_lengths:
                running_sum += length
                batch_offsets_list.append(running_sum)
            return batch_offsets_list
        else:
            return [0, *accumulate(seq_lengths, operator.add)]


@torch.jit.script
def batch_offsets_to_indices(
    batch_offsets: Union[Tensor, list[int]],
    total_seq_length: Optional[int] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    """Converts batch offsets to tensor of batch indices,
    e.g. [0, 5, 9] -> [0, 0, 0, 0, 0, 1, 1, 1, 1]"""
    if isinstance(device, str):
        device = torch.device(device)
    if total_seq_length is not None:
        # Normalize batch_offsets
        batch_offsets = normalize_batch_offsets(batch_offsets, total_seq_length)
    if isinstance(batch_offsets, list):
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.long, device=device)
    else:
        batch_offsets = batch_offsets.to(device=device)

    seq_lengths = batch_offsets_to_seq_lengths(batch_offsets)
    assert isinstance(seq_lengths, Tensor)
    values = torch.arange(batch_offsets.size(0) - 1, device=batch_offsets.device)
    out = torch.repeat_interleave(values, seq_lengths)
    return out


@torch.jit.script
def batch_indices_to_offsets(batch_indices: Tensor) -> Tensor:
    """Converts batch indices to batch offsets,
    e.g. [0, 0, 0, 0, 0, 1, 1, 1, 1] -> [0, 5, 9]"""
    assert not torch.is_floating_point(batch_indices)

    if batch_indices.numel() == 0:  # empty case
        return torch.zeros(1, device=batch_indices.device, dtype=batch_indices.dtype)

    max_batch_index = int(batch_indices.max().item())
    batch_size = max_batch_index + 1

    counts = torch.bincount(batch_indices, minlength=batch_size)

    out = seq_lengths_to_batch_offsets(counts)
    assert isinstance(out, Tensor)

    return out


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
    validate_atleast_nd(tensor, 2)
    if not batch_offsets.ndim == 1:
        raise ValueError(f"Expected batch_offsets to be 1D, got {batch_offsets.ndim}")

    # add the total length to the end of the batch offsets if needed
    batch_offsets_normed = normalize_batch_offsets(batch_offsets, tensor.shape[0])
    assert isinstance(batch_offsets_normed, Tensor)
    batch_offsets = batch_offsets_normed

    seq_lens = batch_offsets_to_seq_lengths(batch_offsets)
    assert isinstance(seq_lens, Tensor)
    batch_size = batch_offsets.shape[0] - 1
    max_len = int(torch.max(seq_lens))

    feature_dims = tensor.shape[1:]
    out_shape = (batch_size, max_len) + feature_dims

    # Fast path: If all sequences are equal length can just return a view
    if torch.all(seq_lens == max_len):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        out = tensor.view(out_shape)
        padding_mask = torch.zeros(
            batch_size, max_len, device=tensor.device, dtype=torch.bool
        )
        return out, padding_mask

    # Full path: Construct the padded outputs and fill them with elements from
    # the input tensor
    out = tensor.new_full(out_shape, pad_value)
    padding_mask = torch.ones(
        (batch_size, max_len), device=tensor.device, dtype=torch.bool
    )

    # Construct indices for vectorized scatter operation

    # indices pointing to the batch each token lives in
    batch_indices = batch_offsets_to_indices(batch_offsets, tensor.shape[0])

    # indices of each token's position within its batch
    arange = torch.arange(tensor.shape[0], device=tensor.device)
    token_indices = arange - batch_offsets[batch_indices]

    # Scatter into the output tensors
    out.index_put_((batch_indices, token_indices), tensor, accumulate=False)
    padding_mask.index_put_(
        (batch_indices, token_indices),
        torch.zeros_like(token_indices, dtype=torch.bool),
        accumulate=False,
    )

    return out, padding_mask


@torch.jit.script
def remove_batch_dim_and_concat(
    tensor: Tensor, padding_mask: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    """Converts batched and padded sequences to concatenated sequences.

    Args:
        tensor (Tensor): A tensor of shape (batch_size, max_seq_length, D1, D2, ..., Dn)
        padding_mask (Tensor, optional): Optional boolean tensor of shape
            (batch_size, max_seq_length) where True indicates padded positions. If None,
            this function assumes that "tensor" has no padding.

    Returns:
        out (Tensor): A tensor of shape (total_seq_length, D1, D2, ..., Dn)
        batch_offsets (Tensor): A 1D tensor indicating where each batch element starts
    """
    validate_atleast_nd(tensor, 3)
    batch_size, max_len = tensor.shape[:2]
    feature_dims = tensor.shape[2:]

    # Early return for empty tensor
    if batch_size == 0 or max_len == 0:
        out = torch.empty((0,) + feature_dims, dtype=tensor.dtype, device=tensor.device)
        batch_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.long, device=tensor.device
        )
        return out, batch_offsets

    # Early return for no padding: All sequences are same length so can just reshape it
    if padding_mask is None or not padding_mask.any():
        total_len = batch_size * max_len
        out_shape = (total_len,) + feature_dims
        out = tensor.reshape(out_shape)
        batch_offsets = torch.arange(0, total_len + 1, max_len, device=tensor.device)

        return out, batch_offsets

    if padding_mask.ndim != 2:
        raise ValueError(f"Expected padding_mask to be 2D, got {padding_mask.ndim}")
    if padding_mask.shape[0] != batch_size:
        raise ValueError("Batch size mismatch between tensor and padding_mask")
    if padding_mask.shape[1] != max_len:
        raise ValueError("Sequence length mismatch between tensor and padding_mask")

    nonpad_mask = padding_mask.logical_not()
    seq_lens = nonpad_mask.sum(-1).to(torch.long)
    batch_offsets = seq_lengths_to_batch_offsets(seq_lens)
    assert isinstance(batch_offsets, Tensor)
    total_len = int(batch_offsets[-1])

    out_shape = (total_len,) + feature_dims
    out = tensor.new_empty(out_shape)

    # meshgrid-like indices
    batch_indices = torch.arange(batch_size, device=tensor.device)
    batch_indices = batch_indices.unsqueeze(1).expand(batch_size, max_len)

    token_indices = torch.arange(max_len, device=tensor.device)
    token_indices = token_indices.unsqueeze(0).expand(batch_size, max_len)

    # select non-padding indices: shape sum(seq_lens)
    sel_batch_indices = batch_indices[nonpad_mask]
    sel_token_indices = token_indices[nonpad_mask]

    # Compute destination indices and extract values
    dest_indices = (batch_offsets[sel_batch_indices] + sel_token_indices).to(torch.long)
    values = tensor[nonpad_mask]

    # Scatter values into the output tensor
    out.index_put_((dest_indices,), values, accumulate=False)

    return out, batch_offsets


# @torch.compiler.disable
@torch.jit.script
def batch_dim_to_leading_index(tensor: Tensor) -> Tensor:
    batch_size = tensor.shape[0]
    last_dim = tensor.shape[-1]
    other_dims = torch._shape_as_tensor(tensor)[1:-1].to(tensor.device)
    batch_index = torch.repeat_interleave(
        torch.arange(batch_size, device=tensor.device), torch.prod(other_dims), 0
    )
    flattened = torch.concat([batch_index.unsqueeze(-1), tensor.view(-1, last_dim)], -1)
    new_shape = list(tensor.shape)
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
        number of nonzero elements in the sparse tensor. Assumes the sparse
        tensor has been coalesce()d.

    Returns:
        torch.Tensor: A 1D tensor with elements corresponding the the first
        incidence of each unique element in the first position of the M axis,
        i.e., the batch offsets if the first element is the batch index.
    """
    assert not torch.is_floating_point(indices_tensor)

    if indices_tensor.shape[1] == 0:  # empty case
        return torch.zeros(1, device=indices_tensor.device, dtype=indices_tensor.dtype)

    batch_indices = indices_tensor[0]
    out = batch_indices_to_offsets(batch_indices)

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
