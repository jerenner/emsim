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


def concatted_to_nested_tensor(tensor: Tensor, batch_offsets: Tensor) -> Tensor:
    assert batch_offsets.ndim == 1
    split_tensor = split_batch_concatted_tensor(tensor, batch_offsets)
    return torch.nested.as_nested_tensor(list(*split_tensor))


# @torch.compiler.disable
@torch.jit.script
def remove_batch_dim_and_concat(
    tensor: Tensor, padding_mask: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    """Converts batched and padded sequences to concatenated sequences.

    Args:
        tensor (Tensor): A tensor of shape (batch_size, max_seq_length, D1, D2, ..., Dn)
        batch_offsets (Tensor, optional): Optional boolean tensor of shape
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
            raise ValueError("Sequence length mismatch between tesnor and padding_mask")

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
def unstack_batch(batch: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    split = {
        k: split_batch_concatted_tensor(batch[k], batch["electron_batch_offsets"])
        for k in (
            "electron_ids",
            "normalized_incidence_points_xy",
            "incidence_points_pixels_rc",
            "normalized_centers_of_mass_xy",
        )
    }
    split.update(
        {
            k: [im.squeeze() for im in batch[k].split(1)]
            for k in ("image", "noiseless_image", "image_size_pixels_rc")
        }
    )
    split["image_sparsified"] = batch["image_sparsified"].unbind()
    out: list[dict[str, Tensor]] = []
    for i in range(len(batch["electron_batch_offsets"])):
        out.append({k: v[i] for k, v in split.items()})
    return out


@torch.jit.script
def unstack_model_output(output: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    split = {
        k: split_batch_concatted_tensor(output[k], output["query_batch_offsets"])
        for k in (
            "pred_logits",
            "pred_positions",
            "pred_std_dev_cholesky",
        )
    }
    split["pred_segmentation_logits"] = output["pred_segmentation_logits"].unbind()
    out: list[dict[str, Tensor]] = []
    for i in range(len(output["query_batch_offsets"])):
        out.append({k: v[i] for k, v in split.items()})
    return out
