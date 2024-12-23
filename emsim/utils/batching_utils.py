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
    assert tensor.ndim == 2
    assert batch_offsets.ndim == 1
    if batch_offsets[-1] != tensor.shape[0]:
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
    feature_dim = tensor.shape[-1]
    out_shape = torch.Size([batchsize, max_len, feature_dim])

    out = tensor.new_full(out_shape, pad_value)
    padding_mask = torch.ones(
        torch.Size([batchsize, max_len]), device=tensor.device, dtype=torch.bool
    )
    # for b, out_b, mask_b in zip(range(batchsize), out, padding_mask):
    for b in torch.arange(batchsize, device=tensor.device):
        start = batch_offsets[b]
        end = batch_offsets[b + 1]
        out[b, : end - start] = tensor[start:end]
        padding_mask[b, : end - start] = False

    return out, padding_mask


def concatted_to_nested_tensor(tensor: Tensor, batch_offsets: Tensor):
    assert batch_offsets.ndim == 1
    split_tensor = split_batch_concatted_tensor(tensor, batch_offsets)
    return torch.nested.as_nested_tensor(list(*split_tensor))


# @torch.compiler.disable
@torch.jit.script
def remove_batch_dim_and_concat(
    tensor: Tensor, padding_mask: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    assert tensor.ndim == 3
    batch_size = tensor.shape[0]
    max_len = tensor.shape[1]
    if padding_mask is None:
        padding_mask = torch.zeros(
            batch_size, max_len, device=tensor.device, dtype=torch.bool
        )
    assert padding_mask.ndim == 2
    nonpadded_batch_sizes = padding_mask.shape[-1] - padding_mask.sum(-1)
    batch_offsets = torch.cat(
        [nonpadded_batch_sizes.new_zeros([1]), nonpadded_batch_sizes.cumsum(-1)]
    )
    # out_shape = [
    #     nonpadded_batch_sizes.sum(),
    #     torch.tensor(tensor.shape[2], device=nonpadded_batch_sizes.device, dtype=nonpadded_batch_sizes.dtype),
    # ]
    total_len = int(nonpadded_batch_sizes.sum())
    out_shape = torch.Size([total_len, tensor.shape[2]])
    out = torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)

    assert tensor.shape[0] == len(batch_offsets[:-1])

    for b, (batch, nonpadded_size, batch_start_index, batch_end_index) in enumerate(
        zip(tensor, nonpadded_batch_sizes, batch_offsets[:-1], batch_offsets[1:])
    ):
        assert nonpadded_size == batch_end_index - batch_start_index
        out[batch_start_index:batch_end_index] = batch[:nonpadded_size]

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
