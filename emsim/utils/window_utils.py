from emsim.utils.sparse_utils.indexing.indexing import batch_sparse_index


import torch
from torch import Tensor


def key_offset_grid(window_height: int, window_width: int, device=None):
    assert window_height % 2 == 1
    assert window_width % 2 == 1
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-(window_height // 2), window_height // 2 + 1, device=device),
        torch.arange(-(window_width // 2), window_width // 2 + 1, device=device),
        indexing="ij",
    )
    offsets = torch.stack([offsets_y, offsets_x], -1)
    return offsets


def key_index_grid(query_indices: Tensor, key_offsets: int):
    # add batch dim with 0 batch offset
    key_offsets = torch.cat(
        [
            key_offsets.new_zeros([key_offsets.shape[0], key_offsets.shape[1], 1]),
            key_offsets,
        ],
        -1,
    )
    return query_indices.unsqueeze(1).unsqueeze(1) + key_offsets


def windowed_keys_for_queries(
    query_indices: Tensor,
    image_feature_tensor: Tensor,
    window_height: int,
    window_width: int,
):
    key_offsets = key_offset_grid(window_height, window_width, query_indices.device)
    key_indices = key_index_grid(query_indices, key_offsets)

    key_pad_mask = torch.logical_or(
        torch.logical_or(
            key_indices[..., 1] < 0, key_indices[..., 1] > image_feature_tensor.shape[1]
        ),
        torch.logical_or(
            key_indices[..., 2] < 0, key_indices[..., 2] > image_feature_tensor.shape[2]
        ),
    )

    key_indices[..., 1] = torch.clamp(
        key_indices[..., 1], 0, image_feature_tensor.shape[1] - 1
    )
    key_indices[..., 2] = torch.clamp(
        key_indices[..., 2], 0, image_feature_tensor.shape[2] - 1
    )

    # queries x H x W x feat
    keys, is_specified_mask = batch_sparse_index(image_feature_tensor, key_indices)
    return keys, key_indices, key_offsets, key_pad_mask, is_specified_mask
