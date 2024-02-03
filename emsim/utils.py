from copy import copy
from math import floor
from random import randint
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from scipy import sparse
import spconv.pytorch as spconv

from emsim.dataclasses import BoundingBox, PixelSet


def random_chunks(x: List[Any], min_size: int, max_size: int):
    out = []
    x = copy(x)
    while x:
        part = []
        out.append(part)
        partition_size = randint(min_size, max_size)
        while len(part) < partition_size:
            try:
                part.append(x.pop(0))
            except IndexError:
                return out

    return out


def torch_sparse_to_spconv(tensor: torch.Tensor):
    assert tensor.is_sparse
    spatial_shape = tensor.shape[1:-1]
    batch_size = tensor.shape[0]
    indices_th = tensor.indices().permute(1, 0).contiguous().int()
    features_th = tensor.values()
    return spconv.SparseConvTensor(features_th, indices_th, spatial_shape, batch_size)


def spconv_to_torch_sparse(tensor: spconv.SparseConvTensor):
    assert isinstance(tensor, spconv.SparseConvTensor)
    size = [tensor.batch_size] + tensor.spatial_shape + [tensor.features.shape[-1]]
    indices = tensor.indices.transpose(0, 1)
    values = tensor.features
    return torch.sparse_coo_tensor(
        indices, values, size, device=tensor.features.device, dtype=tensor.features.dtype,
        requires_grad=tensor.features.requires_grad
    )


def tensors_same_size(tensors: list[torch.Tensor]) -> bool:
    shapes = [x.shape for x in tensors]
    return len(set(shapes)) <= 1


def sparsearray_from_pixels(
    pixelset: PixelSet,
    shape: Tuple[int],
    offset_x: Optional[int] = None,
    offset_y: Optional[int] = None,
    dtype=None
):
    x_indices, y_indices, data = [], [], []
    for p in pixelset:
        x_index, y_index = p.index()
        x_indices.append(x_index)
        y_indices.append(y_index)
        data.append(p.data)
    x_indices = np.array(x_indices)  # columns
    y_indices = np.array(y_indices)  # rows
    if offset_x is not None:
        x_indices = x_indices - offset_x
    if offset_y is not None:
        y_indices = y_indices - offset_y
    array = sparse.coo_array((np.array(data, dtype=dtype), (y_indices, x_indices)), shape=shape)
    return array.tocsr()


def normalize_boxes(boxes: list[BoundingBox], image_width: int, image_height: int):
    center_format_boxes = np.stack([box.center_format() for box in boxes], 0)
    center_format_boxes /= np.array(
        [image_width, image_height, image_width, image_height]
    )
    return center_format_boxes


def make_image(pixels: PixelSet, bbox: BoundingBox):
    pixels = pixels.crop_to_bounding_box(bbox)
    array = sparsearray_from_pixels(
        pixels,
        (floor(bbox.width() + 1), floor(bbox.height() + 1)),
        offset_x=bbox.xmin,
        offset_y=bbox.ymin,
    )
    return torch.tensor(array.todense(), dtype=torch.float)
