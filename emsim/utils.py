from copy import copy
from math import floor
from random import randint
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from scipy import sparse

from emsim.dataclasses import BoundingBox
from emsim.multiscale.dataclasses import PixelSet


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
    x_indices = np.array(x_indices)
    y_indices = np.array(y_indices)
    if offset_x is not None:
        x_indices = x_indices - offset_x
    if offset_y is not None:
        y_indices = y_indices - offset_y
    array = sparse.coo_array((np.array(data, dtype=dtype), (x_indices, y_indices)), shape=shape)
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
