from typing import Optional, Tuple

import numpy as np
from scipy import sparse

from emsim.dataclasses import PixelSet


def make_array(
    pixelset: PixelSet, shape: Tuple[int],
    offset_x: Optional[int] = None, offset_y: Optional[int] = None
):
    x, y, e = [], [], []
    for p in pixelset.pixels:
        x.append(p.x)
        y.append(p.y)
        e.append(p.ionization_electrons)
    x = np.array(x)
    y = np.array(y)
    if offset_x is not None:
        x = x - offset_x
    if offset_y is not None:
        y = y - offset_y
    array = sparse.coo_array((np.array(e), (x, y)), shape=shape)
    return array.tocsr()


def xy_pixel_to_mm(pixel_x, pixel_y, pixel_x_max, pixel_y_max, mm_x_min, mm_x_max, mm_y_min, mm_y_max):
    x_mm = np.interp(pixel_x, [0, pixel_x_max], [mm_x_min, mm_x_max])
    y_mm = np.interp(pixel_y, [0, pixel_y_max], [mm_y_min, mm_y_max])
    return x_mm, y_mm


def xy_mm_to_pixel(mm_x, mm_y, mm_x_min, mm_x_max, mm_y_min, mm_y_max, pixel_x_max, pixel_y_max):
    x_pixel = np.interp(mm_x, [mm_x_min, mm_x_max], [0, pixel_x_max])
    y_pixel = np.interp(mm_y, [mm_y_min, mm_y_max], [0, pixel_y_max])
    return x_pixel, y_pixel
