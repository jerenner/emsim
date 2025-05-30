from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional

import torch
import numpy as np
from math import floor, ceil
from scipy import sparse


@dataclass
class Rectangle:
    xmin: Union[int, float]
    ymin: Union[int, float]
    xmax: Union[int, float]
    ymax: Union[int, float]

    def width(self):
        return self.xmax - self.xmin

    def height(self):
        return self.ymax - self.ymin

    def center_x(self):
        return (self.xmax + self.xmin) / 2

    def center_y(self):
        return (self.ymax + self.ymin) / 2

    def as_indices(self):
        return (floor(self.xmin), floor(self.ymin), floor(self.xmax), floor(self.ymax))


@dataclass
class IncidencePoint:
    id: int
    x: float
    y: float
    z: float
    e0: float

    def __post_init__(self):
        self.id = int(self.id)
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)
        self.e0 = float(self.e0)

    def normalize_origin(self, x_min: float, y_min: float):
        return IncidencePoint(self.id, self.x - x_min, self.y - y_min, self.z, self.e0)

    def tensor_xy(self, device="cpu"):
        return torch.tensor([self.x, self.y], dtype=torch.float, device=device)


@dataclass
class Pixel:
    x: int
    y: int

    def __post_init__(self):
        assert int(self.x) == float(self.x)
        assert int(self.y) == float(self.y)
        self.x = int(self.x)
        self.y = int(self.y)

    def in_box(self, box: Rectangle):
        return box.xmin <= self.x <= box.xmax and box.ymin <= self.y <= box.ymax

    def center_coordinate(self):
        return float(self.x) + 0.5, float(self.y) + 0.5

    def index(self):
        return (floor(self.x), floor(self.y))

    def xy(self):
        return np.array([self.x, self.y])

    def ij(self):
        return np.array([self.y, self.x])


@dataclass
class IonizationElectronPixel(Pixel):
    ionization_electrons: int

    def __post_init__(self):
        super().__post_init__()
        self.ionization_electrons = int(self.ionization_electrons)

    @property
    def data(self):
        return self.ionization_electrons


@dataclass
class EnergyLossPixel(Pixel):
    energy_loss: float

    def __post_init__(self):
        super().__post_init__()
        self.energy_loss = float(self.energy_loss)

    @property
    def data(self):
        return self.energy_loss


@dataclass
class PixelSet:
    _pixels: List[Pixel] | list[IonizationElectronPixel] | list[EnergyLossPixel] = (
        field(default_factory=list)
    )

    def __getitem__(self, item):
        return self._pixels[item]

    def __len__(self):
        return len(self._pixels)

    def __iter__(self):
        yield from self._pixels

    def append(self, item):
        self._pixels.append(item)

    def get_bounding_box(self):
        return bounding_box(self._pixels)

    def crop_to_bounding_box(self, bounding_box):
        new_pixels = [pixel for pixel in self._pixels if pixel.in_box(bounding_box)]
        return PixelSet(new_pixels)

    def xmin(self):
        return min([p.x for p in self._pixels])

    def xmax(self):
        return max([p.x for p in self._pixels])

    def ymin(self):
        return min([p.y for p in self._pixels])

    def ymax(self):
        return max([p.y for p in self._pixels])


@dataclass
class BoundingBox(Rectangle):
    def asarray(self):
        return np.asarray([self.xmin, self.ymin, self.xmax, self.ymax])

    def center_format(self):
        """center_x, center_y, width, height"""
        return np.asarray(
            [self.center_x(), self.center_y(), self.width(), self.height()]
        )

    def rescale_to_multiple(self, x_scale, y_scale):
        xmin = x_scale * floor(self.xmin) + 0.5
        xmax = x_scale * ceil(self.xmax) - 0.5
        ymin = y_scale * floor(self.ymin) + 0.5
        ymax = y_scale * ceil(self.ymax) - 0.5

        return BoundingBox(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

    def scale_to_mm(self, pixel_x_max, pixel_y_max, mm_x_max, mm_y_max):
        xmin, xmax = np.interp([self.xmin, self.xmax], [0, pixel_x_max], [0, mm_x_max])
        ymin, ymax = np.interp([self.ymin, self.ymax], [0, pixel_y_max], [0, mm_y_max])
        return BoundingBox(xmin, ymin, xmax, ymax)


@dataclass
class Event:
    incidence: IncidencePoint
    pixelset: PixelSet = field(default_factory=PixelSet)
    array: Optional[Union[np.ndarray, sparse.spmatrix]] = None
    bounding_box: BoundingBox = None

    def compute_bounding_box(self, pixel_margin: int):
        self.bounding_box = bounding_box(self.pixelset._pixels, pixel_margin)


def bounding_box(pixels: List[Pixel] | PixelSet, pixel_margin=0):
    if isinstance(pixels, list):
        x = np.array([p.x for p in pixels])
        y = np.array([p.y for p in pixels])
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    elif isinstance(pixels, PixelSet):
        xmin, xmax = pixels.xmin(), pixels.xmax()
        ymin, ymax = pixels.ymin(), pixels.ymax()
    else:
        raise ValueError(f"Invalid type for argument `pixels`: {type(pixels)=}")
    xmin = xmin - pixel_margin
    xmax = xmax + pixel_margin
    ymin = ymin - pixel_margin
    ymax = ymax + pixel_margin

    return BoundingBox(xmin, ymin, xmax, ymax)
