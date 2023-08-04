from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np
from scipy import sparse


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


@dataclass
class Rectangle:
    xmin: Union[int, float]
    xmax: Union[int, float]
    ymin: Union[int, float]
    ymax: Union[int, float]

    def width(self):
        return self.xmax - self.xmin

    def height(self):
        return self.ymax - self.ymin

    def center_x(self):
        return (self.xmax + self.xmin) / 2

    def center_y(self):
        return (self.ymax + self.ymin) / 2


@dataclass
class Pixel:
    x: int
    y: int
    ionization_electrons: int

    def __post_init__(self):
        self.x = int(self.x)
        self.y = int(self.y)
        self.ionization_electrons = int(self.ionization_electrons)

    def in_box(self, box: Rectangle):
        return box.xmin <= self.x < box.xmax and box.ymin <= self.y < box.ymax


@dataclass
class PixelSet:
    pixels: List[Pixel] = field(default_factory=list)

    def get_bounding_box(self):
        return bounding_box(self)

    def crop_to_bounding_box(self, bounding_box):
        new_pixels = [
            pixel for pixel in self.pixels if pixel.in_box(bounding_box)
            ]
        return PixelSet(new_pixels)


@dataclass
class MultiscaleEvent:
    incidence: IncidencePoint
    lowres_image_size: Rectangle
    highres_image_size: Rectangle
    size_mm: Rectangle
    lowres_pixelset: PixelSet = field(default_factory=PixelSet)
    highres_pixelset: PixelSet = field(default_factory=PixelSet)

    @property
    def id(self):
        return self.incidence.id


@dataclass
class BoundingBox(Rectangle):
    def asarray(self):
        return np.asarray([self.xmin, self.xmax, self.ymin, self.ymax])

    def corners_format(self):
        """[top_left_x, top_left_y, bottom_right_x, bottom_right_y]"""
        return np.asarray([self.xmin, self.ymax, self.xmax, self.ymin])

    def center_format(self):
        """center_x, center_y, width, height"""
        return np.asarray([
            self.center_x(),
            self.center_y(),
            self.width(),
            self.height()
        ])

    def rescale(self, x_scale, y_scale):
        return BoundingBox(
            xmin=self.xmin * x_scale,
            xmax=self.xmax * x_scale,
            ymin=self.ymin * y_scale,
            ymax=self.ymax * y_scale
        )


@dataclass
class Event:
    incidence: IncidencePoint
    pixelset: PixelSet = field(default_factory=PixelSet)
    array: Union[np.ndarray, sparse.spmatrix] = None
    bounding_box: BoundingBox = None

    def compute_bounding_box(self, pixel_margin: int):
        self.bounding_box = bounding_box(self.pixelset, pixel_margin)


def bounding_box(pixelset: PixelSet, pixel_margin=0):
    x = np.array([p.x for p in pixelset.pixels])
    y = np.array([p.y for p in pixelset.pixels])
    xmin, xmax = x.min() - pixel_margin, x.max() + pixel_margin
    ymin, ymax = y.min() - pixel_margin, y.max() + pixel_margin
    return BoundingBox(xmin, xmax+1, ymin, ymax+1)
