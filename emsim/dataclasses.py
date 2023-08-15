from dataclasses import dataclass, field
from typing import List, Tuple, Union
from functools import cached_property

import numpy as np
from math import floor
from scipy import sparse


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
class MultiscaleFrame:
    mm: Rectangle
    lowres: Rectangle
    highres: Rectangle

    @cached_property
    def highres_pixel_size_mm(self):
        return self.mm.width() / self.highres.width()

    @cached_property
    def lowres_pixel_size_mm(self):
        return self.mm.width() / self.lowres.width()

    def mm_to_highres(self, x_mm: float, y_mm: float):
        x = (x_mm - self.mm.xmin) / self.highres_pixel_size_mm
        y = (y_mm - self.mm.ymin) / self.highres_pixel_size_mm
        return x, y

    def mm_to_lowres(self, x_mm: float, y_mm: float):
        x = (x_mm - self.mm.xmin) / self.lowres_pixel_size_mm
        y = (y_mm - self.mm.ymin) / self.lowres_pixel_size_mm
        return x, y

    def lowres_coord_to_mm(
        self, x_lowres: Union[int, float], y_lowres: Union[int, float],
    ):
        x = x_lowres * self.lowres_pixel_size_mm + self.mm.xmin
        y = y_lowres * self.lowres_pixel_size_mm + self.mm.ymin
        return x, y

    def lowres_index_to_mm(
        self, x_lowres: int, y_lowres: int
    ):
        if not isinstance(x_lowres, int) or not isinstance(y_lowres, int):
            raise ValueError(
                f"Got a non-int value for a pixel index: {(x_lowres, y_lowres)=}")
        x = x_lowres + 0.5
        y = y_lowres + 0.5
        return self.lowres_coord_to_mm(x, y)

    def highres_coord_to_mm(
        self, x_highres: Union[int, float], y_highres: Union[int, float],
    ):
        x = x_highres * self.highres_pixel_size_mm + self.mm.xmin
        y = y_highres * self.highres_pixel_size_mm + self.mm.ymin
        return x, y

    def highres_index_to_mm(
        self, x_highres: int, y_highres: int
    ):
        if not isinstance(x_highres, int) or not isinstance(y_highres, int):
            raise ValueError(
                f"Got a non-int value for a pixel index: {(x_highres, y_highres)=}")
        x = x_highres + 0.5
        y = y_highres + 0.5
        return self.highres_coord_to_mm(x, y)

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

    def in_pixel_scale(self, frame: MultiscaleFrame, scale: str):
        if "lo" in scale and "hi" not in scale:
            # scale_factor = frame.lowres_pixel_size_mm
            return frame.mm_to_lowres(self.x, self.y)
        elif "hi" in scale:
            # scale_factor = frame.highres_pixel_size_mm
            return frame.mm_to_highres(self.x, self.y)
        else:
            raise ValueError(f"Unknown scale {scale=}")
        # distance_to_pixel_center = scale_factor / 2
        # x = (self.x - frame.mm.xmin + distance_to_pixel_center) / scale_factor
        # y = (self.y - frame.mm.ymin + distance_to_pixel_center) / scale_factor
        # return x, y


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

    def scale_to_mm(self, pixel_x_max, pixel_y_max, mm_x_max, mm_y_max):
        xmin, xmax = np.interp([self.xmin, self.xmax], [0, pixel_x_max], [0, mm_x_max])
        ymin, ymax = np.interp([self.ymin, self.ymax], [0, pixel_y_max], [0, mm_y_max])
        return BoundingBox(xmin, xmax, ymin, ymax)


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
