from copy import copy
from dataclasses import dataclass, field
from functools import cached_property
from random import shuffle, randint
from typing import List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader, IterableDataset


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
class Event:
    incidence: IncidencePoint
    pixelset: PixelSet = field(default_factory=PixelSet)
    array: Union[np.ndarray, sparse.spmatrix] = None
    bounding_box: BoundingBox = None

    def finalize(self, shape: Tuple[int], bbox_pixel_margin: int = 10):
        self.make_array(shape)
        self.compute_bounding_box(bbox_pixel_margin)

    def make_array(self, shape):
        self.array = make_array(self, shape)

    def compute_bounding_box(self, pixel_margin: int):
        self.bounding_box = bounding_box(self.pixelset, pixel_margin)


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


class MultiscaleElectronDataset(IterableDataset):
    def __init__(
        self, highres_filename: str, lowres_filename: str,
        highres_image_size: Tuple[int], lowres_image_size: Tuple[int],
        mm_x_range: Tuple[float], mm_y_range: Tuple[float],
        events_per_image_range: Tuple[int],
        noise_std: float = 1.
    ):
        self.highres_image_shape = Rectangle(0, highres_image_size[0], 0, highres_image_size[1])
        self.lowres_image_shape = Rectangle(0, lowres_image_size[0], 0, lowres_image_size[1])
        self.size_mm = Rectangle(mm_x_range[0], mm_x_range[1], mm_y_range[0], mm_y_range[1])
        self.noise_std = noise_std

        self.low_to_high_scale_factor = (
            round(self.highres_image_shape.xmax / self.lowres_image_shape.xmax),
            round(self.highres_image_shape.ymax / self.lowres_image_shape.ymax)
        )

        assert len(events_per_image_range) == 2
        self.events_per_image_range = events_per_image_range

        self.events = read_multiscale_data(
            highres_filename, lowres_filename, highres_image_size,
            lowres_image_size, mm_x_range, mm_y_range
        )

        # validate data
        for event in self.events:
            assert self.size_mm.xmin <= event.incidence.x <= self.size_mm.xmax
            assert self.size_mm.ymin <= event.incidence.y <= self.size_mm.ymax

    def __iter__(self):
        event_indices = list(range(len(self.events)))
        shuffle(event_indices)
        partitions = _partition(event_indices, self.events_per_image_range[0], self.events_per_image_range[1])

        for part in partitions:
            events = [self.events[i] for i in part]
            image = self.make_composite_lowres_image(events)

            boxes = [event.lowres_pixelset.get_bounding_box() for event in events]
            boxes_pixels = np.stack([box.asarray() for box in boxes], 0)
            boxes_normalized = self.normalize_boxes(boxes)
            box_interiors = self.get_lowres_box_interiors(image, boxes)
            box_sizes = np.stack(
                [np.array([box.xmax - box.xmin, box.ymax - box.ymin])for box in boxes],
                0
            )

            boxes_highres = [
                self.bounding_box_lowres_to_highres(box) for box in boxes
            ]
            boxes_pixels_highres = np.stack([box.asarray() for box in boxes_highres], 0)
            highres_images = self.highres_images(events, boxes_highres)
            highres_image_sizes = torch.stack([torch.tensor(image.shape) for image in highres_images])
            local_highres_pixel_incidences = self.local_highres_pixel_incidences(events, boxes_highres)

            yield {
                "image": image,
                "event_ids": torch.tensor([event.id for event in events], dtype=torch.int64),
                "boxes_pixels": boxes_pixels,
                "boxes_normalized": boxes_normalized,
                "box_interiors": box_interiors,
                "box_sizes": box_sizes,
                "boxes_pixels_highres": boxes_pixels_highres,
                "highres_images": highres_images,
                "highres_image_sizes": highres_image_sizes,
                "local_incidence_locations_pixels": local_highres_pixel_incidences,
            }

    def make_composite_lowres_image(self, events: List[MultiscaleEvent], add_noise=True) -> torch.Tensor:
        arrays = [
            make_array(
                event.lowres_pixelset,
                (self.lowres_image_shape.width(), self.lowres_image_shape.height()))
            for event in events
        ]

        image = torch.tensor(sum(arrays).todense(), dtype=torch.float)
        if add_noise and self.noise_std > 0:
            image = image + torch.normal(
                0., self.noise_std, size=image.shape, device=image.device
            )

        return image

    def highres_images(self, events: List[MultiscaleEvent], boxes: List[BoundingBox]) -> List[torch.Tensor]:
        def make_image(event: MultiscaleEvent, bbox: BoundingBox):
            pixelset = event.highres_pixelset.crop_to_bounding_box(bbox)
            array = make_array(
                pixelset, (round(bbox.width()), round(bbox.height())),
                offset_x=bbox.xmin, offset_y=bbox.ymin)
            return torch.tensor(array.todense(), dtype=torch.float)

        images = [make_image(event, bbox) for event, bbox in zip(events, boxes)]
        return images

    def local_highres_pixel_incidences(self, events: List[MultiscaleEvent], boxes: List[BoundingBox]) -> torch.Tensor:
        points = []
        for event, bbox in zip(events, boxes):
            pixel_location = xy_mm_to_pixel(
                event.incidence.x, event.incidence.y,
                event.size_mm.xmin, event.size_mm.xmax,
                event.size_mm.ymin, event.size_mm.ymax,
                event.highres_image_size.xmax, event.highres_image_size.ymax)
            pixel_location = [
                pixel_location[0] - bbox.xmin,
                pixel_location[1] - bbox.ymin
            ]
            points.append(pixel_location)
        return torch.tensor(np.stack(points), dtype=torch.float)

    def normalize_boxes(
        self,
        boxes: List[BoundingBox]
    ) -> np.ndarray:
        center_format_boxes = np.stack([box.center_format() for box in boxes], 0)
        center_format_boxes /= np.array(
            [self.lowres_image_shape.width(), self.lowres_image_shape.height(),
             self.lowres_image_shape.width(), self.lowres_image_shape.height()],
            dtype=np.float32
        )
        return center_format_boxes

    def get_lowres_box_interiors(
        self,
        lowres_composite_image: torch.Tensor,
        boxes: List[BoundingBox]
    ) -> List[torch.Tensor]:
        images = []
        for box in boxes:
            image = lowres_composite_image[box.xmin:box.xmax+1, box.ymin:box.ymax+1]
            images.append(image)
        return images

    def bounding_box_lowres_to_highres(self, bbox: BoundingBox):
        x_scaling = self.highres_image_shape.width() / self.lowres_image_shape.width()
        y_scaling = self.highres_image_shape.height() / self.lowres_image_shape.height()

        return bbox.rescale(x_scaling, y_scaling)

def _partition(x: List, min_size: int, max_size: int):
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


def read_file(filename: str) -> List[Event]:
    events = []
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            if "EV" in line:
                _, electron_id, electron_x, electron_y, electron_z, electron_e0 = line.split(" ")
                event = Event(IncidencePoint(electron_id, electron_x, electron_y, electron_z, electron_e0))
                events.append(event)
            else:
                pixel_x, pixel_y, ion_elecs = line.split(" ")
                pixel = Pixel(pixel_x, pixel_y, ion_elecs)
                event.pixelset.pixels.append(pixel)
    return events


def read_multiscale_data(
    highres_filename: str,
    lowres_filename: str,
    highres_shape: Tuple[int],
    lowres_shape: Tuple[int],
    mm_x_range: Tuple[float],
    mm_y_range: Tuple[float]
) -> List[MultiscaleEvent]:
    highres_data: List[Event] = read_file(highres_filename)
    lowres_data: List[Event] = read_file(lowres_filename)
    assert len(highres_data) == len(lowres_data)

    multiscale_events = []
    for high, low in zip(highres_data, lowres_data):
        assert high.incidence == low.incidence
        event = MultiscaleEvent(
            high.incidence,
            lowres_image_size=Rectangle(0, lowres_shape[0], 0, lowres_shape[1]),
            highres_image_size=Rectangle(0, highres_shape[0], 0, highres_shape[1]),
            size_mm=Rectangle(mm_x_range[0], mm_x_range[1], mm_y_range[0], mm_y_range[1]),
            lowres_pixelset=low.pixelset,
            highres_pixelset=high.pixelset
            )
        multiscale_events.append(event)
    return multiscale_events


def make_array(
    pixelset: PixelSet, shape: Tuple[int],
    offset_x: Optional[int] = None, offset_y: Optional[int] = None
):
    x, y, e = [], [], []
    for p in pixelset.pixels:
        x.append(p.x)
        y.append(p.y)
        e.append(p.ionization_electrons)
    x = np.array(x) - offset_x if offset_x else np.array(x)
    y = np.array(y) - offset_y if offset_y else np.array(y)
    array = sparse.coo_array((np.array(e), (x, y)), shape=shape)
    return array.tocsr()


def bounding_box(pixelset: PixelSet, pixel_margin=0):
    x = np.array([p.x for p in pixelset.pixels])
    y = np.array([p.y for p in pixelset.pixels])
    xmin, xmax = x.min() - pixel_margin, x.max() + pixel_margin
    ymin, ymax = y.min() - pixel_margin, y.max() + pixel_margin
    return BoundingBox(xmin, xmax, ymin, ymax)


def xy_pixel_to_mm(pixel_x, pixel_y, pixel_x_max, pixel_y_max, mm_x_min, mm_x_max, mm_y_min, mm_y_max):
    x_mm = np.interp(pixel_x, [0, pixel_x_max], [mm_x_min, mm_x_max])
    y_mm = np.interp(pixel_y, [0, pixel_y_max], [mm_y_min, mm_y_max])
    return x_mm, y_mm


def xy_mm_to_pixel(mm_x, mm_y, mm_x_min, mm_x_max, mm_y_min, mm_y_max, pixel_x_max, pixel_y_max):
    x_pixel = np.interp(mm_x, [mm_x_min, mm_x_max], [0, pixel_x_max])
    y_pixel = np.interp(mm_y, [mm_y_min, mm_y_max], [0, pixel_y_max])
    return x_pixel, y_pixel
