from copy import copy
from random import randint, shuffle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from emsim.dataclasses import BoundingBox, MultiscaleEvent, Rectangle, MultiscaleFrame
from emsim.multiscale.io import read_multiscale_data
from emsim.multiscale.utils import make_array, xy_mm_to_pixel


class MultiscaleElectronDataset(IterableDataset):
    def __init__(
        self, highres_filename: str, lowres_filename: str,
        highres_image_size: Tuple[int], lowres_image_size: Tuple[int],
        mm_x_range: Tuple[float], mm_y_range: Tuple[float],
        events_per_image_range: Tuple[int],
        noise_std: float = 1.
    ):
        highres_scale = Rectangle(0, highres_image_size[0], 0, highres_image_size[1])
        lowres_scale = Rectangle(0, lowres_image_size[0], 0, lowres_image_size[1])
        mm_scale = Rectangle(mm_x_range[0], mm_x_range[1], mm_y_range[0], mm_y_range[1])
        self.frame = MultiscaleFrame(mm_scale, lowres_scale, highres_scale)
        self.noise_std = noise_std

        self.low_to_high_scale_factor = (
            round(self.frame.highres.xmax / self.frame.lowres.xmax),
            round(self.frame.highres.ymax / self.frame.lowres.ymax)
        )

        assert len(events_per_image_range) == 2
        self.events_per_image_range = events_per_image_range

        self.events = read_multiscale_data(
            highres_filename, lowres_filename, highres_image_size,
            lowres_image_size, mm_x_range, mm_y_range
        )

        # validate data
        for event in self.events:
            assert self.frame.mm.xmin <= event.incidence.x <= self.frame.mm.xmax
            assert self.frame.mm.ymin <= event.incidence.y <= self.frame.mm.ymax

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

            incidence_pixels_lowres = torch.tensor(np.stack([
                e.incidence.in_pixel_scale(self.frame, "lowres") for e in events
            ], 0))
            incidence_pixels_highres = torch.tensor(np.stack([
                e.incidence.in_pixel_scale(self.frame, "highres") for e in events
            ], 0))

            local_lowres_pixel_incidences = self.local_lowres_pixel_incidences(events, boxes)
            local_highres_pixel_incidences = local_lowres_pixel_incidences * torch.tensor(
                self.low_to_high_scale_factor)



            yield {
                "events": events,
                "image": image,
                "event_ids": torch.tensor([event.id for event in events], dtype=torch.int64),
                "boxes_pixels": boxes_pixels,
                "boxes_normalized": boxes_normalized,
                "box_interiors": box_interiors,
                "box_sizes": box_sizes,
                "boxes_pixels_highres": boxes_pixels_highres,
                "highres_images": highres_images,
                "highres_image_sizes": highres_image_sizes,
                "incidence_pixels_lowres": incidence_pixels_lowres,
                "incidence_pixels_highres": incidence_pixels_highres,
                "local_highres_incidence_locations_pixels": local_highres_pixel_incidences,
                "local_lowres_incidence_locations_pixels": local_lowres_pixel_incidences,
            }

    def make_composite_lowres_image(self, events: List[MultiscaleEvent], add_noise=True) -> torch.Tensor:
        arrays = [
            make_array(
                event.lowres_pixelset,
                (self.frame.lowres.width(), self.frame.lowres.height()))
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
            # pixel_location = xy_mm_to_pixel(
            #     event.incidence.x, event.incidence.y,
            #     self.scale.mm.xmin, self.scale.mm.xmax,
            #     self.scale.mm.ymin, self.scale.mm.ymax,
            #     self.scale.highres.xmax, self.scale.highres.ymax)
            pixel_location = event.incidence.in_pixel_scale(self.frame, "highres")
            pixel_location = [
                pixel_location[0] - bbox.xmin,
                pixel_location[1] - bbox.ymin
            ]
            points.append(pixel_location)
        return torch.tensor(np.stack(points), dtype=torch.float)

    def local_lowres_pixel_incidences(self, events: List[MultiscaleEvent], boxes: List[BoundingBox]) -> torch.Tensor:
        points = []
        for event, bbox in zip(events, boxes):
            pixel_location = event.incidence.in_pixel_scale(self.frame, "lowres")
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
            [self.frame.lowres.width(), self.frame.lowres.height(),
             self.frame.lowres.width(), self.frame.lowres.height()],
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
            image = lowres_composite_image[box.xmin:box.xmax, box.ymin:box.ymax]
            images.append(image)
        return images

    def bounding_box_lowres_to_highres(self, bbox: BoundingBox):
        return bbox.rescale(*self.low_to_high_scale_factor)


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
