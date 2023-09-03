from random import shuffle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate
from typing import Callable, Any, Optional

from emsim.dataclasses import BoundingBox
from emsim.geant.dataclasses import GeantElectron
from emsim.geant.io import read_files
from emsim.utils import (
    random_chunks,
    sparsearray_from_pixels,
    normalize_boxes,
    make_image,
)


_KEYS_TO_BATCH = (
    "image",
    "boxes_normalized",
    "box_interiors",
    "undiffused_box_interiors",
    "local_trajectories_pixels",
    "local_incidences_pixels",
)

_TO_DEFAULT_COLLATE = (
    "image",
)

_PAD_STACK_WITHIN_EXAMPLE = (
    "box_interiors",
    "undiffused_box_interiors",
)


class GeantElectronDataset(IterableDataset):
    def __init__(
        self,
        trajectory_file: str,
        pixels_file: str,
        undiffused_file: str,
        events_per_image_range: tuple[int],
        noise_std: float = 1.0,
        shuffle=True,
    ):
        self.electrons = read_files(trajectory_file, pixels_file, undiffused_file)
        self.grid = self.electrons[0].grid

        assert len(events_per_image_range) == 2
        self.events_per_image_range = events_per_image_range
        self.noise_std = noise_std
        self.shuffle = shuffle

    def __iter__(self):
        elec_indices = list(range(len(self.electrons)))
        if self.shuffle:
            shuffle(elec_indices)
        chunks = random_chunks(elec_indices, *self.events_per_image_range)

        for chunk in chunks:
            elecs: list[GeantElectron] = [self.electrons[i] for i in chunk]
            image = self.make_composite_image(elecs)

            boxes = [elec.pixels.get_bounding_box() for elec in elecs]
            boxes_normalized = torch.tensor(
                normalize_boxes(boxes, self.grid.xmax_pixel, self.grid.ymax_pixel),
                dtype=torch.float32,
            )
            box_interiors = self.box_interiors(image, boxes)
            undiffused_box_interiors = self.undiffused_box_interiors(elecs, boxes)

            trajectory_tensors = [
                torch.tensor(elec.trajectory.as_array(), dtype=torch.float)
                for elec in elecs
            ]

            incidence_points = torch.stack(
                [torch.tensor([elec.incidence.x, elec.incidence.y]) for elec in elecs]
            )

            box_offsets = torch.stack(
                [torch.tensor([box.xmin, box.ymin]) for box in boxes]
            )
            offsets_mm = box_offsets * (torch.tensor(self.grid.pixel_size_um) / 1000)

            local_trajectories = [
                torch.tensor(
                    elec.trajectory.localize(offset[0], offset[1]).as_array(),
                    dtype=torch.float,
                )
                for elec, offset in zip(elecs, offsets_mm)
            ]
            local_incidences = incidence_points - offsets_mm

            mm_to_pixel = 1 / (torch.tensor(self.grid.pixel_size_um) / 1000)
            local_incidences_pixels = local_incidences * mm_to_pixel

            trajectory_to_pixels = torch.cat([mm_to_pixel, torch.tensor([1.0, 1.0])])
            local_trajectories_pixels = [
                traj * trajectory_to_pixels for traj in local_trajectories
            ]

            yield {
                "image": image,
                "electron_ids": torch.tensor(
                    [elec.id for elec in elecs], dtype=torch.int
                ),
                "boxes": boxes,
                "boxes_normalized": boxes_normalized,
                "box_interiors": box_interiors,
                "undiffused_box_interiors": undiffused_box_interiors,
                "trajectories": trajectory_tensors,
                "incidence_points": incidence_points,
                "local_trajectories": local_trajectories,
                "local_incidences": local_incidences,
                "local_trajectories_pixels": local_trajectories_pixels,
                "local_incidences_pixels": local_incidences_pixels,
            }

    def make_composite_image(self, elecs: list[GeantElectron]) -> torch.Tensor:
        arrays = [
            sparsearray_from_pixels(
                elec.pixels, (self.grid.xmax_pixel, self.grid.ymax_pixel)
            )
            for elec in elecs
        ]
        image = torch.tensor(sum(arrays).todense(), dtype=torch.float)

        if self.noise_std > 0:
            image = image + torch.normal(0.0, self.noise_std, size=image.shape)

        return image

    def box_interiors(
        self, image: torch.Tensor, boxes: list[BoundingBox]
    ) -> list[torch.Tensor]:
        images = []
        for box in boxes:
            box_pixels = box.as_indices()
            xmin, ymin, xmax, ymax = box_pixels
            box_interior = image[xmin : xmax + 1, ymin : ymax + 1]
            images.append(box_interior)
        return images

    def undiffused_box_interiors(
        self, elecs: list[GeantElectron], boxes: list[BoundingBox]
    ) -> list[torch.Tensor]:
        images = [
            make_image(elec.undiffused_pixels, bbox) for elec, bbox in zip(elecs, boxes)
        ]
        return images


def electron_collate_fn(
    batch: list[dict[str, Any]],
    pad_to_multiple_of=None, # e.g. 8 for tensor cores
    default_collate_fn: Callable = torch.utils.data.dataloader.default_collate,
) -> dict[str, Any]:
    batch = [{k: example[k] for k in _KEYS_TO_BATCH} for example in batch]

    out_batch = {}
    to_default_collate = [{} for _ in range(len(batch))]

    first = batch[0]
    for key in first:
        if key in _TO_DEFAULT_COLLATE:
            for to_default, electron in zip(to_default_collate, batch):
                to_default[key] = electron[key]
        elif key in _PAD_STACK_WITHIN_EXAMPLE:
            stacked_list = []
            mask_list = []
            for electron in batch:
                stacked, mask = pad_and_stack_electron_boxes(electron[key], pad_to_multiple_of)
                stacked_list.append(stacked)
                mask_list.append(mask)
            out_batch[key] = stacked_list
            out_batch[key + "_pad_mask"] = mask_list
        else:
            out_batch[key] = [electron[key] for electron in batch]

    out_batch.update(default_collate_fn(to_default_collate))

    return out_batch


def pad_and_stack_electron_boxes(
    tensors: list[torch.Tensor], pad_to_multiple_of: Optional[int] = None
    ) -> (torch.Tensor, torch.Tensor):

    shapes = [t.shape for t in tensors]
    bsz = len(tensors)

    all_same_size = all(shape == shapes[0] for shape in shapes)
    if all_same_size:
        stacked_padded = torch.stack(tensors, 0)
        mask = stacked_padded.new_ones(stacked_padded.shape, dtype=torch.bool)
        return stacked_padded, mask

    pad_to = np.stack(shapes).max(0)
    if pad_to_multiple_of is not None:
        pad_to = ((pad_to // pad_to_multiple_of) + 1) * pad_to_multiple_of

    stacked_padded = torch.zeros((bsz, *pad_to), dtype=torch.float)
    pad_mask = stacked_padded.new_zeros(stacked_padded.shape, dtype=torch.bool)

    for source, target, mask in zip(tensors, stacked_padded, pad_mask):
        width, height = source.shape
        target[:width, :height] = source
        mask[:width, :height] = True

    return stacked_padded, mask