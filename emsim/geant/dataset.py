from random import shuffle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate
from typing import Callable, Any, Optional

from emsim.dataclasses import BoundingBox, IonizationElectronPixel
from emsim.geant.dataclasses import GeantElectron
from emsim.geant.io import read_files
from emsim.utils import (
    random_chunks,
    sparsearray_from_pixels,
    make_image,
)


# keys in here will not be batched in the collate fn
_KEYS_TO_NOT_BATCH = ("maps",)

# these keys get passed to the default collate_fn, everything else uses custom batching logic
_TO_DEFAULT_COLLATE = ("image",)


class GeantElectronDataset(IterableDataset):
    def __init__(
        self,
        pixels_file: str,
        undiffused_file: str,
        events_per_image_range: tuple[int],
        pixel_patch_size: int = 5,
        processor: Callable = None,
        transform: Callable = None,
        noise_std: float = 1.0,
        shuffle=True,
    ):
        self.electrons = read_files(
            pixels_file=pixels_file, undiffused_pixels_file=undiffused_file
        )
        self.grid = self.electrons[0].grid

        assert len(events_per_image_range) == 2
        self.events_per_image_range = events_per_image_range
        if pixel_patch_size % 2 == 0:
            raise ValueError(f"pixel_patch_size should be odd, got {pixel_patch_size}")
        self.pixel_patch_size = pixel_patch_size
        self.noise_std = noise_std
        self.shuffle = shuffle
        self.processor = processor
        self.transform = transform

    def __iter__(self):
        elec_indices = list(range(len(self.electrons)))
        if self.shuffle:
            shuffle(elec_indices)
        chunks = random_chunks(elec_indices, *self.events_per_image_range)

        for chunk in chunks:
            elecs: list[GeantElectron] = [self.electrons[i] for i in chunk]
            image = self.make_composite_image(elecs)

            patches, patch_coords = get_pixel_patches(
                image, elecs, self.pixel_patch_size
            )
            patch_coords_mm = (
                patch_coords * np.array(self.grid.pixel_size_um * 2) / 1000
            )

            maps = [
                elec.get_segmentation_map(inst_id) for inst_id, elec in enumerate(elecs)
            ]

            incidence_points = np.stack(
                [
                    np.array([elec.incidence.x, elec.incidence.y], dtype=np.float32)
                    for elec in elecs
                ]
            )
            local_incidence_points_mm = incidence_points - patch_coords_mm[:, :2]
            local_incidence_points_pixels = local_incidence_points_mm / (
                np.array(self.grid.pixel_size_um) / 1000
            ).astype(np.float32)

            instance_seg = None

            # image = np.array(image.convert("RGB"))

            if self.transform is not None:
                image = image.numpy()
                instance_seg = np.array(maps.segmentation_map for i in range(len(maps)))
                transformed = self.transform(image=image, mask=instance_seg)
                image, instance_seg = transformed["image"], transformed["mask"]

                # "convert to C, H, W". wth does this do
                image = image.transpose(2, 0, 1)

            if self.processor is not None:
                if instance_seg is None:
                    instance_seg = np.array(
                        maps.segmentation_map for i in range(len(maps))
                    )

                inputs = self.processor([image], [instance_seg], return_tensors="pt")
                inputs = {
                    k: v.squeeze() if isinstance(v, torch.Tensor) else v[0]
                    for k, v in inputs.items()
                }

            if self.processor is not None or self.transform is not None:
                yield inputs
            else:
                yield {
                    "image": image.astype(np.float32),
                    "electron_ids": np.array([elec.id for elec in elecs], dtype=int),
                    "maps": maps,
                    "incidence_points": incidence_points,
                    "local_incidence_points_pixels": local_incidence_points_pixels,
                    "pixel_patches": patches,
                }

    def make_composite_image(self, elecs: list[GeantElectron]) -> np.ndarray:
        arrays = [
            sparsearray_from_pixels(
                elec.pixels, (self.grid.xmax_pixel, self.grid.ymax_pixel)
            )
            for elec in elecs
        ]
        image = np.array(sum(arrays).todense(), dtype=np.float32)

        if self.noise_std > 0:
            image = image + np.random.normal(
                0.0, self.noise_std, size=image.shape
            ).astype(np.float32)

        return image

    def undiffused_box_interiors(
        self, elecs: list[GeantElectron], boxes: list[BoundingBox]
    ) -> list[np.ndarray]:
        images = [
            make_image(elec.undiffused_pixels, bbox) for elec, bbox in zip(elecs, boxes)
        ]
        return images


def get_pixel_patches(
    image: np.ndarray,
    electrons: list[GeantElectron],
    patch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    patches = []
    patch_coordinates = []

    for electron in electrons:
        peak_pixel: IonizationElectronPixel = max(
            electron.pixels, key=lambda x: x.ionization_electrons
        )

        x_min = peak_pixel.x - patch_size // 2
        x_max = peak_pixel.x + patch_size // 2 + 1
        y_min = peak_pixel.y - patch_size // 2
        y_max = peak_pixel.y + patch_size // 2 + 1
        patch_coordinates.append(np.array([x_min, y_min, x_max, y_max]))

        # compute padding if any
        if x_min < 0:
            pad_x_left = -x_min
            x_min = 0
        else:
            pad_x_left = 0
        if y_min < 0:
            pad_y_top = -y_min
            y_min = 0
        else:
            pad_y_top = 0

        pad_y_bottom = max(0, x_max - image.shape[0])
        pad_x_right = max(0, y_max - image.shape[1])

        patch = image[x_min:x_max, y_min:y_max]
        patch = np.pad(patch, ((pad_x_left, pad_x_right), (pad_y_top, pad_y_bottom)))

        patches.append(patch)
    return patches, np.stack(patch_coordinates, 0)


def electron_collate_fn(
    batch: list[dict[str, Any]],
    default_collate_fn: Callable = default_collate,
) -> dict[str, Any]:
    batch = [
        {k: example[k] for k in example if k not in _KEYS_TO_NOT_BATCH}
        for example in batch
    ]

    out_batch = {}
    to_default_collate = [{} for _ in range(len(batch))]

    first = batch[0]
    for key in first:
        if key in _TO_DEFAULT_COLLATE:
            for to_default, electron in zip(to_default_collate, batch):
                to_default[key] = electron[key]
        else:
            lengths = [len(sample[key]) for sample in batch]
            out_batch[key] = torch.as_tensor(
                np.concatenate([sample[key] for sample in batch], axis=0)
            )
            out_batch[key + "_batch_index"] = torch.cat(
                [
                    torch.repeat_interleave(torch.as_tensor(i), length)
                    for i, length in enumerate(lengths)
                ]
            )

    out_batch.update(default_collate_fn(to_default_collate))

    return out_batch


def plot_pixel_patch_and_incidence_point(
    pixel_patch: np.ndarray, incidence_point: np.ndarray
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    map = ax.imshow(
        pixel_patch.T,
        extent=(0, pixel_patch.shape[0], pixel_patch.shape[1], 0),
        interpolation=None,
    )
    fig.colorbar(map, ax=ax)
    ax.scatter(*incidence_point, c="r")
    return fig, ax
