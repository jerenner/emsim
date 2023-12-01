from math import ceil, floor
from random import shuffle
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate

from emsim.dataclasses import BoundingBox, IonizationElectronPixel
from emsim.geant.dataclasses import GeantElectron, Trajectory
from emsim.geant.io import read_files
from emsim.utils import (
    make_image,
    random_chunks,
    sparsearray_from_pixels,
)

# keys in here will not be batched in the collate fn
_KEYS_TO_NOT_BATCH = ("maps", "local_trajectories_pixels")

# these keys get passed to the default collate_fn, everything else uses custom batching logic
_TO_DEFAULT_COLLATE = ("image",)


ELECTRON_IONIZATION_MEV = 3.6e-6

class GeantElectronDataset(IterableDataset):
    def __init__(
        self,
        pixels_file: str,
        events_per_image_range: tuple[int],
        pixel_patch_size: int = 5,
        trajectory_file: str = None,
        train_percentage: float = 0.95,
        split: str = "train",
        processor: Callable = None,
        transform: Callable = None,
        noise_std: float = 1.0,
        shuffle=True,
    ):
        assert 0 < train_percentage <= 1
        assert split in ("train", "test")
        self.electrons = read_files(
            pixels_file=pixels_file, trajectory_file=trajectory_file
        )
        train_test_split = int(len(self.electrons) * train_percentage)
        if split == "train":
            self.electrons = self.electrons[:train_test_split]
        else:
            self.electrons = self.electrons[train_test_split:]

        self.grid = self.electrons[0].grid
        self.pixel_to_mm = np.array(self.grid.pixel_size_um) / 1000

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
            patch_coords_mm = patch_coords * np.concatenate(
                [self.pixel_to_mm, self.pixel_to_mm]
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
            local_incidence_points_pixels = (
                local_incidence_points_mm / self.pixel_to_mm.astype(np.float32)
            )

            local_centers_of_mass_pixels = charge_2d_center_of_mass(patches)

            local_trajectories = [
                elec.trajectory.localize(coords[0], coords[1]).as_array()
                for elec, coords in zip(elecs, patch_coords_mm)
            ]
            trajectory_mm_to_pixels = np.concatenate([1 / self.pixel_to_mm, np.array([1.0, 1.0])]).astype(np.float32)
            local_trajectories_pixels = [traj * trajectory_mm_to_pixels for traj in local_trajectories]

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
                    "incidence_points": incidence_points.astype(np.float32),
                    "local_incidence_points_pixels": local_incidence_points_pixels.astype(
                        np.float32
                    ),
                    "pixel_patches": patches.astype(np.float32),
                    "local_centers_of_mass_pixels": local_centers_of_mass_pixels.astype(
                        np.float32
                    ),
                    "local_trajectories_pixels": local_trajectories_pixels
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

        # add channel dimension
        image = np.expand_dims(image, 0)

        return image


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

        pad_y_bottom = max(0, x_max - image.shape[-2])
        pad_x_right = max(0, y_max - image.shape[-1])

        patch = image[...,x_min:x_max, y_min:y_max]
        patch = np.pad(patch, ((0, 0), (pad_x_left, pad_x_right), (pad_y_top, pad_y_bottom)))

        patches.append(patch)
    return np.stack(patches, 0), np.stack(patch_coordinates, 0)


def charge_2d_center_of_mass(patches: np.ndarray, com_patch_size=3):
    if com_patch_size % 2 == 0:
        raise ValueError(f"'com_patch_size' should be odd, got '{com_patch_size}'")
    if patches.ndim == 2:
        # add batch dim
        patches = np.expand_dims(patches, 0)

    # permute from rows by columns to x by y
    patches = patches.transpose(0, 1, 3, 2)

    patch_x_len = patches.shape[-2]
    patch_y_len = patches.shape[-1]
    coord_grid = np.stack(
        np.meshgrid(
            np.arange(0.5, patches.shape[-2], 1), np.arange(0.5, patches.shape[-1], 1)
        )
    )

    patch_x_mid = patch_x_len / 2
    patch_y_mid = patch_y_len / 2

    patch_radius = com_patch_size // 2

    patches = patches[
        ...,
        floor(patch_x_mid) - patch_radius : ceil(patch_x_mid) + patch_radius,
        floor(patch_y_mid) - patch_radius : ceil(patch_y_mid) + patch_radius,
    ]
    coord_grid = coord_grid[
        ...,
        floor(patch_x_mid) - patch_radius : ceil(patch_x_mid) + patch_radius,
        floor(patch_y_mid) - patch_radius : ceil(patch_y_mid) + patch_radius,
    ]

    # patches: batch * 1 * x * y
    # coord grid: 1 * 2 * x * y
    coord_grid = np.expand_dims(coord_grid, 0)

    weighted_grid = patches * coord_grid
    return weighted_grid.sum((-1, -2)) / patches.sum((-1, -2))


def trajectory_to_ionization_electron_points(trajectory: Trajectory):
    traj_array = trajectory.to_array()
    energy_deposition_points = traj_array[traj_array[:, -1] != 0.0]
    n_elecs = energy_deposition_points[:, -1] / ELECTRON_IONIZATION_MEV
    points = np.concatenate([traj_array[:, :2], n_elecs[..., None]], 1)
    return points


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


def plot_pixel_patch_and_points(
    pixel_patch: np.ndarray,
    points: list[np.ndarray],
    point_labels: Optional[list[str]] = None,
    contour: Optional[np.ndarray] = None,
):
    import matplotlib.pyplot as plt

    pixel_patch = pixel_patch.squeeze()
    fig, ax = plt.subplots()
    map = ax.imshow(
        pixel_patch.T,
        extent=(0, pixel_patch.shape[0], pixel_patch.shape[1], 0),
        interpolation=None,
    )
    fig.colorbar(map, ax=ax)
    for point in points:
        ax.scatter(*point)
    if point_labels is not None:
        ax.legend(point_labels)
    if contour is not None:
        ax.contour(
            np.linspace(0, pixel_patch.shape[0], contour.shape[0]),
            np.linspace(0, pixel_patch.shape[1], contour.shape[1]),
            contour,
            levels=[1 - 0.997, 1 - 0.95, 1 - 0.68], # 3, 2, 1 std devs
            colors="k",
        )
    return fig, ax
