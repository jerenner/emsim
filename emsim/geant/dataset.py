from math import ceil, floor
from random import shuffle
from typing import Any, Callable, Optional

import numpy as np
import sparse
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate

from emsim.dataclasses import IonizationElectronPixel
from emsim.geant.dataclasses import GeantElectron, Trajectory
from emsim.geant.io import read_files
from emsim.utils import (
    random_chunks,
    normalize_boxes,
    sparsearray_from_pixels,
)

# keys in here will not be batched in the collate fn
_KEYS_TO_NOT_BATCH = ("local_trajectories_pixels", "hybrid_sparse_tensors")

# these keys get passed to the default collate_fn, everything else uses custom batching logic
_TO_DEFAULT_COLLATE = ("image", "batch_size")

# these sparse arrays get stacked with an extra batch dimension
_SPARSE_STACK = ("segmentation_background", "image_sparsified", "segmentation_mask")

# these sparse arrays get concatenated, with no batch dimension
_SPARSE_CONCAT = []

ELECTRON_IONIZATION_MEV = 3.6e-6


class GeantElectronDataset(IterableDataset):
    def __init__(
        self,
        pixels_file: str,
        events_per_image_range: tuple[int],
        pixel_patch_size: int = 5,
        hybrid_sparse_tensors: bool = True,
        trajectory_file: Optional[str] = None,
        train_percentage: float = 0.95,
        split: str = "train",
        processor: Callable = None,
        transform: Callable = None,
        noise_std: float = 1.0,
        shuffle=True,
    ):
        assert 0 < train_percentage <= 1
        assert split in ("train", "test")
        self.pixels_file = pixels_file
        self.trajectory_file = trajectory_file
        self.hybrid_sparse_tensors = hybrid_sparse_tensors

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
            batch = {}
            batch["electron_ids"] = np.array([elec.id for elec in elecs], dtype=int)
            batch["batch_size"] = len(elecs)
            batch["hybrid_sparse_tensors"] = self.hybrid_sparse_tensors

            # composite electron image
            batch["image"] = self.make_composite_image(elecs)

            # bounding boxes
            boxes = [elec.pixels.get_bounding_box() for elec in elecs]
            boxes_normalized = normalize_boxes(
                boxes, self.grid.xmax_pixel, self.grid.ymax_pixel
            )
            boxes_array = [box.asarray() for box in boxes]
            batch["bounding_boxes"] = boxes_normalized.astype(np.float32)
            batch["bounding_boxes_unnormalized"] = np.array(
                boxes_array, dtype=np.float32
            )

            # patches centered on high-energy pixels
            patches, patch_coords = get_pixel_patches(
                batch["image"], elecs, self.pixel_patch_size
            )
            patch_coords_mm = patch_coords * np.concatenate(
                [self.pixel_to_mm, self.pixel_to_mm]
            )
            batch["pixel_patches"] = patches.astype(np.float32)

            # actual point the geant electron hit the detector surface
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
            batch["incidence_points"] = incidence_points.astype(np.float32)
            batch[
                "local_incidence_points_pixels"
            ] = local_incidence_points_pixels.astype(np.float32)

            # center of mass for each patch
            local_centers_of_mass_pixels = charge_2d_center_of_mass(patches)
            batch["local_centers_of_mass_pixels"] = local_centers_of_mass_pixels.astype(
                np.float32
            )

            # whole trajectories, if trajectory file is given
            if self.trajectory_file is not None:
                local_trajectories = [
                    elec.trajectory.localize(coords[0], coords[1]).as_array()
                    for elec, coords in zip(elecs, patch_coords_mm)
                ]
                trajectory_mm_to_pixels = np.concatenate(
                    [1 / self.pixel_to_mm, np.array([1.0, 1.0])]
                ).astype(np.float32)
                local_trajectories_pixels = [
                    traj * trajectory_mm_to_pixels for traj in local_trajectories
                ]
                batch["local_trajectories_pixels"] = local_trajectories_pixels

            # per-electron segmentation mask
            segmentation_mask, background = make_soft_segmentation_mask(elecs)
            batch["segmentation_mask"] = segmentation_mask
            batch["segmentation_background"] = background

            if self.processor is not None:
                batch = self.processor(batch)

            if self.transform is not None:
                batch = self.transform(batch)

            yield batch

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


def make_soft_segmentation_mask(electrons: list[GeantElectron], dtype=np.float32):
    grid = electrons[0].grid
    single_electron_arrays = [
        sparsearray_from_pixels(
            electron.pixels, (grid.xmax_pixel, grid.ymax_pixel), dtype=dtype
        )
        for electron in electrons
    ]

    # convert to "sparse" package objects
    single_electron_arrays = [
        sparse.COO.from_scipy_sparse(array) for array in single_electron_arrays
    ]
    sparse_array = sparse.stack(single_electron_arrays, -1)
    sparse_sum = sparse_array.sum(-1, keepdims=True)

    background = ~sparse_sum.astype(bool)
    denom = sparse_sum + background

    sparse_soft_segmap = sparse_array / denom

    return sparse_soft_segmap, background


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

        patch = image[..., x_min:x_max, y_min:y_max]
        patch = np.pad(
            patch, ((0, 0), (pad_x_left, pad_x_right), (pad_y_top, pad_y_bottom))
        )

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
    out_batch = {}
    to_default_collate = [{} for _ in range(len(batch))]

    first = batch[0]
    for key in first:
        if key in _KEYS_TO_NOT_BATCH:
            continue
        if key in _TO_DEFAULT_COLLATE:
            for to_default, electron in zip(to_default_collate, batch):
                to_default[key] = electron[key]
        else:
            lengths = [len(sample[key]) for sample in batch]
            if isinstance(first[key], sparse.SparseArray):
                if key in _SPARSE_CONCAT:
                    sparse_batched = sparse.concatenate(
                        [sample[key] for sample in batch], axis=0
                    )
                elif key in _SPARSE_STACK:
                    to_stack = _sparse_pad([sample[key] for sample in batch])
                    sparse_batched = sparse.stack(to_stack, axis=0)

                if first.get("hybrid_sparse_tensors", False):
                    out_batch[key] = sparse_to_torch_hybrid(sparse_batched)
                else:
                    out_batch[key] = torch.sparse_coo_tensor(
                        sparse_batched.coords, sparse_batched.data, sparse_batched.shape
                    ).coalesce()
            else:
                out_batch[key] = torch.as_tensor(
                    np.concatenate([sample[key] for sample in batch], axis=0)
                )

            if key not in _SPARSE_STACK + _TO_DEFAULT_COLLATE:
                out_batch[key + "_batch_index"] = torch.cat(
                    [
                        torch.repeat_interleave(torch.as_tensor(i), length)
                        for i, length in enumerate(lengths)
                    ]
                )

    out_batch.update(default_collate_fn(to_default_collate))

    return out_batch


def sparse_to_torch_hybrid(sparse_array: sparse.SparseArray, n_hybrid_dims=1):
    dtype = sparse_array.dtype
    hybrid_indices = []
    hybrid_values = []
    hybrid_value_shape = sparse_array.shape[-n_hybrid_dims:]
    for index, value in zip(sparse_array.coords.T, sparse_array.data):
        torch_index = index[:-n_hybrid_dims]
        torch_value = np.zeros(hybrid_value_shape, dtype=dtype)
        torch_value[index[-n_hybrid_dims:]] = value
        hybrid_indices.append(torch_index)
        hybrid_values.append(torch_value)

    hybrid_indices = np.stack(hybrid_indices, -1)
    hybrid_values = np.stack(hybrid_values, 0)
    return torch.sparse_coo_tensor(hybrid_indices, hybrid_values, sparse_array.shape).coalesce()


def sparse_to_torch_hybrid_old(sparse_array: sparse.SparseArray):
    nonempty_coords = sparse_array.any(-1).coords
    values = [sparse_array[*coords].todense() for coords in nonempty_coords.T]
    values = np.stack(values)
    return torch.sparse_coo_tensor(
        torch.as_tensor(nonempty_coords),
        torch.as_tensor(values),
        size=sparse_array.shape,
    ).coalesce()


def _sparse_pad(items: list[sparse.SparseArray]):
    if len({x.shape for x in items}) > 1:
        max_shape = np.stack([item.shape for item in items], 0).max(0)
        items = [
            sparse.pad(
                item,
                [
                    (0, max_len - item_len)
                    for max_len, item_len in zip(max_shape, item.shape)
                ],
            )
            for item in items
        ]
    return items


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
            levels=[1 - 0.997, 1 - 0.95, 1 - 0.68],  # 3, 2, 1 std devs
            colors="k",
        )
    return fig, ax
