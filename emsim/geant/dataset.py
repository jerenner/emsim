from math import ceil, floor
from typing import Any, Callable, Optional, Tuple

import numpy as np
import sparse
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate

from emsim.dataclasses import BoundingBox, IonizationElectronPixel, PixelSet
from emsim.geant.dataclasses import GeantElectron, Trajectory, GeantGridsize
from emsim.geant.io import read_files
from emsim.utils.misc_utils import (
    random_chunks,
)

# keys in here will not be batched in the collate fn
_KEYS_TO_NOT_BATCH = ("local_trajectories_pixels", "hybrid_sparse_tensors")

# these keys get passed to the default collate_fn, everything else uses custom batching logic
_TO_DEFAULT_COLLATE = (
    "image",
    "batch_size",
    "noiseless_image",
)

# these sparse arrays get stacked with an extra batch dimension
_SPARSE_STACK = (
    "segmentation_background",
    "image_sparsified",
    "segmentation_mask",
    "electron_count_map_1/1",
    "electron_count_map_1/2",
    "electron_count_map_1/4",
    "electron_count_map_1/8",
    "electron_count_map_1/16",
    "peak_normalized_noiseless_image_1/1",
    "peak_normalized_noiseless_image_1/2",
    "peak_normalized_noiseless_image_1/4",
    "peak_normalized_noiseless_image_1/8",
    "peak_normalized_noiseless_image_1/16",
    "peak_normalized_noiseless_image_1/32",
    "peak_normalized_noiseless_image_1/64",
)

# these sparse arrays get concatenated, with no batch dimension
_SPARSE_CONCAT = []

ELECTRON_IONIZATION_MEV = 3.6e-6


def make_test_train_datasets(
    pixels_file: str,
    events_per_image_range: tuple[int],
    pixel_patch_size: int = 5,
    hybrid_sparse_tensors: bool = True,
    trajectory_file: Optional[str] = None,
    train_percentage: float = 0.95,
    processor: Callable = None,
    transform: Callable = None,
    noise_std: float = 1.0,
    shuffle: bool = True,
    seed: Optional[int] = None,
):
    electrons = read_files(
        pixels_file=pixels_file, trajectory_file=trajectory_file
    )
    assert 0 < train_percentage <= 1
    train_test_split = int(len(electrons) * train_percentage)
    train_electrons = electrons[:train_test_split]
    test_electrons = electrons[train_test_split:]

    if len(train_electrons) > 0:
        train_dataset = GeantElectronDataset(
            electrons=train_electrons,
            events_per_image_range=events_per_image_range,
            pixel_patch_size=pixel_patch_size,
            hybrid_sparse_tensors=hybrid_sparse_tensors,
            processor=processor,
            transform=transform,
            noise_std=noise_std,
            shuffle=shuffle,
            seed=seed
        )
    else:
        train_dataset = None

    if len(test_electrons) > 0:
        test_dataset = GeantElectronDataset(
            electrons=test_electrons,
            events_per_image_range=events_per_image_range,
            pixel_patch_size=pixel_patch_size,
            hybrid_sparse_tensors=hybrid_sparse_tensors,
            processor=processor,
            transform=transform,
            noise_std=noise_std,
            shuffle=False,
            seed=seed
        )
    else:
        test_dataset = None

    return train_dataset, test_dataset

class GeantElectronDataset(IterableDataset):
    def __init__(
        self,
        electrons: list[GeantElectron],
        events_per_image_range: tuple[int],
        pixel_patch_size: int = 5,
        hybrid_sparse_tensors: bool = True,
        processor: Callable = None,
        transform: Callable = None,
        noise_std: float = 1.0,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.hybrid_sparse_tensors = hybrid_sparse_tensors
        self.electrons = electrons
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
        self._rng = np.random.default_rng(seed=seed)

    def __iter__(self):
        elec_indices = list(range(len(self.electrons)))
        if self.shuffle:
            self._rng.shuffle(elec_indices)
        chunks = random_chunks(elec_indices, *self.events_per_image_range)

        for chunk in chunks:
            elecs: list[GeantElectron] = [self.electrons[i] for i in chunk]
            batch = {}
            batch["electron_ids"] = np.array([elec.id for elec in elecs], dtype=int)
            batch["batch_size"] = len(elecs)
            batch["hybrid_sparse_tensors"] = self.hybrid_sparse_tensors

            # ionization electron statistics
            ionization_electrons = [
                np.array(
                    [pixel.ionization_electrons for pixel in elec.pixels],
                    dtype=np.float32,
                )
                for elec in elecs
            ]
            batch["total_ionization_electrons"] = np.array(
                [ions.sum() for ions in ionization_electrons]
            )
            batch["peak_ionization_electrons"] = np.array(
                [ions.max() for ions in ionization_electrons]
            )

            # stacked sparse arrays of single electron strikes
            stacked_sparse_arrays = sparse_single_electron_arrays(elecs)

            # composite electron image
            batch["image"] = self.make_composite_image(
                stacked_sparse_arrays, self.noise_std
            )
            batch["noiseless_image"] = self.make_composite_image(
                stacked_sparse_arrays, 0.0
            )
            batch.update(multiscale_peak_normalized_maps(stacked_sparse_arrays))

            # image size info
            batch["image_size_pixels_rc"] = np.array([self.grid.ymax_pixel, self.grid.xmax_pixel])
            batch["image_size_um_xy"] = np.array([self.grid.xmax_um, self.grid.ymax_um])

            # bounding boxes
            boxes = [elec.pixels.get_bounding_box() for elec in elecs]
            boxes_normalized = normalize_boxes(
                boxes, self.grid.xmax_pixel, self.grid.ymax_pixel
            )
            boxes_array = [box.asarray() for box in boxes]
            batch["bounding_boxes_pixels_xyxy"] = np.array(
                boxes_array, dtype=np.float32
            )
            batch["bounding_boxes_normalized_xyxy"] = np.array(
                boxes_normalized, dtype=np.float32
            )
            batch["bounding_boxes_pixels_cxcywh"] = np.array(
                [box.center_format() for box in boxes], dtype=np.float32
            )

            # patches centered on high-energy pixels
            patches, patch_coords = get_pixel_patches(
                batch["image"], elecs, self.pixel_patch_size
            )
            patch_coords_mm = patch_coords * np.concatenate(
                [self.pixel_to_mm, self.pixel_to_mm]
            )
            batch["pixel_patches"] = patches.astype(np.float32)
            batch["patch_coords_xyxy"] = patch_coords.astype(int)

            # actual point the geant electron hit the detector surface
            incidence_points_xy = np.stack(
                [np.array([elec.incidence.x, elec.incidence.y]) for elec in elecs]
            )
            normalized_incidence_points_xy = (
                incidence_points_xy
                * 1000
                / batch["image_size_um_xy"]
            )
            incidence_points_pixels_rc = np.fliplr(
                incidence_points_xy
            ) / self.pixel_to_mm.astype(np.float32)
            local_incidence_points_mm = incidence_points_xy - patch_coords_mm[:, :2]
            local_incidence_points_pixels = (
                local_incidence_points_mm / self.pixel_to_mm.astype(np.float32)
            )
            batch["incidence_points_xy"] = incidence_points_xy.astype(np.float32)
            batch["normalized_incidence_points_xy"] = normalized_incidence_points_xy
            batch["incidence_points_pixels_rc"] = incidence_points_pixels_rc
            batch["local_incidence_points_pixels_xy"] = (
                local_incidence_points_pixels.astype(np.float32)
            )

            # center of mass for each patch
            local_centers_of_mass_pixels = charge_2d_center_of_mass(patches)
            batch["local_centers_of_mass_pixels_xy"] = local_centers_of_mass_pixels
            batch["normalized_centers_of_mass_xy"] = (
                local_centers_of_mass_pixels + patch_coords[:, :2]
            ) / np.flip(batch["image_size_pixels_rc"], -1)

            # whole trajectories, if trajectory file is given
            # Note: could have x/y out of order, need to check if needed
            # if self.trajectory_file is not None:
            #     local_trajectories = [
            #         elec.trajectory.localize(coords[0], coords[1]).as_array()
            #         for elec, coords in zip(elecs, patch_coords_mm)
            #     ]
            #     trajectory_mm_to_pixels = np.concatenate(
            #         [1 / self.pixel_to_mm, np.array([1.0, 1.0])]
            #     ).astype(np.float32)
            #     local_trajectories_pixels = [
            #         traj * trajectory_mm_to_pixels for traj in local_trajectories
            #     ]
            #     batch["local_trajectories_pixels"] = local_trajectories_pixels

            # per-electron segmentation mask
            segmentation_mask, background = make_soft_segmentation_mask(
                stacked_sparse_arrays
            )
            batch["segmentation_mask"] = segmentation_mask
            batch["segmentation_background"] = background

            # multiscale incidence count maps
            incidence_points_rc = incidence_points_xy[..., ::-1]
            incidence_map = incident_pixel_map(incidence_points_rc, elecs[0].grid)
            count_maps = multiscale_electron_count_maps(incidence_map)
            batch.update(count_maps)

            if self.processor is not None:
                batch = self.processor(batch)

            if self.transform is not None:
                batch = self.transform(batch)

            yield batch

    def make_composite_image(
        self, stacked_sparse_arrays: sparse.SparseArray, noise_std: float = 0.0
    ) -> np.ndarray:
        summed = stacked_sparse_arrays.sum(-1)
        image = summed.todense()

        if noise_std > 0:
            image = image + self._rng.normal(
                0.0, self.noise_std, size=image.shape
            ).astype(image.dtype)

        # add channel dimension
        image = np.expand_dims(image, 0)

        return image


def sparse_single_electron_arrays(electrons: list[GeantElectron], dtype=np.float32):
    grid = electrons[0].grid
    single_electron_arrays = [
        sparsearray_from_pixels(
            electron.pixels, (grid.xmax_pixel, grid.ymax_pixel), dtype=dtype
        )
        for electron in electrons
    ]
    sparse_array = sparse.stack(single_electron_arrays, -1)
    return sparse_array


def sparsearray_from_pixels(
    pixelset: PixelSet,
    shape: Tuple[int],
    offset_x: Optional[int] = None,
    offset_y: Optional[int] = None,
    dtype=None,
):
    x_indices, y_indices, data = [], [], []
    for p in pixelset:
        x_index, y_index = p.index()
        x_indices.append(x_index)
        y_indices.append(y_index)
        data.append(p.data)
    x_indices = np.array(x_indices)  # columns
    y_indices = np.array(y_indices)  # rows
    if offset_x is not None:
        x_indices = x_indices - offset_x
    if offset_y is not None:
        y_indices = y_indices - offset_y
    # array = scipy.sparse.coo_array((np.array(data, dtype=dtype), (y_indices, x_indices)), shape=shape)
    array = sparse.COO(
        np.stack([y_indices, x_indices], 0), np.array(data, dtype=dtype), shape=shape
    )
    return array


def normalize_boxes(boxes: list[BoundingBox], image_width: int, image_height: int):
    center_format_boxes = np.stack([box.center_format() for box in boxes], 0)
    center_format_boxes /= np.array(
        [image_width, image_height, image_width, image_height]
    )
    return center_format_boxes


def make_soft_segmentation_mask(stacked_sparse_arrays: sparse.SparseArray):
    sparse_sum = stacked_sparse_arrays.sum(-1, keepdims=True)

    background = ~sparse_sum.astype(bool)
    denom = sparse_sum + background

    sparse_soft_segmap = stacked_sparse_arrays / denom

    return sparse_soft_segmap, background


def incident_pixel_map(
    incidence_points: np.ndarray, grid: GeantGridsize
) -> sparse.SparseArray:
    incidence_pixels = incidence_points // (grid.pixel_size_um / np.array(1000))
    incidence_pixels = incidence_pixels.astype(int)
    pixels, counts = np.unique(incidence_pixels, return_counts=True, axis=0)
    out = sparse.COO(pixels.T, counts, shape=(grid.xmax_pixel, grid.ymax_pixel))
    return out


def multiscale_electron_count_maps(
    incidence_map: sparse.SparseArray,
    downscaling_levels: list[int] = [1, 2, 4, 8, 16],
) -> list[sparse.SparseArray]:
    height, width = incidence_map.shape

    out = {}
    for ds in downscaling_levels:
        array = incidence_map.copy()

        if ds > 1:
            # prune the last row/columns if not evenly divisible

            height_remainder = height % ds
            width_remainder = width % ds
            array = array[: height - height_remainder, : width - width_remainder]

            # sum up over windows
            array = array.reshape([height // ds, ds, width // ds, ds]).sum([1, 3])

        out[f"electron_count_map_1/{ds}"] = array

    return out


def multiscale_peak_normalized_maps(
    stacked_sparse_arrays: sparse.SparseArray,
    downscaling_levels: list[int] = [1, 2, 4, 8, 16, 32, 64],
) -> list[sparse.SparseArray]:
    height, width, n_elecs = stacked_sparse_arrays.shape
    out = {}
    for ds in downscaling_levels:
        array = stacked_sparse_arrays.copy()
        if ds > 1:
            height_remainder = height % ds
            width_remainder = width % ds
            array = array[: height - height_remainder, : width - width_remainder]

            array = array.reshape([height // ds, ds, width // ds, ds, n_elecs]).sum(
                [1, 3]
            )

        peaks = array.max([0, 1]).todense()
        array = array / peaks
        out[f"peak_normalized_noiseless_image_1/{ds}"] = array.max(-1)

    return out


def get_pixel_patches(
    image: np.ndarray,
    electrons: list[GeantElectron],
    patch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    patches = []
    patch_coordinates = []

    for electron in electrons:
        peak_pixel: IonizationElectronPixel = max(
            electron.pixels, key=lambda x: x.ionization_electrons
        )

        row_min = peak_pixel.y - patch_size // 2
        row_max = peak_pixel.y + patch_size // 2 + 1
        col_min = peak_pixel.x - patch_size // 2
        col_max = peak_pixel.x + patch_size // 2 + 1
        patch_coordinates.append(np.array([col_min, row_min, col_max, row_max]))

        # compute padding if any
        if row_min < 0:
            pad_top = -row_min
            row_min = 0
        else:
            pad_top = 0
        if col_min < 0:
            pad_left = -col_min
            col_min = 0
        else:
            pad_left = 0

        pad_bottom = max(0, row_max - image.shape[-2])
        pad_right = max(0, col_max - image.shape[-1])

        patch = image[..., row_min:row_max, col_min:col_max]
        patch = np.pad(patch, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))

        patches.append(patch)
    return np.stack(patches, 0), np.stack(patch_coordinates, 0)


def charge_2d_center_of_mass(patches: np.ndarray, com_patch_size=3):
    if com_patch_size % 2 == 0:
        raise ValueError(f"'com_patch_size' should be odd, got '{com_patch_size}'")
    if patches.ndim == 2:
        # add batch dim
        patches = np.expand_dims(patches, 0)

    patch_r_len = patches.shape[-2]
    patch_c_len = patches.shape[-1]
    coord_grid = np.stack(
        np.meshgrid(
            np.arange(0.5, patches.shape[-2], 1), np.arange(0.5, patches.shape[-1], 1)
        )
    )

    patch_r_mid = patch_r_len / 2
    patch_c_mid = patch_c_len / 2

    patch_radius = com_patch_size // 2

    patches = patches[
        ...,
        floor(patch_r_mid) - patch_radius : ceil(patch_r_mid) + patch_radius,
        floor(patch_c_mid) - patch_radius : ceil(patch_c_mid) + patch_radius,
    ]
    coord_grid = coord_grid[
        ...,
        floor(patch_r_mid) - patch_radius : ceil(patch_r_mid) + patch_radius,
        floor(patch_c_mid) - patch_radius : ceil(patch_c_mid) + patch_radius,
    ]

    # patches: batch * 1 * r * c
    # coord grid: 1 * 2 * r * c
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
            # if isinstance(first[key], (list, tuple)):
            #     sub_batch = [
            #         {
            #             str(i):
            #         }
            #     ]
            #     batch[key] = e[{}]
            lengths = [len(sample[key]) for sample in batch]
            if isinstance(first[key], sparse.SparseArray):
                if key in _SPARSE_CONCAT:
                    sparse_batched = sparse.concatenate(
                        [sample[key] for sample in batch], axis=0
                    )
                elif key in _SPARSE_STACK:
                    to_stack = _sparse_pad([sample[key] for sample in batch])
                    sparse_batched = sparse.stack(to_stack, axis=0)

                out_batch[key + "_indices"] = torch.tensor(sparse_batched.coords)
                out_batch[key + "_values"] = torch.tensor(sparse_batched.data)
                out_batch[key + "_shape"] = sparse_batched.shape

                # batch offsets for the nonzero points in the sparse tensor
                # (can find this later by finding first appearance of each batch
                # index but more efficient to do it here)
                out_batch[key + "_batch_offsets"] = torch.as_tensor(
                    np.cumsum([0] + [item.nnz for item in to_stack][:-1])
                )
            else:
                out_batch[key] = torch.as_tensor(
                    np.concatenate([sample[key] for sample in batch], axis=0)
                )

            # if key not in _SPARSE_STACK + _TO_DEFAULT_COLLATE:
            #     out_batch[key + "_batch_index"] = torch.cat(
            #         [
            #             torch.repeat_interleave(torch.as_tensor(i), length)
            #             for i, length in enumerate(lengths)
            #         ]
            #     )

    out_batch.update(default_collate_fn(to_default_collate))

    out_batch["electron_batch_offsets"] = torch.as_tensor(
        np.cumsum([0] + [len(item["electron_ids"]) for item in batch][:-1])
    )

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
    return torch.sparse_coo_tensor(
        hybrid_indices, hybrid_values, sparse_array.shape
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
