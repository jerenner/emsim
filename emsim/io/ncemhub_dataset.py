import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import Dataset, default_collate

import stempy.io as stio

__raw_file_regex = re.compile(".*/?data_scan(\d+).h5")
__counted_file_regex = re.compile(".*/?data_scan(\d+)_id\d+_electrons.h5")


@dataclass
class _Scan:
    id: int
    raw_filename: Path
    counted_filename: Path
    frames_shape: Optional[tuple[int]] = None
    first_frame_index: Optional[int] = 0
    _raw_ptr: Optional[h5py.File] = None
    _counted_ptr: Optional[h5py.File] = None

    def __post_init__(self):
        if self.frames_shape is None:
            with h5py.File(self.raw_filename) as f:
                self.frames_shape = f["frames"].shape

    @property
    def n_frames(self):
        return np.prod(self.frames_shape[:-2])

    @property
    def frame_size(self):
        return self.frames_shape[-2:]

    @property
    def raw_ptr(self):
        if self._raw_ptr is None:
            self._raw_ptr = h5py.File(self.raw_filename)
        return self._raw_ptr

    @property
    def counted_ptr(self):
        if self._counted_ptr is None:
            self._counted_ptr = h5py.File(self.counted_filename)
        return self._counted_ptr

    def __del__(self):
        if self._raw_ptr and hasattr(self._raw_ptr, "close"):
            self._raw_ptr.close()
        if self._counted_ptr and hasattr(self._counted_ptr, "close"):
            self._counted_ptr.close()

    def raw_frame(self, frame_index) -> np.ndarray:
        return self.raw_ptr["frames"][frame_index]

    def counted_frame(self, frame_index) -> np.ndarray:
        data = self.counted_ptr["electron_events/frames"][frame_index]
        assert np.all(
            np.equal(
                self.frame_size,
                (
                    self.counted_ptr["electron_events/frames"].attrs.get("Nx"),
                    self.counted_ptr["electron_events/frames"].attrs.get("Ny"),
                ),
            )
        )

        frame = np.zeros(self.frame_size, dtype=bool)
        frame.reshape(-1)[data] = 1
        return frame


def _parse_data_dirs(raw_folder, counted_folder) -> list[_Scan]:
    def parse_scan_dir(directory, regex):
        filenames = glob.glob(os.path.join(directory, "*.h5"))
        files = [re.match(regex, f) for f in filenames]
        files = [f for f in files if f]
        files = {int(f.group(1)): os.path.abspath(f.group(0)) for f in files}
        return files

    raw_scans = parse_scan_dir(raw_folder, __raw_file_regex)
    counted_scans = parse_scan_dir(counted_folder, __counted_file_regex)

    scans = [
        _Scan(scan_id, raw_filename, counted_scans[scan_id])
        for scan_id, raw_filename in raw_scans.items()
        if scan_id in counted_scans
    ]

    scans = sorted(scans, key=lambda x: x.id)
    total_frames = 0
    for scan in scans:
        scan.first_frame_index = total_frames
        total_frames += scan.n_frames
    return scans


class NCEMHubDataset(Dataset):
    def __init__(self, raw_directory, counted_directory, electron_window_size=3):
        self.raw_folder = raw_directory
        self.counted_folder = counted_directory
        self.electron_window_size = electron_window_size

        self.scans: list[_Scan] = _parse_data_dirs(raw_directory, counted_directory)

    def __len__(self):
        return sum([scan.n_frames for scan in self.scans])

    def _start_indices(self):
        return [scan.first_frame_index for scan in self.scans]

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        # Get the last scan with first_frame_index lower than idx
        scan = next(
            scan for scan in reversed(self.scans) if idx >= scan.first_frame_index
        )
        local_index = idx - scan.first_frame_index
        raw_frame = scan.raw_frame(local_index).astype(np.float16)
        counted_frame = scan.counted_frame(local_index).astype(bool)

        windows = windowed_electrons_for_frame(raw_frame, counted_frame)
        energies = np.sum(windows, (-2, -1))
        sparsified_raw_frame = sparsify_raw_frame_from_counted_frame(
            raw_frame, counted_frame, window_size=self.electron_window_size
        )

        return {
            "raw_frame": raw_frame,
            "counted_frame": counted_frame,
            "sparsified_raw_frame": sparsified_raw_frame,
            "scan_id": scan.id,
            "local_index": local_index,
            "index": idx,
            "energies": energies,
            "windows": windows,
        }

    def get_all_energies_of_scan(self, scan_index):
        scan = self.scans[scan_index]
        raw_frames = scan.raw_frame(slice(None))
        counted_frames = stio.load_electron_counts(scan.counted_filename).ravel_scans()
        windows = extract_surrounding_windows(raw_frames, counted_frames)
        frame_window_energies = [
            np.sum(frame_windows, (-2, -1)) for frame_windows in windows
        ]
        return np.concatenate(frame_window_energies)


def sparsify_raw_frame_from_counted_frame(raw_frame, counted_frame, window_size=3):
    if raw_frame.ndim == 2:
        window = np.ones([window_size, window_size], dtype=bool)
    elif raw_frame.ndim == 3:
        window = np.zeros([3, window_size, window_size], dtype=bool)
        window[1] = True
    else:
        raise ValueError("Expected frames of dimensionality 2 or 3")

    # dilate the single pixels in the counted frame to the specified size
    dilated_counted = ndimage.binary_dilation(counted_frame, window)

    # create an all-zero array and fill in the entries from the raw frame
    out = np.zeros_like(raw_frame)
    out[dilated_counted.nonzero()] = raw_frame[dilated_counted.nonzero()]
    return out


def compute_indices(center, array_size, window_size=np.array([3, 3])):
    low = center - window_size // 2
    high = center + window_size // 2 + window_size % 2

    low_below_0 = low < 0
    high_above_len = high > array_size
    on_edge = np.any(low_below_0 | high_above_len, -1)

    low = low[~on_edge]
    high = high[~on_edge]

    return low, high


def extract_surrounding_windows(
    raw_frames: np.ndarray, counted_frames: np.ndarray, window_size=np.array((3, 3))
) -> np.ndarray:
    if raw_frames.ndim == 2:
        raw_frames = np.expand_dims(raw_frames, 0)
    if counted_frames.ndim == 2:
        counted_frames = np.expand_dims(counted_frames, 0)
    windows = [
        windowed_electrons_for_frame(raw, counted, window_size)
        for raw, counted, in zip(raw_frames, counted_frames)
    ]
    return windows


def windowed_electrons_for_frame(
    raw_frame: np.ndarray, counted_frame: np.ndarray, window_size=np.array((3, 3))
):
    assert raw_frame.ndim == 2
    assert counted_frame.ndim == 2
    frame_windows = []
    window_centers = np.argwhere(counted_frame)
    window_low_indices, window_high_indices = compute_indices(
        window_centers, raw_frame.shape, window_size
    )
    frame_windows = []
    for low, high in zip(window_low_indices, window_high_indices):
        window = raw_frame[low[0] : high[0], low[1] : high[1]]
        frame_windows.append(window)

    return np.stack(frame_windows, 0)


def collate(batch):
    windows = [sample.pop("windows") for sample in batch]
    energies = [sample.pop("energies") for sample in batch]

    batch = default_collate(batch)

    lengths = torch.as_tensor([en.shape[0] for en in energies])
    batch_tensor = torch.cat(
        [
            torch.repeat_interleave(torch.as_tensor(i), length)
            for i, length in enumerate(lengths)
        ]
    )

    batch["energies"] = torch.as_tensor(np.concatenate(energies, 0))
    batch["windows"] = torch.as_tensor(np.concatenate(windows, 0))
    batch["batch_indices"] = batch_tensor

    return batch
