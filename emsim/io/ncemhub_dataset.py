import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

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
        super().__del__(self)

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
    def __init__(self, raw_directory, counted_directory):
        self.raw_folder = raw_directory
        self.counted_folder = counted_directory

        self.scans: list[_Scan] = _parse_data_dirs(raw_directory, counted_directory)

    def __len__(self):
        return sum([scan.n_frames for scan in self.scans])

    def _start_indices(self):
        return [scan.first_frame_index for scan in self.scans]

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        # Get the last scan with first_frame_index lower than idx
        scan = next(
            scan for scan in reversed(self.scans) if idx >= scan.first_frame_index
        )
        local_index = idx - scan.first_frame_index
        raw_frame = scan.raw_frame(local_index).astype(np.float16)
        counted_frame = scan.counted_frame(local_index).astype(np.float16)

        return {
            "raw_frame": raw_frame,
            "counted_frame": counted_frame,
            "scan_id": scan.id,
            "local_index": local_index,
            "index": idx,
        }


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
        raw_frames = raw_frames.unsqueeze(0)
    if counted_frames.ndim == 2:
        counted_frames = counted_frames.unsqueeze(0)
    windows = []
    for raw, counted in zip(raw_frames, counted_frames):
        frame_windows = []
        window_centers = np.argwhere(counted)
        window_low_indices, window_high_indices = compute_indices(
            window_centers, raw.shape, window_size
        )
        frame_windows = []
        for low, high in zip(window_low_indices, window_high_indices):
            window = raw[low[0] : high[0], low[1] : high[1]]
            frame_windows.append(window)

        windows.append(np.stack(frame_windows))

    return windows


def get_summed_windowed_energies(
    raw_frames, counted_frames, window_size=np.array((3, 3))
):
    windows = extract_surrounding_windows(raw_frames, counted_frames, window_size)
    energies = [np.sum(frame_windows, (-2, -1)) for frame_windows in windows]
    return energies
