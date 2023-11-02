import numpy as np

import h5py
from torch.utils.data import Data


def compute_indices(center, array_size, window_size=np.array([3, 3])):
    low = center - window_size // 2
    high = center + window_size // 2 + window_size % 2

    low_below_0 = low < 0
    high_above_len = high > array_size
    on_edge = np.any(low_below_0 | high_above_len, -1)

    low = low[~on_edge]
    high = high[~on_edge]

    return low, high


def extract_surrounding_windows(raw_frames, counted_frames, window_size=np.array((3, 3))):
    windows = []
    for raw, counted in zip(raw_frames, counted_frames):
        frame_windows = []
        window_centers = np.argwhere(counted)
        window_low_indices, window_high_indices = compute_indices(window_centers, raw.shape, window_size)
        frame_windows = []
        for low, high in zip(window_low_indices, window_high_indices):
            window = raw[low[0]:high[0], low[1]:high[1]]
            frame_windows.append(window)

        windows.append(np.stack(frame_windows))

    return windows


def get_summed_windowed_energies(raw_frames, counted_frames, window_size=np.array((3, 3))):
    windows = extract_surrounding_windows(raw_frames, counted_frames, window_size)
    energies = [np.sum(frame_windows, (-2, -1)) for frame_windows in windows]
    return energies
