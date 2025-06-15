from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Configuration for the dataset loader."""

    directory: str = "./data"
    pixels_file: str = "pixelated_5um_tracks_thinned_4um_back_1M_300keV.txt"
    num_workers: int = 8
    train_percentage: float = 0.95
    events_per_image_range: list[int] = field(default_factory=lambda: [100, 200])
    noise_std: float = 1.0

    # sparsification parameters
    pixel_patch_size: int = 7
    n_sigma_sparsify: float = 3
    max_pixels_to_keep: int = 50000

    # if not None, re-map the electrons onto a new grid
    new_grid_size: Optional[list[int]] = None

    # shared seed among all workers for dataset shuffling
    shared_shuffle_seed: int = "${system.seed}"  # pyright: ignore[reportAssignmentType]
