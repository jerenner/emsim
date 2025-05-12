from dataclasses import dataclass, field

from .model import EMModelConfig
from .dataset import DatasetConfig
from .training import TrainingConfig


@dataclass
class DDPConfig:
    """Configuration for distributed data parallel"""

    nodes: int = 2
    devices: int = 2

    # debugging settings
    find_unused_parameters: bool = False  # leave off if not needed
    detect_anomaly: bool = False  # sets torch.autograd.set_detect_anomaly for finding nans

@dataclass
class AppConfig:
    """Top-level application configuration."""
    seed: int = 1234
    debug: bool = False
    device: str = "cuda"

    spatial_dimension: int = 2

    ddp: DDPConfig = field(default_factory=DDPConfig)

    # Components
    model: EMModelConfig = field(default_factory=EMModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
