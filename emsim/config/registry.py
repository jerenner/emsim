from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from .system import SystemConfig
from .model import EMModelConfig
from .dataset import DatasetConfig
from .training import TrainingConfig
from .resolvers import register_resolvers


@dataclass
class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: EMModelConfig = field(default_factory=EMModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    spatial_dimension: int = 2
    debug: bool = False


def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="base_config", node=Config)

    register_resolvers()
