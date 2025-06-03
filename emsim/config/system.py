from dataclasses import dataclass, field


@dataclass
class DDPConfig:
    """Configuration for distributed data parallel"""

    nodes: int = 1
    devices: int = 1

    # debugging settings
    find_unused_parameters: bool = False  # leave off if not needed
    detect_anomaly: bool = False  # sets torch.autograd.set_detect_anomaly (nan finder)


@dataclass
class SystemConfig:
    """Top-level application configuration."""

    seed: int = 1234
    device: str = "cuda"

    ddp: DDPConfig = field(default_factory=DDPConfig)
