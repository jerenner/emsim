from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters"""

    lr: float = 0.0001
    max_grad_norm: float = 0.01
    batch_size: int = 1
    num_steps: int = 10000  # total training steps
    warmup_percentage: float = 0.1

    cpu_only: bool = False  # set True to not use cuda

    eval_steps: int = 1000  # evaluation interval

    seed: int = 1234
    print_interval: int = 10

    resume_file: Optional[str] = None  # checkpoint file to resume from

    log_tensorboard: bool = True
    tensorboard_name: Optional[str] = None  # tensorboard tag

    # interval to clear the cuda cache as recommended by MinkowskiEngine
    clear_cache_interval: int = 1000

    log_level: str = "INFO"  # passed to logger
    compile: bool = False  # whether to use torch.compile
