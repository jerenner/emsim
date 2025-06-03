from dataclasses import dataclass, field

from .backbone import BackboneConfig
from .transformer import TransformerConfig
from .criterion import CriterionConfig
from .denoising import DenoisingConfig


@dataclass
class EMModelConfig:
    """Top-level configuration for the EM Model."""

    backbone: BackboneConfig = field(default_factory=lambda: BackboneConfig())
    transformer: TransformerConfig = field(default_factory=lambda: TransformerConfig())
    criterion: CriterionConfig = field(default_factory=lambda: CriterionConfig())
    denoising: DenoisingConfig = field(default_factory=lambda: DenoisingConfig())
    predict_box: bool = False
    include_aux_outputs: bool = True
