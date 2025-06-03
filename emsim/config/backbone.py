# pyright: reportAssignmentType=false
from dataclasses import dataclass, field

@dataclass
class BackboneEncoderConfig:
    """Configuration for UNet encoder."""
    layers: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    drop_path_rate: float = 0.0

    # To be computed
    output_stride: int = field(init=False)

    # Values from parent config
    in_channels: int = "${model.backbone.in_channels}"
    stem_channels: int = "${model.backbone.stem_channels}"
    stem_kernel_size: int = "${model.backbone.stem_kernel_size}"

    dimension: int = "${model.backbone.dimension}"
    bias: bool = "${model.backbone.bias}"
    act_layer: str = "${model.backbone.act_layer}"
    norm_layer: str = "${model.backbone.norm_layer}"

    def __post_init__(self):
        """Calculate output stride based on number of encoder layers."""
        # Check if layers is resolved yet
        if not isinstance(self.layers, str) and hasattr(self, 'layers'):
            try:
                self.output_stride = 2 ** len(self.layers)
            except (TypeError, ValueError):
                self.output_stride = 16  # Default (2^4)
        else:
            self.output_stride = 16  # Default (2^4)

@dataclass
class BackboneDecoderConfig:
    """Configuration for UNet decoder."""
    layers: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    channels: list[int] = field(default_factory=lambda: [256, 128, 64, 32])
    drop_path_rate: float = 0.0

    # Values from parent config
    dimension: int = "${model.backbone.dimension}"
    bias: bool = "${model.backbone.bias}"
    act_layer: str = "${model.backbone.act_layer}"
    norm_layer: str = "${model.backbone.norm_layer}"

@dataclass
class BackboneConfig:
    """Configuration for the backbone network (UNet)."""
    dimension: int = "${spatial_dimension}"
    in_channels: int = 1
    stem_channels: int = 16
    stem_kernel_size: int = 7
    bias: bool = True

    encoder: BackboneEncoderConfig = field(default_factory=lambda: BackboneEncoderConfig())
    decoder: BackboneDecoderConfig = field(default_factory=lambda: BackboneDecoderConfig())

    act_layer: str = "relu"
    norm_layer: str = "batchnorm1d"
    convert_sync_batch_norm: bool = True
