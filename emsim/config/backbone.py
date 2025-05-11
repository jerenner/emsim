from dataclasses import dataclass, field

@dataclass
class BackboneEncoderConfig:
    """Configuration for UNet encoder."""
    layers: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    drop_path_rate: float = 0.0

    output_stride = "${backbone_encoder_output_stride:}"

    # Values from parent config
    in_channels: int = "${model.backbone.in_channels}"
    stem_channels: int = "${model.backbone.stem_channels}"
    stem_kernel_size: int = "${model.backbone.stem_kernel_size}"

    dimension: int = "${model.backbone.dimension}"
    bias: bool = "${model.backbone.bias}"
    act_layer: str = "${model.backbone.act_layer}"
    norm_layer: str = "${model.backbone.norm_layer}"

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

    encoder: BackboneEncoderConfig = field(default_factory=BackboneEncoderConfig)
    decoder: BackboneDecoderConfig = field(default_factory=BackboneDecoderConfig)

    act_layer: str = "relu"
    norm_layer: str = "batchnorm1d"
    convert_sync_batch_norm: bool = True
