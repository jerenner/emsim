# pyright: reportAssignmentType=false
from dataclasses import dataclass, field

from emsim.networks.positional_encoding.rope import FreqGroupPattern


@dataclass
class RoPEConfig:
    """Configuration for Rotary Position Embedding."""

    spatial_dimension: int = "${spatial_dimension}"
    spatial_base_theta: float = 100.0
    level_base_theta: float = 10.0
    share_heads: bool = False
    # freq_group_pattern: str = "partition"  # ["single", "partition", "closure"]
    freq_group_pattern: FreqGroupPattern = FreqGroupPattern.PARTITION
    enforce_freq_groups_equal: bool = True


@dataclass
class TransformerEncoderConfig:
    """Configuration for the transformer encoder."""

    n_layers: int = 6

    use_ms_deform_attn: bool = False
    use_neighborhood_attn: bool = True

    layer_filter_ratio: list[float] = "${model.transformer.layer_filter_ratio}"
    max_tokens_sa: int = 1000
    max_tokens_non_sa: int = 10000

    use_rope: bool = True


@dataclass
class ClassificationHeadConfig:
    """Configuration for the classification head."""

    hidden_dim: int = "${model.transformer.d_model}"
    n_layers: int = 2
    activation_fn: str = "gelu"


@dataclass
class PositionHeadConfig:
    """Configuration for the position offset head."""

    hidden_dim: int = "${model.transformer.d_model}"
    n_layers: int = 2
    activation_fn: str = "gelu"


@dataclass
class StdDevHeadConfig:
    """Configuration for the standard deviation head."""

    hidden_dim: int = "${model.transformer.d_model}"
    n_layers: int = 2
    activation_fn: str = "gelu"
    scaling_factor: float = 0.001
    eps: float = 1e-6


@dataclass
class SegmentationHeadConfig:
    """Configuration for the segmentation head."""

    hidden_dim: int = "${model.transformer.d_model}"
    n_layers: int = 2
    activation_fn: str = "gelu"
    query_patch_diameter: int = 7

    rope: RoPEConfig = "${model.transformer.rope}"


@dataclass
class TransformerDecoderConfig:
    """Configuration for the transformer decoder."""

    n_layers: int = 6

    use_ms_deform_attn: bool = False
    use_neighborhood_attn: bool = True
    use_full_cross_attn: bool = False

    look_forward_twice: bool = True
    detach_updated_positions: bool = True
    layers_share_heads: bool = False
    predict_box: bool = "${model.predict_box}"

    use_rope: bool = True

    classification_head: ClassificationHeadConfig = field(
        default_factory=lambda: ClassificationHeadConfig()
    )
    position_head: PositionHeadConfig = field(
        default_factory=lambda: PositionHeadConfig()
    )
    std_dev_head: StdDevHeadConfig = field(default_factory=lambda: StdDevHeadConfig())
    segmentation_head: SegmentationHeadConfig = field(
        default_factory=lambda: SegmentationHeadConfig()
    )


@dataclass
class TransformerConfig:
    """Configuration for the transformer model."""

    # Architecture parameters
    spatial_dimension: int = "${spatial_dimension}"
    d_model: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    dim_feedforward: int = 1024
    activation_fn: str = "gelu"
    attn_proj_bias: bool = False
    norm_first: bool = True

    # MS Deform Attention parameters
    backbone_decoder_layers: list[int] = "${model.backbone.decoder.layers}"
    n_feature_levels: int = field(init=False)
    n_deformable_points: int = 4

    # Neighborhood Attention parameters
    neighborhood_sizes: list[int] = field(default_factory=lambda: [3, 5, 7, 9])

    # Query embeddings (should be at least as large as the maximum number of
    # objects in an image)
    query_embeddings: int = 500

    # Salience filtering parameters
    level_filter_ratio: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 1.0, 1.0]
    )
    layer_filter_ratio: list[float] = field(
        default_factory=lambda: [1.0, 0.8, 0.6, 0.6, 0.4, 0.2]
    )

    # predict box instead of point
    predict_box: bool = "${model.predict_box}"
    # Denoising handling parameter
    mask_main_queries_from_denoising: bool = (
        "${model.denoising.mask_main_queries_from_denoising}"
    )

    # Nested configurations
    rope: RoPEConfig = field(default_factory=lambda: RoPEConfig())
    encoder: TransformerEncoderConfig = field(
        default_factory=lambda: TransformerEncoderConfig()
    )
    decoder: TransformerDecoderConfig = field(
        default_factory=lambda: TransformerDecoderConfig()
    )

    def __post_init__(self):
        if not isinstance(self.backbone_decoder_layers, str):
            try:
                self.n_feature_levels = len(self.backbone_decoder_layers)
            except (TypeError, ValueError):
                self.n_feature_levels = 4
        else:
            self.n_feature_levels = 4

        if isinstance(self.layer_filter_ratio, float):
            self.layer_filter_ratio = [self.layer_filter_ratio] * self.encoder.n_layers
        if len(self.layer_filter_ratio) != self.encoder.n_layers:
            raise ValueError(
                "layer_filter_ratio must have length equal to number of encoder layers"
            )

        if isinstance(self.level_filter_ratio, float):
            self.level_filter_ratio = [self.level_filter_ratio] * self.n_feature_levels
        if len(self.level_filter_ratio) != self.n_feature_levels:
            raise ValueError(
                "level_filter_ratio must have length equal to number of feature levels"
            )

        if any([not 0.0 <= ratio <= 1.0 for ratio in self.level_filter_ratio]):
            raise ValueError(
                "level_filter_ratio elements must all be in [0, 1], got "
                f"{self.level_filter_ratio}"
            )
        if any([not 0.0 <= ratio <= 1.0 for ratio in self.layer_filter_ratio]):
            raise ValueError(
                "layer_filter_ratio elements must all be in [0, 1], got "
                f"{self.layer_filter_ratio}"
            )

        if self.encoder.use_neighborhood_attn or self.decoder.use_neighborhood_attn:
            if len(self.neighborhood_sizes) != self.n_feature_levels:
                raise ValueError(
                    "len(neighborhood_sizes) must equal n_feature_levels, but got "
                    f"{len(self.neighborhood_sizes)} and {self.n_feature_levels}"
                )
