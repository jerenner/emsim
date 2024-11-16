from torch import nn

class EMModelConfig:
    def __init__(
        self,
        unet_encoder_layers: list[int] = [2, 2, 2, 2],
        unet_decoder_layers: list[int] = [2, 2, 2, 2],
        unet_encoder_channels: list[int] = [32, 64, 128, 256],
        unet_decoder_channels: list[int] = [256, 128, 64, 32],
        unet_stem_channels: int = 16,
        unet_act_layer: nn.Module = nn.ReLU,
        unet_norm_layer: nn.Module = nn.BatchNorm1d,
        unet_encoder_drop_path_rate: float = 0.0,
        unet_decoder_drop_path_rate: float = 0.0,
        pixel_max_occupancy: int = 5,
        transformer_d_model: int = 256,
        transformer_hidden_dim: int = 1024,
        transformer_n_heads: int = 8,
        transformer_n_deformable_points: int = 4,
        transformer_dropout: float = 0.1,
        transformer_activation_fn: str = "gelu",
        transformer_encoder_layers: int = 6,
        transformer_decoder_layers: int = 6,
        transformer_level_filter_ratio: tuple[float] = (0.25, 0.5, 1.0, 1.0),
        transformer_layer_filter_ratio: tuple[float] = (1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
        transformer_encoder_max_tokens: int = 10000,
        transformer_n_query_embeddings: int = 1000,
        matcher_cost_coef_class: float = 1.0,
        matcher_cost_coef_mask: float = 1.0,
        matcher_cost_coef_dice: float = 1.0,
        matcher_cost_coef_dist: float = 1.0,
        matcher_cost_coef_nll: float = 1.0,
        loss_coef_class: float = 1.0,
        loss_coef_mask_bce: float = 1.0,
        loss_coef_mask_dice: float = 1.0,
        loss_coef_incidence_nll: float = 1.0,
        loss_coef_incidence_huber: float = 1.0,
        loss_no_electron_weight: float = 0.1,
        aux_loss=True,
    ):
        self.unet_encoder_layers = unet_encoder_layers
        self.unet_decoder_layers = unet_decoder_layers
        self.unet_encoder_channels = unet_encoder_channels
        self.unet_decoder_channels = unet_decoder_channels
        self.unet_stem_channels = unet_stem_channels
        self.unet_act_layer = unet_act_layer
        self.unet_norm_layer = unet_norm_layer
