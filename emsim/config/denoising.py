# pyright: reportAssignmentType=false
from dataclasses import dataclass


@dataclass
class DenoisingConfig:
    """Configuration for the denoising generator."""

    use_denoising: bool = False
    embed_dim: int = "${model.transformer.d_model}"
    max_electrons_per_image: int = 400
    max_total_denoising_queries: int = 1200
    position_noise_variance: float = 1.0
    pos_neg_queries_share_embedding: bool = False
    mask_main_queries_from_denoising: bool = False
    denoising_loss_weight: float = 1.0
