# pyright: reportAssignmentType=false
from dataclasses import dataclass, field


@dataclass
class DenoisingConfig:
    """Configuration for the denoising generator."""

    use_denoising: bool = False # toggles denoising training
    denoising_loss_weight: float = 1.0

    embed_dim: int = "${model.transformer.d_model}"

    max_electrons_per_image: int = 400 # max unique embeddings to keep

    # determines how much room for duplicate denoising groups
    max_total_denoising_queries: int = 1200

    # cap on number of denoising groups
    max_denoising_groups: int = 100

    # noise statistics
    position_noise_std: float = 5.0

    # how much further out the negative points should be
    negative_noise_mult_range: list[float] = field(default_factory=lambda: [1.5, 2.5])

    # technical details of handling embeddings in transformer
    pos_neg_queries_share_embedding: bool = False
    mask_main_queries_from_denoising: bool = False

    def __post_init__(self):
        if self.negative_noise_mult_range[0] < 1.0:
            raise ValueError(
                "Expected lower bound of negative_noise_mult_range to be >= 1, got "
                f"{self.negative_noise_mult_range}"
            )
