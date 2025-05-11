from dataclasses import dataclass, field


@dataclass
class SalienceConfig:
    """Configuration for focal loss parameters."""

    alpha: float = 0.25
    gamma: float = 2.0


@dataclass
class MatcherConfig:
    """Configuration for the bipartite matcher."""

    cost_coef_class: float = 1.0
    cost_coef_mask: float = 3.0
    cost_coef_dice: float = 5.0
    cost_coef_dist: float = 1.0
    cost_coef_nll: float = 0.0
    cost_coef_likelihood: float = 0.0
    cost_coef_box_l1: float = 1.0
    cost_coef_box_giou: float = 1.0


@dataclass
class AuxLossConfig:
    """Configuration for auxiliary losses."""

    use_aux_loss: bool = True
    use_final_matches: bool = False
    aux_loss_weight: float = 1.0
    n_aux_losses: int = "${n_aux_losses:}"


@dataclass
class CriterionConfig:
    """Configuration for the loss criterion."""

    loss_coef_class: float = 10.0
    loss_coef_mask_bce: float = 3.0
    loss_coef_mask_dice: float = 5.0
    loss_coef_incidence_nll: float = 0.1
    loss_coef_incidence_likelihood: float = 10.0
    loss_coef_incidence_huber: float = 100.0
    loss_coef_box_l1: float = 1.0
    loss_coef_box_giou: float = 1.0
    no_electron_weight: float = 1.0
    detach_likelihood_mean: bool = False

    detection_metric_distance_thresholds: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 5.0]
    )
    detection_metric_interval: int = "${training.print_interval}"

    # inherited denoising params
    use_denoising_loss: bool = "${model.denoising.use_denoising}"
    denoising_loss_weight: float = "${model.denoising.denoising_loss_weight}"

    # Nested configurations
    salience: SalienceConfig = field(default_factory=SalienceConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    aux_loss: AuxLossConfig = field(default_factory=AuxLossConfig)
