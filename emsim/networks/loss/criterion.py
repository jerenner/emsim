import time
from typing import Any, Optional

import torch
from torch import Tensor, nn

from emsim.config.criterion import AuxLossConfig, CriterionConfig

from .loss_calculator import LossCalculator
from .matcher import HungarianMatcher
from .metrics import MetricManager
from .utils import (
    Mode,
    prep_detection_inputs,
    unstack_batch,
    unstack_model_output,
)


class EMCriterion(nn.Module):
    def __init__(self, config: CriterionConfig):
        super().__init__()
        self.config = config

        self.matcher = HungarianMatcher(config.matcher)
        self.loss_calculator = LossCalculator(config)
        self.metric_manager = MetricManager(config)

        # Instantiate aux loss and denoising handlers if applicable
        if config.aux_loss.use_aux_loss:
            self.aux_loss_handler = AuxLossHandler(config.aux_loss)
        else:
            self.aux_loss_handler = None
        if config.use_denoising_loss:
            self.denoising_handler = DenoisingHandler(config)
        else:
            self.denoising_handler = None

        self.step_counter = 0

    def forward(
        self,
        predicted_dict: dict[str, Any],
        target_dict: dict[str, Any],
    ):
        # Match queries to targets
        device = predicted_dict["pred_logits"].device
        matched_indices = self.matcher(predicted_dict, target_dict)
        matched_indices = [indices.to(device) for indices in matched_indices]

        # Compute main losses
        loss_dict, extras_dict = self.loss_calculator(
            predicted_dict, target_dict, matched_indices
        )
        assert isinstance(loss_dict, dict)
        assert isinstance(extras_dict, dict)

        # Compute aux losses if required
        if self.aux_loss_handler is not None and "aux_outputs" in predicted_dict:
            if self.aux_loss_handler.use_final_matches:
                aux_loss_matches = matched_indices
            else:
                aux_loss_matches = None
            aux_loss_dict, _ = self.aux_loss_handler(
                predicted_dict["aux_outputs"],
                target_dict,
                self.loss_calculator,
                self.matcher,
                aux_loss_matches,
            )
            loss_dict.update(aux_loss_dict)

        # Compute denoising losses if required
        if self.denoising_handler is not None and "denoising_output" in predicted_dict:
            dn_loss_dict = self.denoising_handler(
                predicted_dict["denoising_output"], target_dict, self.aux_loss_handler
            )
            loss_dict.update(dn_loss_dict)
            self.metric_manager.update_detection_metrics(
                Mode.DENOISING, dn_loss_dict, target_dict
            )

        # Compute total loss
        loss_dict["loss"] = self._compute_total_loss(loss_dict)

        # Update metrics
        self.metric_manager.update_from_dict(Mode.TRAIN, loss_dict)
        self.metric_manager.update_detection_metrics(
            Mode.TRAIN, predicted_dict, target_dict
        )
        self.metric_manager.update_classification_metrics(
            Mode.TRAIN, predicted_dict, extras_dict
        )
        self.metric_manager.update_localization_metrics(
            Mode.TRAIN, extras_dict["position_data"]
        )

        return loss_dict, {"matched_indices": matched_indices}

    def _compute_total_loss(self, loss_dict: dict[str, Tensor]) -> Tensor:
        loss_dict = self.loss_calculator.apply_loss_weights(loss_dict)
        if self.aux_loss_handler is not None:
            loss_dict = self.aux_loss_handler.apply_loss_weights(loss_dict)
        if self.denoising_handler is not None:
            loss_dict = self.denoising_handler.apply_loss_weights(loss_dict)

        stacked_losses = torch.stack([v for v in loss_dict.values()])
        return stacked_losses.sum()


class AuxLossHandler(nn.Module):
    def __init__(self, config: AuxLossConfig):
        super().__init__()
        self.n_aux_losses = config.n_aux_losses
        self.use_final_matches = config.use_final_matches
        self.register_buffer("aux_loss_weight", torch.tensor(config.aux_loss_weight))

    def forward(
        self,
        aux_outputs: list[dict[str, Any]],
        target_dict: dict[str, Tensor],
        loss_calculator: LossCalculator,
        matcher: Optional[HungarianMatcher] = None,
        final_matched_indices: Optional[list[Tensor]] = None,
    ):
        aux_loss_dict = {}
        matched_indices_dict = {}
        assert len(aux_outputs) == self.n_aux_losses
        for idx, output_i in enumerate(aux_outputs):
            if final_matched_indices is not None:
                matched_indices_i = final_matched_indices
            else:
                assert matcher is not None
                matched_indices_i = matcher(output_i, target_dict)
                matched_indices_i = [
                    matched.to(output_i["pred_logits"].device)
                    for matched in matched_indices_i
                ]

            # compute aux losses using same loss calculator
            aux_losses_i, _ = loss_calculator(output_i, target_dict, matched_indices_i)

            # label aux losses as such
            for k, v in aux_losses_i.items():
                aux_loss_dict[f"aux_losses/{idx}/{k}"] = v
            matched_indices_dict[f"{idx}"] = matched_indices_i

        return aux_loss_dict, matched_indices_dict

    def apply_loss_weights(self, loss_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Applies the aux loss weight to tensors with the substring 'aux_loss' in
        their name in the loss dict, leaving other tensors unmodified.
        """
        out_dict = {}
        for loss_name, loss in loss_dict.items():
            if "aux_loss" in loss_name:
                out_dict[loss_name] = loss * self.aux_loss_weight
            else:
                out_dict[loss_name] = loss
        return out_dict


class DenoisingHandler(nn.Module):
    def __init__(self, config: CriterionConfig):
        self.register_buffer(
            "denoising_loss_weight", torch.tensor(config.denoising_loss_weight)
        )

    def forward(
        self,
        denoising_output: dict[str, Any],
        target_dict: dict[str, Any],
        aux_loss_handler: Optional[AuxLossHandler] = None,
    ):
        # Compute denoising losses using same loss calculator
        denoising_losses = self.main_criterion.loss_calculator(
            denoising_output,
            target_dict,
            matched_indices=denoising_output["denoising_matched_indices"],
        )

        # Compute aux losses using same aux loss calculator
        if "aux_outputs" in denoising_output:
            assert aux_loss_handler is not None
            denoising_aux_losses, _ = aux_loss_handler(
                denoising_output["aux_outputs"],
                target_dict,
                final_matched_indices=denoising_output["denoising_matched_indices"],
            )
            denoising_losses.update(denoising_aux_losses)

        # label denoising losses as such
        denoising_losses = {f"dn/{k}": v for k, v in denoising_losses}
        return denoising_losses

    def apply_loss_weights(self, loss_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Applies the denoising loss weight to tensors whose key in the loss dict
        begin with the substring 'dn/', leaving other tensors unmodified
        """
        out_dict = {}
        for loss_name, loss in loss_dict.items():
            if loss_name.startswith("dn/"):
                out_dict[loss_name] = loss * self.denoising_loss_weight
            else:
                out_dict[loss_name] = loss
        return out_dict
