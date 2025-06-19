from typing import Any, Optional

import torch
from torch import Tensor, nn

from emsim.config.criterion import AuxLossConfig, CriterionConfig
from emsim.utils.misc_utils import is_str_dict, is_tensor_list

from .loss_calculator import LossCalculator
from .matcher import HungarianMatcher
from .metrics import MetricManager
from .utils import Mode


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
        self.encoder_out_handler = EncoderOutHandler(config)

        self.step_counter = 0

    def forward(
        self,
        predicted_dict: dict[str, Any],
        target_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, list[Tensor]]]:
        # Match queries to targets
        device = predicted_dict["pred_logits"].device
        matched_indices = self.matcher(predicted_dict, target_dict)
        matched_indices = [indices.to(device) for indices in matched_indices]

        # Compute main losses
        loss_dict, extras_dict = self.loss_calculator(
            predicted_dict, target_dict, matched_indices
        )
        assert is_str_dict(loss_dict)
        assert is_str_dict(extras_dict)

        # Compute aux losses if required
        if self.aux_loss_handler is not None and "aux_outputs" in predicted_dict:
            aux_loss_dict, _ = self.aux_loss_handler(
                predicted_dict["aux_outputs"],
                target_dict,
                self.loss_calculator,
                self.matcher,
                matched_indices if self.aux_loss_handler.use_final_matches else None,
            )
            loss_dict.update(aux_loss_dict)

        # Compute encoder output losses
        encoder_out_loss_dict, _ = self.encoder_out_handler(
            predicted_dict["encoder_out"],
            target_dict,
            self.loss_calculator,
            self.matcher,
            matched_indices if self.encoder_out_handler.use_final_matches else None,
        )
        loss_dict.update(encoder_out_loss_dict)

        # Compute denoising losses if required
        if self.denoising_handler is not None and "denoising_output" in predicted_dict:
            dn_loss_dict, dn_extras_dict = self.denoising_handler(
                predicted_dict["denoising_output"],
                target_dict,
                self.loss_calculator,
                self.aux_loss_handler,
            )
            loss_dict.update(dn_loss_dict)
            # update denoising metrics
            self.metric_manager.update_classification_metrics(
                Mode.DENOISING, predicted_dict["denoising_output"], dn_extras_dict
            )
            self.metric_manager.update_localization_metrics(
                Mode.DENOISING, dn_extras_dict["position_data"]
            )

        # Compute total loss
        loss_dict["loss"] = self._compute_total_loss(loss_dict)

        # Update metrics
        self.metric_manager.update_from_loss_dict(loss_dict)
        self.metric_manager.update_classification_metrics(
            Mode.TRAIN, predicted_dict, extras_dict
        )
        self.metric_manager.update_localization_metrics(
            Mode.TRAIN, extras_dict["position_data"]
        )

        # detection metrics are slow, so only calculate intermittently
        # (default: right before logging)
        if (
            self.step_counter % self.config.detection_metric_interval == 0
            and self.step_counter > 0
        ):
            self._update_training_detection_metrics(predicted_dict, target_dict)

        self.step_counter += 1

        return loss_dict, {"matched_indices": matched_indices}

    def evaluate_batch(self, predicted_dict, target_dict):
        """Computes evaluation metrics on the batch and updates eval metric state"""
        self.metric_manager.update_detection_metrics(
            Mode.EVAL, predicted_dict, target_dict
        )

    def _update_training_detection_metrics(self, predicted_dict, target_dict):
        self.metric_manager.update_detection_metrics(
            Mode.TRAIN, predicted_dict, target_dict
        )
        if self.denoising_handler is not None and "denoising_output" in predicted_dict:
            self.metric_manager.update_detection_metrics(
                Mode.DENOISING, predicted_dict["denoising_output"], target_dict
            )

    def _compute_total_loss(self, loss_dict: dict[str, Tensor]) -> Tensor:
        loss_dict = self.loss_calculator.apply_loss_weights(loss_dict)
        if self.aux_loss_handler is not None:
            loss_dict = self.aux_loss_handler.apply_loss_weights(loss_dict)
        loss_dict = self.encoder_out_handler.apply_loss_weights(loss_dict)
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
    ) -> tuple[dict[str, Tensor], dict[str, list[Tensor]]]:
        aux_loss_dict: dict[str, Tensor] = {}
        matched_indices_dict: dict[str, list[Tensor]] = {}
        assert len(aux_outputs) == self.n_aux_losses
        for idx, output_i in enumerate(aux_outputs):
            if final_matched_indices is not None:
                matched_indices_i = final_matched_indices
            else:
                assert matcher is not None
                matched_indices_i = matcher(output_i, target_dict)
                assert is_tensor_list(matched_indices_i)
                matched_indices_i = [
                    matched.to(output_i["pred_logits"].device)
                    for matched in matched_indices_i
                ]

            # compute aux losses using same loss calculator
            aux_losses_i, _ = loss_calculator(output_i, target_dict, matched_indices_i)
            assert is_str_dict(aux_losses_i)

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


class EncoderOutHandler(nn.Module):
    def __init__(self, config: CriterionConfig):
        super().__init__()
        self.use_final_matches = config.aux_loss.use_final_matches
        self.register_buffer(
            "encoder_out_loss_weight", torch.tensor(config.encoder_out_loss_weight)
        )

    def forward(
        self,
        encoder_output: dict[str, Tensor],
        target_dict: dict[str, Tensor],
        loss_calculator: LossCalculator,
        matcher: Optional[HungarianMatcher] = None,
        final_matched_indices: Optional[list[Tensor]] = None,
    ) -> tuple[dict[str, Tensor], list[Tensor]]:
        if final_matched_indices is not None:
            matched_indices = final_matched_indices
        else:
            assert matcher is not None
            matched_indices = matcher(encoder_output, target_dict)
            assert is_tensor_list(matched_indices)
            matched_indices = [
                mi.to(encoder_output["pred_logits"].device) for mi in matched_indices
            ]

        # compute encoder output losses using same loss calculator
        encoder_out_loss_dict, _ = loss_calculator(
            encoder_output, target_dict, matched_indices
        )

        # label encoder output losses as such
        encoder_out_loss_dict = {
            f"encoder_out/{k}": v for k, v in encoder_out_loss_dict.items()
        }

        return encoder_out_loss_dict, matched_indices

    def apply_loss_weights(self, loss_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Applies the encoder output loss weight to tensors with the substring
        'encoder_out' in their name in the loss dict, leaving other tensors unmodified.
        """
        out_dict = {}
        for loss_name, loss in loss_dict.items():
            if "encoder_out" in loss_name:
                out_dict[loss_name] = loss * self.encoder_out_loss_weight
            else:
                out_dict[loss_name] = loss
        return out_dict


class DenoisingHandler(nn.Module):
    def __init__(self, config: CriterionConfig):
        super().__init__()
        self.register_buffer(
            "denoising_loss_weight", torch.tensor(config.denoising_loss_weight)
        )

    def forward(
        self,
        denoising_output: dict[str, Any],
        target_dict: dict[str, Any],
        loss_calculator: LossCalculator,
        aux_loss_handler: Optional[AuxLossHandler] = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        # Compute denoising losses using same loss calculator
        denoising_losses, denoising_extas_dict = loss_calculator(
            denoising_output,
            target_dict,
            matched_indices=denoising_output["denoising_matched_indices"],
        )
        assert is_str_dict(denoising_losses)

        # Compute aux losses using same aux loss calculator
        if "aux_outputs" in denoising_output:
            assert aux_loss_handler is not None
            denoising_aux_losses, _ = aux_loss_handler(
                denoising_output["aux_outputs"],
                target_dict,
                loss_calculator=loss_calculator,
                final_matched_indices=denoising_output["denoising_matched_indices"],
            )
            denoising_losses.update(denoising_aux_losses)

        # label denoising losses as such
        denoising_losses = {f"dn/{k}": v for k, v in denoising_losses.items()}
        return denoising_losses, denoising_extas_dict

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
