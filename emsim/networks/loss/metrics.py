import time
from typing import Any, Union
import re

import torch
from torch import Tensor, nn
from torchmetrics import MeanMetric, Metric, MetricCollection
from torchmetrics.aggregation import MaxMetric, MinMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)

from emsim.config.criterion import CriterionConfig

from .detection_ap import match_detections
from .utils import (
    Mode,
    resolve_mode,
    unstack_batch,
    unstack_model_output,
)

__dn = re.compile(r"(?<=dn/).*")


def _search_dn_match(string: str) -> str:
    match = __dn.search(string)
    assert match is not None
    return match.group(0)


class MetricManager(nn.Module):
    def __init__(self, config: CriterionConfig):
        super().__init__()
        self.config = config
        self.train_metrics = self._build_train_metrics()
        self.eval_metrics = self._build_eval_metrics()
        if config.use_denoising_loss:
            self.denoising_metrics = self._build_denoising_metrics()
        else:
            self.register_module("denoising_metrics", None)

    def _build_train_metrics(self) -> nn.ModuleDict:
        metrics = {
            "loss": MeanMetric(),
            "query_classification": self._build_classification_metrics("query"),
            "mask_classification": self._build_classification_metrics("mask"),
            "localization": self._build_localization_metrics(),
            "detection": self._build_detection_metrics(),
        }

        metrics.update(self._build_loss_metrics())
        metrics["loss_salience"] = MeanMetric()

        metrics.update(self._build_encoder_out_metrics())

        if self.config.aux_loss.use_aux_loss:
            metrics.update(self._build_aux_metrics())

        if self.config.use_denoising_loss:
            metrics.update(self._build_denoising_metrics())

        return nn.ModuleDict(metrics)

    def _build_eval_metrics(self) -> nn.ModuleDict:
        metrics = {"detection": self._build_detection_metrics()}
        return nn.ModuleDict(metrics)

    def _build_loss_metrics(self) -> dict[str, MeanMetric]:
        losses = {}
        loss_names = [
            "classification",
            "mask_bce",
            "mask_dice",
            "distance_nll",
            "distance_likelihood",
            "distance_huber",
        ]
        if self.config.predict_box:
            loss_names.extend(["box_l1", "box_giou"])
        for name in loss_names:
            losses[f"loss_{name}"] = MeanMetric()
        return losses

    def _build_classification_metrics(self, prefix: str):
        return MetricCollection(
            [BinaryAccuracy(), BinaryPrecision(), BinaryRecall()], prefix=f"{prefix}/"
        )

    def _build_localization_metrics(self):
        return nn.ModuleDict(
            {
                "error": MetricCollection(
                    [MinMetric(), MeanMetric(), MaxMetric()],
                    prefix="localization_error/",
                ),
                "centroid_error": MetricCollection(
                    [MinMetric(), MeanMetric(), MaxMetric()], prefix="centroid_error/"
                ),
                "relative_error": MetricCollection(
                    [MinMetric(), MeanMetric(), MaxMetric()], prefix="relative_error/"
                ),
            }
        )

    def _build_detection_metrics(self):
        return DetectionMetrics(self.config.detection_metric_distance_thresholds)

    def _build_aux_metrics(self) -> dict[str, MeanMetric]:
        assert self.config.aux_loss.n_aux_losses > 0
        metrics = {}
        for i in range(self.config.aux_loss.n_aux_losses):
            metrics_i = self._build_loss_metrics()
            for k, v in metrics_i.items():
                metrics[f"aux_losses/{i}/{k}"] = v
        return metrics

    def _build_encoder_out_metrics(self) -> dict[str, MeanMetric]:
        metrics = self._build_loss_metrics()
        metrics = {f"encoder_out/{k}": v for k, v in metrics.items()}
        return metrics

    def _build_denoising_metrics(self) -> nn.ModuleDict:
        # loss trackers
        metrics: dict[str, nn.Module] = {}
        metrics.update(self._build_loss_metrics().items())
        if self.config.aux_loss.use_aux_loss:
            metrics.update(self._build_aux_metrics())

        # metrics
        metrics["query_classification"] = self._build_classification_metrics("query")
        metrics["mask_classification"] = self._build_classification_metrics("mask")
        metrics["localization"] = self._build_localization_metrics()
        metrics["detection"] = self._build_detection_metrics()

        return nn.ModuleDict(metrics)

    def get_metrics(self, mode: Union[str, Mode]) -> nn.ModuleDict:
        mode = resolve_mode(mode)
        assert isinstance(mode, Mode)
        if mode == mode.TRAIN:
            return self.train_metrics
        elif mode == mode.EVAL:
            return self.eval_metrics
        elif mode == mode.DENOISING:
            assert self.denoising_metrics is not None
            return self.denoising_metrics

    @torch.no_grad()
    def update_detection_metrics(
        self,
        mode: Union[str, Mode],
        predicted_dict: dict[str, Any],
        target_dict: dict[str, Any],
    ):
        metrics = self.get_metrics(mode)["detection"]
        assert isinstance(metrics, DetectionMetrics)
        metrics(predicted_dict, target_dict)

    @torch.no_grad()
    def update_from_loss_dict(
        self,
        loss_dict: dict[str, Tensor],
    ):
        for k, v in loss_dict.items():
            if k.startswith("dn/"):
                metrics = self.get_metrics(Mode.DENOISING)
                key = _search_dn_match(k)
            else:
                metrics = self.get_metrics(Mode.TRAIN)
                key = k
            metrics[key].update(v)

    @torch.no_grad()
    def update_classification_metrics(
        self,
        mode: Mode,
        predicted_dict: dict[str, Any],
        extras_dict: dict[str, Tensor],
    ):
        metrics = self.get_metrics(mode)
        # Query classification
        metrics["query_classification"].update(
            predicted_dict["pred_logits"], extras_dict["classification_labels"]
        )

        # mask classification
        metrics["mask_classification"].update(
            extras_dict["unioned_predicted_mask"].values(),
            extras_dict["unioned_true_mask"].values().bool().int(),
        )

    @torch.no_grad()
    def update_localization_metrics(
        self,
        mode: Union[str, Mode],
        position_data: dict[str, Tensor],
    ):
        metrics = self.get_metrics(mode)["localization"]
        assert isinstance(metrics, nn.ModuleDict)
        localization_error = torch.linalg.vector_norm(
            position_data["pred_positions"] - position_data["true_positions"], dim=-1
        )
        centroid_error = torch.linalg.vector_norm(
            position_data["centroid_positions"] - position_data["true_positions"],
            dim=-1,
        )
        metrics["error"].update(localization_error)
        metrics["centroid_error"].update(centroid_error)
        metrics["relative_error"].update(localization_error - centroid_error)

    def reset(self, mode: Union[str, Mode]):
        metrics = self.get_metrics(mode)
        for metric in metrics.values():
            recursive_reset(metric)

    @torch.no_grad()
    def get_logs(self, mode: Union[str, Mode], reset: bool = True) -> dict[str, Tensor]:
        mode = resolve_mode(mode)
        metrics = self.get_metrics(mode)
        log_dict = _flatten_metrics(metrics)
        if reset:
            self.reset(mode)
        log_dict = {k.replace(",", "."): v for k, v in log_dict.items()}
        if mode == Mode.DENOISING:
            log_dict = {"dn/" + k: v for k, v in log_dict.items()}
        elif mode == Mode.EVAL:
            log_dict = {"eval/" + k: v for k, v in log_dict.items()}
        return log_dict

    @staticmethod
    def make_log_str(log_dict: dict):
        return " ".join([f"{k}: {v}" for k, v in log_dict.items()])

    @staticmethod
    def format_log_keys(log_dict: dict):
        log_dict = {k.replace("loss_", "loss/"): v for k, v in log_dict.items()}
        return log_dict


class DetectionMetrics(nn.ModuleDict):
    def __init__(self, distance_thresholds: list[float]):
        super().__init__()
        self.thresholds = distance_thresholds
        detection_metrics = {}
        for threshold in distance_thresholds:
            # can't have periods in submodule names
            name = str(threshold).replace(".", ",")
            detection_metrics[name] = MetricCollection(
                [BinaryPrecision(), BinaryRecall(), BinaryAveragePrecision()],
                prefix=f"detection/{name}/",
            )
        detection_metrics["eval_time"] = MeanMetric()
        self.update(detection_metrics)

    def forward(
        self,
        predicted_dict: dict[str, Any],
        target_dict: dict[str, Any],
    ):
        start = time.time()
        pred_dict_list = unstack_model_output(
            {k: v for k, v in predicted_dict.items() if isinstance(v, Tensor)}
        )
        tgt_dict_list = unstack_batch(target_dict)
        assert len(pred_dict_list) == len(tgt_dict_list)

        pred_positions = [pred["pred_positions"] for pred in pred_dict_list]
        pred_logits = [pred["pred_logits"] for pred in pred_dict_list]
        target_positions = [
            tgt["incidence_points_pixels_rc"].to(pred)
            for tgt, pred in zip(tgt_dict_list, pred_positions)
        ]

        for threshold in self.thresholds:
            detection_inputs, _, _ = match_detections(
                pred_positions, pred_logits, target_positions, threshold, min_score=0.01
            )
            key = str(threshold).replace(".", ",")
            for scores, labels in detection_inputs:
                self[key].update(scores, labels)

        self["eval_time"].update(time.time() - start)


@torch.no_grad()
def _flatten_metrics(metric_dict) -> dict[str, Tensor]:
    out = {}
    for k, v in metric_dict.items():
        if isinstance(v, (nn.ModuleDict, MetricCollection)):
            out.update(_flatten_metrics(v))
        else:
            assert isinstance(v, Metric)
            out[k] = v.compute()
    return out


def recursive_reset(
    metric_or_moduledict: Union[nn.ModuleDict, Metric, MetricCollection, nn.Module],
):
    if not hasattr(metric_or_moduledict, "reset"):
        for metric in metric_or_moduledict.values():
            recursive_reset(metric)
    else:
        metric_or_moduledict.reset()
