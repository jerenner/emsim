from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.aggregation import MaxMetric, MinMetric
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.wrappers import Running, MultitaskWrapper
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
)

from .salience_criterion import ElectronSalienceCriterion
from .matcher import HungarianMatcher
from ...utils.sparse_utils import (
    gather_from_sparse_tensor,
    union_sparse_indices,
    minkowski_to_torch_sparse,
    sparse_squeeze_dense_dim,
    sparse_index_select,
    sparse_resize,
)
from ...utils.batching_utils import split_batch_concatted_tensor


class EMCriterion(nn.Module):
    def __init__(
        self,
        loss_coef_class: float = 1.0,
        loss_coef_mask_bce: float = 1.0,
        loss_coef_mask_dice: float = 1.0,
        loss_coef_incidence_nll: float = 1.0,
        loss_coef_incidence_likelihood: float = 1.0,
        loss_coef_incidence_huber: float = 1.0,
        loss_coef_salience: float = 1.0,
        no_electron_weight: float = 0.1,
        salience_alpha: float = 0.25,
        salience_gamma: float = 2.0,
        matcher_cost_coef_class: float = 1.0,
        matcher_cost_coef_mask: float = 1.0,
        matcher_cost_coef_dice: float = 1.0,
        matcher_cost_coef_dist: float = 1.0,
        matcher_cost_coef_nll: float = 1.0,
        matcher_cost_coef_likelihood: float = 1.0,
        use_aux_loss=True,
        aux_loss_use_final_matches=False,
        aux_loss_weight: float = 1.0,
        n_aux_losses: int = 0,
        detach_likelihood_mean: bool = False,
        use_denoising_loss: bool = False,
        denoising_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_coef_class = loss_coef_class
        self.loss_coef_mask_bce = loss_coef_mask_bce
        self.loss_coef_mask_dice = loss_coef_mask_dice
        self.loss_coef_incidence_nll = loss_coef_incidence_nll
        self.loss_coef_incidence_likelihood = loss_coef_incidence_likelihood
        self.loss_coef_incidence_huber = loss_coef_incidence_huber
        self.loss_coef_salience = loss_coef_salience
        self.no_electron_weight = no_electron_weight
        self.aux_loss = use_aux_loss
        self.aux_loss_use_final_matches = aux_loss_use_final_matches
        self.aux_loss_weight = aux_loss_weight
        self.detach_likelihood_mean = detach_likelihood_mean
        self.use_denoising_loss = use_denoising_loss
        self.denoising_loss_weight = denoising_loss_weight

        self.salience_criterion = ElectronSalienceCriterion(
            salience_alpha, salience_gamma
        )

        self.matcher = HungarianMatcher(
            cost_coef_class=matcher_cost_coef_class,
            cost_coef_mask=matcher_cost_coef_mask,
            cost_coef_dice=matcher_cost_coef_dice,
            cost_coef_dist=matcher_cost_coef_dist,
            cost_coef_nll=matcher_cost_coef_nll,
            cost_coef_likelihood=matcher_cost_coef_likelihood,
        )

        self.train_losses = nn.ModuleDict(
            {
                "loss_class": MeanMetric(),
                "loss_bce": MeanMetric(),
                "loss_dice": MeanMetric(),
                "loss_incidence_nll": MeanMetric(),
                "loss_incidence_likelihood": MeanMetric(),
                "loss_incidence_huber": MeanMetric(),
            }
        )
        if use_aux_loss:
            assert n_aux_losses > 0
            for i in range(n_aux_losses):
                for loss_name in [
                    "loss_class",
                    "loss_bce",
                    "loss_dice",
                    "loss_incidence_nll",
                    "loss_incidence_likelihood",
                    "loss_incidence_huber",
                ]:
                    self.train_losses.update(
                        {f"aux_losses/{i}/{loss_name}": MeanMetric()}
                    )
        if use_denoising_loss:
            self.train_losses.update(
                {"dn/" + k: MeanMetric() for k in self.train_losses}
            )
        self.train_losses.update(
            {
                "loss_salience": MeanMetric(),
                "loss": MeanMetric(),
            }
        )

        self.train_metrics = nn.ModuleDict(
            {
                "query_classification": MetricCollection(
                    [BinaryAccuracy(), BinaryPrecision(), BinaryRecall()],
                    prefix="query/",
                ),
                "mask_classification": MetricCollection(
                    [BinaryAccuracy(), BinaryPrecision(), BinaryRecall()],
                    prefix="mask/",
                ),
                "localization_error": MetricCollection(
                    [MinMetric(), MeanMetric(), MaxMetric()],
                    prefix="localization_error/",
                ),
                # "mask_map": MeanAveragePrecision(iou_type="segm"),
                "centroid_error": MetricCollection(
                    [MinMetric(), MeanMetric(), MaxMetric()], prefix="centroid_error/"
                ),
                "localization_minus_centroid_error": MetricCollection(
                    [MinMetric(), MeanMetric(), MaxMetric()],
                    prefix="localization_minus_centroid_error/",
                ),
            },
        )
        if use_denoising_loss:
            self.dn_metrics = nn.ModuleDict(
                {
                    "query_classification": MetricCollection(
                        [BinaryAccuracy(), BinaryPrecision(), BinaryRecall()],
                        prefix="dn/query/",
                    ),
                    "mask_classification": MetricCollection(
                        [BinaryAccuracy(), BinaryPrecision(), BinaryRecall()],
                        prefix="dn/mask/",
                    ),
                    "localization_error": MetricCollection(
                        [MinMetric(), MeanMetric(), MaxMetric()],
                        prefix="dn/localization_error/",
                    ),
                    "localization_minus_centroid_error": MetricCollection(
                        [MinMetric(), MeanMetric(), MaxMetric()],
                        prefix="dn/localization_minus_centroid_error/",
                    ),
                },
            )
        else:
            self.dn_metrics = None

        self.eval_metrics = MultitaskWrapper(
            {
                "query_classification": MetricCollection(
                    [BinaryAccuracy(), BinaryPrecision(), BinaryRecall()],
                    prefix="query/",
                ),
                "classification_accuracy": BinaryAccuracy(),
                "localization_error": MeanMetric(),
                "dice": MeanMetric(),
                "centroid_error": MeanMetric(),
            },
            prefix="eval/",
        )

    def compute_losses(
        self,
        predicted_dict: dict[str, Tensor],
        target_dict: dict[str, Tensor],
        matched_indices: Optional[Tensor] = None,
        update_metrics: bool = False,
        is_denoising: bool = False,
    ):
        if matched_indices is None:
            matched_indices = [
                indices.to(device=predicted_dict["pred_logits"].device)
                for indices in self.matcher(predicted_dict, target_dict)
            ]
            assert not is_denoising

        n_gt_electrons_per_image = target_dict["batch_size"]
        if is_denoising:
            n_gt_electrons_per_image = (
                n_gt_electrons_per_image
                * predicted_dict["dn_batch_mask_dict"]["n_denoising_groups"]
            )
        if update_metrics:
            if is_denoising:
                metrics = self.dn_metrics
            else:
                metrics = self.train_metrics
        else:
            metrics = None

        class_loss = self.compute_class_loss(
            predicted_dict,
            matched_indices,
            self.no_electron_weight,
            metrics=metrics,
        )

        # reorder query and electron tensors in order of matched indices
        sorted_predicted_logits, sorted_true_segmentation = _sort_predicted_true_maps(
            predicted_dict["pred_segmentation_logits"],
            target_dict["segmentation_mask"],
            matched_indices,
        )

        bce_loss = self.compute_mask_bce_loss(
            sorted_predicted_logits,
            sorted_true_segmentation,
            metrics=metrics,
        )
        dice_loss = self.compute_mask_dice_loss(
            sorted_predicted_logits,
            sorted_true_segmentation,
        )

        pred_positions = _sort_tensor(
            predicted_dict["pred_positions"],
            predicted_dict["query_batch_offsets"],
            [inds[0] for inds in matched_indices],
        )
        pred_std_cholesky = _sort_tensor(
            predicted_dict["pred_std_dev_cholesky"],
            predicted_dict["query_batch_offsets"],
            [inds[0] for inds in matched_indices],
        )
        true_positions = _sort_tensor(
            target_dict["normalized_incidence_points_xy"].to(pred_positions),
            target_dict["electron_batch_offsets"],
            [inds[1] for inds in matched_indices],
        )
        centroid_positions = _sort_tensor(
            target_dict["normalized_centers_of_mass_xy"].to(true_positions),
            target_dict["electron_batch_offsets"],
            [inds[1] for inds in matched_indices],
        )

        image_size = (
            target_dict["image_size_pixels_rc"].flip(-1).to(pred_positions.device)
        )
        n_queries = torch.cat(
            [
                predicted_dict["query_batch_offsets"],
                predicted_dict["query_batch_offsets"].new_tensor(
                    [predicted_dict["pred_logits"].shape[0]]
                ),
            ]
        ).diff()
        image_size_per_query = torch.cat(
            [
                size.expand(n_queries, -1)
                for size, n_queries in zip(image_size, n_queries)
            ],
            0,
        )
        # for scaling the positions from normalized coords to pixels
        image_size_per_electron = torch.cat(
            [
                size.expand(n_elecs, -1)
                for size, n_elecs in zip(image_size, n_gt_electrons_per_image)
            ],
            0,
        )
        distance_nll_loss = self.get_distance_nll_loss(
            pred_positions, pred_std_cholesky, true_positions, image_size_per_electron
        )
        distance_likelihood_loss = self.get_distance_likelihood_loss(
            pred_positions, pred_std_cholesky, true_positions, image_size_per_electron
        )
        huber_loss = self.get_distance_huber_loss(pred_positions, true_positions)

        if update_metrics:
            if not is_denoising:
                metrics = self.train_metrics
            else:
                metrics = self.dn_metrics
            with torch.no_grad():
                localization_error = torch.linalg.vector_norm(
                    (pred_positions - true_positions) * image_size_per_electron,
                    dim=-1,
                )
                metrics["localization_error"].update(localization_error)
                centroid_error = torch.linalg.vector_norm(
                    (centroid_positions - true_positions) * image_size_per_electron,
                    dim=-1,
                )
                if "centroid_error" in metrics:
                    metrics["centroid_error"].update(centroid_error)
                metrics["localization_minus_centroid_error"].update(
                    localization_error - centroid_error
                )

        loss_dict = {
            "loss_class": class_loss,
            "loss_bce": bce_loss,
            "loss_dice": dice_loss,
            "loss_incidence_nll": distance_nll_loss,
            "loss_incidence_likelihood": distance_likelihood_loss,
            "loss_incidence_huber": huber_loss,
        }

        return loss_dict, {"matched_indices": matched_indices}

    def forward(
        self,
        predicted_dict: dict[str, Tensor],
        target_dict: dict[str, Tensor],
    ):
        loss_dict, matched_indices = self.compute_losses(
            predicted_dict, target_dict, update_metrics=True
        )

        if "aux_outputs" in predicted_dict:
            aux_loss_dict, aux_matched_indices = self.compute_aux_losses(
                predicted_dict["aux_outputs"],
                target_dict,
                matched_indices=(
                    matched_indices["matched_indices"]
                    if self.aux_loss_use_final_matches
                    else None
                ),
            )
            loss_dict.update(aux_loss_dict)
            matched_indices.update(aux_matched_indices)

        if self.use_denoising_loss and "denoising_output" in predicted_dict:
            denoising_output = predicted_dict["denoising_output"]
            denoising_losses, _ = self.compute_losses(
                denoising_output,
                target_dict,
                matched_indices=denoising_output["denoising_matched_indices"],
                update_metrics=True,
                is_denoising=True,
            )
            denoising_losses = {"dn/" + k: v for k, v in denoising_losses.items()}
            if "aux_outputs" in denoising_output:
                dn_aux_loss_dict, _ = self.compute_aux_losses(
                    denoising_output["aux_outputs"],
                    target_dict,
                    matched_indices=denoising_output["denoising_matched_indices"],
                    is_denoising=True,
                )
                denoising_losses.update(
                    {"dn/" + k: v for k, v in dn_aux_loss_dict.items()}
                )
            assert all([k not in loss_dict for k in denoising_losses])
            loss_dict.update(denoising_losses)

        loss_dict["loss_salience"] = self.compute_salience_loss(
            predicted_dict, target_dict
        )
        loss_dict["loss"] = self.compute_total_loss(loss_dict)
        self.log_losses(loss_dict)

        return loss_dict, matched_indices

    def compute_aux_losses(
        self,
        aux_outputs: list[dict],
        target_dict: dict[str, Tensor],
        matched_indices: Optional[Tensor] = None,
        is_denoising: bool = False,
    ):
        aux_loss_dict = {}
        matched_indices_dict = {}
        for i, aux_output_dict in enumerate(aux_outputs):
            aux_losses_i, aux_indices_i = self.compute_losses(
                aux_output_dict,
                target_dict,
                matched_indices,
                is_denoising=is_denoising,
            )
            for k, v in aux_losses_i.items():
                loss_str = f"aux_losses/{i}/{k}"
                aux_loss_dict[loss_str] = v
            matched_indices_dict.update(
                {k + f"/{i}": v for k, v in aux_indices_i.items()}
            )
        return aux_loss_dict, matched_indices_dict

    def compute_class_loss(
        self,
        predicted_dict: dict[str, Tensor],
        matched_indices: list[Tensor],
        no_electron_weight: float,
        metrics: Optional[nn.Module] = None,
    ) -> Tensor:
        pred_logits, labels, weights = get_query_class_preds_targets_weights(
            predicted_dict, matched_indices, no_electron_weight
        )
        loss = F.binary_cross_entropy_with_logits(pred_logits, labels.float(), weights)
        if metrics:
            metrics["query_classification"].update(pred_logits, labels)
        return loss

    def compute_mask_bce_loss(
        self,
        sorted_predicted_logits: Tensor,
        sorted_true: Tensor,
        metrics: Optional[nn.Module] = None,
    ) -> Tensor:
        assert sorted_predicted_logits.shape == sorted_true.shape

        unioned_predicted, unioned_true = union_sparse_indices(
            sorted_predicted_logits, sorted_true
        )
        assert torch.equal(unioned_predicted.indices(), unioned_true.indices())

        loss = F.binary_cross_entropy_with_logits(
            unioned_predicted.values(), unioned_true.values()
        )
        if metrics:
            self.__update_mask_class_metric(metrics, unioned_predicted, unioned_true)
        return loss

    @staticmethod
    @torch.compiler.disable
    def __update_mask_class_metric(
        metrics: nn.ModuleDict, unioned_predicted: Tensor, unioned_true: Tensor
    ):
        metrics["mask_classification"].update(
            unioned_predicted.values(), unioned_true.values().bool().int()
        )

    def compute_mask_dice_loss(
        self,
        sorted_predicted_logits: Tensor,
        sorted_true: Tensor,
    ) -> Tensor:
        assert sorted_predicted_logits.shape == sorted_true.shape

        predicted_segmentation: Tensor = torch.sparse.softmax(sorted_predicted_logits, -1)

        losses = []
        num = 2 * predicted_segmentation * sorted_true
        den = predicted_segmentation + sorted_true
        for num_i, den_i in zip(num, den):
            num_sum = num_i.coalesce().values().sum()
            den_sum = den_i.coalesce().values().sum()
            losses.append(1 - (num_sum + 1) / (den_sum + 1))
        loss = torch.stack(losses).mean()
        return loss

    def get_distance_nll_loss(
        self,
        pred_centers: Tensor,
        pred_std_cholesky: Tensor,
        true_incidence_points: Tensor,
        image_size_per_electron: Tensor,
    ) -> Tensor:
        if self.detach_likelihood_mean:
            pred_centers = pred_centers.detach()
        loss = -torch.distributions.MultivariateNormal(
            pred_centers * image_size_per_electron, scale_tril=pred_std_cholesky
        ).log_prob(true_incidence_points * image_size_per_electron)
        # print(torch.isinf(loss).any())
        # loss = loss.clamp(-1e7, 1e7)
        # loss = torch.clamp_max(loss, 1e7)
        return loss.mean()

    def get_distance_likelihood_loss(
        self,
        pred_centers: Tensor,
        pred_std_cholesky: Tensor,
        true_incidence_points: Tensor,
        image_size_per_query: Tensor,
    ) -> Tensor:
        if self.detach_likelihood_mean:
            pred_centers = pred_centers.detach()
        loss = torch.distributions.MultivariateNormal(
            pred_centers * image_size_per_query, scale_tril=pred_std_cholesky
        ).log_prob(true_incidence_points * image_size_per_query)
        loss = 1 - loss.exp()
        return loss.mean()

    def get_distance_huber_loss(
        self,
        pred_centers: Tensor,
        true_incidence_points: Tensor,
        huber_delta: float = 0.1,
    ) -> Tensor:
        loss = F.huber_loss(pred_centers, true_incidence_points, delta=huber_delta)
        return loss

    def compute_salience_loss(
        self,
        predicted_dict: dict[str, Tensor],
        target_dict: dict[str, Tensor],
    ):
        predicted_foreground_masks = [
            sparse_squeeze_dense_dim(
                minkowski_to_torch_sparse(
                    mask,
                    full_scale_spatial_shape=target_dict["image_size_pixels_rc"].max(0)[
                        0
                    ],
                )
            )
            for mask in predicted_dict["score_dict"]["score_feature_maps"]
        ]

        downsample_scales = [1, 2, 4, 8, 16, 32, 64]
        downsample_scales = downsample_scales[: len(predicted_foreground_masks)]
        downsample_scales.reverse()
        peak_normalized_images = [
            target_dict[f"peak_normalized_noiseless_image_1/{ds}"]
            for ds in downsample_scales
        ]

        return self.salience_criterion(
            predicted_foreground_masks, peak_normalized_images
        )

    def __reweight_loss(self, loss_name, loss_tensor):
        if "class" in loss_name:
            loss_tensor = loss_tensor * self.loss_coef_class
        elif "bce" in loss_name:
            loss_tensor = loss_tensor * self.loss_coef_mask_bce
        elif "dice" in loss_name:
            loss_tensor = loss_tensor * self.loss_coef_mask_dice
        elif "incidence_nll" in loss_name:
            loss_tensor = loss_tensor * self.loss_coef_incidence_nll
        elif "incidence_likelihood" in loss_name:
            loss_tensor = loss_tensor * self.loss_coef_incidence_likelihood
        elif "incidence_huber" in loss_name:
            loss_tensor = loss_tensor * self.loss_coef_incidence_huber
        elif "salience" in loss_name:
            loss_tensor = loss_tensor * self.loss_coef_salience
        else:
            raise ValueError(f"Unrecognized loss {loss_name}: {loss_tensor}")

        if "aux_loss" in loss_name and self.aux_loss_weight != 1.0:
            loss_tensor = loss_tensor * self.aux_loss_weight
        if "dn" in loss_name and self.denoising_loss_weight != 1.0:
            loss_tensor = loss_tensor * self.denoising_loss_weight
        return loss_tensor

    def compute_total_loss(self, loss_dict: dict[str, Tensor]) -> Tensor:
        weighted_loss_dict = {
            k: self.__reweight_loss(k, v) for k, v in loss_dict.items()
        }
        total_loss = sum([loss for loss in weighted_loss_dict.values()])
        return total_loss

    def log_losses(self, loss_dict: dict[str, Tensor]) -> None:
        for k, v in self.train_losses.items():
            v.update(loss_dict[k])

    def get_logs(self, reset_metrics: bool = True) -> None:
        def flatten_metrics(metric_dict):
            out = {}
            for k, v in metric_dict.items():
                if isinstance(v, MetricCollection):
                    out.update(flatten_metrics(v))
                else:
                    out[k] = v.compute()
            return out

        log_dict = {}
        log_dict.update(flatten_metrics(self.train_losses))
        log_dict.update(flatten_metrics(self.train_metrics))
        if reset_metrics:
            self.reset_metrics()

        return log_dict

    @staticmethod
    def make_log_str(log_dict: dict):
        return " ".join([f"{k}: {v}" for k, v in log_dict.items()])

    @staticmethod
    def tensorboardify_keys(log_dict: dict):
        log_dict = {k.replace("loss_", "loss/"): v for k, v in log_dict.items()}
        return log_dict

    def reset_metrics(self) -> None:
        for loss_tracker in self.train_losses.values():
            loss_tracker.reset()
        for metric_tracker in self.train_metrics.values():
            metric_tracker.reset()
        if self.dn_metrics is not None:
            for metric_tracker in self.dn_metrics.values():
                metric_tracker.reset()
        for metric_tracker in self.eval_metrics.values():
            metric_tracker.reset()


def get_query_class_preds_targets_weights(
    predicted_dict: dict[str, Tensor],
    matched_indices: list[Tensor],
    no_electron_weight: Optional[float] = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    pred_logits = predicted_dict["pred_logits"]
    query_batch_offsets = predicted_dict["query_batch_offsets"]
    labels = torch.zeros_like(pred_logits, dtype=torch.int)
    weights = torch.ones_like(pred_logits)

    true_entries = torch.cat(
        [
            indices[0] + offset
            for indices, offset in zip(
                matched_indices,
                query_batch_offsets.to(matched_indices[0].device),
            )
        ]
    )
    labels[true_entries] = 1
    if no_electron_weight != 1.0:
        weights[labels.logical_not()] = no_electron_weight
    return pred_logits, labels, weights


def _sort_predicted_true_maps(
    predicted_segmentation_logits: Tensor,
    true_segmentation_map: Tensor,
    matched_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    assert predicted_segmentation_logits.is_sparse
    assert true_segmentation_map.is_sparse

    reordered_predicted = []
    reordered_true = []
    max_elecs = max([indices.shape[1] for indices in matched_indices])
    for predicted_map, true_map, indices in zip(
        predicted_segmentation_logits.unbind(0),
        true_segmentation_map.unbind(0),
        matched_indices,
    ):

        reordered_predicted.append(sparse_index_select(predicted_map, 2, indices[0]))
        reordered_true.append(sparse_index_select(true_map, 2, indices[1]))

    def restack_sparse_segmaps(segmaps: list[Tensor]):
        segmaps = [
            sparse_resize(
                segmap,
                [*segmap.shape[:-1], max_elecs],
            )
            for segmap in segmaps
        ]
        return torch.stack(segmaps, 0).coalesce()

    reordered_predicted = restack_sparse_segmaps(reordered_predicted).coalesce()
    reordered_true = restack_sparse_segmaps(reordered_true).coalesce()

    return reordered_predicted, reordered_true


@torch.jit.script
def _sort_tensor(
    batch_concatted_tensor: Tensor, batch_offsets: Tensor, matched_indices: list[Tensor]
) -> Tensor:
    stacked_matched = torch.cat(
        [matched + offset for matched, offset in zip(matched_indices, batch_offsets)]
    )
    out = batch_concatted_tensor[stacked_matched]
    return out


# def get_occupancy_loss(
#     predicted_dict: dict[str, Tensor],
#     target_dict: dict[str, Tensor],
#     no_electron_weight: float,
# ):
#     logits = predicted_dict["occupancy_logits"]
#     targets = target_dict["electron_count_map_1/1"]
#     gathered_targets = gather_from_sparse_tensor(targets, logits.indices().T)[0]
#     weights = gathered_targets.new_ones((logits.shape[-1],), dtype=torch.float)
#     # weights[0] = 1 - (torch.count_nonzero(gathered_targets) / gathered_targets.numel())
#     weights[0] = no_electron_weight
#     loss = F.cross_entropy(logits.values(), gathered_targets, weights)
#     acc = (
#         torch.count_nonzero(logits.values().argmax(-1) == gathered_targets)
#         / gathered_targets.numel()
#     )
#     return loss, acc


def _mse(pred_centers: Tensor, true_points: Tensor) -> Tensor:
    return F.mse_loss(
        pred_centers,
        true_points,
    )
