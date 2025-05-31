from enum import Enum
from typing import Union

import torch
from torch import Tensor, nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAveragePrecision

from emsim.utils.sparse_utils.indexing.sparse_index_select import sparse_index_select
from emsim.utils.sparse_utils.shape_ops import sparse_resize


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"
    DENOISING = "denoising"


def resolve_mode(mode: Union[Mode, str]) -> Mode:
    if isinstance(mode, Mode):
        return mode
    try:
        return Mode(mode)
    except ValueError:
        try:
            return Mode[mode.upper()]
        except KeyError:
            valid_modes = [m.value for m in Mode]
            raise ValueError(f"Invalid mode '{mode}'. Valid mode: {valid_modes}")


def sort_detections_by_score(
    predicted_scores: Tensor, predicted_xy: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    if predicted_scores.ndim == 2:
        predicted_scores = predicted_scores.squeeze(1)
    assert len(predicted_scores) == len(predicted_xy)
    sorted_scores, sorted_indices = torch.sort(predicted_scores, descending=True)
    sorted_xy = predicted_xy[sorted_indices]
    return sorted_scores, sorted_xy, sorted_indices


def match_single_detection(
    predicted_xy: Tensor, gt_xys: Tensor, distance_threshold: float
) -> Tensor:
    if len(gt_xys) == 0:
        return torch.tensor(-1, device=gt_xys.device)
    distances = torch.cdist(predicted_xy.view(1, 2), gt_xys)
    min_dist, match_index = distances.min(-1)
    if min_dist <= distance_threshold:
        return match_index.squeeze()
    else:
        return torch.tensor(-1, device=match_index.device)


@torch.jit.script
def find_matches(
    predicted_xy: Tensor,
    predicted_scores: Tensor,
    target_xy: Tensor,
    distance_threshold: float,
):
    device = predicted_xy.device
    sorted_scores, sorted_xy, sorted_indices = sort_detections_by_score(
        predicted_scores, predicted_xy
    )
    target_indices = torch.arange(len(target_xy), device=device)
    matches = []
    no_matches = []
    match_scores = []
    no_match_scores = []
    for index, score, xy in zip(sorted_indices, sorted_scores, sorted_xy):
        match_index = match_single_detection(xy, target_xy, distance_threshold)
        # if score <= score_threshold:
        #     break
        if match_index == -1:
            no_matches.append(index)
            no_match_scores.append(score)
        else:
            matches.append(torch.stack([index, target_indices[match_index]]))
            match_scores.append(score)
            target_indices = target_indices[
                target_indices != target_indices[match_index]
            ]
            target_xy = target_xy[
                torch.arange(len(target_xy), device=device) != match_index
            ]

    if len(matches) > 0:
        true_positives = torch.stack(matches)
        tp_scores = torch.stack(match_scores)
    else:
        true_positives = torch.zeros((0,), dtype=torch.int, device=device)
        tp_scores = torch.zeros((0,), dtype=torch.float, device=device)
    if len(no_matches) > 0:
        unmatched = torch.stack(no_matches)
        unmatched_scores = torch.stack(no_match_scores)
    else:
        unmatched = torch.zeros((0,), dtype=torch.int, device=device)
        unmatched_scores = torch.zeros((0,), dtype=torch.float, device=device)
    false_negatives = target_indices

    return true_positives, tp_scores, unmatched, unmatched_scores, false_negatives


@torch.jit.script
def prep_detection_inputs(
    predicted_dict_list: list[dict[str, Tensor]],
    target_dict_list: list[dict[str, Tensor]],
    distance_threshold_pixels: float,
):
    detection_inputs: list[tuple[Tensor, Tensor]] = []
    true_positives: list[Tensor] = []
    unmatched: list[Tensor] = []
    for predicted, target in zip(predicted_dict_list, target_dict_list):
        image_size_pixels_xy = target["image_size_pixels_rc"].flip(-1)
        true_positives_i, tp_scores, unmatched_i, unmatched_scores, false_negatives = (
            find_matches(
                predicted["pred_positions"] * image_size_pixels_xy,
                predicted["pred_logits"].sigmoid(),
                target["normalized_incidence_points_xy"] * image_size_pixels_xy,
                distance_threshold_pixels,
            )
        )
        scores = torch.cat(
            [
                tp_scores,
                unmatched_scores,
                torch.zeros_like(false_negatives, dtype=tp_scores.dtype),
            ]
        )
        labels = torch.cat(
            [
                torch.ones_like(tp_scores, dtype=torch.int),
                torch.zeros_like(unmatched_scores, dtype=torch.int),
                torch.ones_like(false_negatives, dtype=torch.int),
            ]
        )
        detection_inputs.append((scores, labels))
        true_positives.append(true_positives_i)
        unmatched.append(unmatched_i)

    return detection_inputs, true_positives, unmatched


class PixelAP(BinaryAveragePrecision):
    def __init__(self, pixel_threshold: float):
        super().__init__()
        self.pixel_threshold = pixel_threshold

    def update(
        self,
        predicted_dict_list: list[dict[str, Tensor]],
        target_dict_list: list[dict[str, Tensor]],
    ):
        ap_inputs, _, _ = prep_detection_inputs(
            predicted_dict_list, target_dict_list, self.pixel_threshold
        )
        for scores, labels in ap_inputs:
            super().update(scores, labels)


@torch.jit.script
def sort_tensor(
    batch_concatted_tensor: Tensor, batch_offsets: Tensor, matched_indices: list[Tensor]
) -> Tensor:
    stacked_matched = torch.cat(
        [matched + offset for matched, offset in zip(matched_indices, batch_offsets)]
    )
    out = batch_concatted_tensor[stacked_matched]
    return out


@torch.jit.script
def _restack_sparse_segmaps(segmaps: list[Tensor], max_elecs: int):
    outs = []
    for segmap in segmaps:
        shape = list(segmap.shape)
        shape[-1] = max_elecs
        outs.append(sparse_resize(segmap, shape))
    return torch.stack(outs, 0).coalesce()


@torch.jit.script
def sort_predicted_true_maps(
    predicted_segmentation_logits: Tensor,
    true_segmentation_map: Tensor,
    matched_indices: list[Tensor],
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

    reordered_predicted = _restack_sparse_segmaps(
        reordered_predicted, max_elecs
    ).coalesce()
    reordered_true = _restack_sparse_segmaps(reordered_true, max_elecs).coalesce()

    return reordered_predicted, reordered_true


def _flatten_metrics(metric_dict):
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
