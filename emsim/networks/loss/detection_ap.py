"""Detection matching utilities for position-based object detection.

This module provides functions to match predicted detections with ground truth targets
based on spatial distance rather than bounding box IoU, suitable for point-based
object detection tasks.
"""

from typing import Optional

import torch
from torch import Tensor


def sort_detections_by_score(
    predicted_scores: Tensor, predicted_positions: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Sort detections by confidence score in descending order.

    Args:
        predicted_scores (Tensor): Detection confidence scores, shape (N,) or (N, 1)
        predicted_positions (Tensor): Detection positions, shape (N, D), where D is
            the spatial dimension

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - sorted_scores
            - sorted_positions
            - sorted_indices

    Raises:
        ValueError: If scores and positions have different lengths
    """
    if predicted_scores.ndim == 2:
        predicted_scores = predicted_scores.squeeze(1)
    if len(predicted_scores) != len(predicted_positions):
        raise ValueError(
            "Expected `predicted_scores` and `predicted_positions` to have same len,"
            f"but got shapes {predicted_scores.shape} and {predicted_positions.shape}"
        )
    sorted_scores, sorted_indices = torch.sort(predicted_scores, descending=True)
    sorted_positions = predicted_positions[sorted_indices]
    return sorted_scores, sorted_positions, sorted_indices


@torch.jit.script
def match_single_detection(
    predicted_pos: Tensor, gt_positions: Tensor, distance_threshold: float
) -> Tensor:
    """Match a single predicted position to the nearest ground truth position.

    Args:
        predicted_pos (Tensor): Single predicted position, shape (D,), where D is
            the spatial dimension
        gt_positions (Tensor): Ground truth positions, shape (M, D)
        distance_threshold (float): Maximum distance for a valid match

    Returns:
        Tensor: Scalar tensor with index of matched ground truth position, or -1 if no
            match found
    """
    if len(gt_positions) == 0:  # No available targets
        return torch.tensor(-1, device=gt_positions.device)
    gt_positions = gt_positions.to(predicted_pos)
    distances = torch.cdist(predicted_pos.view(1, -1), gt_positions)
    min_dist, match_index = distances.min(-1)
    if min_dist <= distance_threshold:
        return match_index.squeeze()
    else:
        return torch.tensor(-1, device=match_index.device)


def _match_loop(
    sorted_indices: Tensor,
    sorted_positions: Tensor,
    sorted_scores: Tensor,
    target_positions: Tensor,
    distance_threshold: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = sorted_positions.device
    n_targets = target_positions.size(0)
    # Compute pairwise distances for detections above score threshold
    pairwise_distances = torch.cdist(sorted_positions, target_positions)  # (N, M)
    distances_within_threshold = pairwise_distances <= distance_threshold

    # Trackers for matched targets
    target_indices = torch.arange(n_targets, device=device)
    target_matched = torch.zeros(n_targets, device=device, dtype=torch.bool)

    matches = []  # Pairs (pred, tgt) for matches
    match_scores = []  # Prediction scores for matches
    no_matches = []  # Predictions with no matches
    no_match_scores = []  # Scores of unmatched predictions
    for i, (pred_idx, score) in enumerate(zip(sorted_indices, sorted_scores)):
        targets_in_range = distances_within_threshold[i] & ~target_matched
        if not targets_in_range.any():  # no match among targets
            no_matches.append(pred_idx)
            no_match_scores.append(score)
        else:
            # Find closest target
            match_index = pairwise_distances[i][targets_in_range].argmin()
            # Convert local argmin to global target index
            matched_target_id = target_indices[targets_in_range][match_index]

            # record matched pair
            matches.append(torch.stack([pred_idx, matched_target_id]))
            match_scores.append(score)

            # Mark target as matched
            target_matched[matched_target_id] = True

    # Stack outputs
    if len(matches) > 0:
        true_positives = torch.stack(matches)
        tp_scores = torch.stack(match_scores)
    else:
        true_positives = torch.zeros((0, 2), dtype=torch.long, device=device)
        tp_scores = torch.zeros((0,), dtype=sorted_scores.dtype, device=device)
    if len(no_matches) > 0:
        unmatched = torch.stack(no_matches)
        unmatched_scores = torch.stack(no_match_scores)
    else:
        unmatched = torch.zeros((0,), dtype=torch.long, device=device)
        unmatched_scores = torch.zeros((0,), dtype=sorted_scores.dtype, device=device)

    false_negatives = target_indices[~target_matched]

    return true_positives, tp_scores, unmatched, unmatched_scores, false_negatives


@torch.jit.script
def find_matches_for_image(
    predicted_positions: Tensor,
    predicted_scores: Tensor,
    target_positions: Tensor,
    distance_threshold: float,
    min_score: Optional[float] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Find matches between predicted and target positions using greedy assignment.

    Greedily matches predictions to targets in order of decreasing confidence score.

    Args:
        predicted_positions (Tensor): Predicted positions, shape (N, D) where D is
            the spatial dimension
        predicted_scores (Tensor): Prediction confidence scores, shape (N,)
        target_positions (Tensor): Ground truth positions, shape (M, D)
        distance_threshold (float): Maximum distance for a valid match
        min_score (Optional[float]): If set, scores under this threshold are not
            attempted to be matched to any target. Useful to avoid wasting time
            iterating through many non-detections.


    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            - true_positives: Matched (pred_idx, target_idx) pairs, shape (K, 2)
            - tp_scores: Scores for true positives, shape (K,)
            - unmatched: Indices of unmatched predictions, shape (U,)
            - unmatched_scores: Scores for unmatched predictions, shape (U,)
            - false_negatives: Indices of unmatched targets, shape (V,)

    Raises:
        ValueError: If predicted and target positions have incompatible dimensions,
            or min_score is not in [0, 1)
    """
    # Validate dims
    if predicted_positions.ndim != 2 or target_positions.ndim != 2:
        raise ValueError(
            "Expected predicted and target position tensors to be 2D, got shapes "
            f"{predicted_positions.shape} and {target_positions.shape}."
        )
    if (
        len(target_positions) > 0
        and predicted_positions.shape[1] != target_positions.shape[1]
    ):
        raise ValueError(
            "Expected predicted and target positions to have same spatial dim, but got"
            f"{predicted_positions.shape[-1]} and {target_positions.shape[-1]}."
        )
    if min_score is not None and not 0.0 <= min_score < 1.0:
        raise ValueError(f"Expected `min_score` to be in [0, 1), got {min_score}")

    device = predicted_positions.device

    sorted_scores, sorted_positions, sorted_indices = sort_detections_by_score(
        predicted_scores, predicted_positions
    )

    n_targets = len(target_positions)

    # Early exit for no targets
    if n_targets == 0:
        true_positives = torch.zeros((0, 2), dtype=torch.long, device=device)
        tp_scores = torch.zeros((0,), dtype=predicted_scores.dtype, device=device)
        unmatched = sorted_indices
        unmatched_scores = sorted_scores
        false_negatives = torch.zeros((0,), dtype=torch.long, device=device)
        return true_positives, tp_scores, unmatched, unmatched_scores, false_negatives

    # Filter out "detections" that fall below min score threshold
    if min_score is None:
        keep_mask = torch.ones_like(sorted_scores, dtype=torch.bool)
    else:
        keep_mask = sorted_scores >= min_score

    kept_indices = sorted_indices[keep_mask]
    kept_positions = sorted_positions[keep_mask]
    kept_scores = sorted_scores[keep_mask]

    skipped_indices = sorted_indices[~keep_mask]
    skipped_scores = sorted_scores[~keep_mask]

    if kept_indices.numel() == 0:  # All "detections" were under the score threshold
        true_positives = torch.zeros((0, 2), dtype=torch.long, device=device)
        tp_scores = torch.zeros((0,), dtype=predicted_scores.dtype, device=device)
        unmatched = skipped_indices
        unmatched_scores = skipped_scores
        false_negatives = torch.arange(n_targets, device=device)
        return true_positives, tp_scores, unmatched, unmatched_scores, false_negatives

    # actually do the matching
    true_positives, tp_scores, unmatched, unmatched_scores, false_negatives = (
        _match_loop(
            kept_indices,
            kept_positions,
            kept_scores,
            target_positions,
            distance_threshold,
        )
    )

    # handle skipped predictions
    if skipped_indices.size(0) > 0:
        unmatched = torch.cat([unmatched, skipped_indices])
        unmatched_scores = torch.cat([unmatched_scores, skipped_scores])

    return true_positives, tp_scores, unmatched, unmatched_scores, false_negatives


@torch.no_grad()
@torch.jit.script
def match_detections(
    predicted_positions: list[Tensor],
    predicted_logits: list[Tensor],
    target_positions: list[Tensor],
    distance_threshold_pixels: float,
    min_score: Optional[float] = None,
) -> tuple[list[tuple[Tensor, Tensor]], list[Tensor], list[Tensor]]:
    """Match predictions to targets across a batch of images for detection metrics.

    Processes each image in the batch to create single-class score-label pairs suitable
    for computing detection metrics like precision-recall curves.
    Scores are in [0, 1], and labels are {0, 1}.

    Args:
        predicted_positions (list[Tensor]): List of predicted positions, each of shape
            (N_i, D), where i is the image index and D is the spatial dimension
        predicted_logits (list[Tensor]): List of prediction logits, each of shape (N_i,)
        target_positions (list[Tensor]): List of ground-truth target position tensors,
            each of shape (M_i, D)
        distance_threshold (float): Maximum distance for a valid match
        min_score (Optional[float]): If set, scores under this threshold are not
            attempted to be matched to any target. Useful to avoid wasting time
            iterating through many non-detections.

    Returns:
        Tuple[list[tuple[Tensor, Tensor]], list[Tensor], list[Tensor]]:
            - detection_inputs: List of (scores, labels) tuples for metrics calculation
            - true_positives: List of matched prediction-target pairs per image
            - unmatched: List of unmatched prediction indices per image
    """
    detection_inputs: list[tuple[Tensor, Tensor]] = []
    true_positives: list[Tensor] = []
    unmatched: list[Tensor] = []
    if not len(predicted_positions) == len(predicted_logits) == len(target_positions):
        raise ValueError(
            "Expected `pred_positions`, `pred_logits`, and `target_positions` to all "
            "be the same length, but got lengths"
            f"{len(predicted_positions)}, {len(predicted_logits)}, and {len(target_positions)}."
        )
    for pred_pos, pred_logits, tgt_pos in zip(
        predicted_positions, predicted_logits, target_positions
    ):
        true_positives_i, tp_scores, unmatched_i, unmatched_scores, false_negatives = (
            find_matches_for_image(
                pred_pos,
                pred_logits.sigmoid(),
                tgt_pos,
                distance_threshold_pixels,
                min_score=min_score,
            )
        )

        num_tp = len(tp_scores)
        num_unmatched = len(unmatched_scores)  # false positives
        num_fn = len(false_negatives)
        total_len = num_tp + num_unmatched + num_fn

        # Create scores and labels tensors
        scores = torch.zeros(total_len, dtype=tp_scores.dtype, device=tp_scores.device)
        scores[:num_tp] = tp_scores
        scores[num_tp : num_tp + num_unmatched] = unmatched_scores
        # scores[num_tp + num_unmatched:] (false negatives) remain 0

        labels = torch.ones(total_len, dtype=torch.long, device=tp_scores.device)
        labels[num_tp : num_tp + num_unmatched] = 0  # false positives
        # labels for true positives and false negatives remain 1

        detection_inputs.append((scores, labels))
        true_positives.append(true_positives_i)
        unmatched.append(unmatched_i)

    return detection_inputs, true_positives, unmatched
