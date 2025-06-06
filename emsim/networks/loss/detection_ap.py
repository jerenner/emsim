import torch
from torch import Tensor


def sort_detections_by_score(
    predicted_scores: Tensor, predicted_positions: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    if predicted_scores.ndim == 2:
        predicted_scores = predicted_scores.squeeze(1)
    assert len(predicted_scores) == len(predicted_positions)
    sorted_scores, sorted_indices = torch.sort(predicted_scores, descending=True)
    sorted_positions = predicted_positions[sorted_indices]
    return sorted_scores, sorted_positions, sorted_indices


def match_single_detection(
    predicted_pos: Tensor, gt_positions: Tensor, distance_threshold: float
) -> Tensor:
    if len(gt_positions) == 0:
        return torch.tensor(-1, device=gt_positions.device)
    gt_positions = gt_positions.to(predicted_pos)
    distances = torch.cdist(predicted_pos.view(1, 2), gt_positions)
    min_dist, match_index = distances.min(-1)
    if min_dist <= distance_threshold:
        return match_index.squeeze()
    else:
        return torch.tensor(-1, device=match_index.device)


@torch.jit.script
def find_matches(
    predicted_positions: Tensor,
    predicted_scores: Tensor,
    target_positions: Tensor,
    distance_threshold: float,
):
    device = predicted_positions.device
    sorted_scores, sorted_positions, sorted_indices = sort_detections_by_score(
        predicted_scores, predicted_positions
    )
    target_indices = torch.arange(len(target_positions), device=device)
    matches = []
    no_matches = []
    match_scores = []
    no_match_scores = []
    for index, score, pos in zip(sorted_indices, sorted_scores, sorted_positions):
        match_index = match_single_detection(pos, target_positions, distance_threshold)
        # if score <= score_threshold:
        #     break
        if match_index == -1:
            no_matches.append(index)
            no_match_scores.append(score)
        else:
            # store matched target
            matched_target_id = target_indices[match_index]
            matches.append(torch.stack([index, matched_target_id]))
            match_scores.append(score)

            # remove matched target from available targets
            target_indices = target_indices[target_indices != matched_target_id]
            target_positions = target_positions[
                torch.arange(len(target_positions), device=device) != match_index
            ]

    if len(matches) > 0:
        true_positives = torch.stack(matches)
        tp_scores = torch.stack(match_scores)
    else:
        true_positives = torch.zeros((0, 2), dtype=torch.int, device=device)
        tp_scores = torch.zeros((0,), dtype=torch.float, device=device)
    if len(no_matches) > 0:
        unmatched = torch.stack(no_matches)
        unmatched_scores = torch.stack(no_match_scores)
    else:
        unmatched = torch.zeros((0,), dtype=torch.int, device=device)
        unmatched_scores = torch.zeros((0,), dtype=torch.float, device=device)
    false_negatives = target_indices  # remaining target indices

    return true_positives, tp_scores, unmatched, unmatched_scores, false_negatives


@torch.jit.script
def match_detections(
    predicted_dict_list: list[dict[str, Tensor]],
    target_dict_list: list[dict[str, Tensor]],
    distance_threshold_pixels: float,
):
    detection_inputs: list[tuple[Tensor, Tensor]] = []
    true_positives: list[Tensor] = []
    unmatched: list[Tensor] = []
    for predicted, target in zip(predicted_dict_list, target_dict_list):
        true_positives_i, tp_scores, unmatched_i, unmatched_scores, false_negatives = (
            find_matches(
                predicted["pred_positions"],
                predicted["pred_logits"].sigmoid(),
                target["incidence_points_pixels_rc"],
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
