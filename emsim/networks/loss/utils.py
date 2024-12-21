import torch
from torch import Tensor


from emsim.utils.batching_utils import unstack_batch, unstack_model_output


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
def prep_ap_inputs(
    predicted_dict: dict[str, Tensor],
    target_dict: dict[str, Tensor],
    distance_threshold: float,
):
    predicted_split = unstack_model_output(predicted_dict)
    target_split = unstack_batch(target_dict)
    ap_inputs: list[tuple[Tensor, Tensor]] = []
    true_positives: list[Tensor] = []
    unmatched: list[Tensor] = []
    for predicted, target in zip(predicted_split, target_split):
        true_positives_i, tp_scores, unmatched_i, unmatched_scores, false_negatives = (
            find_matches(
                predicted["pred_positions"],
                predicted["pred_logits"].sigmoid(),
                target["normalized_incidence_points_xy"],
                distance_threshold,
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
        ap_inputs.append((scores, labels))
        true_positives.append(true_positives_i)
        unmatched.append(unmatched_i)

    return ap_inputs, true_positives, unmatched
