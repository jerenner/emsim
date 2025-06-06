from enum import Enum
from typing import Union

import torch
from torch import Tensor, nn
from torchmetrics import Metric, MetricCollection

from emsim.utils.sparse_utils.batching.batch_utils import split_batch_concatted_tensor
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


@torch.jit.script
def unstack_batch(batch: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    split = {
        k: split_batch_concatted_tensor(batch[k], batch["electron_batch_offsets"])
        for k in (
            "electron_ids",
            "normalized_incidence_points_xy",
            "incidence_points_pixels_rc",
            "normalized_centers_of_mass_xy",
        )
    }
    split.update(
        {
            k: [im.squeeze() for im in batch[k].split(1)]
            for k in ("image", "noiseless_image", "image_size_pixels_rc")
        }
    )
    split["image_sparsified"] = batch["image_sparsified"].unbind()
    out: list[dict[str, Tensor]] = []
    for i in range(len(batch["electron_batch_offsets"])):
        out.append({k: v[i] for k, v in split.items()})
    return out


@torch.jit.script
def unstack_model_output(output: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    split = {
        k: split_batch_concatted_tensor(output[k], output["query_batch_offsets"])
        for k in (
            "pred_logits",
            "pred_positions",
            "pred_std_dev_cholesky",
        )
    }
    split["pred_segmentation_logits"] = output["pred_segmentation_logits"].unbind()
    out: list[dict[str, Tensor]] = []
    for i in range(len(output["query_batch_offsets"]) - 1):
        out.append({k: v[i] for k, v in split.items()})
    return out
