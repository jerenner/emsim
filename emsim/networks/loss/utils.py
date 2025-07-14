from enum import Enum
from typing import Union, Optional

import torch
from torch import Tensor, nn
from torchmetrics import Metric, MetricCollection

from emsim.utils.sparse_utils.batching.batch_utils import split_batch_concatted_tensor
from emsim.utils.sparse_utils.indexing.sparse_index_select import sparse_index_select
from emsim.utils.sparse_utils.shape_ops import sparse_resize, sparse_flatten


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
def index_predicted_maps_denoising(
    pred_logits: Tensor,
    dn_info_dict: dict[str, Union[Tensor, list[Tensor], int, list[int]]],
    matched_indices_b: Tensor,
    batch_index: int,
) -> Tensor:
    """Index into the sparse segmentation map tensor, accounting for the extra
    denoising group dimension.
    """
    # get the number of objects in this image, with a lot of type annotations to
    # make torchscript happy
    n_obj_per_image = dn_info_dict["n_objects_per_image"]
    assert isinstance(n_obj_per_image, list)
    assert torch.jit.isinstance(n_obj_per_image, list[int])
    n_obj = n_obj_per_image[batch_index]
    assert isinstance(n_obj, int)

    n_dn_groups = pred_logits.shape[-2]
    padded_dn_group_size = pred_logits.shape[-1]

    # flatten out the [dn group, query] indices to 1D
    pred_with_dn_group_flattened: Tensor = sparse_flatten(pred_logits, -2, -1)
    assert pred_with_dn_group_flattened.ndim == 3

    # account for background query in flattened dim
    groups_arange = torch.repeat_interleave(
        torch.arange(n_dn_groups, device=pred_logits.device), n_obj
    )
    rel_inds = matched_indices_b[0] - groups_arange * (n_obj * 2)
    inds = rel_inds + groups_arange * padded_dn_group_size

    out = sparse_index_select(pred_with_dn_group_flattened, -1, inds)
    return out


@torch.jit.script
def sort_predicted_true_maps(
    predicted_segmentation_logits: Tensor,
    true_segmentation_map: Tensor,
    matched_indices: list[Tensor],
    dn_info_dict: Optional[
        dict[str, Union[Tensor, list[Tensor], int, list[int]]]
    ] = None,
) -> tuple[Tensor, Tensor]:
    assert predicted_segmentation_logits.is_sparse
    assert true_segmentation_map.is_sparse

    reordered_predicted = []
    reordered_true = []
    max_elecs = max([indices.shape[1] for indices in matched_indices])
    for b, (predicted_map, true_map, indices) in enumerate(
        zip(
            predicted_segmentation_logits.unbind(0),
            true_segmentation_map.unbind(0),
            matched_indices,
        )
    ):
        if dn_info_dict is None:
            reordered_predicted.append(
                sparse_index_select(predicted_map, 2, indices[0])
            )
        else:
            reordered_predicted.append(
                index_predicted_maps_denoising(predicted_map, dn_info_dict, indices, b)
            )
        reordered_true.append(sparse_index_select(true_map, 2, indices[1]))

    reordered_predicted = _restack_sparse_segmaps(
        reordered_predicted, max_elecs
    ).coalesce()
    reordered_true = _restack_sparse_segmaps(reordered_true, max_elecs).coalesce()

    return reordered_predicted, reordered_true


@torch.jit.script
def unstack_batch(batch: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    split = {
        k: split_batch_concatted_tensor(batch[k], batch["electron_batch_offsets"])
        for k in (
            "electron_ids",
            "normalized_incidence_points_xy",
            "incidence_points_pixels_rc",
            "normalized_centers_of_mass_xy",
            "centers_of_mass_rc",
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
    for i in range(len(batch["electron_batch_offsets"]) - 1):
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
