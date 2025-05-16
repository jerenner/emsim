import torch
from torch import Tensor

from emsim.utils.sparse_utils.batching.batch_utils import split_batch_concatted_tensor


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
    for i in range(len(output["query_batch_offsets"])):
        out.append({k: v[i] for k, v in split.items()})
    return out
