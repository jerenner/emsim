# Based on https://github.com/xiuqhou/Salience-DETR/blob/main/models/detectors/salience_detr.py

import torch
from torch import nn, Tensor

from torchvision.ops import sigmoid_focal_loss

from emsim.utils.batching_utils import split_batch_concatted_tensor
from emsim.utils.sparse_utils import gather_from_sparse_tensor, union_sparse_indices


def pixel_coord_grid(height: int, width: int, stride: Tensor, device: torch.device):
    coord_y, coord_x = torch.meshgrid(
        torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device)
        * stride[0],
        torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device)
        * stride[1],
        indexing="ij",
    )
    return coord_y, coord_x


class ElectronSalienceCriterion(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        predicted_foreground_masks: list[Tensor],
        peak_normalized_images: list[Tensor],
    ):
        assert len(predicted_foreground_masks) == len(peak_normalized_images)

        predicted_pixels = []
        true_pixels = []

        for predicted_mask, true_mask in zip(
            predicted_foreground_masks, peak_normalized_images
        ):
            assert predicted_mask.shape == true_mask.shape
            predicted_unioned, true_unioned = union_sparse_indices(
                predicted_mask, true_mask
            )
            assert torch.equal(predicted_unioned.indices(), true_unioned)
            predicted_pixels.append(predicted_unioned.values())
            true_pixels.append(true_pixels.values())

        predicted_pixels = torch.cat(predicted_pixels)
        true_pixels = torch.cat(true_pixels)

        num_pos = (true_pixels > 0.5).sum().clamp_min_(1)
        loss = sigmoid_focal_loss(
            predicted_pixels, true_pixels, alpha=self.alpha, gamma=self.gamma
        )
        loss = loss.sum() / num_pos
        return loss
