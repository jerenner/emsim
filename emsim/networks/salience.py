# Based on https://github.com/xiuqhou/Salience-DETR/blob/main/models/detectors/salience_detr.py

import torch
from torch import nn, Tensor

def pixel_coord_grid(height: int, width: int, stride: Tensor, device: torch.device):
    coord_y, coord_x = torch.meshgrid(
        torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device) * stride[0],
        torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device) * stride[1],
        indexing="ij"
    )
    return coord_y, coord_x
