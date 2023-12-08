import torch
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import binary_dilation


def torch_sigma_energy_threshold_sparsify(
    image: torch.Tensor, background_threshold_n_sigma=4, window_size=7
):
    if image.ndim > 3:
        reduce_dims = [-1, -2, -3]
    else:
        reduce_dims = [-1, -2]

    mean = image.mean(reduce_dims, keepdims=True)
    std = image.std(reduce_dims, keepdims=True)
    thresholded = image > mean + background_threshold_n_sigma * std

    kernel = torch.ones((1, 1, window_size, window_size), device=thresholded.device)

    conved = F.conv2d(thresholded.float(), kernel, padding="same")
    indices = conved.nonzero(as_tuple=True)
    values = image[indices]
    out = torch.sparse_coo_tensor(torch.stack(indices), values, size=image.shape)
    return out.coalesce()
