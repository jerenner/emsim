import numpy as np
import sparse
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation


class NSigmaSparsifyTransform:
    def __init__(self, background_threshold_n_sigma=4, window_size=7):
        self.background_threshold_n_sigma = background_threshold_n_sigma
        self.window_size = window_size

    def __call__(self, batch):
        image = batch["image"]
        if isinstance(image, np.ndarray):
            sparsified = numpy_sigma_energy_threshold_sparsify(
                image,
                background_threshold_n_sigma=self.background_threshold_n_sigma,
                window_size=self.window_size
            )
        elif isinstance(image, torch.Tensor):
            sparsified = torch_sigma_energy_threshold_sparsify(
                image,
                background_threshold_n_sigma=self.background_threshold_n_sigma,
                window_size=self.window_size,
            )
        batch["image_sparsified"] = sparsified
        return batch


def torch_sigma_energy_threshold_sparsify(
    image: torch.Tensor, background_threshold_n_sigma=4, window_size=7
):
    if window_size % 2 != 1:
        raise ValueError(f"Expected an odd `window_size`, got {window_size=}")
    if image.ndim > 3:
        reduce_dims = (-1, -2, -3)
    else:
        reduce_dims = (-1, -2)

    mean = torch.mean(image, reduce_dims, keepdim=True)
    std = torch.std(image, reduce_dims, keepdim=True)
    thresholded = image > mean + background_threshold_n_sigma * std

    kernel = torch.ones((1, 1, window_size, window_size), device=thresholded.device)

    conved = F.conv2d(thresholded.float(), kernel, padding="same")
    indices = conved.nonzero(as_tuple=True)
    values = image[indices]
    out = torch.sparse_coo_tensor(torch.stack(indices), values, size=image.shape)
    return out.coalesce()


def numpy_sigma_energy_threshold_sparsify(
    image: np.ndarray, background_threshold_n_sigma=4, window_size=7
):
    if window_size % 2 != 1:
        raise ValueError(f"Expected an odd `window_size`, got {window_size=}")
    if image.ndim == 4:
        reduce_dims = (-1, -2, -3)
        kernel = np.ones((1, 1, window_size, window_size), dtype=bool)
    elif image.ndim == 3:
        reduce_dims = (-1, -2)
        kernel = np.ones((1, window_size, window_size), dtype=bool)
    else:
        raise ValueError

    mean = np.mean(image, reduce_dims, keepdims=True)
    std = np.std(image, reduce_dims, keepdims=True)
    thresholded = image > mean + background_threshold_n_sigma * std

    conved = binary_dilation(thresholded, kernel)
    indices = conved.nonzero()
    values = image[indices]
    out = sparse.COO(indices, values, shape=image.shape)
    return out
