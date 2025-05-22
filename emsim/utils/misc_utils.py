import numpy as np
from typing import Any, List, Union, Optional

import sparse
import torch
from torch import Tensor, nn


def random_chunks(
    x: List[Any],
    min_size: int,
    max_size: int,
    rng: Optional[np.random.Generator] = None,
):
    if min_size == max_size:
        chunk_sizes = np.full(shape=[len(x) // min_size], fill_value=min_size)
    else:
        if rng is None:
            rng = np.random.default_rng()
        chunk_sizes = rng.integers(min_size, max_size, size=len(x) // min_size)
    chunk_sizes = np.concatenate([[0], chunk_sizes], 0)
    start_indices = np.cumsum(chunk_sizes)
    chunked = [
        x[start:stop] for start, stop in zip(start_indices[:-1], start_indices[1:])
    ]
    chunked = [c for c in chunked if len(c) > 0]
    return chunked


def tensors_same_size(tensors: list[torch.Tensor]) -> bool:
    shapes = [x.shape for x in tensors]
    return len(set(shapes)) <= 1


def inverse_sigmoid(x, eps: float = 1e-6):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@torch.jit.ignore
def can_broadcast_shapes(*shapes) -> bool:
    """
    Returns True if the shapes can be broadcasted.

    Args:
        *shapes: Shapes to check.

    Returns:
        bool: True if shapes can be broadcasted, False if not
    """
    # Handle case where shapes is a single list of shapes
    if (
        len(shapes) == 1
        and hasattr(shapes[0], "__iter__")
        and not isinstance(shapes[0], (torch.Size, tuple))
    ):
        shapes = shapes[0]

    try:
        torch.broadcast_shapes(*shapes)
        return True
    except RuntimeError:
        return False


def get_layer(layer: Union[str, type]) -> type[nn.Module]:
    if isinstance(layer, type):
        return layer
    if layer.lower() == "relu":
        return nn.ReLU
    elif layer.lower() == "gelu":
        return nn.GELU
    elif layer.lower() == "batchnorm1d":
        return nn.BatchNorm1d
    else:
        raise ValueError(f"Unexpected layer {layer}")


@torch.jit.script
def multilevel_normalized_xy(sparse_tensor: Tensor, spatial_shapes: Tensor) -> Tensor:
    assert sparse_tensor.ndim == 5  # batch, i, j, level, feature
    assert spatial_shapes.ndim == 2  # i, j
    spatial_shapes_per_token = spatial_shapes[sparse_tensor.indices()[3]]
    normalized_shapes = (
        sparse_tensor.indices()[1:3].T / spatial_shapes_per_token
    ).flip(-1)
    return normalized_shapes


def _trim(subtensor: Tensor) -> Tensor:
    subtensor = subtensor.coalesce()
    indices, values = subtensor.indices(), subtensor.values()
    shape = subtensor.shape
    n_electrons = indices[0].max().item() + 1
    new_shape = (n_electrons, *shape[1:])
    return torch.sparse_coo_tensor(indices, values, new_shape).coalesce()


def bhwn_to_nhw_iterator_over_batches_torch(tensor: Tensor) -> list[Tensor]:
    assert tensor.is_sparse
    tensor = tensor.permute(0, 3, 1, 2).coalesce()

    return [_trim(t) for t in tensor.unbind()]


def bhwn_to_nhw_iterator_over_batches_pydata_sparse(array: sparse.SparseArray):
    assert isinstance(array, sparse.SparseArray)

    array = array.transpose([0, 3, 1, 2])

    def trim(subarray: sparse.SparseArray):
        max_electron_index = subarray.coords[0].max() + 1
        return subarray[:max_electron_index]

    return [trim(subarray) for subarray in array]
