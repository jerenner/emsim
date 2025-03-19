import numpy as np
from typing import Any, List, Union, Optional

import torch
from torch import nn


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


def _get_layer(layer: Union[str, type]):
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
