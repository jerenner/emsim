import numpy as np
from typing import Any, List

import torch


def random_chunks(x: List[Any], min_size: int, max_size: int):
    chunk_sizes = np.concatenate(
        [[0], np.random.randint(min_size, max_size, size=len(x) // min_size)], 0
    )
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
