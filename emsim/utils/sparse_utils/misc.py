import re

import torch
from torch import Tensor

from . import utils

_pytorch_atleast_2_5 = utils.check_pytorch_version("2.5")


def unpack_sparse_tensors(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """
    Takes in a batch dict and converts packed sparse tensors (with separate
    indices and values tensors, and shape tuple) into sparse torch.Tensors.
    Not needed as of Pytorch 2.5 as pinned_memory now supports sparse tensors
    (https://github.com/pytorch/pytorch/pull/129645)

    Args:
        batch (dict[str, Tensor]): Input batch dict

    Returns:
        dict[str, Tensor]: Input batch dict with sparse tensors unpacked into
        sparse torch.Tensor format
    """
    if _pytorch_atleast_2_5:
        raise DeprecationWarning(
            "`unpack_sparse_tensors` is no longer needed as of Pytorch 2.5",
            "which added native support for pinned_memory=True for sparse tensors",
        )
    prefixes_indices = [
        match[0]
        for match in [re.match(".+(?=_indices$)", key) for key in batch.keys()]
        if match is not None
    ]
    prefixes_values = [
        match[0]
        for match in [re.match(".+(?=_values$)", key) for key in batch.keys()]
        if match is not None
    ]
    prefixes_shape = [
        match[0]
        for match in [re.match(".+(?=_shape$)", key) for key in batch.keys()]
        if match is not None
    ]
    prefixes = list(set(prefixes_indices) & set(prefixes_values) & set(prefixes_shape))
    for prefix in prefixes:
        assert not batch[prefix + "_values"].requires_grad
        shape = batch[prefix + "_shape"]
        if isinstance(shape, Tensor):
            shape = shape.tolist()
        batch[prefix] = torch.sparse_coo_tensor(
            batch[prefix + "_indices"],
            batch[prefix + "_values"],
            shape,
            dtype=batch[prefix + "_values"].dtype,
            device=batch[prefix + "_values"].device,
        ).coalesce()
        del batch[prefix + "_indices"]
        del batch[prefix + "_values"]
        del batch[prefix + "_shape"]
    return batch
