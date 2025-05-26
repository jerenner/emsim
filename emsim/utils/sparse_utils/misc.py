import math
import re
from typing import TYPE_CHECKING, overload

import torch
from torch import Tensor

from . import imports

if TYPE_CHECKING:
    _overload = overload
else:
    _overload = torch.jit._overload

_pytorch_atleast_2_5 = imports.check_pytorch_version("2.5")


@_overload
def prod(x: Tensor) -> Tensor: ...
@_overload
def prod(x: list[int]) -> int: ...  # noqa: F811
@_overload
def prod(x: list[float]) -> float: ...  # noqa: F811
@_overload
def prod(x: tuple[int, ...]) -> int: ...  # noqa: F811
@_overload
def prod(x: tuple[float, ...]) -> float: ...  # noqa: F811


def prod(x):  # noqa: F811
    """Computes the product of elements of a tensor or list.

    Args:
        x: (Tensor | list): Tensor or list to take the product over.

    Returns:
        Tensor | int | float: Resulting product.
    """
    if isinstance(x, Tensor):
        return x.prod()

    if torch.jit.is_scripting():  # type: ignore
        if isinstance(x[0], float):
            result = 1.0
        else:
            result = 1
        for element in x:
            result *= element
        return result
    return math.prod(x)


@torch.jit.script
def sparse_tensor_to_dense_with_mask(sparse_tensor: Tensor) -> tuple[Tensor, Tensor]:
    """
    Convert a sparse tensor to dense with a mask indicating stored indices.

    Args:
        sparse_tensor: A PyTorch sparse tensor (can be hybrid with dense dimensions)

    Returns:
        dense_tensor: The dense version of the input tensor
        mask: A boolean tensor with True at sparse indices where values were stored
    """
    sparse_tensor = sparse_tensor.coalesce()
    dense_tensor = sparse_tensor.to_dense()

    indices = sparse_tensor.indices()

    mask = torch.sparse_coo_tensor(
        indices,
        indices.new_ones(indices.size(1), dtype=torch.bool),
        sparse_tensor.shape[: sparse_tensor.sparse_dim()],
    ).to_dense()
    return dense_tensor, mask


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
