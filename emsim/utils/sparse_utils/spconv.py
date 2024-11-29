from functools import reduce
from typing import List

import torch
from torch import Tensor
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor


def torch_sparse_to_spconv(tensor: torch.Tensor):
    """Converts a sparse torch.Tensor to an equivalent spconv SparseConvTensor

    Args:
        tensor (torch.Tensor): Sparse tensor to be converted

    Returns:
        SparseConvTensor: Converted spconv tensor
    """
    if isinstance(tensor, spconv.SparseConvTensor):
        return tensor
    assert tensor.is_sparse
    spatial_shape = tensor.shape[1:-1]
    batch_size = tensor.shape[0]
    indices_th = tensor.indices()
    features_th = tensor.values()
    if features_th.ndim == 1:
        features_th = features_th.unsqueeze(-1)
        indices_th = indices_th[:-1]
    indices_th = indices_th.permute(1, 0).contiguous().int()
    return spconv.SparseConvTensor(features_th, indices_th, spatial_shape, batch_size)


def spconv_to_torch_sparse(tensor: spconv.SparseConvTensor, squeeze=False):
    """Converts an spconv SparseConvTensor to a sparse torch.Tensor

    Args:
        tensor (spconv.SparseConvTensor): spconv tensor to be converted
        squeeze (bool): If the spconv tensor has a feature dimension of 1,
            setting this to true squeezes it out so that the resulting
            sparse Tensor has a dense_dim() of 0. Raises an error if the spconv
            feature dim is not 1.

    Returns:
        torch.Tensor: Converted sparse torch.Tensor
    """
    if isinstance(tensor, Tensor) and tensor.is_sparse:
        return tensor
    assert isinstance(tensor, spconv.SparseConvTensor)
    if squeeze:
        if tensor.features.shape[-1] != 1:
            raise ValueError(
                "Got `squeeze`=True, but the spconv tensor has a feature dim of "
                f"{tensor.features.shape[-1]}, not 1"
            )
        size = [tensor.batch_size] + tensor.spatial_shape
        values = tensor.features.squeeze(-1)
    else:
        size = [tensor.batch_size] + tensor.spatial_shape + [tensor.features.shape[-1]]
        values = tensor.features
    indices = tensor.indices.transpose(0, 1)
    out = torch.sparse_coo_tensor(
        indices,
        values,
        size,
        device=tensor.features.device,
        dtype=tensor.features.dtype,
        requires_grad=tensor.features.requires_grad,
        check_invariants=True,
    )
    out = out.coalesce()
    return out


def spconv_sparse_mult(*tens: SparseConvTensor):
    """reuse torch.sparse. the internal is sort + unique"""
    max_num_indices = 0
    max_num_indices_idx = 0
    ten_ths: List[torch.Tensor] = []
    first = tens[0]

    for i, ten in enumerate(tens):
        assert ten.spatial_shape == tens[0].spatial_shape
        assert ten.batch_size == tens[0].batch_size
        assert ten.features.shape[1] in (tens[0].features.shape[1], 1)
        if max_num_indices < ten.features.shape[0]:
            max_num_indices_idx = i
            max_num_indices = ten.features.shape[0]
        res_shape = [ten.batch_size, *ten.spatial_shape, ten.features.shape[1]]
        ten_ths.append(
            torch.sparse_coo_tensor(
                ten.indices.T, ten.features, res_shape, requires_grad=True
            ).coalesce()
        )

    ## hacky workaround sparse_mask bug...
    if all([torch.equal(ten_ths[0].indices(), ten.indices()) for ten in ten_ths]):
        c_th = torch.sparse_coo_tensor(
            ten_ths[0].indices(),
            reduce(lambda x, y: x * y, [ten.values() for ten in ten_ths]),
            max([ten.shape for ten in ten_ths]),
            requires_grad=True,
        ).coalesce()
    else:
        c_th = reduce(lambda x, y: torch.mul(x, y), ten_ths).coalesce()

    c_th_inds = c_th.indices().T.contiguous().int()
    c_th_values = c_th.values()
    assert c_th_values.is_contiguous()

    res = SparseConvTensor(
        c_th_values,
        c_th_inds,
        first.spatial_shape,
        first.batch_size,
        benchmark=first.benchmark,
    )
    if c_th_values.shape[0] == max_num_indices:
        res.indice_dict = tens[max_num_indices_idx].indice_dict
    res.benchmark_record = first.benchmark_record
    res._timer = first._timer
    res.thrust_allocator = first.thrust_allocator
    return res
