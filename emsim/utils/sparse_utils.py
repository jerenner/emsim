from .batching_utils import batch_dim_to_leading_index
import numpy as np
from torch import Tensor
import spconv.pytorch as spconv


import torch


def torch_sparse_to_spconv(tensor: torch.Tensor):
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


def spconv_to_torch_sparse(tensor: spconv.SparseConvTensor):
    assert isinstance(tensor, spconv.SparseConvTensor)
    size = [tensor.batch_size] + tensor.spatial_shape + [tensor.features.shape[-1]]
    indices = tensor.indices.transpose(0, 1)
    values = tensor.features
    return torch.sparse_coo_tensor(
        indices, values, size, device=tensor.features.device, dtype=tensor.features.dtype,
        requires_grad=tensor.features.requires_grad
    )


def gather_from_sparse_tensor(sparse_tensor: Tensor, index_tensor: Tensor):
    """Batch selection of elements from a torch sparse tensor. Should be
    equivalent to sparse_tensor[index_tensor]. It works by flattening the sparse
    tensor's sparse dims and the index tensor to 1D (and converting n-d indices
    to raveled indices), then using index_select along the flattened sparse tensor.

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension ..., M; where ... are
        S leading sparse dimensions and M is the dense dimension
        index_tensor (Tensor): Long tensor of dimension ..., S; where ... are
        leading batch dimensions.

    Returns:
        Tensor: Tensor of dimension ..., M; where the leading dimensions are
        the same as the batch dimensions from `index_tensor`
    """
    sparse_shape = sparse_tensor.shape[:sparse_tensor.sparse_dim()]
    dim_linear_offsets = index_tensor.new_tensor(
        [np.prod(sparse_shape[i+1:] + (1,)) for i in range(len(sparse_shape))]
    )

    sparse_tensor_indices_linear = (sparse_tensor.indices() * dim_linear_offsets.unsqueeze(-1)).sum(0, keepdim=True)
    linear_shape = tuple(np.prod(sparse_shape, keepdims=True)) + sparse_tensor.shape[-sparse_tensor.dense_dim():]
    sparse_tensor_linearized = torch.sparse_coo_tensor(
        sparse_tensor_indices_linear,
        sparse_tensor.values(),
        linear_shape,
        dtype=sparse_tensor.dtype,
        device=sparse_tensor.device,
        requires_grad=sparse_tensor.requires_grad
    )

    if index_tensor.shape[-1] != sparse_tensor.sparse_dim():
        assert index_tensor.shape[-1] == sparse_tensor.sparse_dim() - 1
        index_tensor = batch_dim_to_leading_index(index_tensor)

    index_tensor_shape = index_tensor.shape
    index_tensor_linearized = (index_tensor * dim_linear_offsets).sum(-1)
    index_tensor_linearized = index_tensor_linearized.reshape(-1, )

    selected = sparse_tensor_linearized.index_select(0, index_tensor_linearized).to_dense()
    selected = selected.reshape(*index_tensor_shape[:-1], selected.shape[-1])
    return selected


def batch_offsets_from_sparse_tensor_indices(indices_tensor: Tensor):
    """Gets the batch offsets from an index tensor where the first element of the
    first dimension is the batch index, e.g. the indices() tensor of a sparse
    torch.Tensor.

    Args:
        indices_tensor (torch.Tensor): A tensor of shape (M x nnz), where M is
        the number of dimensions of the underlying sparse tensor and nnz is the
        number of nonzero elements in the sparse tensor. Assumes the sparse
        tensor has been coalesce()d.

    Returns:
        torch.Tensor: A 1D tensor with elements corresponding the the first
        incidence of each unique element in the first position of the M axis,
        i.e., the batch offsets if the first element is the batch index.
    """
    assert not torch.is_floating_point(indices_tensor)
    batch_indices = indices_tensor[0]
    max_batch_index = batch_indices.max()
    matching_indices = batch_indices.unsqueeze(-1) == torch.arange(
        max_batch_index + 1, device=batch_indices.device, dtype=batch_indices.dtype
    )
    out = matching_indices.to(torch.uint8).argmax(0)
    return out
