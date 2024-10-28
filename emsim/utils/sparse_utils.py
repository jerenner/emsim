import re
from functools import reduce
from typing import List

import numpy as np
import sparse
import spconv.pytorch as spconv
import torch
from spconv.pytorch import SparseConvTensor
from torch import Tensor

from .batching_utils import batch_dim_to_leading_index


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


def torch_sparse_to_pydata_sparse(tensor: Tensor):
    assert tensor.is_sparse
    tensor = tensor.detach().coalesce().cpu()
    assert tensor.is_coalesced
    nonzero_values = tensor.values().nonzero(as_tuple=True)
    return sparse.COO(
        tensor.indices()[:, nonzero_values[0]],
        tensor.values()[nonzero_values],
        tensor.shape,
        has_duplicates=False,
    )


def sparse_select(tensor: Tensor, axis: int, index: int):
    assert tensor.is_sparse
    tensor = tensor.coalesce()
    index_mask = tensor.indices()[axis] == index
    values = tensor.values()[index_mask]
    indices = torch.cat(
        [tensor.indices()[:axis, index_mask], tensor.indices()[axis + 1 :, index_mask]]
    )
    return torch.sparse_coo_tensor(
        indices, values, tensor.shape[:axis] + tensor.shape[axis + 1 :]
    ).coalesce()


def sparse_index_select(tensor: Tensor, axis: int, index: Tensor):
    assert tensor.is_sparse
    tensor = tensor.coalesce()
    assert tensor.ndim <= 1
    index_masks = [tensor.indices()[axis] == i for i in index]
    new_values = [tensor.values()[mask] for mask in index_masks]
    new_indices = [
        torch.cat(
            [
                tensor.indices()[:axis, mask],
                tensor.indices().new_full((1, mask.count_nonzero()), i),
                tensor.indices()[axis + 1 :, mask],
            ]
        )
        for i, mask in enumerate(index_masks)
    ]
    return torch.sparse_coo_tensor(
        torch.cat(new_indices, -1),
        torch.cat(new_values, 0),
        (*tensor.shape[:axis], len(new_indices), *tensor.shape[axis + 1 :]),
    ).coalesce()


def sparse_flatten_hw(tensor: Tensor):
    assert tensor.is_sparse
    tensor = tensor.coalesce()
    indices = tensor.indices()
    i = indices[1]
    j = indices[2]
    H = tensor.shape[1]
    W = tensor.shape[2]
    ij = (i * W + j).unsqueeze(0)
    new_shape = tensor.shape[:1] + (H * W,) + tensor.shape[3:]
    new_indices = torch.cat([indices[:1], ij, indices[3:]], 0).long()
    return torch.sparse_coo_tensor(new_indices, tensor.values(), new_shape).coalesce()


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
            max([ten.shape for ten in ten_ths]), requires_grad=True
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


def unpack_sparse_tensors(batch: dict[str, Tensor]):
    """
    Takes in a batch dict and converts packed sparse tensors (with separate
    indices and values tensors, and shape tuple) into sparse torch.Tensors

    Args:
        batch (dict[str, Tensor]): Input batch dict

    Returns:
        dict[str, Tensor]: Input batch dict with sparse tensors unpacked into
        sparse torch.Tensor format
    """
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


def gather_from_sparse_tensor(
    sparse_tensor: Tensor, index_tensor: Tensor, check_all_specified=False
):
    """Batch selection of elements from a torch sparse tensor. Should be
    equivalent to sparse_tensor[index_tensor]. It works by flattening the sparse
    tensor's sparse dims and the index tensor to 1D (and converting n-d indices
    to raveled indices), then using searchsorted along the flattened sparse
    tensor indices.

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension ..., M; where ... are
        S leading sparse dimensions and M is the dense dimension
        index_tensor (Tensor): Long tensor of dimension ..., S; where ... are
        leading batch dimensions.

    Returns:
        Tensor: Tensor of dimension ..., M; where the leading dimensions are
        the same as the batch dimensions from `index_tensor`
        Tensor: Boolean tensor of dimension ...; where each element is True if
        the corresponding index is a specified (nonzero) element of the sparse
        tensor and False if not
    """
    (
        sparse_tensor_indices_linearized,
        sparse_tensor_values,
        index_tensor_linearized,
        index_tensor_shape,
        is_specified_mask,
    ) = linearize_sparse_and_index_tensors(sparse_tensor, index_tensor)

    index_search = torch.searchsorted(
        sparse_tensor_indices_linearized, index_tensor_linearized
    ).clamp_max(sparse_tensor_indices_linearized.shape[0] - 1)
    selected = sparse_tensor_values[index_search]
    if sparse_tensor.dense_dim() > 0:
        selected = selected.view(
            *index_tensor_shape[:-1], *selected.shape[-sparse_tensor.dense_dim() :]
        )
    else:
        selected = selected.view(*index_tensor_shape[:-1])
    is_specified_mask = is_specified_mask.view(
        *index_tensor_shape[:-1], *[1] * sparse_tensor.dense_dim()
    )
    if not is_specified_mask.all():
        # selected = torch.masked_fill(selected, is_specified_mask.logical_not(), 0.0)
        selected[(~is_specified_mask).expand_as(selected)] = 0.0
    is_specified_mask = is_specified_mask.view(*index_tensor_shape[:-1])

    if check_all_specified:
        if not is_specified_mask.all():
            raise ValueError(
                "`check_all_specified` was set to True but not all gathered values "
                "were specified"
            )

    return selected, is_specified_mask


def linearize_sparse_and_index_tensors(sparse_tensor: Tensor, index_tensor: Tensor):
    if index_tensor.shape[-1] != sparse_tensor.sparse_dim():
        if (
            sparse_tensor.sparse_dim() - 1 == index_tensor.shape[-1]
            and sparse_tensor.shape[-1] == 1
            and sparse_tensor.dense_dim() == 0
        ):
            sparse_tensor = sparse_tensor[..., 0].coalesce()
        else:
            raise ValueError(
                "Expected last dim of `index_tensor` to be the same as "
                f"`sparse_tensor.sparse_dim()`, got {index_tensor.shape[-1]=} "
                f"and {sparse_tensor.sparse_dim()=}"
            )
    sparse_shape = sparse_tensor.shape[: sparse_tensor.sparse_dim()]
    dim_linear_offsets = index_tensor.new_tensor(
        [np.prod((sparse_shape + (1,))[i + 1 :]) for i in range(len(sparse_shape))]
    )

    sparse_tensor_indices_linear = (
        sparse_tensor.indices() * dim_linear_offsets.unsqueeze(-1)
    ).sum(0)

    if index_tensor.shape[-1] != sparse_tensor.sparse_dim():
        assert index_tensor.shape[-1] == sparse_tensor.sparse_dim() - 1
        index_tensor = batch_dim_to_leading_index(index_tensor)

    index_tensor_shape = index_tensor.shape
    index_tensor_linearized = (index_tensor * dim_linear_offsets).sum(-1)
    index_tensor_linearized = index_tensor_linearized.reshape(-1)

    is_specified_mask = torch.isin(
        index_tensor_linearized, sparse_tensor_indices_linear
    )

    assert index_tensor_linearized.min() >= 0
    assert index_tensor_linearized.max() <= np.prod(sparse_shape)

    return (
        sparse_tensor_indices_linear,
        sparse_tensor.values(),
        index_tensor_linearized,
        index_tensor_shape,
        is_specified_mask,
    )


def scatter_to_sparse_tensor(
    sparse_tensor: Tensor, index_tensor: Tensor, values: Tensor
):
    """Batch updating of elements in a torch sparse tensor. Should be
    equivalent to sparse_tensor[index_tensor] = values. It works by flattening
    the sparse tensor's sparse dims and the index tensor to 1D (and converting
    n-d indices to raveled indices), then using index_put along the flattened
    sparse tensor.

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension ..., M; where ... are
        S leading sparse dimensions and M is the dense dimension
        index_tensor (Tensor): Long tensor of dimension ..., S; where ... are
        leading batch dimensions.
        values (Tensor): Tensor of dimension ..., M; where ... are leading
        batch dimensions and M is the dense dimension

    Returns:
        Tensor: sparse_tensor with the new values scattered into it
    """
    assert index_tensor.shape[0] == values.shape[0]
    assert sparse_tensor.dense_dim() == values.ndim - 1

    (
        sparse_tensor_indices_linearized,
        sparse_tensor_values,
        index_tensor_linearized,
        _,
        is_specified_mask,
    ) = linearize_sparse_and_index_tensors(sparse_tensor, index_tensor)

    index_search = torch.searchsorted(
        sparse_tensor_indices_linearized, index_tensor_linearized
    ).clamp_max(sparse_tensor_indices_linearized.shape[0] - 1)

    new_values = sparse_tensor_values.clone()
    new_values[index_search[is_specified_mask]] = values[is_specified_mask]
    new_values = torch.cat([new_values, values[~is_specified_mask]], 0)
    new_indices = torch.cat(
        [sparse_tensor.indices(), index_tensor[~is_specified_mask].T], -1
    )

    out = torch.sparse_coo_tensor(
        new_indices,
        new_values,
        sparse_tensor.shape,
        dtype=sparse_tensor.dtype,
        device=sparse_tensor.device,
    ).coalesce()
    return out


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
    out = matching_indices.to(torch.uint8).argmax(0).cpu()
    return out


def union_sparse_indices(sparse_tensor_1: Tensor, sparse_tensor_2: Tensor):
    assert sparse_tensor_1.is_sparse
    assert sparse_tensor_2.is_sparse
    assert sparse_tensor_1.sparse_dim() == sparse_tensor_2.sparse_dim()
    assert sparse_tensor_1.dense_dim() == sparse_tensor_2.dense_dim()

    M = sparse_tensor_1.sparse_dim()
    K = sparse_tensor_1.dense_dim()

    if sparse_tensor_1.shape != sparse_tensor_2.shape:
        max_shape = max([tensor.shape for tensor in (sparse_tensor_1, sparse_tensor_2)])
        sparse_tensor_1 = sparse_tensor_1.sparse_resize_(max_shape, M, K)
        sparse_tensor_2 = sparse_tensor_2.sparse_resize_(max_shape, M, K)

    sparse_tensor_1 = sparse_tensor_1.coalesce()
    sparse_tensor_2 = sparse_tensor_2.coalesce()

    indices_1 = sparse_tensor_1.indices()
    values_1 = sparse_tensor_1.values()
    indices_2 = sparse_tensor_2.indices()
    values_2 = sparse_tensor_2.values()

    indices_2_2_1 = torch.cat([indices_2, indices_2, indices_1], -1)
    uniques, counts = torch.unique(indices_2_2_1, dim=-1, return_counts=True)
    tensor_1_exclusives = uniques[:, counts == 1]
    tensor_2_exclusives = uniques[:, counts == 2]

    tensor_2_unioned = torch.sparse_coo_tensor(
        torch.cat([indices_2, tensor_1_exclusives], -1),
        torch.cat(
            [
                values_2,
                values_2.new_zeros(
                    (tensor_1_exclusives.shape[-1], *sparse_tensor_2.shape[M : M + K]),
                ),
            ],
            0,
        ),
        size=sparse_tensor_2.shape,
        device=sparse_tensor_2.device,
    ).coalesce()

    tensor_1_unioned = torch.sparse_coo_tensor(
        torch.cat([indices_1, tensor_2_exclusives], -1),
        torch.cat(
            [
                values_1,
                values_1.new_zeros(
                    (tensor_2_exclusives.shape[-1], *sparse_tensor_1.shape[M : M + K])
                ),
            ],
            0,
        ),
        size=sparse_tensor_1.shape,
        device=sparse_tensor_1.device,
    ).coalesce()

    assert torch.equal(tensor_1_unioned.indices(), tensor_2_unioned.indices())

    return tensor_1_unioned, tensor_2_unioned


def __trim(subtensor: Tensor):
    subtensor = subtensor.coalesce()
    indices, values = subtensor.indices(), subtensor.values()
    shape = subtensor.shape
    n_electrons = indices[0].max().item() + 1
    new_shape = (n_electrons, *shape[1:])
    return torch.sparse_coo_tensor(indices, values, new_shape).coalesce()


def bhwn_to_nhw_iterator_over_batches_torch(tensor: Tensor):
    assert tensor.is_sparse
    tensor = tensor.permute(0, 3, 1, 2).coalesce()

    return [__trim(t) for t in tensor.unbind()]


def bhwn_to_nhw_iterator_over_batches_pydata_sparse(array: sparse.SparseArray):
    assert isinstance(array, sparse.SparseArray)

    array = array.transpose([0, 3, 1, 2])

    def trim(subarray: sparse.SparseArray):
        max_electron_index = subarray.coords[0].max() + 1
        return subarray[:max_electron_index]

    return [trim(subarray) for subarray in array]


def flatten_multi_level_sparse_maps_to_nested(tensors: list[Tensor]):
    # tensors is a list of pytorch sparse tensors of dimension batch x height x width x channel
    assert all(isinstance(tensor, Tensor) for tensor in tensors)
    assert all(tensor.is_sparse for tensor in tensors)
    feature_dims = [tensor.shape[-1] for tensor in tensors]
    if len(set(feature_dims)) != 1:
        raise ValueError(
            f"Expected all feature maps to have same feature dimension, got {feature_dims}"
        )

    batch_size = [tensor.shape[0] for tensor in tensors]
    if len(set(batch_size)) != 1:
        raise ValueError(
            f"Expected all feature maps to have same batch size, got {batch_size}"
        )
    batch_size = batch_size[0]

    tensors_unbatched = [tensor.unbind() for tensor in tensors]
    flattened_feature_tensors = [[] for _ in range(batch_size)]
    flattened_level_indices = [[] for _ in range(batch_size)]
    flattened_pixel_indices = [[] for _ in range(batch_size)]
    for level_index, level_tensors in enumerate(tensors_unbatched):
        for b in range(batch_size):
            tensor = level_tensors[b].coalesce()
            flattened_feature_tensors[b].append(tensor.values())
            flattened_level_indices[b].append(
                torch.full(
                    [tensor.values().shape[0]], level_index, device=tensor.device
                )
            )
            flattened_pixel_indices[b].append(tensor.indices().T)

    flattened_feature_tensors = [
        torch.cat(values) for values in flattened_feature_tensors
    ]
    flattened_level_indices = [
        torch.cat(indices) for indices in flattened_level_indices
    ]
    flattened_pixel_indices = [
        torch.cat(indices) for indices in flattened_pixel_indices
    ]

    features = torch.nested.as_nested_tensor(flattened_feature_tensors)
    levels = torch.nested.as_nested_tensor(flattened_level_indices)
    indices = torch.nested.as_nested_tensor(flattened_pixel_indices)

    return features, levels, indices


def nested_flattened_tensors_to_sparse_tensors(
    features: Tensor,
    levels: Tensor,
    indices: Tensor,
    level_spatial_shapes: list[list[int]],
):
    all_levels = torch.unique(torch.cat([torch.unique(lev) for lev in levels]))
    batch_size = features.size(0)
    assert levels.size(0) == indices.size(0) == batch_size

    features_unbatched = features.unbind()
    levels_unbatched = levels.unbind()
    indices_unbatched = indices.unbind()

    values_per_level = [[] for _ in range(len(all_levels))]
    indices_per_level = [[] for _ in range(len(all_levels))]

    for level_index, level_vals, level_inds in zip(
        all_levels, values_per_level, indices_per_level
    ):
        for batch_index, (feat_b, level_b, index_b) in enumerate(
            zip(features_unbatched, levels_unbatched, indices_unbatched)
        ):
            in_level = level_b == level_index
            n_elements = torch.count_nonzero(in_level)
            level_vals.append(feat_b[in_level])
            level_inds.append(
                torch.cat(
                    [
                        index_b.new_full([1, n_elements], batch_index),
                        index_b[in_level].T,
                    ],
                    0,
                )
            )

    values_per_level = [torch.cat(values, 0) for values in values_per_level]
    indices_per_level = [torch.cat(indices, 1) for indices in indices_per_level]

    out = [
        torch.sparse_coo_tensor(
            level_inds, level_vals, [batch_size] + level_shape + [level_vals.shape[-1]]
        ).coalesce()
        for level_inds, level_vals, level_shape in zip(
            indices_per_level, values_per_level, level_spatial_shapes
        )
    ]
    return out
