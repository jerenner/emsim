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


def spconv_to_torch_sparse(tensor: spconv.SparseConvTensor):
    """Converts an spconv SparseConvTensor to a sparse torch.Tensor

    Args:
        tensor (spconv.SparseConvTensor): spconv tensor to be converted

    Returns:
        torch.Tensor: Converted sparse torch.Tensor
    """
    if isinstance(tensor, Tensor) and tensor.is_sparse:
        return tensor
    assert isinstance(tensor, spconv.SparseConvTensor)
    size = [tensor.batch_size] + tensor.spatial_shape + [tensor.features.shape[-1]]
    indices = tensor.indices.transpose(0, 1)
    values = tensor.features
    return torch.sparse_coo_tensor(
        indices,
        values,
        size,
        device=tensor.features.device,
        dtype=tensor.features.dtype,
        requires_grad=tensor.features.requires_grad,
    ).coalesce()


def torch_sparse_to_pydata_sparse(tensor: Tensor):
    assert tensor.is_sparse
    tensor = tensor.detach().coalesce().cpu()
    assert tensor.is_coalesced
    return sparse.COO(
        tensor.indices(),
        tensor.values(),
        tensor.shape,
        has_duplicates=False,
    )


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
            )
        )

    c_th = reduce(lambda x, y: x * y, ten_ths).coalesce()
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
        sparse_tensor_linearized,
        index_tensor_linearized,
        index_tensor_shape,
        is_specified_mask,
    ) = linearize_sparse_and_index_tensors(sparse_tensor, index_tensor)

    index_search = torch.searchsorted(
        sparse_tensor_linearized.indices().squeeze(0), index_tensor_linearized
    ).clamp_max(sparse_tensor_linearized.indices().shape[1] - 1)
    selected = sparse_tensor_linearized.values()[index_search]
    if sparse_tensor.dense_dim() > 0:
        selected = selected.reshape(*index_tensor_shape[:-1], selected.shape[-1])
    else:
        selected = selected.reshape(*index_tensor_shape[:-1])
    is_specified_mask = is_specified_mask.reshape(
        *index_tensor_shape[:-1], *[1] * sparse_tensor.dense_dim()
    )
    selected = torch.masked_fill(selected, is_specified_mask.logical_not(), 0.0)
    is_specified_mask = is_specified_mask.reshape(*index_tensor_shape[:-1])

    if check_all_specified:
        if not is_specified_mask.all():
            raise ValueError(
                "`check_all_specified` was set to True but not all gathered values "
                "were specified"
            )

    return selected, is_specified_mask


def gather_from_sparse_tensor_old(sparse_tensor: Tensor, index_tensor: Tensor):
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
        Tensor: Boolean tensor of dimension ...; where each element is True if
        the corresponding index is a specified (nonzero) element of the sparse
        tensor and False if not
    """
    (
        sparse_tensor_linearized,
        index_tensor_linearized,
        index_tensor_shape,
        is_specified_mask,
    ) = linearize_sparse_and_index_tensors(sparse_tensor, index_tensor)
    selected = sparse_tensor_linearized.index_select(
        0, index_tensor_linearized
    ).to_dense()
    if sparse_tensor.dense_dim() > 0:
        selected = selected.reshape(*index_tensor_shape[:-1], selected.shape[-1])
    else:
        selected = selected.reshape(*index_tensor_shape[:-1])
    is_specified_mask = is_specified_mask.reshape(*index_tensor_shape[:-1])
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
    ).sum(0, keepdim=True)
    if sparse_tensor.dense_dim() > 0:
        linear_shape = (
            tuple(np.prod(sparse_shape, keepdims=True))
            + sparse_tensor.shape[-sparse_tensor.dense_dim() :]
        )
    else:
        linear_shape = tuple(np.prod(sparse_shape, keepdims=True))
    sparse_tensor_linearized = torch.sparse_coo_tensor(
        sparse_tensor_indices_linear,
        sparse_tensor.values(),
        linear_shape,
        dtype=sparse_tensor.dtype,
        device=sparse_tensor.device,
        requires_grad=sparse_tensor.requires_grad,
    ).coalesce()

    if index_tensor.shape[-1] != sparse_tensor.sparse_dim():
        assert index_tensor.shape[-1] == sparse_tensor.sparse_dim() - 1
        index_tensor = batch_dim_to_leading_index(index_tensor)

    index_tensor_shape = index_tensor.shape
    index_tensor_linearized = (index_tensor * dim_linear_offsets).sum(-1)
    index_tensor_linearized = index_tensor_linearized.reshape(
        -1,
    )

    is_specified_mask = torch.isin(
        index_tensor_linearized, sparse_tensor_indices_linear, assume_unique=True
    )

    assert index_tensor_linearized.min() >= 0
    assert index_tensor_linearized.max() <= sparse_tensor_linearized.shape[0]

    return (
        sparse_tensor_linearized,
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
        sparse_tensor_linearized,
        index_tensor_linearized,
        _,
        is_specified_mask,
    ) = linearize_sparse_and_index_tensors(sparse_tensor, index_tensor)

    index_search = torch.searchsorted(
        sparse_tensor_linearized.indices().squeeze(0), index_tensor_linearized
    ).clamp_max(sparse_tensor_linearized.indices().shape[1] - 1)

    new_values = sparse_tensor_linearized.values().clone()
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
    out = matching_indices.to(torch.uint8).argmax(0)
    return out


def union_sparse_indices(
    predicted_occupancy_logits: Tensor, groundtruth_occupancy: Tensor
):
    assert groundtruth_occupancy.is_sparse
    assert predicted_occupancy_logits.is_sparse

    if not groundtruth_occupancy.is_coalesced():
        groundtruth_occupancy = groundtruth_occupancy.coalesce()
    if not predicted_occupancy_logits.is_coalesced():
        predicted_occupancy_logits = predicted_occupancy_logits.coalesce()

    groundtruth_indices = groundtruth_occupancy.indices()
    groundtruth_values = groundtruth_occupancy.values()
    predicted_indices = predicted_occupancy_logits.indices()
    predicted_values = predicted_occupancy_logits.values()

    indices_gt_gt_predicted = torch.cat(
        [groundtruth_indices, groundtruth_indices, predicted_indices], -1
    )
    uniques, counts = torch.unique(indices_gt_gt_predicted, dim=-1, return_counts=True)
    predicted_exclusives = uniques[:, counts == 1]
    groundtruth_exclusives = uniques[:, counts == 2]

    groundtruth_unioned = torch.sparse_coo_tensor(
        torch.cat([groundtruth_indices, predicted_exclusives], -1),
        torch.cat(
            [
                groundtruth_values,
                groundtruth_values.new_zeros(
                    predicted_exclusives.shape[-1],
                ),
            ],
            0,
        ),
        size=groundtruth_occupancy.shape[:3],
        device=groundtruth_occupancy.device,
    ).coalesce()

    stacked_zero_logits = predicted_values.new_zeros(
        [groundtruth_exclusives.shape[-1], predicted_values.shape[-1]],
    )
    stacked_zero_logits[:, 0] = 1
    predicted_unioned = torch.sparse_coo_tensor(
        torch.cat([predicted_indices, groundtruth_exclusives], -1),
        torch.cat([predicted_values, stacked_zero_logits], 0),
        size=predicted_occupancy_logits.shape,
        device=predicted_occupancy_logits.device,
    ).coalesce()

    return predicted_unioned, groundtruth_unioned


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
