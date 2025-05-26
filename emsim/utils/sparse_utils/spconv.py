from functools import reduce
from typing import List

import torch

from . import imports
from .imports import SparseConvTensor


@imports.requires_spconv
def spconv_sparse_mult(*tens) -> SparseConvTensor:
    """This is more or less a line-for-line copy of spconv's Fsp.sparse_add
    function, except it replaces the elementwise addition reduction with an
    elementwise multiplication."""
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
