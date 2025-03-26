from typing import Optional, Union

import torch
from torch import Tensor, nn

from .rope import RoPEEncodingNDGroupedFreqs, prep_multilevel_positions
from emsim.utils.sparse_utils import (
    gather_from_sparse_tensor,
    batch_offsets_from_sparse_tensor_indices,
)
from emsim.utils.batching_utils import (
    deconcat_add_batch_dim,
    remove_batch_dim_and_concat,
)


class SparseNeighborhoodAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int = 4,
        neighborhood_sizes: Union[Tensor, list[int]] = [3, 5, 7, 9],
        position_dim: int = 2,
        dropout: float = 0.0,
        bias: bool = False,
        norm_first: bool = True,
        rope_theta: float = 10.0,
        rope_dtype: torch.dtype = torch.double,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm_first = norm_first

        self.norm = nn.LayerNorm(d_model)

        self.q_in_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_in_proj = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.pos_encoding = RoPEEncodingNDGroupedFreqs(
            position_dim + 1,
            d_model,
            n_heads,
            [0] * position_dim + [1],
            [rope_theta] * position_dim + [rope_theta / 100],
            dtype=rope_dtype,
        )
        self.attn_drop_rate = dropout
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj_drop = nn.Dropout(dropout)

        self.n_levels = n_levels
        self.neighborhood_sizes = torch.tensor(neighborhood_sizes, dtype=torch.int)

    def forward(
        self,
        query: Tensor,
        query_positions_bijl: Tensor,
        query_batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
    ):
        assert query.ndim == 2
        assert query_positions_bijl.shape == (query.shape[0], 4)
        n_queries = query.shape[0]
        value_batch_offsets = batch_offsets_from_sparse_tensor_indices(
            stacked_feature_maps.indices()
        )

        # gather the neighborhood of each query
        value_bijl = get_multilevel_neighborhood(
            query_positions_bijl, level_spatial_shapes, self.neighborhood_sizes
        )
        keys_per_query = sum(self.neighborhood_sizes**2)
        assert value_bijl.shape == (n_queries, keys_per_query, 4)
        value, value_is_specified = gather_from_sparse_tensor(
            stacked_feature_maps, value_bijl
        )
        assert value.shape == (n_queries, keys_per_query, self.d_model)
        assert value_is_specified.shape == (n_queries, keys_per_query)

        residual = query
        if self.norm_first:
            query = self.norm(query)
        q = self.q_in_proj(query)
        k, v = self.kv_in_proj(value).chunk(2, dim=-1)

        query_ijl = prep_multilevel_positions(
            query_positions_bijl, level_spatial_shapes
        )[:, 1:]
        key_ijl = prep_multilevel_positions(
            value_bijl.view(-1, 4), level_spatial_shapes
        ).view(n_queries, keys_per_query, 4)[:, :, 1:]

        q, query_pad_mask = deconcat_add_batch_dim(q, query_batch_offsets)
        k, key_pad_mask = deconcat_add_batch_dim(k, value_batch_offsets)
        v, value_pad_mask = deconcat_add_batch_dim(v, value_batch_offsets)
        value_is_specified, _ = deconcat_add_batch_dim(
            value_is_specified, value_batch_offsets
        )
        assert torch.equal(key_pad_mask, value_pad_mask)
        # pad value of 1.0 instead of 0.0 to suppress a false warning
        query_ijl, _ = deconcat_add_batch_dim(
            query_ijl,
            query_batch_offsets,
            query_ijl.new_ones([]),
        )
        key_ijl, _ = deconcat_add_batch_dim(
            key_ijl,
            value_batch_offsets,
            key_ijl.new_ones([]),
        )
        assert k.shape == (q.shape[0], q.shape[1], keys_per_query, self.d_model)

        q, k = self.pos_encoding(
            q,
            query_ijl,
            k.view(q.shape[0], -1, self.d_model),
            key_ijl.view(q.shape[0], -1, 3),
        )
        bsz, tgt_seq_len, n_heads, head_dim = q.shape
        q = q.transpose(1, 2)
        k = k.view(bsz, tgt_seq_len, keys_per_query, n_heads, head_dim).permute(
            0, 3, 1, 2, 4
        )
        v = v.view(bsz, tgt_seq_len, keys_per_query, n_heads, head_dim).permute(
            0, 3, 1, 2, 4
        )

        attn_scores = torch.einsum("bhqd,bhqkd->bhqk")
        attn_scores = torch.masked_fill(
            attn_scores, value_is_specified.unsqueeze(1).logical_not(), -torch.inf
        )
        attn_weights = torch.softmax(attn_scores, -1)
        attn_weights = torch.dropout(
            attn_weights, self.attn_drop_rate, train=self.training
        )
        x = torch.einsum("bhqk,bhqkd->bhqd", attn_weights, v)
        x = x.transpose(1, 2)
        x = x.reshape(bsz, tgt_seq_len, self.d_model)
        x, query_batch_offsets_2 = remove_batch_dim_and_concat(x, query_pad_mask)
        assert torch.equal(query_batch_offsets, query_batch_offsets_2)

        x = self.out_proj(x)
        x = self.out_proj_drop(x)

        x = x + residual

        if not self.norm_first:
            x = self.norm(x)
        return x

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.q_in_proj.reset_parameters()
        self.kv_in_proj.reset_parameters()
        self.pos_encoding.reset_parameters()
        self.out_proj.reset_parameters()


def get_multilevel_neighborhood(
    bijl_positions: Tensor,
    level_spatial_shapes: Tensor,
    neighborhood_sizes: Union[Tensor, list[int]] = [3, 5, 7, 9],
):
    assert bijl_positions.ndim == 2
    assert level_spatial_shapes.ndim == 2
    assert bijl_positions.shape[-1] == 4
    assert torch.all(torch.tensor(neighborhood_sizes) % 2 == 1)  # all odd

    n_queries = bijl_positions.shape[0]

    spatial_scalings = level_spatial_shapes / level_spatial_shapes[-1]
    query_spatial_scalings = spatial_scalings[bijl_positions[:, -1].int()]

    query_fullscale_ij = bijl_positions[:, 1:-1] * query_spatial_scalings
    query_multilevel_ijs = query_fullscale_ij.unsqueeze(1) * spatial_scalings

    neighborhood_offsets = [
        torch.stack(
            torch.meshgrid(
                torch.arange(size, device=ij.device, dtype=torch.int),
                torch.arange(size, device=ij.device, dtype=torch.int),
                indexing="ij",
            ),
            -1,
        )
        - (size - 1) / 2
        for size, ij in zip(neighborhood_sizes, query_multilevel_ijs)
    ]
    query_multilevel_ij_neighborhoods = [
        ij[:, None, None, :].floor().int() + offset[None]
        for ij, offset in zip(query_multilevel_ijs, neighborhood_offsets)
    ]
    query_multilevel_ij_neighborhoods = [
        torch.cat(
            [
                neighborhood,
                neighborhood.new_full([], i).expand(*neighborhood.shape[:-1], 1),
            ],
            -1,
        )
        for i, neighborhood in enumerate(query_multilevel_ij_neighborhoods)
    ]
    ijl_neighborhoods_concat = torch.cat(
        [
            nhood.view(n_queries, -1, nhood.shape[-1])
            for nhood in query_multilevel_ij_neighborhoods
        ],
        1,
    )
    bijl_neighborhood_indices = torch.cat(
        [
            bijl_positions[:, 0][:, None, None].expand(
                -1, ijl_neighborhoods_concat.shape[1], 1
            ),
            ijl_neighborhoods_concat,
        ],
        -1,
    ).int()
    return bijl_neighborhood_indices
