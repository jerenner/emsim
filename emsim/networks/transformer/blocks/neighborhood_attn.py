from typing import Optional, Union

import torch
from torch import Tensor, nn

from .rope import RoPEEncodingNDGroupedFreqs


class SparseNeighborhoodAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int,
        neighborhood_size: int = 7,
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

    def forward(
        self,
        query: Tensor,
        query_positions_bijl: Tensor,
        query_batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
        attn_mask: Optional[Tensor] = None,
    ):
        assert query.ndim == 2
        assert query_positions_bijl.shape == (query.shape[0], 4)

        # gather the neighborhood of each query

        residual = query
        if self.norm_first:
            query = self.norm(query)

        q = self.q_in_proj(query)


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
                torch.arange(size, device=ij.device, dtype=ij.dtype),
                torch.arange(size, device=ij.device, dtype=ij.dtype),
                indexing="ij",
            ),
            -1,
        )
        - (size - 1) / 2
        for size, ij in zip(neighborhood_sizes, query_multilevel_ijs)
    ]
    query_multilevel_ij_neighborhoods = [
        ij[:, None, None, :].floor() + offset[None]
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
