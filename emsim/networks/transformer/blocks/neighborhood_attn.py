from typing import Union

import torch
from torch import Tensor, nn

from emsim.utils.sparse_utils.batching.batching import (
    deconcat_add_batch_dim,
    remove_batch_dim_and_concat,
    batch_offsets_to_indices,
    seq_lengths_to_batch_offsets,
)
from emsim.utils.sparse_utils.ops.linear.linear import batch_sparse_index_linear
from emsim.utils.sparse_utils.ops.subset_attn.subset_attn import (
    batch_sparse_index_subset_attn,
)

from emsim.networks.positional_encoding.rope import (
    RoPEEncodingND,
    prep_multilevel_positions,
    get_multilevel_freq_group_pattern,
)
from emsim.utils.sparse_utils.validation import validate_nd


class SparseNeighborhoodAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_levels: int = 4,
        neighborhood_sizes: Union[Tensor, list[int]] = [3, 5, 7, 9],
        position_dim: int = 2,
        dropout: float = 0.0,
        bias: bool = False,
        norm_first: bool = True,
        rope_spatial_base_theta: float = 100.0,
        rope_level_base_theta: float = 10.0,
        rope_share_heads: bool = False,
        rope_freq_group_pattern: str = "single",
        rope_enforce_freq_groups_equal: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.position_dim = position_dim
        self.norm_first = norm_first

        self.norm = nn.LayerNorm(embed_dim)

        self.q_in_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_in_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.pos_encoding = RoPEEncodingND(
            position_dim + 1,  # +1 for level dimension
            embed_dim,
            n_heads,
            rope_share_heads,
            get_multilevel_freq_group_pattern(position_dim, rope_freq_group_pattern),
            enforce_freq_groups_equal=rope_enforce_freq_groups_equal,
            rope_base_theta=[
                [rope_spatial_base_theta] * position_dim + [rope_level_base_theta]
            ],
        )
        self.attn_drop_rate = dropout
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj_drop = nn.Dropout(dropout)

        self.n_levels = n_levels
        self.register_buffer(
            "rope_spatial_base_theta", torch.tensor(rope_spatial_base_theta)
        )
        self.register_buffer(
            "rope_level_base_theta", torch.tensor(rope_level_base_theta)
        )
        self.neighborhood_sizes = torch.tensor(neighborhood_sizes, dtype=torch.int)

    def forward(
        self,
        query: Tensor,
        query_spatial_positions: Tensor,
        query_batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
    ) -> Tensor:
        validate_nd(query, 2, "query")
        n_queries = query.shape[0]

        residual = query
        if self.norm_first:
            query = self.norm(query)
        q = self.q_in_proj(query)

        # Prep query
        max_spatial_level = level_spatial_shapes.argmax(-2).unique()
        assert max_spatial_level.numel() == 1

        # Add level "dimension" to query position for pos encoding
        query_spatial_level_positions = query_spatial_positions.new_zeros(
            (n_queries, self.pos_encoding.position_dim)
        )
        query_spatial_level_positions[:, :-1] = query_spatial_positions
        query_spatial_level_positions[:, -1] = max_spatial_level

        # Position encode queries
        query_rotated = self.pos_encoding(q, query_spatial_level_positions)

        # Prepare key data:
        # Compute the neighborhood indices of each query
        nhood_spatial_indices, nhood_level_indices = get_multilevel_neighborhoods(
            query_spatial_positions, level_spatial_shapes, self.neighborhood_sizes
        )
        keys_per_query = nhood_spatial_indices.size(1)

        # Initialize the full sparse indices tensor for keys:
        # (batch, *spatial_dims, level)
        key_index_tensor = query_spatial_positions.new_zeros(
            n_queries, keys_per_query, self.position_dim + 2
        )
        # fill in batch indices
        for i in range(query_batch_offsets.size(0) - 1):
            batch_start, batch_end = query_batch_offsets[i], query_batch_offsets[i + 1]
            key_index_tensor[batch_start:batch_end, :, 0] = i
        key_index_tensor[:, :, 1:-1] = nhood_spatial_indices
        key_index_tensor[:, :, -1] = nhood_level_indices

        # Get the key RoPE components
        key_rope_freqs = self.pos_encoding.grouped_rope_freqs_tensor(
            self.pos_encoding.freqs
        )
        key_positions = prep_multilevel_positions(
            key_index_tensor[..., 1:-1],
            key_index_tensor[:, 0],
            key_index_tensor[:, -1],
            level_spatial_shapes,
        )

        # Get weight tensors for attention call
        key_weight, value_weight = self.kv_in_proj.weight.chunk(2, -1)
        if self.kv_in_proj.bias is not None:
            key_bias, value_bias = self.kv_in_proj.bias.chunk(2, -1)
        else:
            key_bias, value_bias = None, None

        assert nhood_spatial_indices.shape == (
            n_queries,
            keys_per_query,
            query_spatial_positions.size(-1),
        )

        x = batch_sparse_index_subset_attn(
            stacked_feature_maps,
            key_index_tensor,
            query_rotated,
            self.n_heads,
            key_weight,
            value_weight,
            key_bias,
            value_bias,
            key_positions=key_positions,
            rope_freqs=key_rope_freqs,
        )

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


def get_multilevel_neighborhoods(
    query_fullscale_spatial_positions: Tensor,
    level_spatial_shapes: Tensor,
    neighborhood_sizes: Union[Tensor, list[int]] = [3, 5, 7, 9],
) -> Tensor:
    """Computes multi-resolution neighborhood indices for query positions.

    Generates neighborhood indices at multiple resolution levels for each query
    position, with configurable neighborhood sizes for each level. This enables
    hierarchical feature aggregation by defining sampling regions around each query
    point at different scales.

    Args:
        query_fullscale_spatial_positions (Tensor): Query positions of shape
            [n_queries, position_dim], where each row contains the N-D position of a
            query point at the full scale resolution.
        level_spatial_shapes (Tensor): Tensor of shape [num_levels, position_dim]
            specifying the spatial dimensions of each resolution level.
        neighborhood_sizes (Union[Tensor, list[int]]): List or tensor of odd integers
            specifying the neighborhood size (window width) at each level.
            Default: [3, 5, 7, 9].

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - multilevel_neighborhood_indices: Tensor of shape
                [n_queries, sum(neighborhood_sizes^position_dim), position_dim]
                containing the spatial indices of all neighborhood points for each
                query across all levels.
            - level_indices: Tensor of shape [sum(neighborhood_sizes^position_dim)]
                mapping each neighborhood position to its corresponding resolution
                level.

    Raises:
        ValueError: If input tensors don't have the expected shape or dimensions, or
            if any neighborhood size is not an odd number.
    """
    validate_nd(
        query_fullscale_spatial_positions, 2, "query_fullscale_spatial_positions"
    )
    n_queries, position_dim = query_fullscale_spatial_positions.shape

    device = query_fullscale_spatial_positions.device

    neighborhood_sizes = torch.as_tensor(neighborhood_sizes, device=device)
    if any(neighborhood_sizes % 2 != 1):
        raise ValueError(
            f"Expected all odd neighborhood_sizes, got {neighborhood_sizes}"
        )

    spatial_scalings = level_spatial_shapes / level_spatial_shapes.max(-2)[0]

    # query x level x position_dim
    query_multilevel_spatial_positions = (
        query_fullscale_spatial_positions.unsqueeze(1) * spatial_scalings
    )

    # Compute neighborhood cardinality for each level
    n_neighborhood_elements = neighborhood_sizes.pow(position_dim)

    # Create the centered neighborhood offset grids for each level
    # [size^position_dim x position_dim] * n_level
    neighborhood_offset_grids = []
    for size in neighborhood_sizes:
        axes = [torch.arange(size, device=device)] * position_dim
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        offsets = grid.flatten(0, -2) - (size - 1) / 2
        neighborhood_offset_grids.append(offsets)

    # Prepare level indexing
    level_indices = torch.repeat_interleave(
        torch.arange(neighborhood_sizes.size(0), device=device), n_neighborhood_elements
    )
    level_offsets = seq_lengths_to_batch_offsets(n_neighborhood_elements)

    # Initialize output tensor holding all neighborhood indices
    multilevel_neighborhood_indices = torch.zeros(
        n_queries,
        n_neighborhood_elements.sum(),
        query_fullscale_spatial_positions.size(-1),
        device=device,
        dtype=torch.long,
    )

    # Compute the neighborhood indices and fill in the output tensor
    for level, level_positions in enumerate(
        query_multilevel_spatial_positions.unbind(1)
    ):
        level_start = level_offsets[level]
        level_end = level_offsets[level + 1]
        nhood_grid = neighborhood_offset_grids[level]

        multilevel_neighborhood_indices[:, level_start:level_end, :] = (
            level_positions.unsqueeze(1).floor().long() + nhood_grid.unsqueeze(0)
        )

    return multilevel_neighborhood_indices, level_indices
