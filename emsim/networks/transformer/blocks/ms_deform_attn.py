from emsim.networks.sparse_ms_deform_attn import SparseMSDeformableAttention


from torch import Tensor, nn


class SparseDeformableAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int,
        n_points_per_level_per_head: int,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm_first = norm_first

        self.attn = SparseMSDeformableAttention(
            d_model, n_levels, n_heads, n_points_per_level_per_head
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,
        query_pos_encoding: Tensor,
        query_normalized_xy_positions: Tensor,
        batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        spatial_shapes: Tensor,
    ):
        if self.norm_first:
            residual = queries
            queries = queries + query_pos_encoding
            queries = self.norm(queries)
            queries = self.attn(
                queries,
                batch_offsets,
                query_normalized_xy_positions,
                stacked_feature_maps,
                spatial_shapes,
            )
            queries = self.dropout(queries)
            queries = queries + residual
        else:
            queries = queries + self.dropout(
                self.attn(
                    queries + query_pos_encoding,
                    batch_offsets,
                    query_normalized_xy_positions,
                    stacked_feature_maps,
                    spatial_shapes,
                )
            )
            queries = self.norm(queries)
        return queries

    def reset_parameters(self):
        self.attn.reset_parameters()
        self.norm.reset_parameters()


class SparseDeformableAttentionBlockWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int,
        n_points_per_level_per_head: int,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm_first = norm_first

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
