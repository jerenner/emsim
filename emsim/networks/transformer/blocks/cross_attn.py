from typing import Optional

from torch import Tensor, nn

from emsim.utils.sparse_utils import multilevel_normalized_xy, sparse_tensor_to_batched


class MultilevelCrossAttentionBlockWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_key_levels: int,
        dropout: float = 0.0,
        bias: bool = False,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm_first = norm_first

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout, bias, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        query_bijl_positions: Tensor,
        query_batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        spatial_shapes: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        assert query.ndim == 2 # (stacked sequences x d_model)
        residual = query
        if self.norm_first:
            query = self.norm(query)


        value, value_indices, value_pad_mask = sparse_tensor_to_batched(
            stacked_feature_maps
        )
        value_pos_normalized_xy = multilevel_normalized_xy(
            stacked_feature_maps, spatial_shapes
        )
        if self.norm_first:
            query_with_pos_embed = self.norm(query_with_pos_embed)
            query = residual + self.dropout(
                self.attn(
                    query_with_pos_embed,
                    value,
                    value,
                    key_padding_mask=value_pad_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
            )
        else:
            query = residual + self.dropout(
                self.attn(
                    query_with_pos_embed,
                    value,
                    value,
                    key_padding_mask=value_pad_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
            )
            query = self.norm(query)
        return query

    def reset_parameters(self):
        self.attn._reset_parameters()
        self.norm.reset_parameters()
