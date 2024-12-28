from typing import Optional

from torch import Tensor, nn
import torch
import torch.nn.functional as F

from emsim.networks.positional_encoding.rope import RoPEEncoding2D
from emsim.networks.sparse_ms_deform_attn import SparseMSDeformableAttention
from emsim.utils.batching_utils import (
    deconcat_add_batch_dim,
    remove_batch_dim_and_concat,
)
from emsim.utils.sparse_utils import sparse_tensor_to_batched, multilevel_normalized_xy


class FFNBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation_fn: nn.Module = nn.GELU,
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor):
        if self.norm_first:
            x = x + self.mlp(self.norm(x))
        else:
            x = self.norm(x + self.mlp(x))
        return x

    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
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

    def forward(self, query, query_pos_embedding, pad_mask=None, attn_mask=None):
        residual = query
        query_with_pos_embed = query + query_pos_embedding
        if self.norm_first:
            query_with_pos_embed = self.norm(query_with_pos_embed)
            query = residual + self.dropout(
                self.attn(
                    query_with_pos_embed,
                    query_with_pos_embed,
                    query,
                    key_padding_mask=pad_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
            )
        else:
            query = residual + self.dropout(
                self.attn(
                    query_with_pos_embed,
                    query_with_pos_embed,
                    query,
                    key_padding_mask=pad_mask,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
            )
            query = self.norm(query)
        return query

    def reset_parameters(self):
        self.attn._reset_parameters()
        self.norm.reset_parameters()


class SparseTensorCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
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
        query_pos_encoding: Tensor,
        query_normalized_xy_positions: Tensor,
        stacked_feature_maps: Tensor,
        spatial_shapes: Tensor,
        pad_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        residual = query
        query_with_pos_embed = query + query_pos_encoding

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


class SelfAttentionBlockWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm_first = norm_first

        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.pos_encoding = RoPEEncoding2D(d_model, n_heads, dtype=torch.double)
        self.attn_drop_rate = dropout
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, positions: Tensor, batch_offsets: Tensor):
        residual = x
        if self.norm_first:
            x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, -1)

        q, pad_mask = deconcat_add_batch_dim(q, batch_offsets)
        k, _ = deconcat_add_batch_dim(k, batch_offsets)
        v, _ = deconcat_add_batch_dim(v, batch_offsets)
        positions, _ = deconcat_add_batch_dim(positions, batch_offsets)

        q, k = self.pos_encoding(q, positions, k)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=pad_mask, dropout_p=self.attn_drop_rate
        )
        x, batch_offsets_2 = remove_batch_dim_and_concat(x, pad_mask)
        assert torch.equal(batch_offsets, batch_offsets_2)

        x = self.out_proj(x)
        x = self.out_proj_drop(x)
        x = x + residual
        if not self.norm_first:
            x = self.norm(x)

        return x

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.qkv.reset_parameters()
        self.pos_encoding.reset_parameters()
        self.out_proj.reset_parameters()


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


# class FP64Linear(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         bias: bool = True,
#         device: torch.device = None,
#     ):
#         super().__init__()
#         self.linear = nn.Linear(
#             in_features, out_features, bias=bias, device=device, dtype=torch.float64
#         )

#     def forward(self, x):
#         return self.linear(x.to(torch.float64))
