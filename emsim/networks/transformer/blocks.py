from typing import Optional

from torch import Tensor, nn
import torch

from emsim.networks.transformer.attn import SelfAttentionWithRoPE
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

        self.attn = SelfAttentionWithRoPE(d_model, n_heads, dropout, bias)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, positions: Tensor, batch_offsets: Tensor):
        x, pad_mask = deconcat_add_batch_dim(x, batch_offsets)
        positions, _ = deconcat_add_batch_dim(positions, batch_offsets)

        if self.norm_first:
            residual = x
            x = self.norm(x)
            x = self.attn(x, positions, pad_mask)
            x = self.dropout(x)
            x = x + residual
        else:
            x = x + self.dropout(self.attn(x, positions, pad_mask))
            x = self.norm(x)

        x, batch_offsets_2 = remove_batch_dim_and_concat(x, pad_mask)
        assert torch.equal(batch_offsets, batch_offsets_2)

        return x

    def reset_parameters(self):
        self.attn.reset_parameters()
        self.norm.reset_parameters()


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
