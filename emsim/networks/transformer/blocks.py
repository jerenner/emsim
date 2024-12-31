from typing import Optional
import math

from torch import Tensor, nn
import torch
import torch.nn.functional as F

from emsim.networks.positional_encoding.rope import RoPEEncodingND
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

    def forward_batched(
        self,
        query: Tensor,
        query_pos_embedding: Tensor,
        pad_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        if pad_mask is not None:
            assert pad_mask.ndim == 2  # batch, seq_len
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

    def forward(
        self,
        query: Tensor,
        query_pos_embedding: Tensor,
        batch_offsets: Tensor,
        attn_mask: Optional[Tensor] = None,
        pad_mask: Optional[Tensor] = None,
    ):
        if query.ndim == 3:
            assert query_pos_embedding.ndim == 3
            assert pad_mask is not None
            return self.forward_batched(
                query, query_pos_embedding, pad_mask=pad_mask, attn_mask=attn_mask
            )
        assert batch_offsets is not None
        residual = query
        query_with_pos_embed = query + query_pos_embedding
        if self.norm_first:
            query_with_pos_embed = self.norm(query_with_pos_embed)
        query, pad_mask = deconcat_add_batch_dim(query, batch_offsets)
        query_with_pos_embed, pad_mask_2 = deconcat_add_batch_dim(
            query_with_pos_embed, batch_offsets
        )
        assert torch.equal(pad_mask, pad_mask_2)
        query = self.dropout(
            self.attn(
                query_with_pos_embed,
                query_with_pos_embed,
                query,
                key_padding_mask=pad_mask,
                need_weights=False,
                attn_mask=attn_mask,
            )[0]
        )
        query, batch_offsets_2 = remove_batch_dim_and_concat(query, pad_mask)
        assert torch.equal(batch_offsets, batch_offsets_2)
        query = query + residual
        if not self.norm_first:
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


class MultilevelSelfAttentionBlockWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int,
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
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.pos_encoding = MultilevelRoPE(
            n_levels, d_model, n_heads, rope_base_theta=rope_theta, dtype=rope_dtype
        )
        self.attn_drop_rate = dropout
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        batch_offsets: Tensor,
    ):
        assert x.ndim == 2  # (stacked sequences x d_model)
        residual = x
        if self.norm_first:
            x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q, pad_mask = deconcat_add_batch_dim(q, batch_offsets)
        k, _ = deconcat_add_batch_dim(k, batch_offsets)
        v, _ = deconcat_add_batch_dim(v, batch_offsets)
        positions, _ = deconcat_add_batch_dim(positions, batch_offsets)

        q, k = self.pos_encoding(q, positions, k)
        # (batch x seq_len x n_heads x head_dim) -> (batch x n_heads x seq_len x head_dim)
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 3
        bsz, seq_len, n_heads, head_dim = q.shape
        q: Tensor = q.transpose(1, 2)
        k: Tensor = k.transpose(1, 2)
        v: Tensor = v.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)

        if pad_mask is not None and pad_mask.any():
            assert pad_mask.shape == (bsz, seq_len)
            #  F.scaled_dot_product_attention wants attn mask broadcastable to
            #  (N, num_heads, L, S)
            attn_mask = pad_mask.view(bsz, 1, seq_len, 1)
            attn_mask = torch.where(attn_mask, -torch.inf, 0.0).contiguous()
            # attn_mask = attn_mask.expand(-1, -1, -1, seq_len)
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_rate if self.training else 0.0,
        )
        # (batch x n_heads x seq_len x head_dim) ->
        # (batch x seq_len x n_heads x head_dim)
        x = x.transpose(1, 2)

        x = x.reshape(bsz, seq_len, self.d_model)
        x, batch_offsets_2 = remove_batch_dim_and_concat(x, pad_mask)
        assert torch.equal(batch_offsets, batch_offsets_2)

        x = self.out_proj(x)
        x = self.out_proj_drop(x)

        x = x + residual

        if not self.norm_first:
            x = self.norm(x)
        return x

    # for testing and debugging purposes
    @staticmethod
    def scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.expand_as(attn_mask).clone().masked_fill_(
                    attn_mask.logical_not(), float("-inf")
                )
            else:
                attn_bias = attn_bias + attn_mask

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.qkv.reset_parameters()
        self.pos_encoding.reset_parameters()
        self.out_proj.reset_parameters()


class MultilevelRoPE(nn.Module):
    def __init__(
        self,
        n_levels: int,
        d_model: int,
        n_heads: int,
        rope_base_theta: int,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope_base_theta = rope_base_theta
        self.ropes = nn.ModuleList(
            RoPEEncodingND(
                position_dim=2,
                d_model=d_model,
                n_heads=n_heads,
                rope_base_theta=rope_base_theta,
                dtype=dtype,
            )
            for _ in range(n_levels)
        )

    def forward(
        self,
        q: Tensor,
        q_positions: Tensor,
        k: Tensor,
        k_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        query_leading_dims = q.shape[:-1]
        key_leading_dims = k.shape[:-1]
        split_head_dims = (self.n_heads, self.d_model // self.n_heads)
        level_arange = torch.arange(
            self.n_levels, device=q_positions.device, dtype=q_positions.dtype
        )
        assert torch.all(torch.isin(q_positions[..., -1], level_arange))
        split_position_indices_q = [q_positions[..., -1] == i for i in level_arange]
        if k_positions is not None:
            assert torch.all(torch.isin(k_positions[..., -1], level_arange))
            split_position_indices_k = [k_positions[..., -1] == i for i in level_arange]
        out_q = torch.zeros_like(q).view(query_leading_dims + split_head_dims)
        out_k = torch.zeros_like(k).view(key_leading_dims + split_head_dims)
        for i, rope in enumerate(self.ropes):
            level_q_indices = split_position_indices_q[i]
            level_q = q[level_q_indices]
            level_q_pos = q_positions[level_q_indices][..., -3:-1] + 0.5
            level_k_indices = (
                split_position_indices_k[i]
                if k_positions is not None
                else level_q_indices
            )
            level_k = k[level_k_indices]
            level_k_pos = (
                k_positions[level_k_indices][..., -3:-1] + 0.5
                if k_positions is not None
                else None
            )
            # if level_q.numel() > 0:
            if True:
                level_out_q, level_out_k = rope(
                    level_q, level_q_pos, level_k, level_k_pos
                )
                out_q[level_q_indices] = level_out_q.to(out_q)
                out_k[level_k_indices] = level_out_k.to(out_k)
        return out_q, out_k

    def reset_parameters(self):
        for module in self.ropes:
            module.reset_parameters()


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
