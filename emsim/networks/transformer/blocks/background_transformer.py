from torch import nn, Tensor
import torch
import torch.nn.functional as F

from .ffn import FFNBlock
from ...positional_encoding.rope import RoPEEncodingND

from emsim.utils.sparse_utils.batching.batch_utils import (
    batch_offsets_from_sparse_tensor_indices,
    deconcat_add_batch_dim,
)


class BackgroundEmbeddingTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        norm_first: bool = True,
        rope_base_theta: float = 10.0,
        rope_share_heads: bool = False,
    ):
        super().__init__()
        self.attn = BackgroundEmbeddingAttentionBlock(
            embed_dim,
            n_heads,
            dropout,
            bias,
            norm_first,
            rope_base_theta,
            rope_share_heads,
        )

        self.ffn = FFNBlock(embed_dim, ffn_hidden_dim, dropout, norm_first=norm_first)

    def forward(self, background_embedding: Tensor, stacked_feature_maps: Tensor):
        assert stacked_feature_maps.is_sparse

        bg = self.attn(background_embedding, stacked_feature_maps)
        bg = self.ffn(bg)
        return bg


class BackgroundEmbeddingAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        norm_first: bool = True,
        rope_base_theta: float = 10.0,
        rope_share_heads: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.norm_first = norm_first

        self.q_in_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_in_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)

        self.pos_encoding = RoPEEncodingND(
            1,
            embed_dim,
            n_heads,
            rope_base_theta=rope_base_theta,
            share_heads=rope_share_heads,
        )
        self.attn_drop_rate = dropout
        self.norm_attn = nn.LayerNorm(embed_dim)

        self.attn_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_out_proj_drop = nn.Dropout(dropout)

    def reset_parameters(self):
        self.attn._reset_parameters()
        self.ffn.reset_parameters()
        self.norm_attn.reset_parameters()
        self.norm_ffn.reset_parameters()

    def forward(self, bg: Tensor, stacked_feature_maps: Tensor):
        bsz, n_bg_levels, _ = bg.shape
        residual = bg

        if self.norm_first:
            bg = self.norm_attn(bg)

        query = self.q_in_proj(bg)

        # Get all feature plane tokens and their level
        stacked_feature_maps = stacked_feature_maps.coalesce()
        feature_values = stacked_feature_maps.values()
        # sparse indices: [batch, i, j, level]
        feature_level_indices = stacked_feature_maps.indices()[-1:].T

        key, value = self.kv_in_proj(feature_values).chunk(2, dim=-1)

        # rotate queries and keys
        query_level_indices = torch.repeat_interleave(
            torch.arange(n_bg_levels, device=query.device), bsz
        ).unsqueeze(1)
        query = query.view(-1, self.embed_dim)
        query, key = self.pos_encoding(
            query, query_level_indices, key, feature_level_indices
        )
        query: Tensor = query.view(bsz, n_bg_levels, self.embed_dim)

        # unstack stacked-batch key/value tensors for attention
        kv_batch_offsets = batch_offsets_from_sparse_tensor_indices(
            stacked_feature_maps.indices()
        )
        key, pad_mask = deconcat_add_batch_dim(key, kv_batch_offsets)
        value, _ = deconcat_add_batch_dim(value, kv_batch_offsets)

        # split and transpose heads
        head_dim = self.embed_dim // self.n_heads
        query = query.view(bsz, n_bg_levels, self.n_heads, head_dim).transpose(1, 2)
        key: Tensor = key.view(bsz, -1, self.n_heads, head_dim).transpose(1, 2)
        value: Tensor = value.view(bsz, -1, self.n_heads, head_dim).transpose(1, 2)
        attn_mask: Tensor = pad_mask.view(bsz, 1, 1, pad_mask.size(1)).logical_not()

        # do attention
        query = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_rate if self.training else 0.0,
        )
        query = query.transpose(1, 2).reshape(bsz, n_bg_levels, self.embed_dim)  # stack heads

        query = self.attn_out_proj(query)
        query = self.attn_out_proj_drop(query)

        query = query + residual

        if not self.norm_first:
            query = self.norm_attn(query)
        return query


class BackgroundMaxPooler(nn.Module):
    def forward(self, background_embedding: Tensor):
        return background_embedding.max(1).values


class BackgroundAttentionPooler(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, rope_base_theta: float = 10.0):
        super().__init__()
        self.pooling_query = nn.Embedding(1, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.pos_encoding = RoPEEncodingND(
            1, embed_dim, n_heads, rope_base_theta=rope_base_theta
        )

    def reset_parameters(self):
        self.pooling_query.reset_parameters()
        self.kv_proj.reset_parameters()
        self.pos_encoding.reset_parameters()

    def forward(self, background_embedding: Tensor):
        batch_size, n_levels, _ = background_embedding.shape

        query = self.pooling_query.weight.unsqueeze(0).expand(batch_size, 1, -1)

        key, value = self.kv_proj(background_embedding).chunk(2, dim=-1)

        level_indices = (
            torch.arange(n_levels, device=background_embedding.device)
            .view(1, -1, 1)
            .expand(batch_size, -1, 1)
        )

        # rotate key embeddings
        key = self.pos_encoding(key, level_indices)

        query = F.scaled_dot_product_attention(query, key, value)

        return query.squeeze(1)
