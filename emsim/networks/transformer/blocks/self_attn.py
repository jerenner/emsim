import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from emsim.networks.transformer.blocks import MultilevelRoPE
from emsim.utils.batching_utils import (
    deconcat_add_batch_dim,
    remove_batch_dim_and_concat,
)


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
    ) -> Tensor:
        assert x.ndim == 2  # (stacked sequences x d_model)
        residual = x
        if self.norm_first:
            x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q, pad_mask = deconcat_add_batch_dim(q, batch_offsets)
        k, _ = deconcat_add_batch_dim(k, batch_offsets)
        v, _ = deconcat_add_batch_dim(v, batch_offsets)
        # pad value of 1.0 instead of 0.0 to suppress a false warning
        positions, _ = deconcat_add_batch_dim(positions, batch_offsets, positions.new_ones([]))

        q, k = self.pos_encoding(q, positions, k)
        # (batch x seq_len x n_heads x head_dim) -> (batch x n_heads x seq_len x head_dim)
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 3
        bsz, seq_len, n_heads, head_dim = q.shape
        q: Tensor = q.transpose(1, 2)
        k: Tensor = k.transpose(1, 2)
        v: Tensor = v.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)

        if pad_mask is not None and pad_mask.any():
            # need to split up the batches since F.scaled_dot_product_attention
            # doesn't actually handle attn_mask in a memory efficient manner
            assert pad_mask.shape == (bsz, seq_len)

            attn_mask = pad_mask.logical_not()
            x = torch.zeros_like(q)
            for i, (q_i, k_i, v_i, mask_i) in enumerate(zip(q, k, v, attn_mask)):
                x[i, :, mask_i] = F.scaled_dot_product_attention(
                    q_i[:, mask_i].unsqueeze(0),
                    k_i[:, mask_i].unsqueeze(0),
                    v_i[:, mask_i].unsqueeze(0),
                    dropout_p=self.attn_drop_rate if self.training else 0.0,
                ).squeeze(0)
        else:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
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

    # not used
    def scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
    ) -> torch.Tensor:
        scale_factor = 1 / math.sqrt(query.size(-1))

        query = query * scale_factor
        attn_weight = torch.matmul(query, key.transpose(-1, -2))
        # attn_weight = query @ key.transpose(-2, -1)
        if attn_mask is not None:
            attn_weight = torch.masked_fill(attn_weight, attn_mask, -torch.inf)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=self.training)
        return attn_weight @ value

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.qkv.reset_parameters()
        self.pos_encoding.reset_parameters()
        self.out_proj.reset_parameters()