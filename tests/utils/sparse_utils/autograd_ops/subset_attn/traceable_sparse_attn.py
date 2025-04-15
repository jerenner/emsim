from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from emsim.utils.sparse_utils.indexing.script_funcs import gather_and_mask
from emsim.utils.sparse_utils.ops.subset_attn.rotary_encoding import (
    calculate_rope,
    rotate_embeddings,
)


def traceable_sparse_attention(
    query_tensor: Tensor,
    n_heads: int,
    sparse_tensor_values: Tensor,
    linear_index_tensor: Tensor,
    is_specified_mask: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    key_bias: Optional[Tensor] = None,
    value_bias: Optional[Tensor] = None,
    key_rope_encoding: Optional[Tensor] = None,
    key_positions: Optional[Tensor] = None,
    rope_freqs: Optional[Tensor] = None,
    scale_factor: Optional[float] = None,
    dropout_p: float = 0.0,
    training: bool = True,
):
    """Traceable implementation of sparse attention using standard Pytorch ops.

    This implementation avoids the memory optimizations of the custom op by using
    straightforward operations that are fully traceable by Pytorch autograd.
    """
    n_queries = query_tensor.size(0)
    embed_dim = query_tensor.size(1)
    n_keys_per_query = linear_index_tensor.size(1)
    head_dim = embed_dim // n_heads

    if scale_factor is None:
        scale_factor = embed_dim ** (-1 / 2)

    # Gather values using the same helper as the custom op
    selected = gather_and_mask(
        sparse_tensor_values, linear_index_tensor, is_specified_mask, mask_inplace=False
    )

    # Project keys and values separately (no batching)
    k = F.linear(selected, key_weight, key_bias)
    v = F.linear(selected, value_weight, value_bias)

    # Split heads
    q = query_tensor.view(n_queries, n_heads, head_dim)
    k = k.view(n_queries, n_keys_per_query, n_heads, head_dim)
    v = v.view(n_queries, n_keys_per_query, n_heads, head_dim)

    if key_positions is not None:
        assert rope_freqs is not None
        assert key_rope_encoding is None
        key_rope_encoding = calculate_rope(key_positions, rope_freqs)

    # Apply rotary position encoding if provided
    if key_rope_encoding is not None:
        k = rotate_embeddings(k, key_rope_encoding, needs_autograd=True)

    # Move head dim forward
    q = q.transpose(-2, -3).contiguous()  # (n_heads, n_queries, head_dim)
    k = k.permute(
        2, 0, 1, 3
    ).contiguous()  # (n_heads, n_queries, n_keys_per_query, head_dim)
    v = v.permute(
        2, 0, 1, 3
    ).contiguous()  # (n_heads, n_queries, n_keys_per_query, head_dim)

    # Calculate attention scores
    attn_scores = torch.matmul(
        q.unsqueeze(-2) * scale_factor,  # (n_heads, n_queries, 1, head_dim)
        k.transpose(-1, -2),  # (n_heads, n_queries, head_dim, n_keys_per_query)
    ).squeeze(
        -2
    )  # (n_heads, n_queries, n_keys_per_query)

    # Apply masking and softmax
    attn_scores = attn_scores.masked_fill(~is_specified_mask, -torch.inf)
    attn_weights = attn_scores.softmax(-1)
    attn_weights = attn_weights.nan_to_num(0.0)

    attn_weights = F.dropout(attn_weights, dropout_p, training)

    # Apply attention weights to values
    output = torch.matmul(
        attn_weights.unsqueeze(-2),  # (n_heads, n_queries, 1, n_keys_per_query)
        v,  # (n_heads, n_queries, n_keys_per_query, head_dim)
    ).squeeze(
        -2
    )  # (n_heads, n_queries, head_dim)

    # Reshape output
    output = output.transpose(-2, -3)  # (n_queries, n_heads, head_dim)
    output = output.reshape(n_queries, embed_dim)

    return output
