from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from emsim.utils.sparse_utils.base import _gather_and_mask


def traceable_sparse_attention(
    query_tensor: Tensor,
    n_heads: int,
    sparse_tensor_values: Tensor,
    index_search: Tensor,
    is_specified_mask: Tensor,
    Wk: Tensor,
    Wv: Tensor,
    bias_k: Optional[Tensor] = None,
    bias_v: Optional[Tensor] = None,
    key_pos_encoding: Optional[Tensor] = None,
    scale_factor: Optional[float] = None,
):
    """Traceable implementation of sparse attention using standard PyTorch ops.

    This implementation avoids the memory optimizations of the custom op by using
    straightforward operations that are fully traceable by PyTorch autograd.
    """
    n_queries = query_tensor.size(0)
    embed_dim = query_tensor.size(1)
    n_keys_per_query = index_search.size(1)
    head_dim = embed_dim // n_heads

    if scale_factor is None:
        scale_factor = embed_dim ** (-1/2)

    # Gather values using the same helper as the custom op
    selected = _gather_and_mask(
        sparse_tensor_values, index_search, ~is_specified_mask, mask_inplace=False
    )

    # Project keys and values separately (no batching)
    k = F.linear(selected, Wk, bias_k)
    v = F.linear(selected, Wv, bias_v)

    # Split heads
    q = query_tensor.view(n_queries, n_heads, head_dim)
    k = k.view(n_queries, n_keys_per_query, n_heads, head_dim)
    v = v.view(n_queries, n_keys_per_query, n_heads, head_dim)

    # Move head dim forward
    q = q.transpose(-2, -3).contiguous()  # (n_heads, n_queries, head_dim)
    k = k.permute(2, 0, 1, 3).contiguous()  # (n_heads, n_queries, n_keys_per_query, head_dim)
    v = v.permute(2, 0, 1, 3).contiguous()  # (n_heads, n_queries, n_keys_per_query, head_dim)

    # Apply rotary position encoding if provided
    if key_pos_encoding is not None:
        key_pos_encoding = key_pos_encoding.reshape(
            n_queries, n_keys_per_query, n_heads, head_dim
        )
        key_pos_encoding = key_pos_encoding.permute(2, 0, 1, 3).contiguous()

        # Convert to complex and apply rotation
        k_complex = torch.view_as_complex(k.view(*k.shape[:-1], head_dim // 2, 2))
        key_pos_complex = torch.view_as_complex(
            key_pos_encoding.view(*key_pos_encoding.shape[:-1], head_dim // 2, 2)
        )

        # Complex multiplication for RoPE
        k_rotated = k_complex * key_pos_complex
        k = torch.view_as_real(k_rotated).reshape_as(k)

    # Calculate attention scores
    attn_scores = torch.matmul(
        q.unsqueeze(-2) * scale_factor,  # (n_heads, n_queries, 1, head_dim)
        k.transpose(-1, -2)              # (n_heads, n_queries, head_dim, n_keys_per_query)
    ).squeeze(-2)                        # (n_heads, n_queries, n_keys_per_query)

    # Apply masking and softmax
    attn_scores = attn_scores.masked_fill(~is_specified_mask, -torch.inf)
    attn_weights = attn_scores.softmax(-1)
    attn_weights = attn_weights.nan_to_num(0.0)

    # Apply attention weights to values
    output = torch.matmul(
        attn_weights.unsqueeze(-2),  # (n_heads, n_queries, 1, n_keys_per_query)
        v                            # (n_heads, n_queries, n_keys_per_query, head_dim)
    ).squeeze(-2)                    # (n_heads, n_queries, head_dim)

    # Reshape output
    output = output.transpose(-2, -3)  # (n_queries, n_heads, head_dim)
    output = output.reshape(n_queries, embed_dim)

    return output
