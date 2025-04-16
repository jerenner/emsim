from typing import Optional, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from emsim.utils.sparse_utils.indexing.script_funcs import gather_and_mask
from emsim.utils.sparse_utils.misc import sparse_tensor_to_dense_with_mask
from emsim.utils.sparse_utils.ops.subset_attn.autograd_helpers import (
    project_kv,
)
from emsim.utils.sparse_utils.ops.subset_attn.rotary_encoding import (
    calculate_rope,
    rotate_embeddings,
)


def traceable_subset_attention(
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
    batch_kv_projection: bool = True,
    return_extended_outputs: bool = False,
):
    """Traceable implementation of subset attention using standard Pytorch ops.

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

    if batch_kv_projection:
        keys, values = project_kv(
            selected, key_weight, value_weight, key_bias, value_bias
        )
    else:
        # Project keys and values separately (no batching)
        keys = F.linear(selected, key_weight, key_bias)
        values = F.linear(selected, value_weight, value_bias)

    # Split heads
    queries = query_tensor.view(n_queries, n_heads, head_dim)
    keys = keys.view(n_queries, n_keys_per_query, n_heads, head_dim)
    values = values.view(n_queries, n_keys_per_query, n_heads, head_dim)

    if key_positions is not None:
        assert rope_freqs is not None
        assert key_rope_encoding is None
        key_rope_encoding = calculate_rope(key_positions, rope_freqs)

    # Apply rotary position encoding if provided
    if key_rope_encoding is not None:
        keys = rotate_embeddings(keys, key_rope_encoding, needs_autograd=True)

    # (n_queries, n_keys_per_query, n_heads, head_dim) ->
    # (n_queries, n_heads, n_keys_per_query, head_dim)
    keys = keys.transpose(1, 2).contiguous()
    values = values.transpose(1, 2).contiguous()

    # Calculate attention scores

    attn_scores = torch.matmul(
        queries.unsqueeze(-2) * scale_factor,  # (n_queries, n_heads, 1, head_dim)
        keys.transpose(-1, -2),  # (n_queries, n_heads, head_dim, n_keys_per_query)
    ).squeeze(-2)
    # attn_scores: (n_queries, n_heads, n_keys_per_query)

    # Apply masking and softmax
    assert is_specified_mask.shape == (n_queries, n_keys_per_query)
    attn_scores = attn_scores.masked_fill(~is_specified_mask.unsqueeze(1), -torch.inf)
    attn_weights = attn_scores.softmax(-1)
    attn_weights = attn_weights.nan_to_num(0.0)

    attn_weights = F.dropout(attn_weights, dropout_p, training)

    # Apply attention weights to values
    attn_output = torch.matmul(
        attn_weights.unsqueeze(-2),  # (n_queries, n_heads, 1, n_keys_per_query)
        values,  # (n_queries, n_heads, n_keys_per_query, head_dim)
    ).squeeze(-2)
    # output: (n_queries, n_heads, head_dim)

    # Reshape output
    attn_output = attn_output.reshape(n_queries, embed_dim)

    if not return_extended_outputs:
        return attn_output
    else:
        return {
            "queries": queries,  # n_queries, n_heads, head_dim
            "keys": keys.transpose(
                1, 2
            ),  # n_queries, n_keys_per_query, n_heads, head_dim
            "values": values.transpose(1, 2),
            "is_specified_mask": is_specified_mask,
            "attn_scores": attn_scores,
            "attn_weights": attn_weights,
            "attn_output": attn_output.view(n_queries, n_heads, head_dim),
            "key_positions": key_positions,
            "key_rope_encoding": key_rope_encoding,
            "rope_freqs": rope_freqs,
        }


def traceable_batched_attention(
    query_tensor: Tensor,
    n_heads: int,
    source_tensor: Tensor,
    attn_mask: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    key_bias: Tensor,
    value_bias: Tensor,
    key_rope_encoding: Optional[Tensor] = None,
    key_positions: Optional[Tensor] = None,
    rope_freqs: Optional[Tensor] = None,
    scale_factor: Optional[float] = None,
    dropout_p: float = 0.0,
    training: bool = True,
    return_extended_outputs: bool = False,
):
    batch_size, n_queries, embed_dim = query_tensor.size()
    n_keys = source_tensor.size(1)
    head_dim = embed_dim // n_heads

    keys, values = project_kv(
        source_tensor, key_weight, value_weight, key_bias, value_bias
    )

    # (batch_size, seq_len, n_heads, head_dim)
    queries = query_tensor.reshape(batch_size, n_queries, n_heads, head_dim)
    keys = keys.reshape(batch_size, n_keys, n_heads, head_dim)
    values = values.reshape(batch_size, n_keys, n_heads, head_dim)

    # Compute Rope if needed
    if key_positions is not None:
        assert rope_freqs is not None
        assert key_rope_encoding is None
        key_rope_encoding = calculate_rope(key_positions, rope_freqs)

    # Apply Rope if provided
    if key_rope_encoding is not None:
        keys = rotate_embeddings(keys, key_rope_encoding, needs_autograd=True)

    # (batch_size, n_heads, seq_len, head_dim)
    queries = queries.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    # (batch_size, n_queries, n_keys) -> (batch_size, 1, n_queries, n_keys)
    attn_mask = attn_mask.unsqueeze(1)

    attn_output = F.scaled_dot_product_attention(
        queries,
        keys,
        values,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
        scale=scale_factor,
    )

    # (batch_size, n_queries, n_heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()

    # (batch_size, n_queries, embed_dim)
    attn_output = attn_output.view(batch_size, n_queries, embed_dim)
    if not return_extended_outputs:
        return attn_output
    else:
        return {
            "queries": queries.transpose(1, 2),
            "keys": keys.transpose(1, 2),
            "values": values.transpose(1, 2),
            "attn_mask": attn_mask,
            "attn_output": attn_output.view(batch_size, n_queries, n_heads, head_dim),
            "key_positions": key_positions,
            "key_rope_encoding": key_rope_encoding,
            "rope_freqs": rope_freqs,
        }


def prep_batched_attention(inputs: dict[str, Any]):
    """Preps the inputs for traceable_batched_attention using the dict
    from attention_inputs
    """
    queries = inputs["batched_query_tensor"]
    bsz, n_queries, embed_dim = queries.shape
    n_heads = inputs["n_heads"]

    source_tensor, source_mask = sparse_tensor_to_dense_with_mask(
        inputs["sparse_tensor"]
    )

    # combine masks
    attn_mask = inputs["attn_mask"]  # [batch, query, height, width, level]
    if attn_mask.is_sparse:
        attn_mask = attn_mask.to_dense()

    # [batch, height, width, level] -> [batch, query, height, width, level]
    source_mask = source_mask.unsqueeze(1).expand_as(attn_mask)

    combined_mask = torch.logical_and(attn_mask, source_mask)
    combined_mask = combined_mask.view(bsz, n_queries, -1)  # [batch, query, h*w*l]

    # flatten source tensor
    source_tensor = source_tensor.view(bsz, -1, embed_dim)  # [batch, h*w*l, embed_dim]
    assert combined_mask.size(-1) == source_tensor.size(1)

    key_weight, value_weight = inputs["key_weight"], inputs["value_weight"]
    key_bias, value_bias = inputs["key_bias"], inputs["value_bias"]

    key_rope_encoding = inputs["batched_key_rope_encoding"]
    key_positions = inputs["batched_key_positions"]
    rope_freqs = inputs["rope_freqs"]

    scale_factor = inputs["scale_factor"]
    dropout_p = inputs["dropout_p"]
    training = inputs["training"]

    return {
        "query_tensor": queries,
        "n_heads": n_heads,
        "source_tensor": source_tensor,
        "attn_mask": combined_mask,
        "key_weight": key_weight,
        "value_weight": value_weight,
        "key_bias": key_bias,
        "value_bias": value_bias,
        "key_rope_encoding": key_rope_encoding,
        "key_positions": key_positions,
        "rope_freqs": rope_freqs,
        "scale_factor": scale_factor,
        "dropout_p": dropout_p,
        "training": training,
    }
