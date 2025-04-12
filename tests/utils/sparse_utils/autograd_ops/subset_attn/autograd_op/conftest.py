from typing import Optional, Union, Any

import torch
import numpy as np


DIFFERENTIABLE_TENSOR_NAMES = [
    "query_tensor",
    "sparse_tensor_values",
    "key_weight",
    "value_weight",
    "key_bias",
    "value_bias",
    "key_rope_encoding",
    "key_positions",
    "rope_freqs",
]


def attention_inputs(
    n_queries: int = 4,
    embed_dim: int = 8,
    n_heads: int = 2,
    n_keys_per_query: int = 4,
    num_sparse_values: int = 20,
    use_biases: bool = True,
    use_rope: Union[str, None] = "none",  # none, precomputed, from_freqs
    position_dim: int = 2,
    n_freq_groups: int = 1,
    unspecified_query_indices: Optional[Union[int, list[int]]] = None,
    unspecified_prob: float = 0.25,  # Probability of a key being unspecified
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate inputs for testing GatherAndSubsetAttentionFunction with specific
    parameters."""
    # Ensure embed_dim is divisible by n_heads
    assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
    head_dim = embed_dim // n_heads

    # Ensure head_dim is even for RoPE
    if use_rope != "none":
        assert head_dim % 2 == 0, "head_dim must be even to use RoPE"

    # Create query tensor and sparse tensor values
    query_tensor = torch.randn(n_queries, embed_dim, device=device, dtype=dtype)
    sparse_tensor_values = torch.randn(
        num_sparse_values, embed_dim, device=device, dtype=dtype
    )

    # Generate is_specified_mask randomly with given probability
    is_specified_mask = (
        torch.rand(n_queries, n_keys_per_query, device=device) > unspecified_prob
    )

    # Set up queries with all keys unspecified
    if unspecified_query_indices is not None:
        all_unspecified_indices = (
            torch.tensor(unspecified_query_indices, device=device, dtype=torch.long)
            .clamp(0, n_queries - 1)
            .unique()
        )

        is_specified_mask[all_unspecified_indices] = False

    # Create index tensor for sparse lookups
    index_tensor = torch.where(
        is_specified_mask,
        torch.randint(
            0, num_sparse_values, (n_queries, n_keys_per_query), device=device
        ),
        torch.zeros(n_queries, n_keys_per_query, device=device, dtype=torch.long),
    )

    # Create key and value weights
    key_weight = torch.randn(embed_dim, embed_dim, device=device, dtype=dtype)
    value_weight = torch.randn(embed_dim, embed_dim, device=device, dtype=dtype)

    # Create key and value biases if needed
    key_bias: Optional[torch.Tensor] = (
        torch.randn(embed_dim, device=device, dtype=dtype) if use_biases else None
    )
    value_bias: Optional[torch.Tensor] = (
        torch.randn(embed_dim, device=device, dtype=dtype) if use_biases else None
    )

    # Handle RoPE encodings
    key_rope_encoding: Optional[torch.Tensor] = None
    key_positions: Optional[torch.Tensor] = None
    rope_freqs: Optional[torch.Tensor] = None

    if use_rope == "precomputed":
        # Precomputed RoPE encoding
        key_rope_encoding = torch.randn(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim // 2,
            device=device,
            dtype=dtype,
        )
    elif use_rope == "from_freqs":
        # On-the-fly RoPE encoding with key positions and frequencies
        key_positions = torch.randn(
            n_queries, n_keys_per_query, position_dim, device=device, dtype=dtype
        )
        rope_freqs = torch.rand(
            position_dim,
            n_freq_groups,
            n_heads,
            head_dim // 2,
            device=device,
            dtype=dtype,
        )
        # Scale with random magnitude (0-1000) for variety
        rope_freqs *= torch.randint(1, 1000, (1,), device=device)

        # Simple case for freq_groups: one group per position dim
        freq_mask = torch.zeros(
            position_dim, n_freq_groups, device=device, dtype=torch.bool
        )
        for pos_dim in range(position_dim):
            freq_mask[pos_dim, pos_dim % n_freq_groups] = True

        rope_freqs *= freq_mask

    # Random scale factor or None (defaults to 1/sqrt(embed_dim) in the function)
    scale_factor: Optional[float] = (
        torch.rand(1).item() if np.random.random() > 0.5 else None
    )

    return {
        "query_tensor": query_tensor,
        "n_heads": n_heads,
        "sparse_tensor_values": sparse_tensor_values,
        "index_tensor": index_tensor,
        "is_specified_mask": is_specified_mask,
        "key_weight": key_weight,
        "value_weight": value_weight,
        "key_bias": key_bias,
        "value_bias": value_bias,
        "key_rope_encoding": key_rope_encoding,
        "key_positions": key_positions,
        "rope_freqs": rope_freqs,
        "scale_factor": scale_factor,
        "metadata": {
            "n_queries": n_queries,
            "embed_dim": embed_dim,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "n_keys_per_query": n_keys_per_query,
            "num_sparse_values": num_sparse_values,
            "use_biases": use_biases,
            "use_rope": use_rope,
            "position_dim": position_dim,
            "n_freq_groups": n_freq_groups,
            "unspecified_query_indices": unspecified_query_indices,
            "device": device,
            "dtype": dtype,
        },
    }
