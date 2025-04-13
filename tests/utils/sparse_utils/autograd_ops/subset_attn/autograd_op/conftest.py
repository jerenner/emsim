from typing import Optional, Union, Any, Literal

from hypothesis import strategies as st
import torch
from torch import Tensor
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

MAX_SEED = np.iinfo(np.int32).max


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
    seed: Optional[int] = None,
    **kwargs,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate inputs for testing GatherAndSubsetAttentionFunction with specific
    parameters."""

    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    try:
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
                position_dim,
                n_freq_groups,
                n_heads,
                head_dim // 2,
                device=device,
                dtype=torch.bool,
            )
            for pos_dim in range(position_dim):
                freq_mask[pos_dim, pos_dim % n_freq_groups] = True

            rope_freqs *= freq_mask

        # Random scale factor or None (defaults to 1/sqrt(embed_dim) in the function)
        scale_factor: Optional[float] = (
            torch.rand(1).item() if np.random.random() > 0.5 else None
        )

    finally:
        # reset seed back to what it was
        if seed is not None:
            torch.set_rng_state(torch_rng_state)
            np.random.set_state(np_rng_state)

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


def ordered_inputs(
    inputs: Union[dict[str, Any], tuple[dict[str, Any], dict[str, Any]]],
) -> tuple:
    if isinstance(inputs, tuple):
        inputs = inputs[0]

    return (
        inputs["query_tensor"],
        inputs["n_heads"],
        inputs["sparse_tensor_values"],
        inputs["index_tensor"],
        inputs["is_specified_mask"],
        inputs["key_weight"],
        inputs["value_weight"],
        inputs["key_bias"],
        inputs["value_bias"],
        inputs["key_rope_encoding"],
        inputs["key_positions"],
        inputs["rope_freqs"],
        inputs["scale_factor"],
    )


def set_requires_grad(inputs: dict[str, Any], tensor_names: Union[str, list[str]]):
    """Sets the requires_grad flag to True for specified tensors in the input dict"""
    modified_inputs = inputs.copy()
    if isinstance(tensor_names, str):
        tensor_names = [tensor_names]
    for name in tensor_names:
        if name in modified_inputs and modified_inputs[name] is not None:
            tensor: Tensor = modified_inputs[name].clone()
            modified_inputs[name] = tensor.requires_grad_(True)
    return modified_inputs


def filter_valid_tensor_names(
    use_rope: Union[Literal["none"], Literal["precomputed"], Literal["from_freqs"]],
    use_biases: bool,
) -> list[str]:
    """Filter tensor names based on the given parameters.

    Returns a list of tensor names that are valid for the given combination
    of use_rope and use_biases parameters.
    """
    # Start with all tensors
    valid_tensors = list(DIFFERENTIABLE_TENSOR_NAMES)

    if use_rope != "precomputed":
        # Remove key_rope_encoding if not using precomputed RoPE
        valid_tensors = [t for t in valid_tensors if t != "key_rope_encoding"]

    if use_rope != "from_freqs":
        # Remove position-based RoPE tensors if not computing RoPE from frequencies
        valid_tensors = [
            t for t in valid_tensors if t not in ["key_positions", "rope_freqs"]
        ]

    if not use_biases:
        # Remove bias tensors if not using biases
        valid_tensors = [
            t for t in valid_tensors if t not in ["key_bias", "value_bias"]
        ]

    return valid_tensors


def _draw_shared_attention_params(draw, min_requiring_grads: int = 0):
    """Helper function that does the drawing of base parameters for both strategies"""
    use_rope = draw(st.sampled_from(["none", "precomputed", "from_freqs"]))
    use_biases = draw(st.booleans())

    # Get valid tensor names for these parameters
    available_tensors = filter_valid_tensor_names(use_rope, use_biases)

    # Draw a non-empty subset of available tensors
    tensors_requiring_grads = draw(
        st.lists(
            st.sampled_from(available_tensors),
            min_size=min_requiring_grads,
            max_size=len(available_tensors),
            unique=True,
        )
    )

    # Sample seed
    seed = draw(st.integers(0, MAX_SEED))

    return {
        "use_rope": use_rope,
        "use_biases": use_biases,
        "tensors_requiring_grads": tensors_requiring_grads,
        "seed": seed,
    }


@st.composite
def simple_attention_input_configs(draw, min_requiring_grads: int = 1):
    """Hypothesis strategy for generating valid parameters for attention function
    tests."""
    return _draw_shared_attention_params(draw, min_requiring_grads)


@st.composite
def exhaustive_attention_input_configs(
    draw,
    dtypes: Union[torch.dtype, list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float,
        torch.double,
    ],
    min_requiring_grads: int = 0,
) -> dict[str, Any]:
    """Strategy that generates all of attention_inputs's input args.

    Args:
        dtypes (Union[torch.dtype, list[torch.dtype]]): Specific dtype or list of
            dtypes to sample from. For gradcheck tests, use torch.double for
            numerical stability. Defaults to both 16-bit floats, float, and double
        min_requiring_grads (int): Minimum number of input tensors to set as
            requiring gradients. For gradcheck tests, this should be at least 1.
            Defaults to 0.
    """
    base_params = _draw_shared_attention_params(draw, min_requiring_grads)

    n_queries = draw(st.integers(min_value=1, max_value=8))
    n_heads = draw(st.integers(min_value=1, max_value=4))

    # Make sure embed_dim is divisible by n_heads
    # and even for compatibility with RoPE
    head_dim = draw(st.integers(min_value=1, max_value=4).map(lambda x: x * 2))
    embed_dim = head_dim * n_heads

    n_keys_per_query = draw(st.integers(min_value=1, max_value=8))
    num_sparse_values = draw(
        st.integers(min_value=max(5, n_keys_per_query), max_value=32)
    )

    # These parameters matter only when use_rope="from_freqs"
    position_dim = draw(st.integers(min_value=1, max_value=4))
    n_freq_groups = draw(st.integers(min_value=1, max_value=position_dim))

    # Parameter for sparse attention
    unspecified_prob = draw(st.floats(min_value=0.0, max_value=0.5))

    # Decide if we want queries with all keys unspecified
    has_unspecified_queries = draw(st.booleans())
    unspecified_query_indices = None
    if has_unspecified_queries and n_queries > 0:
        num_unspecified = draw(st.integers(min_value=0, max_value=min(2, n_queries)))
        if num_unspecified > 0:
            unspecified_query_indices = draw(
                st.lists(
                    st.integers(min_value=0, max_value=n_queries - 1),
                    min_size=1,
                    max_size=num_unspecified,
                    unique=True,
                )
            )

    # Sample dtype
    dtypes = [dtypes] if isinstance(dtypes, torch.dtype) else dtypes
    dtype = draw(st.sampled_from(dtypes))


    return {
        "n_queries": n_queries,
        "embed_dim": embed_dim,
        "n_heads": n_heads,
        "n_keys_per_query": n_keys_per_query,
        "num_sparse_values": num_sparse_values,
        "position_dim": position_dim,
        "n_freq_groups": n_freq_groups,
        "unspecified_query_indices": unspecified_query_indices,
        "unspecified_prob": unspecified_prob,
        "dtype": dtype,
        "use_biases": base_params["use_biases"],
        "use_rope": base_params["use_rope"],
        "tensors_requiring_grads": base_params["tensors_requiring_grads"],
        "seed": base_params["seed"],
    }
