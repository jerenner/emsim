from typing import Optional, Union, Literal

import math
import numpy as np
from torch import Tensor
import torch

from emsim.utils.sparse_utils.indexing.script_funcs import get_sparse_index_mapping
from emsim.utils.sparse_utils.batching import remove_batch_dim_and_concat


def attention_inputs(
    n_queries: Union[int, list[int]] = 4,
    embed_dim: int = 16,
    n_heads: int = 4,
    n_keys_per_query: int = 5,
    stacked_query_format: bool = True,  # set to False for functions that expect a batch dim
    use_biases: bool = True,
    use_rope: Union[
        Literal["none"], Literal["precomputed", Literal["from_freqs"]]
    ] = "none",  # none, precomputed, from_freqs
    position_dim: int = 2,
    n_freq_groups: int = 1,
    sparse_height: int = 8,
    sparse_width: int = 8,
    sparse_levels: int = 4,
    sparsity: float = 0.9,
    index_hit_rate: float = 0.75,
    unspecified_query_indices: Optional[Union[int, list[int]]] = None,
    use_2d_sparse_features: bool = False,
    generate_linear_sparse_tensor_directly: bool = False,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    dropout_p: float = 0.1,
    training: bool = True,
    seed: Optional[int] = None,
    **kwargs,
):
    """Generate test inputs for subset_attn"""

    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if isinstance(n_queries, int):
        n_queries = [n_queries]

    # Generate padding mask for converting from batched to stacked format
    query_padding_mask = torch.zeros(
        len(n_queries), max(n_queries), dtype=torch.bool, device=device
    )
    for b, n_queries_b in enumerate(n_queries):
        query_padding_mask[b, n_queries_b:] = True

    query_batch_offsets = torch.tensor(np.cumsum([0] + n_queries)[:-1], device=device)

    # Generate the sparse tensor inputs, either in the form of the sparse tensor
    # itself or in the form of the outputs of get_sparse_index_mapping directly
    if not generate_linear_sparse_tensor_directly:
        sparse_tensor, index_tensor = create_sparse_and_index_tensor(
            n_queries=n_queries,
            height=sparse_height,
            width=sparse_width,
            levels=sparse_levels,
            embed_dim=embed_dim,
            n_keys_per_query=n_keys_per_query,
            sparsity=sparsity,
            index_hit_rate=index_hit_rate,
            use_2d_features=use_2d_sparse_features,
            n_heads=n_heads,
            unspecified_query_indices=unspecified_query_indices,
            device=device,
            dtype=dtype,
            seed=None,  # seeding done in the current function
        )

        stacked_index_tensor, query_batch_offsets_2 = remove_batch_dim_and_concat(
            index_tensor, query_padding_mask
        )
        assert torch.equal(query_batch_offsets, query_batch_offsets_2)  # sanity check

        linear_index_tensor, is_specified_mask = get_sparse_index_mapping(
            sparse_tensor, stacked_index_tensor
        )
        sparse_tensor_values = sparse_tensor.values()

    else:
        if stacked_query_format:
            raise ValueError(
                "stacked query format incompatible with direct generation of linear "
                "sparse tensor"
            )
        total_spatial_indices = (
            len(n_queries) * sparse_height * sparse_width * sparse_levels
        )
        sparse_tensor_values, linear_index_tensor, is_specified_mask = (
            create_linear_sparse_values_and_index_tensor_directly(
                num_sparse_values=math.ceil(total_spatial_indices) * sparsity,
                n_queries=n_queries,
                n_keys_per_query=n_keys_per_query,
                embed_dim=embed_dim,
                unspecified_prob=1 - index_hit_rate,
                device=device,
                dtype=dtype,
            )
        )
        sparse_tensor = None
        index_tensor = None
        query_padding_mask = None
        stacked_index_tensor = None

    # Ensure embed_dim is divisible by n_heads
    assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
    head_dim = embed_dim // n_heads

    # Ensure head_dim is even for RoPE
    if use_rope != "none":
        assert head_dim % 2 == 0, "head_dim must be even to use RoPE"

    # Create query tensor: in batched and padded format
    query_tensor = torch.randn(
        len(n_queries), max(n_queries), embed_dim, device=device, dtype=dtype
    )
    stacked_query_tensor, query_batch_offsets_3 = remove_batch_dim_and_concat(
        query_tensor, query_padding_mask
    )
    assert torch.equal(query_batch_offsets, query_batch_offsets_3)  # sanity check

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

    stacked_key_rope_encoding: Optional[Tensor] = None
    stacked_key_positions: Optional[Tensor] = None

    if use_rope == "precomputed":
        # Precomputed RoPE encoding
        key_rope_encoding = torch.randn(
            len(n_queries),
            max(n_queries),
            n_keys_per_query,
            n_heads,
            head_dim // 2,
            device=device,
            dtype=dtype,
        )
        stacked_key_rope_encoding, _ = remove_batch_dim_and_concat(
            key_rope_encoding, query_padding_mask
        )
    elif use_rope == "from_freqs":
        # On-the-fly RoPE encoding with key positions and frequencies
        key_positions = torch.randn(
            len(n_queries),
            max(n_queries),
            n_keys_per_query,
            position_dim,
            device=device,
            dtype=dtype,
        )
        stacked_key_positions, _ = remove_batch_dim_and_concat(
            key_positions, query_padding_mask
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

    # reset seed
    if seed is not None:
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)

    return {
        "query_tensor": stacked_query_tensor if stacked_query_format else query_tensor,
        "sparse_tensor": sparse_tensor,
        "index_tensor": stacked_index_tensor if stacked_query_format else index_tensor,
        "query_padding_mask": query_padding_mask,
        "query_batch_offsets": query_batch_offsets,
        "n_heads": n_heads,
        "sparse_tensor_values": sparse_tensor_values,
        "linear_index_tensor": linear_index_tensor,
        "is_specified_mask": is_specified_mask,
        "key_weight": key_weight,
        "value_weight": value_weight,
        "key_bias": key_bias,
        "value_bias": value_bias,
        "key_rope_encoding": (
            stacked_key_rope_encoding if stacked_query_format else key_rope_encoding
        ),
        "key_positions": (
            stacked_key_positions if stacked_query_format else key_positions
        ),
        "rope_freqs": rope_freqs,
        "scale_factor": scale_factor,
        "dropout_p": dropout_p,
        "training": training,
        "metadata": {
            "n_queries": n_queries,
            "embed_dim": embed_dim,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "n_keys_per_query": n_keys_per_query,
            "use_biases": use_biases,
            "use_rope": use_rope,
            "position_dim": position_dim,
            "n_freq_groups": n_freq_groups,
            "unspecified_query_indices": unspecified_query_indices,
            "device": device,
            "dtype": dtype,
        },
    }


def create_sparse_and_index_tensor(
    n_queries: Union[int, list[int]] = 4,
    height: int = 8,
    width: int = 8,
    levels: int = 4,
    embed_dim: int = 16,
    n_keys_per_query: int = 8,
    sparsity: float = 0.9,  # Proportion of empty entries in the sparse tensor
    index_hit_rate: float = 0.75,  # Proportion of indexed values that exist in the sparse tensor
    use_2d_features: bool = False,  # If True, use (n_heads, head_dim) for feature dims
    n_heads: Optional[int] = None,
    unspecified_query_indices: Optional[Union[int, list[int]]] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Create a sparse tensor and index tensor for testing batch_sparse_index_subset_attn.

    Args:
        batch_size (int): Number of batches
        height (int): Height of the spatial grid
        width (int): Width of the spatial grid
        levels (int): Number of levels in the hierarchy
        embed_dim (int): Feature dimension
        n_queries (Union[int, list[int]]): Number of queries per batch. Either an int, which
            means a batch_size of 1, or a list of ints with length batch_size.
            For batch_size > 1, the index_tensor will have a pad value of -1 for elements
            index_tensor[i, n_queries_i:]
        n_keys_per_query (int): Number of keys per query
        sparsity (float): Proportion of empty entries in the sparse tensor (0.0-1.0)
        index_hit_rate (float): Proportion of indexed values that exist in the sparse tensor (0.0-1.0)
        use_2d_features (bool): If True, use (n_heads, head_dim) for feature dimensions
        n_heads (int): Number of attention heads (required if use_2d_features=True)
        unspecified_query_indices (Union[int, list[int]], Optional): If given, the
            indicated queries will
        device (Union[str, torch.device]): Device to create tensors on
        dtype (torch.dtype): Data type for the values
        seed (int): Random seed for reproducibility

    Returns:
        tuple[Tensor, Tensor]: The sparse tensor and index tensor
            - sparse_tensor will be of dimension [batch, height, width, levels, embed_dim]
                or [batch, height, width, levels, n_heads, head_dim], with the first 4
                dimensions being sparse dimensions and the last 1 or 2 being dense dims
            - index_tensor will be of dimension
                [batch_size, n_queries_per_batch, n_keys_per_query, 4], with padded and
                designated-unspecified queries having pad values of -1. Note that this
                format has an extra leading batch dimension compared to the stacked-batches
                approach in most of the rest of the code.
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Validate inputs
    if use_2d_features:
        if n_heads is None:
            raise ValueError("n_heads must be provided when use_2d_features=True")
        assert embed_dim % n_heads == 0
        feature_size = (n_heads, embed_dim // n_heads)
    else:
        feature_size = (embed_dim,)

    batch_size = len(n_queries)

    if isinstance(n_queries, int):
        n_queries = [n_queries]
    max_queries = max(n_queries)

    # Calculate the total number of elements in the spatial dimensions
    total_spatial_elements = batch_size * height * width * levels

    # Ensure sparsity is valid
    max_allowed_density = 0.99  # Leave at least 1% of indices as "misses"
    actual_density = min(1.0 - sparsity, max_allowed_density)

    # Decide how many elements will be non-zero based on sparsity
    nnz = int(total_spatial_elements * actual_density)
    nnz = max(1, nnz)  # Ensure at least one element

    # Create indices for all possible positions using meshgrid
    b_range = torch.arange(batch_size, device=device)
    h_range = torch.arange(height, device=device)
    w_range = torch.arange(width, device=device)
    l_range = torch.arange(levels, device=device)

    # Create meshgrid of all possible indices
    b_grid, h_grid, w_grid, l_grid = torch.meshgrid(
        b_range, h_range, w_range, l_range, indexing="ij"
    )

    # Reshape to get a tensor of shape (4, total_elements)
    all_indices = torch.stack(
        [
            b_grid.reshape(-1),
            h_grid.reshape(-1),
            w_grid.reshape(-1),
            l_grid.reshape(-1),
        ],
        dim=0,
    )

    # Randomly select indices for non-zero elements
    perm = torch.randperm(total_spatial_elements, device=device)
    selected_indices = all_indices[:, perm[:nnz]]

    # Generate random values for the sparse tensor
    values = torch.randn((nnz,) + feature_size, dtype=dtype, device=device)

    # Create the sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices=selected_indices,
        values=values,
        size=(batch_size, height, width, levels) + feature_size,
        device=device,
        dtype=dtype,
    ).coalesce()

    index_tensor_batch_shape = (batch_size, max_queries, n_keys_per_query)
    # Now create the index tensor
    # Initialize index tensor with random spatial and level indices
    index_tensor = torch.stack(
        [
            b_range.view(batch_size, 1, 1).expand(-1, max_queries, n_keys_per_query),
            torch.randint(0, height, index_tensor_batch_shape, device=device),
            torch.randint(0, width, index_tensor_batch_shape, device=device),
            torch.randint(0, levels, index_tensor_batch_shape, device=device),
        ],
        dim=-1,
    )

    is_hit = torch.rand(index_tensor_batch_shape, device=device) < index_hit_rate

    # For hits, sample from existing indices with replacement
    for b in range(batch_size):
        indices_in_this_batch_mask = selected_indices[0] == b
        n_indices_in_this_batch = indices_in_this_batch_mask.sum()

        hits_this_batch = is_hit[b]
        n_hits_this_batch = hits_this_batch.sum()

        sampling_indices = torch.randint(
            0, n_indices_in_this_batch, (n_hits_this_batch,), device=device
        )
        sampled_hits = selected_indices[:, indices_in_this_batch_mask][
            :, sampling_indices
        ]

        index_tensor[b, hits_this_batch] = sampled_hits.T

    # double-check all the hits were selected from nonzero indices correctly
    all_hits = index_tensor[is_hit]
    assert (all_hits.unsqueeze(-1) == selected_indices.unsqueeze(0)).all(1).any(1).all()

    # Fill in indices past the specified number of queries per batch with -1 pad value
    for i, n_queries_i in enumerate(n_queries):
        index_tensor[i, n_queries_i:] = -1

    # Fill in the designated unspecified_query_indices (queries with all misses) with -1
    # pad value if requested
    if unspecified_query_indices is not None:
        if isinstance(unspecified_query_indices, int):
            # same single unspecified query per batch
            unspecified_query_indices = [[unspecified_query_indices]] * batch_size
        elif isinstance(unspecified_query_indices[0], int):
            # same one or more unspecified queries per batch
            unspecified_query_indices = [unspecified_query_indices] * batch_size
        assert len(unspecified_query_indices) == len(n_queries) == batch_size
        for b, unspecified_b in enumerate(unspecified_query_indices):
            index_tensor[b, unspecified_b] = -1

    return sparse_tensor, index_tensor


def create_linear_sparse_values_and_index_tensor_directly(
    num_sparse_values: int = 20,
    n_queries: Union[int, list[int]] = 4,
    n_keys_per_query: int = 5,
    embed_dim: int = 8,
    unspecified_prob: float = 0.25,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    # Generate spares tensor values
    sparse_tensor_values = torch.randn(
        num_sparse_values, embed_dim, device=device, dtype=dtype
    )

    # Generate random is_specified_mask
    is_specified_mask = torch.rand(n_queries, n_keys_per_query, device=device)
    is_specified_mask = is_specified_mask > unspecified_prob

    # Generate random linear index tensor
    linear_index_tensor = torch.where(
        is_specified_mask,
        torch.randint(
            0, num_sparse_values, (n_queries, n_keys_per_query), device=device
        ),
        torch.zeros(n_queries, n_keys_per_query, device=device, dtype=torch.long),
    )

    return sparse_tensor_values, linear_index_tensor, is_specified_mask
