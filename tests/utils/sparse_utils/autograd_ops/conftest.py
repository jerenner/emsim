import pytest
import torch

from .constants import (
    BATCH_SIZE,
    SPARSE_DIM_1,
    SPARSE_DIM_2,
    SPARSE_DIM_3,
    EMBED_DIM,
    N_KEYS_PER_QUERY,
    ALWAYS_SPECIFIED,
    ALWAYS_UNSPECIFIED,
)


@pytest.fixture
def setup_sparse_tensor(device):
    """Create a 4D sparse tensor with vectorized operations."""
    # Create coordinates spanning all dimensions
    grid = torch.meshgrid(
        torch.arange(BATCH_SIZE, device=device),
        torch.arange(SPARSE_DIM_1, device=device),
        torch.arange(SPARSE_DIM_2, device=device),
        torch.arange(SPARSE_DIM_3, device=device),
        indexing="ij",
    )
    coords = torch.stack(grid, dim=-1).reshape(-1, 4)

    # Filter out unspecified points with vectorized operations
    mask_per_unspecified = (
        coords.unsqueeze(1) == ALWAYS_UNSPECIFIED.to(device).unsqueeze(0)
    ).all(dim=2)
    match_any_unspecified = mask_per_unspecified.any(dim=1)
    valid_coords = coords[~match_any_unspecified]

    # Sample a random subset
    num_samples = max(50, int(valid_coords.shape[0] * 0.15))  # Fixed 15% sample rate
    indices = torch.randperm(valid_coords.shape[0])[:num_samples]
    sampled_coords = valid_coords[indices]

    # Add specified test points
    final_coords = torch.cat([sampled_coords, ALWAYS_SPECIFIED.to(device)], dim=0)

    # Create sparse tensor
    values = torch.randn(
        final_coords.shape[0],
        EMBED_DIM,
        dtype=torch.double,
        requires_grad=True,
        device=device,
    )
    return torch.sparse_coo_tensor(
        final_coords.t(),
        values,
        (BATCH_SIZE, SPARSE_DIM_1, SPARSE_DIM_2, SPARSE_DIM_3, EMBED_DIM),
    ).coalesce()


@pytest.fixture
def setup_linear_index_tensor(setup_sparse_tensor, device):
    """Create index tensor for linear mapping tests with each index having a 50% chance
    of being from specified indices and 50% chance of being random."""
    # Get indices from the sparse tensor
    sparse_indices = setup_sparse_tensor.indices().t().contiguous()

    # Assert that we have enough sparse indices for meaningful testing
    assert len(sparse_indices) > 0, "Sparse tensor has no indices for testing"

    # Determine total number of indices to generate (excluding test points)
    total_indices = 50
    num_test_points = len(ALWAYS_SPECIFIED) + len(ALWAYS_UNSPECIFIED)
    num_random_indices = total_indices - num_test_points

    # Create a batch-to-indices mapping for quick lookup
    batch_to_sparse_indices = {
        b: sparse_indices[sparse_indices[:, 0] == b] for b in range(BATCH_SIZE)
    }

    # Generate indices with 50% probability of being specified vs random
    random_indices = torch.zeros(num_random_indices, 4, dtype=torch.long, device=device)

    # First assign random batch indices
    random_indices[:, 0].random_(0, BATCH_SIZE)

    # For each index, decide whether to use specified or random values
    use_specified = torch.rand(num_random_indices, device=device) < 0.5

    # Default all spatial dimensions to random values
    random_indices[:, 1].random_(0, SPARSE_DIM_1)
    random_indices[:, 2].random_(0, SPARSE_DIM_2)
    random_indices[:, 3].random_(0, SPARSE_DIM_3)

    # Vectorized replacement of specified indices
    # Process each batch group separately
    for batch_idx in range(BATCH_SIZE):
        # Find indices that should use specified values from this batch
        batch_mask = (random_indices[:, 0] == batch_idx) & use_specified
        num_to_replace = batch_mask.sum().item()

        if num_to_replace > 0 and batch_idx in batch_to_sparse_indices:
            batch_specified = batch_to_sparse_indices[batch_idx]
            if len(batch_specified) > 0:
                # Sample indices with replacement
                sampled_idx = torch.randint(0, len(batch_specified), (num_to_replace,))
                sampled_specified = batch_specified[sampled_idx]

                # Update all matching indices at once
                random_indices[batch_mask, 1:] = sampled_specified[:, 1:]

    # Combine with test points
    combined_indices = torch.cat(
        [
            random_indices,
            ALWAYS_SPECIFIED.to(device),  # Test points known to be in the sparse tensor
            ALWAYS_UNSPECIFIED.to(device),  # Test points known to be unspecified
        ],
        dim=0,
    )

    # Sort by batch dimension for consistent batch-wise processing
    _, sort_indices = torch.sort(combined_indices[:, 0])
    return combined_indices[sort_indices].contiguous()


@pytest.fixture
def setup_attention_index_tensor(setup_sparse_tensor, device):
    """Create attention index tensor with a mixture of specified and random indices."""
    # Get indices from the sparse tensor and ensure contiguous memory layout
    sparse_indices = setup_sparse_tensor.indices().t().contiguous()

    # Create random regular queries
    n_queries = 2
    index_tensor = torch.zeros(
        n_queries, N_KEYS_PER_QUERY, 4, dtype=torch.long, device=device
    )

    # Assign random batch indices and spatial dimensions
    query_batch_indices = torch.randint(0, BATCH_SIZE, (n_queries,), device=device)
    index_tensor[:, :, 0] = query_batch_indices.unsqueeze(1)
    index_tensor[:, :, 1].random_(0, SPARSE_DIM_1)
    index_tensor[:, :, 2].random_(0, SPARSE_DIM_2)
    index_tensor[:, :, 3].random_(0, SPARSE_DIM_3)

    # Randomly decide which keys will use specified indices (50% probability)
    use_specified = torch.rand(n_queries, N_KEYS_PER_QUERY) < 0.5

    # Pre-compute a dictionary mapping each batch to its sparse indices
    # This avoids repeatedly filtering the sparse tensor for each query
    batch_to_sparse_indices = {
        b: sparse_indices[sparse_indices[:, 0] == b] for b in range(BATCH_SIZE)
    }

    # Replace random indices with specified ones where appropriate
    for q in range(n_queries):
        b = query_batch_indices[q].item()
        batch_specified = batch_to_sparse_indices.get(b, None)

        # Skip if no specified indices for this batch or no keys to replace
        if batch_specified is None or len(batch_specified) == 0:
            continue

        # Get keys to replace and count
        specified_key_mask = use_specified[q]
        num_specified_keys = specified_key_mask.sum().item()

        if num_specified_keys > 0:
            # Sample indices with replacement from the known specified indices
            idx = torch.randint(0, len(batch_specified), (num_specified_keys,))
            index_tensor[q, specified_key_mask, 1:] = batch_specified[idx, 1:]

    # Create test case queries with predefined patterns
    test_queries = []

    # For each batch, create two special test queries
    for b in range(BATCH_SIZE):
        # Query 1: First key is a known specified index, rest are random
        q1 = torch.zeros(N_KEYS_PER_QUERY, 4, dtype=torch.long, device=device)
        q1[:, 0] = b  # Set batch index
        q1[:, 1:].random_(0, max(SPARSE_DIM_1, SPARSE_DIM_2, SPARSE_DIM_3))
        q1[0, 1:] = ALWAYS_SPECIFIED[b, 1:].to(
            device
        )  # First key is the specified point

        # Query 2: All keys point to a known unspecified location
        q2 = torch.zeros(N_KEYS_PER_QUERY, 4, dtype=torch.long, device=device)
        q2[:, 0] = b  # Set batch index
        q2[:, 1:] = (
            ALWAYS_UNSPECIFIED[b, 1:].to(device).unsqueeze(0)
        )  # All keys are unspecified

        test_queries.extend([q1, q2])

    # Combine regular and test queries along the query dimension (dim=0)
    # index_tensor shape: [n_queries, N_KEYS_PER_QUERY, 4]
    # torch.stack(test_queries) shape: [2*BATCH_SIZE, N_KEYS_PER_QUERY, 4]
    combined_tensor = torch.cat([index_tensor, torch.stack(test_queries)], dim=0)
    _, sort_indices = torch.sort(combined_tensor[:, 0, 0])

    return combined_tensor[sort_indices].contiguous()
