import pytest
import torch

from ..constants import (
    BATCH_SIZE,
    SPARSE_DIM_1,
    SPARSE_DIM_2,
    SPARSE_DIM_3,
    N_KEYS_PER_QUERY,
    ALWAYS_SPECIFIED,
    ALWAYS_UNSPECIFIED,
)


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
