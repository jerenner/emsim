import torch
import pytest
from torch import Tensor

from emsim.networks.transformer.blocks.neighborhood_attn import (
    get_multilevel_neighborhoods,
    SparseNeighborhoodAttentionBlock,
)


@pytest.mark.cpu_and_cuda
class TestGetMultilevelNeighorhoods:
    def test_get_multilevel_neighborhoods_basic(self, device):
        """Test the basic functionality of get_multilevel_neighborhoods."""
        # Create 2D case with 2 query positions
        query_positions = torch.tensor(
            [
                [10.1, 20.2],  # Query position 1
                [15.3, 25.5],  # Query position 2
            ],
            device=device,
        )

        # Define 2 resolution levels
        level_shapes = torch.tensor(
            [
                [32, 32],  # Level 0 shape
                [16, 16],  # Level 1 shape
            ],
            device=device,
        )

        # Use simple neighborhood sizes
        neighborhood_sizes = [3, 5]

        # Call the function
        multilevel_neighborhood_indices, level_indices = get_multilevel_neighborhoods(
            query_positions, level_shapes, neighborhood_sizes
        )

        # Validate output shapes
        n_queries, position_dim = query_positions.shape
        expected_total_elements = 3**2 + 5**2  # 9 + 25 = 34 elements

        assert multilevel_neighborhood_indices.shape == (
            n_queries,
            expected_total_elements,
            position_dim,
        )
        assert level_indices.shape == (expected_total_elements,)

        # Check level indices are correct
        assert torch.all(level_indices[:9] == 0)  # First 9 elements are from level 0
        assert torch.all(level_indices[9:] == 1)  # Remaining elements are from level 1

    def test_get_multilevel_neighborhoods_validation(self, device):
        """Test error handling and input validation."""
        # Invalid query positions (should be 2D)
        invalid_query = torch.ones(3, 2, 2, device=device)  # 3D tensor
        valid_shapes = torch.tensor([[32, 32], [16, 16]], device=device)

        with pytest.raises((torch.jit.Error, ValueError)):
            get_multilevel_neighborhoods(invalid_query, valid_shapes)

        # Invalid neighborhood sizes (should be odd)
        valid_query = torch.ones(3, 2, device=device)
        even_sizes = [2, 4]  # Even sizes should raise error

        with pytest.raises(
            (torch.jit.Error, ValueError), match="Expected all odd neighborhood_sizes"
        ):
            get_multilevel_neighborhoods(valid_query, valid_shapes, even_sizes)

    def test_get_multilevel_neighborhoods_single_level(self, device):
        """Test with a single level for simplicity."""
        # 1D case with 1 query position
        query_positions = torch.tensor([[5.5]], device=device)

        # Single resolution level
        level_shapes = torch.tensor([[10]], device=device)

        # Use neighborhood size of 3
        neighborhood_sizes = [3]

        # Expected outputs for this simple case:
        # Query at position 5.5 gets floored to 5
        # With neighborhood size 3, we expect indices [4, 5, 6]
        expected_indices = torch.tensor([[[4], [5], [6]]], device=device)
        expected_level_indices = torch.tensor([0, 0, 0], device=device)

        # Call the function
        multilevel_neighborhood_indices, level_indices = get_multilevel_neighborhoods(
            query_positions, level_shapes, neighborhood_sizes
        )

        # Validate outputs match expected values
        assert torch.all(multilevel_neighborhood_indices == expected_indices)
        assert torch.all(level_indices == expected_level_indices)
