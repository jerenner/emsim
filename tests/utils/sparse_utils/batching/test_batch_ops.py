import pytest
import torch
from torch import Tensor

from emsim.utils.sparse_utils.batching.batch_ops import batch_topk
from emsim.utils.sparse_utils.batching.batch_utils import (
    batch_offsets_to_seq_lengths,
    seq_lengths_to_batch_offsets,
)


@pytest.fixture
def uniform_lengths_tensor(device) -> Tensor:
    """Create a tensor with uniform sequence lengths for testing."""
    # Create 3 sequences, each with 4 elements
    return torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        device=device
    )


@pytest.fixture
def uniform_lengths_offsets(device) -> Tensor:
    """Create batch offsets for uniform sequence lengths."""
    return torch.tensor([0, 4, 8, 12], device=device)


@pytest.fixture
def variable_lengths_tensor(device) -> Tensor:
    """Create a tensor with variable sequence lengths for testing."""
    return torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        device=device
    )


@pytest.fixture
def variable_lengths_offsets(device) -> Tensor:
    """Create batch offsets for variable sequence lengths."""
    return torch.tensor([0, 2, 5, 9], device=device)


@pytest.fixture
def multidim_tensor(device) -> Tensor:
    """Create a multi-dimensional tensor for testing."""
    return torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0]
        ],
        device=device
    )


@pytest.fixture
def multidim_offsets(device) -> Tensor:
    """Create batch offsets for multi-dimensional tensor."""
    return torch.tensor([0, 2, 4, 6], device=device)


@pytest.mark.cpu_and_cuda
class TestBatchTopK:
    def test_uniform_length_scalar_k(self, uniform_lengths_tensor, uniform_lengths_offsets, device):
        """Test basic functionality with uniform sequence lengths and scalar k."""
        # Get top 2 elements from each sequence
        topk_indices, topk_offsets = batch_topk(
            uniform_lengths_tensor, uniform_lengths_offsets, k=2
        )

        # Expected indices: [3, 2, 7, 6, 11, 10]
        # These are the indices of the 2 largest elements in each sequence
        expected_indices = torch.tensor([3, 2, 7, 6, 11, 10], device=device)
        expected_offsets = torch.tensor([0, 2, 4, 6], device=device)

        assert torch.equal(topk_indices, expected_indices)
        assert torch.equal(topk_offsets, expected_offsets)

        # Verify the values by indexing into the original tensor
        values = uniform_lengths_tensor[topk_indices]
        expected_values = torch.tensor([4.0, 3.0, 8.0, 7.0, 12.0, 11.0], device=device)
        assert torch.equal(values, expected_values)

    def test_variable_length_scalar_k(self, variable_lengths_tensor, variable_lengths_offsets, device):
        """Test with variable sequence lengths and scalar k."""
        # Get top 2 elements from each sequence (note: first sequence has only 2 elements)
        topk_indices, topk_offsets = batch_topk(
            variable_lengths_tensor, variable_lengths_offsets, k=2
        )

        # Expected indices: [1, 0, 4, 3, 8, 7]
        expected_indices = torch.tensor([1, 0, 4, 3, 8, 7], device=device)
        expected_offsets = torch.tensor([0, 2, 4, 6], device=device)

        assert torch.equal(topk_indices, expected_indices)
        assert torch.equal(topk_offsets, expected_offsets)

        # Verify values
        values = variable_lengths_tensor[topk_indices]
        expected_values = torch.tensor([2.0, 1.0, 5.0, 4.0, 9.0, 8.0], device=device)
        assert torch.equal(values, expected_values)

    def test_tensor_k(self, variable_lengths_tensor, variable_lengths_offsets, device):
        """Test with k as a tensor with different values per batch."""
        # Different k for each sequence: [1, 2, 3]
        k_tensor = torch.tensor([1, 2, 3], device=device)

        topk_indices, topk_offsets = batch_topk(
            variable_lengths_tensor, variable_lengths_offsets, k=k_tensor
        )

        # Expected indices: [1, 4, 3, 8, 7, 6]
        expected_indices = torch.tensor([1, 4, 3, 8, 7, 6], device=device)
        expected_offsets = torch.tensor([0, 1, 3, 6], device=device)

        assert torch.equal(topk_indices, expected_indices)
        assert torch.equal(topk_offsets, expected_offsets)

        # Verify values
        values = variable_lengths_tensor[topk_indices]
        expected_values = torch.tensor([2.0, 5.0, 4.0, 9.0, 8.0, 7.0], device=device)
        assert torch.equal(values, expected_values)

    def test_k_greater_than_length(self, variable_lengths_tensor, variable_lengths_offsets, device):
        """Test with k > sequence length - should clamp to sequence length."""
        topk_indices, topk_offsets = batch_topk(
            variable_lengths_tensor, variable_lengths_offsets, k=10
        )

        # All elements should be returned in each sequence
        expected_indices = torch.tensor([1, 0, 4, 3, 2, 8, 7, 6, 5], device=device)
        expected_offsets = torch.tensor([0, 2, 5, 9], device=device)

        assert torch.equal(topk_indices, expected_indices)
        assert torch.equal(topk_offsets, expected_offsets)

        # All values should be returned
        values = variable_lengths_tensor[topk_indices]
        assert torch.equal(values, variable_lengths_tensor)

    def test_smallest_elements(self, uniform_lengths_tensor, uniform_lengths_offsets, device):
        """Test getting smallest elements instead of largest."""
        topk_indices, topk_offsets = batch_topk(
            uniform_lengths_tensor, uniform_lengths_offsets, k=2, largest=False
        )

        # Expected indices: [0, 1, 4, 5, 8, 9]
        expected_indices = torch.tensor([0, 1, 4, 5, 8, 9], device=device)
        expected_offsets = torch.tensor([0, 2, 4, 6], device=device)

        assert torch.equal(topk_indices, expected_indices)
        assert torch.equal(topk_offsets, expected_offsets)

        # Verify values (smallest 2 from each sequence)
        values = uniform_lengths_tensor[topk_indices]
        expected_values = torch.tensor([1.0, 2.0, 5.0, 6.0, 9.0, 10.0], device=device)
        assert torch.equal(values, expected_values)

    def test_unsorted_elements(self, uniform_lengths_tensor, uniform_lengths_offsets, device):
        """Test with sorted=False - order may not be preserved."""
        topk_indices, topk_offsets = batch_topk(
            uniform_lengths_tensor, uniform_lengths_offsets, k=2, sorted=False
        )

        # Since order is not guaranteed, we sort the indices within each batch
        # and compare after sorting
        for i in range(len(topk_offsets) - 1):
            start, end = topk_offsets[i], topk_offsets[i+1]
            topk_indices[start:end] = torch.sort(topk_indices[start:end])[0]

        # Expected indices after sorting: [2, 3, 6, 7, 10, 11]
        expected_indices = torch.tensor([2, 3, 6, 7, 10, 11], device=device)

        assert torch.equal(topk_indices, expected_indices)

    def test_multidimensional_tensor(self, multidim_tensor, multidim_offsets, device):
        """Test with multi-dimensional tensor."""
        # Get top 1 element from each sequence using the first column
        topk_indices, topk_offsets = batch_topk(
            multidim_tensor[:, 0], multidim_offsets, k=1
        )

        # Expected indices: [1, 3, 5]
        expected_indices = torch.tensor([1, 3, 5], device=device)
        expected_offsets = torch.tensor([0, 1, 2, 3], device=device)

        assert torch.equal(topk_indices, expected_indices)
        assert torch.equal(topk_offsets, expected_offsets)

        # Verify the full tensor values
        values = multidim_tensor[topk_indices]
        expected_values = torch.tensor(
            [[2.0, 20.0], [4.0, 40.0], [6.0, 60.0]], device=device
        )
        assert torch.equal(values, expected_values)

    def test_empty_tensor(self, device):
        """Test with empty tensor."""
        empty_tensor = torch.tensor([], device=device)
        batch_offsets = torch.tensor([0], device=device)

        topk_indices, topk_offsets = batch_topk(empty_tensor, batch_offsets, k=2)

        assert topk_indices.shape[0] == 0
        assert torch.equal(topk_offsets, torch.tensor([0], device=device))

    def test_single_batch(self, device):
        """Test with a single batch."""
        tensor = torch.tensor([3.0, 1.0, 4.0, 2.0], device=device)
        batch_offsets = torch.tensor([0, 4], device=device)

        topk_indices, topk_offsets = batch_topk(tensor, batch_offsets, k=2)

        expected_indices = torch.tensor([2, 0], device=device)
        expected_offsets = torch.tensor([0, 2], device=device)

        assert torch.equal(topk_indices, expected_indices)
        assert torch.equal(topk_offsets, expected_offsets)

        # Verify values
        values = tensor[topk_indices]
        expected_values = torch.tensor([4.0, 3.0], device=device)
        assert torch.equal(values, expected_values)

    def test_different_dim(self, device):
        """Test topk along a different dimension."""
        tensor = torch.tensor([[1.0, 3.0], [4.0, 2.0], [6.0, 5.0]], device=device)
        batch_offsets = torch.tensor([0, 3], device=device)

        # Get top element along dim 1 (columns)
        topk_indices, topk_offsets = batch_topk(
            tensor, batch_offsets, k=1, dim=1
        )

        # This is trickier to test directly since we're doing topk on dim 1
        # Instead, let's verify by checking if the result gives us the correct values
        selected_values = torch.zeros_like(tensor)
        for i, idx in enumerate(topk_indices):
            row = i // 1  # Since k=1
            selected_values[row, idx % tensor.shape[1]] = 1  # Mark selected positions

        # We expect the largest element in each row to be selected
        expected_selections = torch.tensor([[0, 1], [1, 0], [1, 0]], device=device)
        assert torch.equal(selected_values.bool(), expected_selections.bool())

    def test_integration_with_helper_functions(self, device):
        """Test integration with seq_lengths_to_batch_offsets."""
        # Create sequence lengths and convert to batch offsets
        seq_lengths = torch.tensor([3, 4, 2], device=device)
        batch_offsets = seq_lengths_to_batch_offsets(seq_lengths)

        # Create tensor based on these sequence lengths
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device=device)

        # Get top 2 elements from each sequence
        topk_indices, topk_offsets = batch_topk(tensor, batch_offsets, k=2)

        # Convert topk_offsets back to sequence lengths
        result_seq_lengths = batch_offsets_to_seq_lengths(topk_offsets)
        expected_seq_lengths = torch.tensor([2, 2, 2], device=device)

        assert torch.equal(result_seq_lengths, expected_seq_lengths)
