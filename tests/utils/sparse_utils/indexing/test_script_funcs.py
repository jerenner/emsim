import pytest
import torch

# Import the functions to test
from emsim.utils.sparse_utils.indexing.script_funcs import (
    flatten_sparse_indices,
    gather_and_mask,
    get_sparse_index_mapping,
    linearize_sparse_and_index_tensors,
)


@pytest.mark.cpu_and_cuda
class TestFlattenedIndices:
    def test_flattened_indices_basic(self, device):
        """Test basic functionality of flattened_indices."""
        i = torch.tensor([[0, 0, 1], [1, 0, 0], [1, 1, 1], [2, 2, 0]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (3, 3, 2)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Flatten the first two dimensions
        new_indices, new_shape, offsets = flatten_sparse_indices(sparse_tensor, 0, 1)

        # Check results
        assert new_shape.shape[0] == 2  # Should be (9, 2)
        assert new_shape[0] == 9  # 3 * 3
        assert new_shape[1] == 2

        # Check flattening computation (for dims [d0, d1], linear index = d0*3 + d1)
        expected_linear_indices = i[0] * 3 + i[1]
        assert torch.allclose(new_indices[0], expected_linear_indices)
        assert torch.allclose(new_indices[1], i[2])

        # Check offsets - should be [3, 1] for flattening [d0, d1]
        expected_offsets = torch.tensor([3, 1], device=device)
        assert torch.allclose(offsets, expected_offsets)

    def test_flattened_indices_single_dim(self, device):
        """Test flattened_indices with just one dimension."""
        i = torch.tensor([[0, 0], [1, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (3, 3)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Flatten just the first dimension (no change expected)
        new_indices, new_shape, offsets = flatten_sparse_indices(sparse_tensor, 0, 0)

        assert torch.allclose(new_indices, i)
        assert torch.allclose(new_shape, torch.tensor(shape, device=device))
        assert offsets.shape[0] == 1
        assert offsets[0] == 1


@pytest.mark.cpu_and_cuda
class TestLinearizeSparseAndIndexTensors:
    def test_linearize_sparse_and_index_tensors(self, device):
        """Test basic functionality of linearize_sparse_and_index_tensors."""
        i = torch.tensor([[0, 0], [1, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (3, 3)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Create an index tensor
        index_tensor = torch.tensor([[0, 0], [1, 1], [2, 2]], device=device)

        # Linearize both tensors
        sparse_linear, index_linear = linearize_sparse_and_index_tensors(
            sparse_tensor, index_tensor
        )

        # Check output shapes
        assert sparse_linear.shape[0] == 4  # Number of non-zeros
        assert index_linear.shape[0] == 3  # Number of indices

        # Calculate expected linear indices (for 2D: row*ncols + col)
        expected_sparse_linear = i[0] * shape[1] + i[1]
        expected_index_linear = index_tensor[:, 0] * shape[1] + index_tensor[:, 1]

        assert torch.allclose(sparse_linear, expected_sparse_linear)
        assert torch.allclose(index_linear, expected_index_linear)

    def test_linearize_sparse_and_index_tensors_error(self, device):
        """Test error handling in linearize_sparse_and_index_tensors."""
        i = torch.tensor([[0, 0], [1, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (3, 3)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Create an index tensor with wrong last dimension
        index_tensor = torch.tensor([[0], [1], [2]], device=device)

        with pytest.raises((ValueError, torch.jit.Error), match="Expected last dim"):  # type: ignore
            linearize_sparse_and_index_tensors(sparse_tensor, index_tensor)


@pytest.mark.cpu_and_cuda
class TestGetSparseIndexMapping:
    def test_get_sparse_index_mapping(self, device):
        """Test basic functionality of get_sparse_index_mapping."""
        i = torch.tensor([[0, 0], [1, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (3, 3)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Create indices to look up - mix of existing and non-existing positions
        index_tensor = torch.tensor(
            [
                [0, 0],  # exists at position 0 with value 1.0
                [1, 0],  # exists at position 1 with value 2.0
                [1, 1],  # exists at position 2 with value 3.0
                [0, 1],  # doesn't exist
                [2, 2],  # exists at position 3 with value 4.0
            ],
            device=device,
        )

        # Get mapping
        indices, is_specified = get_sparse_index_mapping(sparse_tensor, index_tensor)

        # Check which indices were found
        expected_specified = torch.tensor(
            [True, True, True, False, True], device=device
        )
        assert torch.all(is_specified == expected_specified)

        # For existing indices, check correct mapping to values
        found_values = sparse_tensor.values()[indices[is_specified]]
        expected_values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        assert torch.allclose(found_values, expected_values)

    def test_get_sparse_index_mapping_out_of_bounds(self, device):
        """Test get_sparse_index_mapping with out-of-bounds indices."""
        i = torch.tensor([[0, 0], [1, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (3, 3)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Create indices with out-of-bounds values
        index_tensor = torch.tensor(
            [
                [0, 0],  # valid
                [3, 0],  # out of bounds
                [-1, 0],  # out of bounds
                [0, 3],  # out of bounds
            ],
            device=device,
        )

        _, is_specified = get_sparse_index_mapping(sparse_tensor, index_tensor)

        # Only the first index should be specified
        expected_specified = torch.tensor([True, False, False, False], device=device)
        assert torch.equal(is_specified, expected_specified)


@pytest.mark.cpu_and_cuda
class TestGatherAndMask:
    def test_gather_and_mask_basic(self, device):
        """Test basic functionality of gather_and_mask."""
        # Create source values
        values = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            device=device,
        )

        # Create indices and mask
        indices = torch.tensor([[0, 2, 1], [3, 1, 0]], device=device)
        mask = torch.tensor([[True, False, True], [True, True, False]], device=device)

        # Gather and mask
        result = gather_and_mask(values, indices, mask)

        # Expected result: only masked entries should be preserved
        expected = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
                [[10.0, 11.0, 12.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]],
            ],
            device=device,
        )

        assert torch.allclose(result, expected)

    def test_gather_and_mask_1d_input(self, device):
        """Test gather_and_mask with 1D input values."""
        # Create 1D source values
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)

        # Create indices and mask
        indices = torch.tensor([[0, 2], [3, 1]], device=device)
        mask = torch.tensor([[True, False], [True, True]], device=device)

        # Gather and mask
        result = gather_and_mask(values, indices, mask)

        # Expected result: only masked entries should be preserved
        expected = torch.tensor(
            [
                [1.0, 0.0],
                [4.0, 2.0],
            ],
            device=device,
        )

        assert torch.allclose(result, expected)

    def test_gather_and_mask_3d_input(self, device):
        """Test gather_and_mask with 3D input values."""
        # Create 3D source values
        values = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ],
            device=device,
        )

        # Create indices and mask
        indices = torch.tensor([0, 2, 1], device=device)
        mask = torch.tensor([True, False, True], device=device)

        # Gather and mask
        result = gather_and_mask(values, indices, mask)

        # Expected result: only masked entries should be preserved
        expected = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            device=device,
        )

        assert torch.allclose(result, expected)

    def test_gather_and_mask_complex_shapes(self, device):
        """Test gather_and_mask with complex shapes (2D indices, 3D values)."""
        # Create source values with 3 dimensions
        values = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ],
            device=device,
        )

        # Create 2D indices and mask
        indices = torch.tensor([[0, 2], [3, 1]], device=device)
        mask = torch.tensor([[True, False], [True, True]], device=device)

        # Gather and mask
        result = gather_and_mask(values, indices, mask)

        # Expected result: should have shape (2, 2, 2, 2)
        expected = torch.tensor(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                [
                    [[13.0, 14.0], [15.0, 16.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
            ],
            device=device,
        )

        assert torch.allclose(result, expected)

    def test_gather_and_mask_4d_input(self, device):
        """Test gather_and_mask with 4D input values."""
        # Create 4D source values
        values = torch.zeros((3, 2, 2, 2), device=device)
        values[0] = 1.0
        values[1] = 2.0
        values[2] = 3.0

        # Create indices and mask
        indices = torch.tensor([0, 2, 1], device=device)
        mask = torch.tensor([True, False, True], device=device)

        # Gather and mask
        result = gather_and_mask(values, indices, mask)

        # Expected result
        expected = torch.zeros((3, 2, 2, 2), device=device)
        expected[0] = 1.0  # From values[0], mask is True
        # expected[1] remains zeros (from values[2], mask is False)
        expected[2] = 2.0  # From values[1], mask is True

        assert torch.allclose(result, expected)

    def test_gather_and_mask_1d_gradient(self, device):
        """Test that gradients flow correctly through gather_and_mask with 1D input."""
        # Create 1D source values with gradients
        values = torch.tensor(
            [1.0, 2.0, 3.0, 4.0],
            device=device,
            requires_grad=True,
        )

        # Create indices and mask
        indices = torch.tensor([0, 2, 1], device=device)
        mask = torch.tensor([True, False, True], device=device)

        result = gather_and_mask(values, indices, mask)

        # Compute loss and check gradients
        loss = result.sum()
        loss.backward()

        # Expected gradients
        expected_grad = torch.tensor(
            [1.0, 1.0, 0.0, 0.0],  # 0 and 1 are used and not masked
            device=device,
        )

        assert values.grad is not None
        assert torch.allclose(values.grad, expected_grad)

    def test_gather_and_mask_gradient(self, device):
        """Test that gradients flow correctly through gather_and_mask."""
        # Create source values with gradients
        values = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            device=device,
            requires_grad=True,
        )

        # Create indices and mask
        indices = torch.tensor([0, 1, 0], device=device)
        mask = torch.tensor([True, False, True], device=device)

        result = gather_and_mask(values, indices, mask)

        # Compute loss and check gradients
        loss = result.sum()
        loss.backward()

        # First row gets gradient 2.0 (used twice), second row gets no gradient (masked)
        expected_grad = torch.tensor(
            [
                [2.0, 2.0, 2.0],  # Used at positions 0 and 2, both masked True
                [0.0, 0.0, 0.0],  # Used at position 1, but masked False
            ],
            device=device,
        )

        assert values.grad is not None
        assert torch.allclose(values.grad, expected_grad)

    def test_gather_and_mask_errors(self, device):
        """Test error handling in gather_and_mask."""
        # Test with mismatched indices and mask shapes
        values_2d = torch.ones((2, 3), device=device)
        indices_2 = torch.zeros(2, dtype=torch.long, device=device)
        mask_3 = torch.ones(3, dtype=torch.bool, device=device)

        with pytest.raises(
            (ValueError, torch.jit.Error),  # type: ignore
            match="Expected indices and mask to have same shape",
        ):
            gather_and_mask(values_2d, indices_2, mask_3)
