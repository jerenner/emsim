import pytest
import torch
from torch import Tensor

from hypothesis import strategies as st
from hypothesis import given, settings, assume, HealthCheck

# Import the functions to test
from emsim.utils.sparse_utils.indexing.script_funcs import (
    flatten_sparse_indices,
    gather_mask_and_fill,
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
class TestGatherMaskAndFill:
    def test_basic_functionality(self, device):
        """Test basic functionality with 2D values, including fill_values."""
        # Create source values
        values = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            device=device,
        )

        # Create indices and mask
        indices = torch.tensor([[0, 2, 1], [3, 1, 0]], device=device)
        mask = torch.tensor([[True, False, True], [True, True, False]], device=device)

        # Test default behavior (fill with zeros)
        result = gather_mask_and_fill(values, indices, mask)
        expected = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
                [[10.0, 11.0, 12.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]],
            ],
            device=device,
        )
        assert torch.allclose(result, expected)

        # Test with custom fill_values
        fill_values = torch.tensor([[-1.0, -2.0, -3.0]], device=device)
        result_filled = gather_mask_and_fill(values, indices, mask, fill_values)
        expected_filled = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]],
                [[10.0, 11.0, 12.0], [4.0, 5.0, 6.0], [-1.0, -2.0, -3.0]],
            ],
            device=device,
        )
        assert torch.allclose(result_filled, expected_filled)

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
            gather_mask_and_fill(values_2d, indices_2, mask_3)

    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    @given(
        n_values=st.integers(1, 20),
        values_feature_dims=st.lists(st.integers(0, 20), min_size=1, max_size=3),
        indices_shape=st.lists(st.integers(0, 20), min_size=0, max_size=3),
        # Portion of False values in mask
        mask_sparsity=st.floats(min_value=0.0, max_value=1.0),
        # Fill strategy
        fill_type=st.sampled_from(
            ["none", "scalar", "vector", "per_index_broadcasted", "full"]
        ),
        test_grads=st.booleans(),
    )
    def test_hypothesis(
        self,
        n_values: int,
        values_feature_dims: list[int],
        indices_shape: list[int],
        mask_sparsity: float,
        fill_type: str,
        test_grads: bool,
        device,
    ):
        """Property-based test."""
        # Create values tensor
        values = torch.randn(
            [n_values] + values_feature_dims, device=device, requires_grad=test_grads
        )

        # Create indices and mask
        if n_values > 0:
            indices = torch.randint(0, n_values, indices_shape, device=device)
        else:
            indices = torch.zeros(indices_shape, device=device, dtype=torch.long)
        mask = torch.rand(indices_shape, device=device) > mask_sparsity

        # Determine expected output shape
        if values.ndim == 1:
            expected_output_shape = indices_shape
        else:
            expected_output_shape = indices_shape + values_feature_dims

        # Create fill values based on broadcast type
        if fill_type == "none":
            fill = None
        elif fill_type == "scalar":
            fill = torch.randn([], device=device, requires_grad=test_grads)
        elif fill_type == "vector":
            assume(values_feature_dims[-1] > 0)
            fill = torch.randn(
                values_feature_dims[-1], device=device, requires_grad=test_grads
            )
        elif fill_type == "per_index_broadcasted":
            assume(len(indices_shape) > 0)
            fill_shape = (
                [indices_shape[0]]
                + ([1] * len(indices_shape[:-1]))
                + values_feature_dims
            )
            fill = torch.randn(fill_shape, device=device, requires_grad=test_grads)
        else:
            fill = torch.randn(
                expected_output_shape, device=device, requires_grad=test_grads
            )

        # Execute function
        result = gather_mask_and_fill(values, indices, mask, fill=fill)

        # Check result of correct type
        assert isinstance(result, Tensor)
        assert list(result.shape) == expected_output_shape

        # Get values with equivalent method
        values_copy = values.detach().clone().requires_grad_(test_grads)
        expected = values_copy[indices].clone()
        if fill is None:
            expected[~mask] = 0.0
        else:
            fill_copy = fill.detach().clone().requires_grad_(test_grads)
            expected[~mask] = fill_copy.expand_as(result)[~mask]

        assert torch.equal(result, expected)

        if test_grads:
            result.sum().backward()
            expected.sum().backward()

            assert values.grad is not None
            assert values_copy.grad is not None
            assert torch.equal(values.grad, values_copy.grad)

            if fill is not None:
                assert fill.grad is not None
                assert fill_copy.grad is not None
                assert torch.equal(fill.grad, fill_copy.grad)
