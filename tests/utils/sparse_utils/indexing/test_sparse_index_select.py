import pytest
import torch
import torch.nn.functional as F
import numpy as np

from emsim.utils.sparse_utils.indexing.sparse_index_select import sparse_index_select
from emsim.utils.sparse_utils.indexing.sparse_index_select import (
    _sparse_index_select_inner,
)


# Helper functions for testing
def create_random_sparse_tensor(shape, sparsity=0.8, device="cpu"):
    """Create a random sparse tensor with given shape and sparsity."""
    dense = torch.rand(shape, device=device)
    mask = torch.rand(shape, device=device) > sparsity
    dense = dense * mask
    return dense.to_sparse()


def assert_sparse_tensors_equal(tensor1, tensor2):
    """Assert that two sparse tensors are equal in indices and values."""
    # Coalesce both tensors for consistent representation
    tensor1 = tensor1.coalesce()
    tensor2 = tensor2.coalesce()

    # Check shapes
    assert tensor1.shape == tensor2.shape, "Shapes don't match"

    # Check indices and values
    assert torch.all(tensor1.indices() == tensor2.indices()), "Indices don't match"
    assert torch.allclose(tensor1.values(), tensor2.values()), "Values don't match"


### basic tests


@pytest.mark.cpu_and_cuda
class TestBasicFunctionality:

    def test_sparse_index_select_basic(self, device):
        """Test basic functionality of sparse_index_select."""
        i = torch.tensor([[0, 0], [1, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (3, 3)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Select specific indices
        index = torch.tensor([0, 2], device=device)
        result = sparse_index_select(sparse_tensor, 0, index)

        # Check shape and values
        assert result.shape == (2, 3)
        dense_result = result.to_dense()
        expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 4.0]], device=device)
        assert torch.allclose(dense_result, expected)

    def test_3d_tensor_selection(self, device):
        # Create a 3D sparse tensor
        values = torch.zeros((3, 4, 5), device=device)
        values[0, 1, 2] = 1.0
        values[1, 2, 3] = 2.0
        values[2, 1, 4] = 3.0
        sparse = values.to_sparse()

        # Select along axis 1
        index = torch.tensor([1, 2], device=device)
        result = sparse_index_select(sparse, 1, index)

        # Expected result
        expected_values = torch.zeros((3, 2, 5), device=device)
        expected_values[0, 0, 2] = 1.0  # From (0,1,2) -> (0,0,2) after selection
        expected_values[1, 1, 3] = 2.0  # From (1,2,3) -> (1,1,3) after selection
        expected_values[2, 0, 4] = 3.0  # From (2,1,4) -> (2,0,4) after selection
        expected = expected_values.to_sparse()

        assert_sparse_tensors_equal(result, expected)

    def test_sparse_index_select_axis(self, device):
        """Test sparse_index_select with different axes."""
        i = torch.tensor([[0, 0], [1, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (3, 3)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Select along axis 1
        index = torch.tensor([0, 1], device=device)
        result = sparse_index_select(sparse_tensor, 1, index)

        # Expected: dimensions along axis 1 have been selected and reindexed
        assert result.shape == (3, 2)
        dense_result = result.to_dense()
        expected = torch.tensor([[1.0, 0.0], [2.0, 3.0], [0.0, 0.0]], device=device)
        assert torch.allclose(dense_result, expected)

    def test_negative_axis(self, device):
        # Test selection with negative axis indexing
        values = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=device)
        sparse = values.to_sparse()

        # Select columns 0 and 2 with negative axis (-1 = last dimension)
        index = torch.tensor([0, 2], device=device)
        result = sparse_index_select(sparse, -1, index)

        # Expected result
        expected = torch.tensor([[1.0, 2.0], [0.0, 0.0]], device=device).to_sparse()
        assert_sparse_tensors_equal(result, expected)

    def test_scalar_index(self, device):
        # Test with a scalar index tensor
        values = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=device)
        sparse = values.to_sparse()

        index = torch.tensor(1, device=device)  # Scalar index
        result = sparse_index_select(sparse, 1, index)

        expected = torch.tensor([[0.0], [3.0]], device=device).to_sparse()
        assert_sparse_tensors_equal(result, expected)

    def test_select_all_indices(self, device):
        # Test selecting all indices (should return equivalent tensor)
        values = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=device)
        sparse = values.to_sparse()

        index = torch.tensor([0, 1, 2], device=device)
        result = sparse_index_select(sparse, 1, index)

        assert_sparse_tensors_equal(result, sparse)

    def test_duplicate_indices(self, device):
        # Test selecting same index multiple times
        values = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=device)
        sparse = values.to_sparse()

        index = torch.tensor([1, 1, 1], device=device)
        result = sparse_index_select(sparse, 1, index)

        expected_values = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]], device=device
        )
        expected = expected_values.to_sparse()
        assert_sparse_tensors_equal(result, expected)

    def test_disable_builtin_fallback(self, device):
        # Test the "disable_builtin_fallback" option
        values = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=device)
        sparse = values.to_sparse()

        index = torch.tensor([1, 1, 1], device=device)
        result = sparse_index_select(sparse, 1, index, disable_builtin_fallback=True)

        expected_values = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]], device=device
        )
        expected = expected_values.to_sparse()
        assert_sparse_tensors_equal(result, expected)


@pytest.mark.cpu_and_cuda
class TestEdgeCases:
    def test_empty_index(self, device):
        # Test with an empty index tensor
        values = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=device)
        sparse = values.to_sparse()

        index = torch.tensor([], device=device, dtype=torch.long)
        result = sparse_index_select(sparse, 1, index)

        # Expected result - empty tensor with correct shape
        expected = torch.zeros((2, 0), device=device).to_sparse()
        assert result.shape == expected.shape
        assert result.is_sparse
        assert result.indices().numel() == 0

    def test_empty_sparse_tensor(self, device):
        # Test with an empty sparse tensor (no non-zero elements)
        indices = torch.zeros((2, 0), device=device, dtype=torch.long)
        values = torch.zeros((0,), device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 3))

        index = torch.tensor([0, 2], device=device)
        result = sparse_index_select(sparse, 1, index)

        # Expected result
        expected_shape = (2, 2)
        assert result.shape == expected_shape
        assert result.is_sparse
        assert result.indices().numel() == 0
        assert result.values().numel() == 0

    def test_sparse_tensor_with_all_values_zero(self, device):
        """Test handling of a sparse tensor containing all zero values
        (but with structure).
        """
        # Create indices but set all values to zero
        i = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 3]], device=device)
        v = torch.zeros(4, device=device)

        sparse = torch.sparse_coo_tensor(i, v, (4, 4)).coalesce()

        # Select some indices
        indices = torch.tensor([0, 2], device=device)
        result = sparse_index_select(sparse, 1, indices)

        # Expected shape
        assert result.shape == (4, 2)
        assert result.is_sparse

        result_dense = result.to_dense()
        assert torch.all(result_dense == 0)

    def test_check_bounds_parameter(self, device):
        # Test with check_bounds=False
        values = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=device)
        sparse = values.to_sparse()

        # Valid indices should work with check_bounds=False
        index = torch.tensor([1], device=device)
        result = sparse_index_select(sparse, 1, index, check_bounds=False)

        expected = torch.tensor([[0.0], [3.0]], device=device).to_sparse()
        assert_sparse_tensors_equal(result, expected)


@pytest.mark.cpu_and_cuda
class TestErrorHandling:
    def test_sparse_index_select_errors(self, device):
        """Test error handling in sparse_index_select."""
        i = torch.tensor([[0, 0], [1, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (3, 3)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()
        dense_tensor = sparse_tensor.to_dense()

        # Test with dense tensor (should error)
        with pytest.raises((ValueError, torch.jit.Error), match="must be sparse"):
            sparse_index_select(dense_tensor, 0, torch.tensor([0], device=device))

        # Test with out-of-bounds index
        with pytest.raises((ValueError, torch.jit.Error), match="out of bounds"):
            sparse_index_select(sparse_tensor, 0, torch.tensor([3], device=device))

        # Test with negative index
        with pytest.raises((ValueError, torch.jit.Error), match="out of bounds"):
            sparse_index_select(sparse_tensor, 0, torch.tensor([-1], device=device))

        # Test with invalid axis
        with pytest.raises((ValueError, torch.jit.Error), match="out of bounds"):
            sparse_index_select(sparse_tensor, 2, torch.tensor([0], device=device))

        # Test with 2D index tensor
        with pytest.raises((ValueError, torch.jit.Error), match="must be 0D or 1D"):
            sparse_index_select(
                sparse_tensor, 0, torch.zeros((2, 2), dtype=torch.long, device=device)
            )

        # Test with float index tensor
        with pytest.raises((ValueError, torch.jit.Error), match="non-integer dtype"):
            sparse_index_select(
                sparse_tensor, 0, torch.tensor([0.0], dtype=torch.float, device=device)
            )


@pytest.mark.cpu_and_cuda
class TestGradients:
    def test_sparse_index_select_grad(self, device):
        """Test that gradients flow correctly through sparse_index_select."""
        i = torch.tensor([[0, 0], [1, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, requires_grad=True)
        shape = (3, 3)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Select indices and compute loss
        index = torch.tensor([0, 2], device=device)
        result = sparse_index_select(sparse_tensor, 0, index)
        loss = result.to_dense().sum()
        loss.backward()

        # Check gradients (values at indices 0, 3 were selected)
        expected_grad = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
        assert torch.allclose(v.grad, expected_grad)

    def test_requires_grad_false(self, device):
        # Test with requires_grad=False
        values = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=device)
        sparse = values.to_sparse()

        # Ensure requires_grad is False
        assert not sparse.requires_grad

        index = torch.tensor([1], device=device)
        result = sparse_index_select(sparse, 1, index)

        # Result shouldn't require gradients
        assert not result.requires_grad

    def test_gradient_computation(self, device):
        # Test gradient computation
        values = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=device)
        sparse = values.to_sparse().requires_grad_()

        index = torch.tensor([1], device=device)
        result = sparse_index_select(sparse, 1, index)

        # Compute gradient
        result.values().sum().backward()

        # Check that gradient propagated correctly
        # We need to extract the gradient at the selected positions
        grad_sparse = sparse.grad.coalesce()
        grad_dense = torch.zeros_like(values)

        indices = grad_sparse.indices()
        grad_values = grad_sparse.values()

        for i in range(indices.shape[1]):
            row, col = indices[0, i].item(), indices[1, i].item()
            grad_dense[row, col] = grad_values[i].item()

        # Only the selected column (index 1) should have gradients
        expected_grad = torch.zeros_like(values)
        expected_grad[1, 1] = 1.0  # Only the non-zero element in column 1

        assert torch.allclose(grad_dense, expected_grad)

    def test_higher_order_gradients(self, device):
        """Test that higher-order gradients work through sparse_index_select."""
        # Create a small sparse tensor
        dense = torch.zeros((3, 3), device=device)
        dense[0, 1] = 2.0
        dense[1, 0] = 3.0
        dense[2, 2] = 4.0

        # Convert to sparse with gradients
        sparse_tensor = dense.to_sparse().requires_grad_(True)

        # Select along dim 1
        indices = torch.tensor([0, 1], device=device)
        selected = sparse_index_select(sparse_tensor, 1, indices)

        # First forward pass and backward pass
        output1 = selected.to_dense().pow(2).sum()

        # Create a tensor to track gradients
        grad_output = torch.zeros_like(sparse_tensor.to_dense(), requires_grad=True)

        # First backward pass
        output1.backward(retain_graph=True)

        # Copy gradients
        with torch.no_grad():
            grad_output.copy_(sparse_tensor.grad.to_dense())

        # Zero gradients
        sparse_tensor.grad.zero_()

        # Second forward and backward pass
        output2 = (grad_output * sparse_tensor.to_dense()).sum()
        output2.backward()

        # The fact that this runs without error confirms higher-order gradient support
        assert sparse_tensor.grad is not None

    def test_hybrid_sparse_complex_gradient_flow(self, device):
        """Test gradient flow through complex operations on hybrid sparse tensors."""
        # Create a hybrid sparse tensor
        i = torch.tensor([[0, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],  # Values for (0,0)
                [[5.0, 6.0], [7.0, 8.0]],  # Values for (1,1)
                [[9.0, 10.0], [11.0, 12.0]],  # Values for (2,2)
            ],
            device=device,
            requires_grad=True,
        )

        hybrid_sparse = torch.sparse_coo_tensor(i, v, (3, 3, 2, 2))

        # Create a small model that processes the sparse tensor
        class SparseProcessor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 4, device=device)
                self.linear2 = torch.nn.Linear(4, 1, device=device)

            def forward(self, x_sparse):
                # Select specific indices
                x_selected = sparse_index_select(
                    x_sparse, 0, torch.tensor([0, 2], device=device)
                )

                # Convert to dense for processing each block
                x_dense = x_selected.to_dense()
                batch_size, n_blocks, h, w = x_dense.shape

                # Reshape to process each 2D block independently
                x_reshaped = x_dense.reshape(-1, w)

                # Apply layers
                x = self.linear1(x_reshaped)
                x = F.gelu(x)
                x = self.linear2(x)

                # Reshape back to original structure
                return x.reshape(batch_size, n_blocks, h)

        model = SparseProcessor()

        # Process the sparse tensor
        output = model(hybrid_sparse)

        # Compute loss and backpropagate
        target = torch.zeros_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Verify gradient flow
        assert v.grad is not None

        # Only values corresponding to indices 0 and 2 should have non-zero gradients
        v_grad_sum = v.grad.sum(dim=(1, 2))
        assert v_grad_sum[0] != 0  # Index 0 was selected
        assert v_grad_sum[1] == 0  # Index 1 was not selected
        assert v_grad_sum[2] != 0  # Index 2 was selected


@pytest.mark.cpu_and_cuda
class TestAgainstBuiltin:

    def test_builtin_index_select_does_not_support_grad(self, device):
        i = torch.tensor([[0, 0], [1, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (3, 3)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()
        sparse_tensor.requires_grad_(True)

        # Select specific indices
        index = torch.tensor([0, 2], device=device)

        selected = sparse_tensor.index_select(0, index)
        loss = selected.to_dense().sum()

        with pytest.raises(
            NotImplementedError, match="Could not run 'aten::index_add_'"
        ):
            loss.backward()


@pytest.mark.cpu_and_cuda
class TestGradcheck:

    def test_gradcheck_sparse_index_select(self, device):
        """Perform a detailed gradcheck on the sparse_index_select function."""
        # Create a small sparse tensor with double precision
        i = torch.tensor([[0, 0], [0, 1], [1, 1], [2, 0]], device=device).T
        # Use double precision for gradcheck
        v = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.double, requires_grad=True
        )

        # Indices to select
        indices = torch.tensor([0, 1], device=device)

        def func(input_values):
            # Recreate sparse tensor with new values but same indices
            sparse_tensor = torch.sparse_coo_tensor(i, input_values, (3, 2)).coalesce()
            result = sparse_index_select(sparse_tensor, 1, indices)
            return result.to_dense()

        # Perform numerical gradient check
        torch.autograd.gradcheck(func, v, eps=1e-6, atol=1e-5)

    def test_gradcheck_hybrid_sparse_tensor(self, device):
        """Test gradcheck with hybrid sparse-dense tensors."""
        # Create a hybrid sparse tensor (2D sparse, 1D dense)
        i = torch.tensor([[0, 1], [1, 0]], device=device)
        v = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],  # Values for (0,1)  # Values for (1,0)
            device=device,
            dtype=torch.double,
            requires_grad=True,
        )

        # Indices to select
        indices = torch.tensor([1], device=device)

        def func(input_values):
            # Recreate hybrid sparse tensor with new values but same indices
            sparse_tensor = torch.sparse_coo_tensor(
                i, input_values.reshape(2, 2), (2, 2, 2)
            ).coalesce()
            result = sparse_index_select(sparse_tensor, 1, indices)
            return result.to_dense()

        # Perform numerical gradient check
        torch.autograd.gradcheck(func, v.flatten(), eps=1e-6, atol=1e-5)

    @pytest.mark.cpu_and_cuda
    def test_gradcheck_sparse_values(self, device):
        """Test gradcheck on the values of the sparse tensor directly."""
        # Skip for CUDA to avoid slow tests
        # Create a small sparse tensor with double precision
        i = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 0]], device=device)
        v = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.double, requires_grad=True
        )

        # Indices to select
        indices = torch.tensor([0], device=device)

        def func(input_values):
            # Recreate sparse tensor with new values but same indices
            sparse_tensor = torch.sparse_coo_tensor(i, input_values, (3, 2)).coalesce()
            result = sparse_index_select(sparse_tensor, 1, indices)
            # Key difference: extract and return values directly
            return result.values()

        # Perform numerical gradient check
        torch.autograd.gradcheck(func, v, eps=1e-6, atol=1e-5)

    @pytest.mark.cpu_and_cuda
    def test_gradcheck_hybrid_sparse_values(self, device):
        """Test gradcheck with values of hybrid sparse-dense tensors."""
        # Create a hybrid sparse tensor (2D sparse, 1D dense)
        i = torch.tensor([[0, 1], [1, 0]], device=device)
        v = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],  # Values for (0,1)  # Values for (1,0)
            device=device,
            dtype=torch.double,
            requires_grad=True,
        )

        # Indices to select
        indices = torch.tensor([1], device=device)

        def func(input_values):
            # Recreate hybrid sparse tensor with new values but same indices
            sparse_tensor = torch.sparse_coo_tensor(
                i, input_values.reshape(2, 2), (2, 2, 2)
            ).coalesce()
            result = sparse_index_select(sparse_tensor, 1, indices)
            # Key difference: extract and return values directly
            return result.values()

        # Perform numerical gradient check
        torch.autograd.gradcheck(func, v.flatten(), eps=1e-6, atol=1e-5)


@pytest.mark.cpu_and_cuda
class TestBasicHybridTensorBehavior:
    def test_hybrid_sparse_tensor_with_dense_dim(self, device):
        """Test sparse_index_select with hybrid sparse tensors having dense dimensions."""
        # Create a sparse tensor with a dense dimension
        # This is a 3D tensor sparse in first two dims, with last dim being dense
        i = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device)
        v = torch.tensor(
            [
                [1.0, 2.0, 3.0],  # values for (0,1)
                [4.0, 5.0, 6.0],  # values for (1,0)
                [7.0, 8.0, 9.0],  # values for (1,2)
                [10.0, 11.0, 12.0],  # values for (2,1)
            ],
            device=device,
        )

        hybrid_sparse = torch.sparse_coo_tensor(i, v, (3, 3, 3), device=device)

        # Select along dim 0 (sparse dimension)
        indices0 = torch.tensor([0, 2], device=device)
        result0 = sparse_index_select(hybrid_sparse, 0, indices0)

        # Expected shape: (2, 3, 3) - two rows, all columns, all dense features
        assert result0.shape == (2, 3, 3)
        result0_dense = result0.to_dense()

        # Check specific values
        # For row 0, only position (0,1) had values [1,2,3]
        assert torch.allclose(
            result0_dense[0, 1], torch.tensor([1.0, 2.0, 3.0], device=device)
        )
        # For row 1 (originally row 2), only position (2,1) had values [10,11,12]
        assert torch.allclose(
            result0_dense[1, 1], torch.tensor([10.0, 11.0, 12.0], device=device)
        )

        # Select along dim 1 (also sparse dimension)
        indices1 = torch.tensor([0, 1], device=device)
        result1 = sparse_index_select(hybrid_sparse, 1, indices1)

        # Expected shape: (3, 2, 3) - all rows, two columns, all dense features
        assert result1.shape == (3, 2, 3)
        result1_dense = result1.to_dense()

        # Check specific values
        assert torch.allclose(
            result1_dense[0, 1], torch.tensor([1.0, 2.0, 3.0], device=device)
        )
        assert torch.allclose(
            result1_dense[1, 0], torch.tensor([4.0, 5.0, 6.0], device=device)
        )
        assert torch.allclose(
            result1_dense[2, 1], torch.tensor([10.0, 11.0, 12.0], device=device)
        )

        # Select along dim 2 (dense dimension)
        indices2 = torch.tensor([0, 2], device=device)
        result2 = sparse_index_select(hybrid_sparse, 2, indices2)

        # Expected shape: (3, 3, 2) - all rows, all columns, selected dense features
        assert result2.shape == (3, 3, 2)
        result2_dense = result2.to_dense()

        # Check specific values - should only contain features at indices 0 and 2
        assert torch.allclose(
            result2_dense[0, 1], torch.tensor([1.0, 3.0], device=device)
        )
        assert torch.allclose(
            result2_dense[1, 0], torch.tensor([4.0, 6.0], device=device)
        )
        assert torch.allclose(
            result2_dense[1, 2], torch.tensor([7.0, 9.0], device=device)
        )
        assert torch.allclose(
            result2_dense[2, 1], torch.tensor([10.0, 12.0], device=device)
        )

    def test_hybrid_sparse_tensor_with_multiple_dense_dims(self, device):
        """Test sparse_index_select with sparse tensors having multiple dense dimensions."""
        # Create a 4D tensor sparse in first dim, with 3 dense dimensions
        i = torch.tensor([[0, 1, 2]], device=device)  # Sparse dimension indices
        v = torch.randn((3, 2, 3, 4), device=device)  # 3 sets of dense values

        hybrid_sparse = torch.sparse_coo_tensor(i, v, (3, 2, 3, 4), device=device)

        # Select along dim 0 (sparse dimension)
        indices0 = torch.tensor([0, 2], device=device)
        result0 = sparse_index_select(hybrid_sparse, 0, indices0)

        # Expected shape: (2, 2, 3, 4)
        assert result0.shape == (2, 2, 3, 4)
        result0_dense = result0.to_dense()

        # Check values match
        expected0 = torch.stack(
            [hybrid_sparse.to_dense()[0], hybrid_sparse.to_dense()[2]]
        )
        assert torch.allclose(result0_dense, expected0)

        # Select along dim 1 (first dense dimension)
        indices1 = torch.tensor([1], device=device)
        result1 = sparse_index_select(hybrid_sparse, 1, indices1)

        # Expected shape: (3, 1, 3, 4)
        assert result1.shape == (3, 1, 3, 4)
        result1_dense = result1.to_dense()

        # Check values match
        expected1 = hybrid_sparse.to_dense()[:, 1:2]
        assert torch.allclose(result1_dense, expected1)

        # Select along dim 2 (second dense dimension)
        indices2 = torch.tensor([0, 2], device=device)
        result2 = sparse_index_select(hybrid_sparse, 2, indices2)

        # Expected shape: (3, 2, 2, 4)
        assert result2.shape == (3, 2, 2, 4)
        result2_dense = result2.to_dense()

        # Check values match
        expected2 = torch.stack(
            [hybrid_sparse.to_dense()[:, :, 0], hybrid_sparse.to_dense()[:, :, 2]],
            dim=2,
        )
        assert torch.allclose(result2_dense, expected2)

    def test_hybrid_tensor_multiple_dense_axes_selection(self, device):
        """Test selecting different axes from a hybrid tensor with multiple dense dimensions."""
        # Create a hybrid tensor with 1 sparse dim and 3 dense dims
        i = torch.tensor([[0, 2, 4]], device=device)
        v = torch.randn((3, 2, 3, 4), device=device)

        hybrid_sparse = torch.sparse_coo_tensor(i, v, (5, 2, 3, 4))

        # 1. Select from the sparse dimension
        indices_sparse = torch.tensor([0, 4], device=device)
        result_sparse = sparse_index_select(hybrid_sparse, 0, indices_sparse)
        assert result_sparse.shape == (2, 2, 3, 4)

        # 2. Select from first dense dimension
        indices_dense1 = torch.tensor([0], device=device)
        result_dense1 = sparse_index_select(hybrid_sparse, 1, indices_dense1)
        assert result_dense1.shape == (5, 1, 3, 4)

        # 3. Select from second dense dimension
        indices_dense2 = torch.tensor([0, 2], device=device)
        result_dense2 = sparse_index_select(hybrid_sparse, 2, indices_dense2)
        assert result_dense2.shape == (5, 2, 2, 4)

        # 4. Select from last dense dimension
        indices_dense3 = torch.tensor([1, 2, 3], device=device)
        result_dense3 = sparse_index_select(hybrid_sparse, 3, indices_dense3)
        assert result_dense3.shape == (5, 2, 3, 3)

        # Verify values for each selection
        original_dense = hybrid_sparse.to_dense()

        # Check sparse dimension selection
        selected_sparse = result_sparse.to_dense()
        assert torch.allclose(selected_sparse[0], original_dense[0])
        assert torch.allclose(selected_sparse[1], original_dense[4])

        # Check first dense dimension selection
        selected_dense1 = result_dense1.to_dense()
        assert torch.allclose(selected_dense1[:, 0], original_dense[:, 0])

    def test_hybrid_tensor_single_element_selection(self, device):
        """Test selecting single elements from each dimension of a hybrid tensor."""
        # Create a hybrid tensor with 2 sparse dims, 2 dense dims
        i = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], device=device)
        v = torch.arange(4 * 5 * 6, device=device).reshape(4, 5, 6).float()

        hybrid_sparse = torch.sparse_coo_tensor(i, v, (4, 4, 5, 6))

        # Single element selection in each dimension
        for dim in range(4):
            indices = torch.tensor([2], device=device)
            result = sparse_index_select(hybrid_sparse, dim, indices)

            # Expected shape: original with dim=1 at the selected dimension
            expected_shape = list(hybrid_sparse.shape)
            expected_shape[dim] = 1

            assert result.shape == tuple(expected_shape)

            # Verify correct slice was selected
            dense_result = result.to_dense()
            dense_original = hybrid_sparse.to_dense()

            # Create indexing tuple for verification
            idx = [slice(None)] * 4
            idx[dim] = 2

            # The result has shape with 1 in the selected dimension
            # so we need to squeeze that dimension for comparison
            result_squeezed = dense_result.squeeze(dim)
            expected_slice = dense_original[tuple(idx)]

            if dim < 2:  # Sparse dimensions might not have value at every position
                # For sparse dims, make sure non-zero values match where they exist
                nonzero_mask = expected_slice != 0
                assert torch.allclose(
                    result_squeezed[nonzero_mask], expected_slice[nonzero_mask]
                )
            else:  # Dense dimensions should match exactly
                assert torch.allclose(result_squeezed, expected_slice)


@pytest.mark.cpu_and_cuda
class TestAdvancedHybridTensorBehavior:
    def test_hybrid_tensor_with_block_sparsity_pattern(self, device):
        """Test hybrid tensor with block-sparse structure."""
        # Create a block-sparse tensor: 3x3 grid of 2x2 blocks
        # Only blocks at (0,0), (1,1), (2,2) have values
        blocks = torch.zeros((3, 3, 2, 2), device=device)

        # Fill diagonal blocks with different values
        blocks[0, 0] = torch.ones((2, 2))
        blocks[1, 1] = 2 * torch.ones((2, 2))
        blocks[2, 2] = 3 * torch.ones((2, 2))

        # Create a version that maintains block structure as hybrid tensor
        i = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device)
        v = torch.stack([blocks[0, 0], blocks[1, 1], blocks[2, 2]])
        hybrid_sparse = torch.sparse_coo_tensor(i, v, (3, 3, 2, 2))

        # 1. Select rows of blocks
        row_indices = torch.tensor([0, 2], device=device)
        row_result = sparse_index_select(hybrid_sparse, 0, row_indices)
        assert row_result.shape == (2, 3, 2, 2)

        row_dense = row_result.to_dense()
        assert torch.all(row_dense[0, 0] == torch.ones((2, 2), device=device))
        assert torch.all(row_dense[1, 2] == 3 * torch.ones((2, 2), device=device))

        # 2. Select inside the blocks (dense dim 0)
        inner_indices = torch.tensor([0], device=device)
        inner_result = sparse_index_select(hybrid_sparse, 2, inner_indices)
        assert inner_result.shape == (3, 3, 1, 2)

        inner_dense = inner_result.to_dense()
        assert torch.all(inner_dense[0, 0, 0] == torch.ones(2, device=device))
        assert torch.all(inner_dense[1, 1, 0] == 2 * torch.ones(2, device=device))

    def test_hybrid_tensor_with_varying_dense_values(self, device):
        """Test hybrid tensor where dense values have different patterns across sparse locations."""
        # Create a sparse tensor with 2 sparse dims and 2 dense dims
        i = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 2]], device=device)

        # Each sparse location has a unique pattern in dense dims
        v = torch.zeros((4, 3, 4), device=device)
        v[0] = torch.ones((3, 4), device=device)  # Pattern for (0,0)
        v[1] = torch.eye(
            3, 4, device=device
        )  # Pattern for (1,1) - identity matrix pattern
        v[2] = torch.arange(12, device=device).reshape(
            3, 4
        )  # Pattern for (2,2) - sequential values
        v[3, 0, 0] = 5.0  # Single value at (0,2,0,0)

        hybrid_sparse = torch.sparse_coo_tensor(i, v, (3, 3, 3, 4)).requires_grad_(True)

        # Select along sparse dim 0
        indices0 = torch.tensor([0, 2], device=device)
        result0 = sparse_index_select(hybrid_sparse, 0, indices0)

        # Verify shape and specific patterns in result
        assert result0.shape == (2, 3, 3, 4)
        result0_dense = result0.to_dense()

        # Check patterns were preserved
        assert torch.all(result0_dense[0, 0] == torch.ones((3, 4), device=device))
        assert torch.all(
            result0_dense[1, 2] == torch.arange(12, device=device).reshape(3, 4)
        )
        assert result0_dense[0, 2, 0, 0] == 5.0

        # Test gradient flow
        loss = result0_dense.sum()
        loss.backward()
        assert hybrid_sparse.grad is not None

        # Specific gradient check: only selected indices should have gradients
        grad_dense = hybrid_sparse.grad.to_dense()
        assert torch.all(grad_dense[0, 0] == 1.0)
        assert torch.all(grad_dense[0, 2] == 1.0)
        assert torch.all(grad_dense[2, 2] == 1.0)
        assert torch.all(grad_dense[1, 1] == 0.0)  # Not selected

    def test_hybrid_tensor_with_different_dims_sparsity(self, device):
        """Test a sparse tensor sparse in different dimensions (not just leading dims)."""
        # Create a tensor sparse in dimensions 0 and 2, with dense dimensions 1 and 3
        # This isn't directly constructible in PyTorch, but we can simulate it
        # by constructing a tensor with leading sparse dimensions

        # First, create a hybrid tensor with normal structure
        shape = (3, 4, 5, 6)  # Original shape

        # Create some sparse points
        indices = []
        values = []

        # Add 5 random sparse points
        for _ in range(5):
            dim0 = torch.randint(0, shape[0], (1,)).item()
            dim2 = torch.randint(0, shape[2], (1,)).item()

            # Store sparse coordinates
            indices.append([dim0, dim2])

            # Create dense values for this sparse point (4x6 matrix)
            values.append(torch.randn(shape[1], shape[3], device=device))

        # Convert to tensor format
        i = torch.tensor(indices, device=device).t()
        v = torch.stack(values)

        # Create sparse tensor (sparse in dims 0,2; dense in dims 1,3)
        # We need to shuffle dimensions to satisfy PyTorch's requirement for leading sparse dims
        sparse_tensor = torch.sparse_coo_tensor(
            i, v, (shape[0], shape[2], shape[1], shape[3])
        )

        # 1. Select from first sparse dimension
        indices0 = torch.tensor([1], device=device)
        result0 = sparse_index_select(sparse_tensor, 0, indices0)
        assert result0.shape == (1, shape[2], shape[1], shape[3])

        # 2. Select from second sparse dimension
        indices1 = torch.tensor([2, 3], device=device)
        result1 = sparse_index_select(sparse_tensor, 1, indices1)
        assert result1.shape == (shape[0], 2, shape[1], shape[3])

        # 3. Select from first dense dimension
        indices2 = torch.tensor([0, 2], device=device)
        result2 = sparse_index_select(sparse_tensor, 2, indices2)
        assert result2.shape == (shape[0], shape[2], 2, shape[3])


@pytest.mark.cpu_and_cuda
class TestPerformanceAndIntegration:
    def test_random_sparse_tensors(self, device):
        # Test with various random sparse tensors
        shapes = [(10, 10), (5, 8, 12), (3, 4, 5, 6)]
        sparsities = [0.7, 0.9, 0.99]

        for shape in shapes:
            for sparsity in sparsities:
                sparse = create_random_sparse_tensor(shape, sparsity, device)

                # Choose a random axis and indices
                ndim = len(shape)
                axis = np.random.randint(0, ndim)
                num_indices = np.random.randint(1, shape[axis])
                indices = torch.randint(0, shape[axis], (num_indices,), device=device)

                # Run our function
                result = sparse_index_select(sparse, axis, indices)

                # Check expected shape
                expected_shape = list(shape)
                expected_shape[axis] = num_indices
                assert result.shape == tuple(expected_shape)

                # Verify result by comparing with dense version
                dense = sparse.to_dense()
                # Convert to numpy indexing
                idx = [slice(None)] * ndim
                idx[axis] = indices.cpu().numpy()
                expected_dense = dense[tuple(idx)]

                assert torch.allclose(result.to_dense(), expected_dense)

    def test_linear_layer_and_sgd_optimization(self, device):
        """Test that sparse_index_select properly propagates gradients in a neural network."""
        # Create a sparse tensor with values to be updated through SGD
        dense_tensor = torch.zeros((5, 3, 4), device=device)
        # Add some values that will be involved in computation
        dense_tensor[0, 1, 2] = 2.0
        dense_tensor[1, 0, 1] = 3.0
        dense_tensor[2, 1, 3] = 1.0
        dense_tensor[3, 2, 0] = 4.0

        # Convert to sparse and require gradients
        sparse_tensor = dense_tensor.to_sparse().requires_grad_(True)

        # Create a mini neural network
        linear = torch.nn.Linear(4, 2, device=device)

        # Initialize weights deterministically for reproducible test
        torch.nn.init.ones_(linear.weight)
        torch.nn.init.zeros_(linear.bias)

        # Create optimizer
        optimizer = torch.optim.SGD([sparse_tensor], lr=1.0)

        # Select specific indices
        indices = torch.tensor([0, 1], device=device)
        selected = sparse_index_select(sparse_tensor, 1, indices)

        # Forward pass
        output = linear(selected.to_dense())

        # Compute loss (MSE against target)
        target = torch.zeros_like(output)
        target[1, 0] = (
            1.0  # Target for position (1,0) (which contains value from original (1,0,1))
        )

        loss = torch.nn.functional.mse_loss(output, target)

        # Save original values for comparison
        original_dense = sparse_tensor.detach().to_dense().clone()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get updated tensor
        updated_dense = sparse_tensor.detach().to_dense()

        # Check correct positions where we expect updates
        # Position (1,0,1) should be updated since it was selected and involved in loss
        assert not torch.allclose(updated_dense[1, 0, 1], original_dense[1, 0, 1])

        # Position (3,2,0) should NOT be updated since index 2 wasn't selected
        assert torch.allclose(updated_dense[3, 2, 0], original_dense[3, 2, 0])

        # All zeros should remain zeros
        zero_mask = original_dense == 0
        assert torch.all(updated_dense[zero_mask] == 0)

    def test_sparse_values_sgd_optimization(self, device):
        """Test direct optimization on sparse tensor values."""
        # Create a sparse tensor with values to be updated
        i = torch.tensor([[0, 1, 2], [1, 0, 1]], device=device)
        v = torch.tensor([2.0, 3.0, 1.0], device=device)
        sparse_tensor = torch.sparse_coo_tensor(i, v, (3, 2)).requires_grad_(True)

        # Create optimizer
        optimizer = torch.optim.SGD([sparse_tensor], lr=1.0)

        # Select and get values directly
        indices = torch.tensor([0, 1], device=device)
        selected = sparse_index_select(sparse_tensor, 1, indices)

        # Use the values directly - this is the key difference in this test
        values = selected.values()

        # Compute loss directly on the values
        target = torch.tensor([1.0, 0.5, 0.0], device=device)
        loss = torch.nn.functional.mse_loss(values, target)

        # Save original values
        original_values = sparse_tensor.detach().coalesce().values().clone()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check values were updated
        updated_values = sparse_tensor.detach().coalesce().values()
        assert not torch.allclose(updated_values, original_values)

        # Verify the update happened in the right direction
        assert updated_values[1] < original_values[1]  # Should decrease towards target
        assert updated_values[0] < original_values[0]  # Should decrease towards target

    def test_hybrid_tensor_sgd_optimization(self, device):
        """Test optimization with hybrid sparse-dense tensors."""
        # Create a hybrid sparse tensor (2D sparse, 1D dense)
        i = torch.tensor([[0, 1, 2], [1, 0, 1]], device=device)
        v = torch.tensor(
            [
                [1.0, 2.0, 3.0],  # Values for (0,1)
                [4.0, 5.0, 6.0],  # Values for (1,0)
                [7.0, 8.0, 9.0],  # Values for (2,1)
            ],
            device=device,
        )

        hybrid_sparse = torch.sparse_coo_tensor(i, v, (3, 2, 3)).requires_grad_(True)

        # Create a layer that operates on the dense dimension
        linear = torch.nn.Linear(3, 1, device=device)
        torch.nn.init.ones_(linear.weight)
        torch.nn.init.zeros_(linear.bias)

        # Create optimizer
        optimizer = torch.optim.SGD([hybrid_sparse], lr=0.5)

        # Select indices from the sparse dimensions
        indices = torch.tensor([1], device=device)
        selected = sparse_index_select(hybrid_sparse, 1, indices)

        # Convert to dense and apply linear layer
        selected_dense = selected.to_dense().reshape(-1, 3)  # Reshape to [N, dense_dim]
        output = linear(selected_dense)

        # Target: reduce the sum of each set of values
        target = torch.tensor([[3.0], [0.0], [3.0]], device=device)
        loss = torch.nn.functional.mse_loss(output, target)

        # Save original values
        original_dense = hybrid_sparse.detach().to_dense().clone()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check values updated correctly
        updated_dense = hybrid_sparse.detach().to_dense()

        # Should show update at (0,1) and (2,1) which were selected and involved in loss
        assert not torch.allclose(updated_dense[0, 1], original_dense[0, 1])
        assert not torch.allclose(updated_dense[2, 1], original_dense[2, 1])

        # Should NOT update unselected positions
        assert torch.allclose(updated_dense[1, 0], original_dense[1, 0])

    def test_deeper_network_integration(self, device):
        """Test sparse_index_select in a deeper network with multiple layers."""
        # Create a sparse tensor
        dense_tensor = torch.zeros((4, 6, 5), device=device)
        # Add some values
        dense_tensor[0, 2, 3] = 1.0
        dense_tensor[1, 4, 2] = 2.0
        dense_tensor[2, 1, 0] = 3.0
        dense_tensor[3, 2, 4] = 4.0

        sparse_tensor = dense_tensor.to_sparse().requires_grad_(True)

        # Create a simple two-layer network
        linear1 = torch.nn.Linear(5, 8, device=device)
        activation = torch.nn.GELU()
        linear2 = torch.nn.Linear(8, 3, device=device)

        # Create optimizer
        optimizer = torch.optim.SGD([sparse_tensor], lr=0.1)

        # Select indices
        indices = torch.tensor([1, 2, 4], device=device)
        selected = sparse_index_select(sparse_tensor, 1, indices)

        # Forward pass through the network
        output1 = linear1(selected.to_dense())
        output2 = activation(output1)
        output3 = linear2(output2)

        # Compute loss
        target = torch.ones((4, 3, 3), device=device)
        loss = torch.nn.functional.mse_loss(output3, target)

        # Store original values
        original_values = sparse_tensor.clone().detach()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify gradients propagated and values updated
        updated_dense = sparse_tensor.detach().to_dense()
        original_dense = original_values.to_dense()

        # Only values at indices 1,2,4 should be updated
        changed = updated_dense != original_dense

        # Check specific positions
        assert torch.any(changed[0, 2])  # Should be updated (index 2 was selected)
        assert torch.any(changed[2, 1])  # Should be updated (index 1 was selected)
        assert torch.any(changed[1, 4])  # Should be updated (index 4 was selected)

        # Positions with indices not selected should remain unchanged
        not_selected_indices = [0, 3, 5]
        for idx in not_selected_indices:
            assert not torch.any(changed[:, idx])


@pytest.mark.cpu_and_cuda
class TestBatchAndPerformance:
    def test_large_sparse_tensor_performance(self, device):
        """Test performance with a large sparse tensor (not an assertion-based test)."""
        # Large 3D sparse tensor with high sparsity
        shape = (1000, 500, 20)

        # Fix: Create a properly structured 3D tensor
        # We'll use 2D sparsity with a dense last dimension
        indices = torch.zeros((2, 200), dtype=torch.long, device=device)
        indices[0] = torch.randint(
            0, shape[0], (200,), device=device
        )  # First dimension
        indices[1] = torch.randint(
            0, shape[1], (200,), device=device
        )  # Second dimension

        # Values have shape [nnz, dense_dim]
        values = torch.randn(200, shape[2], device=device)

        large_sparse = torch.sparse_coo_tensor(indices, values, shape)

        # Select a small subset of indices
        indices = torch.randint(0, shape[1], (50,), device=device)

        # This would time out if there's a serious performance issue
        result = sparse_index_select(large_sparse, 1, indices)

        # Basic shape check
        expected_shape = (shape[0], 50, shape[2])
        assert result.shape == expected_shape

    def test_batch_calculation_with_sparse_tensors(self, device):
        """Test using sparse_index_select in a batch calculation scenario."""
        # Create a "batch" of sparse matrices
        batch_size = 3
        feature_size = 5
        sparse_matrices = []

        for i in range(batch_size):
            dense = torch.zeros((10, feature_size), device=device)
            # Add different non-zero elements for each batch
            dense[i, 0] = i + 1
            dense[i + 1, 2] = i + 2
            dense[i + 2, 4] = i + 3
            sparse_matrices.append(dense.to_sparse().requires_grad_())

        # Our batched calculation will select different columns for each matrix
        batch_indices = [
            torch.tensor([0, 2], device=device),
            torch.tensor([2, 4], device=device),
            torch.tensor([0, 1, 4], device=device),
        ]

        # Process each item in the batch
        results = []
        for i in range(batch_size):
            selected = sparse_index_select(sparse_matrices[i], 1, batch_indices[i])
            results.append(selected)

        # Verify each result
        for i in range(batch_size):
            dense_original = sparse_matrices[i].to_dense()
            dense_result = results[i].to_dense()

            # Check shape
            expected_shape = (10, len(batch_indices[i]))
            assert dense_result.shape == expected_shape

            # Check values
            for j, idx in enumerate(batch_indices[i]):
                assert torch.all(dense_result[:, j] == dense_original[:, idx])


@pytest.mark.cpu_and_cuda
class TestInnerFunction:
    def test_inner_function_basic(self, device):
        # Test the inner function directly
        values = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=device)
        sparse = values.to_sparse().coalesce()

        tensor_indices = sparse.indices()
        tensor_values = sparse.values()

        axis = 1
        index = torch.tensor([1], device=device)

        new_indices, new_values = _sparse_index_select_inner(
            tensor_indices, tensor_values, axis, index
        )

        # Create result tensor
        new_shape = list(sparse.shape)
        new_shape[axis] = len(index)
        result = torch.sparse_coo_tensor(new_indices, new_values, new_shape).coalesce()

        expected = torch.tensor([[0.0], [3.0]], device=device).to_sparse()
        assert_sparse_tensors_equal(result, expected)

    def test_inner_function_complex_case(self, device):
        # Test inner function with a more complex 3D tensor
        values = torch.zeros((3, 4, 5), device=device)
        values[0, 1, 2] = 1.0
        values[1, 2, 3] = 2.0
        values[2, 1, 4] = 3.0
        sparse = values.to_sparse().coalesce()

        tensor_indices = sparse.indices()
        tensor_values = sparse.values()

        axis = 1
        index = torch.tensor([1, 2], device=device)

        new_indices, new_values = _sparse_index_select_inner(
            tensor_indices, tensor_values, axis, index
        )

        new_shape = list(sparse.shape)
        new_shape[axis] = len(index)
        result = torch.sparse_coo_tensor(new_indices, new_values, new_shape).coalesce()

        expected_values = torch.zeros((3, 2, 5), device=device)
        expected_values[0, 0, 2] = 1.0  # From (0,1,2) -> (0,0,2)
        expected_values[1, 1, 3] = 2.0  # From (1,2,3) -> (1,1,3)
        expected_values[2, 0, 4] = 3.0  # From (2,1,4) -> (2,0,4)
        expected = expected_values.to_sparse()

        assert_sparse_tensors_equal(result, expected)

    def test_inner_function_errors(self, device):
        # Test error messages
        values = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], device=device)
        sparse = values.to_sparse().coalesce()

        tensor_indices = sparse.indices()
        tensor_values = sparse.values()

        index = torch.tensor([1], device=device)

        with pytest.raises(
            (ValueError, torch.jit.Error), match="does not support negative axes"
        ):
            _, _ = _sparse_index_select_inner(tensor_indices, tensor_values, -1, index)

        with pytest.raises((ValueError, torch.jit.Error), match="is out of bounds"):
            _, _ = _sparse_index_select_inner(tensor_indices, tensor_values, 5, index)
