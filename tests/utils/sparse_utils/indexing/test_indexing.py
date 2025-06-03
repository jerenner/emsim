import pytest
import torch
from torch import Tensor, nn

from emsim.utils.sparse_utils.indexing.indexing import (
    sparse_select,
    batch_sparse_index,
    union_sparse_indices,
)


@pytest.fixture
def simple_sparse_tensor(device):
    # Create a simple 2D sparse tensor wtih no dense dim
    i = torch.tensor([[0, 0], [0, 1], [1, 1], [2, 2]], device=device).T
    v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    shape = (3, 3)
    sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()
    return sparse_tensor


def copy_sparse_tensor_and_set_requires_grad(
    sparse_tensor: Tensor,
) -> tuple[Tensor, Tensor]:
    assert sparse_tensor.is_sparse
    new_values = sparse_tensor.values().clone().requires_grad_(True)
    new_tensor = torch.sparse_coo_tensor(
        sparse_tensor.indices(),
        new_values,
        sparse_tensor.shape,
        device=sparse_tensor.device,
        is_coalesced=sparse_tensor.is_coalesced(),
    )
    return new_tensor, new_values


@pytest.mark.cpu_and_cuda
class TestSparseSelect:
    def test_basic_functionality(
        self, simple_sparse_tensor: Tensor, device: str
    ) -> None:
        """Test basic functionality of sparse_select."""
        sparse_tensor = simple_sparse_tensor
        assert (
            sparse_tensor.device.type == device
        )  # Test fixture places on right device

        # Select row 1
        result = sparse_select(sparse_tensor, 0, 1)

        # Expected: a 1D tensor with entries from row 1
        expected = torch.zeros(3, device=device)
        expected[1] = 3.0
        assert result.shape == (3,)
        assert torch.allclose(result.to_dense(), expected)

        # Select column 1
        result = sparse_select(sparse_tensor, 1, 1)

        # Expected: a 1D tensor with entries from column 1
        expected = torch.zeros(3, device=device)
        expected[0] = 2.0
        expected[1] = 3.0
        assert result.shape == (3,)
        assert torch.allclose(result.to_dense(), expected)

    def test_negative_indexing(self, simple_sparse_tensor: Tensor, device: str) -> None:
        """Test sparse_select with negative indices."""
        sparse_tensor = simple_sparse_tensor

        # Select last row with negative index
        result = sparse_select(sparse_tensor, 0, -1)

        # Expected: same as selecting row 2
        expected = torch.zeros(3, device=device)
        expected[2] = 4.0
        assert result.shape == (3,)
        assert torch.allclose(result.to_dense(), expected)

    def test_3d_tensor(self, device: str) -> None:
        """Test sparse_select with a 3D tensor."""
        # Create a 3D sparse tensor
        i = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]], device=device).T
        i = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (2, 2, 2)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Select along dim 0
        result = sparse_select(sparse_tensor, 0, 0)

        # Expected: a 2D tensor with entries where dim 0 = 0
        expected_shape = (2, 2)
        expected_dense = torch.zeros(expected_shape, device=device)
        expected_dense[0, 0] = 1.0
        expected_dense[1, 1] = 2.0

        assert result.shape == expected_shape
        assert torch.allclose(result.to_dense(), expected_dense)

        # Select along dim 1
        result = sparse_select(sparse_tensor, 1, 1)

        # Expected: a 2D tensor with entries where dim 1 = 1
        expected_dense = torch.zeros(expected_shape, device=device)
        expected_dense[0, 1] = 2.0
        expected_dense[1, 0] = 4.0

        assert result.shape == expected_shape
        assert torch.allclose(result.to_dense(), expected_dense)

    def test_hybrid_tensor(self, device: str) -> None:
        """Test sparse_select with a hybrid tensor (sparse+dense dimensions)."""
        # Create a hybrid tensor: 2 sparse dims, 1 dense dim
        i = torch.tensor([[0, 0], [1, 0], [1, 1]], device=device).T
        v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        shape = (2, 2, 2)  # 2 sparse dims, 1 dense dim of size 2
        hybrid_sparse = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Select along sparse dim 0
        result = sparse_select(hybrid_sparse, 0, 1)

        # Expected: a tensor with shape (2, 2) - remaining sparse dim and dense dim
        expected_shape = (2, 2)
        expected_dense = torch.zeros(expected_shape, device=device)
        expected_dense[0, 0] = 3.0
        expected_dense[0, 1] = 4.0
        expected_dense[1, 0] = 5.0
        expected_dense[1, 1] = 6.0

        assert result.shape == expected_shape
        assert torch.allclose(result.to_dense(), expected_dense)

        # Select along dense dimension
        result = sparse_select(hybrid_sparse, 2, 0)

        # Expected: a tensor with shape (2, 2) - both sparse dims
        expected_shape = (2, 2)
        expected_dense = torch.zeros(expected_shape, device=device)
        expected_dense[0, 0] = 1.0
        expected_dense[1, 0] = 3.0
        expected_dense[1, 1] = 5.0

        assert result.shape == expected_shape
        assert torch.allclose(result.to_dense(), expected_dense)

    def test_error_handling(self, simple_sparse_tensor: Tensor) -> None:
        """Test error handling in sparse_select."""
        sparse_tensor = simple_sparse_tensor

        # Test with non-sparse tensor
        dense_tensor = sparse_tensor.to_dense()
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="not sparse",
        ):
            sparse_select(dense_tensor, 0, 0)

        # Test with invalid axis
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="out of bounds",
        ):
            sparse_select(sparse_tensor, 3, 0)

        # Test with invalid index
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="out of bounds",
        ):
            sparse_select(sparse_tensor, 0, 5)

    def test_gradient_flow(self, simple_sparse_tensor: Tensor, device: str) -> None:
        """Test that gradients flow correctly through sparse_select."""
        sparse_tensor = simple_sparse_tensor
        # Make copy that requires grad
        sparse_tensor, v = copy_sparse_tensor_and_set_requires_grad(sparse_tensor)

        # Select row 1 and compute loss
        result = sparse_select(sparse_tensor, 0, 1)
        loss = result.to_dense().sum()
        loss.backward()

        # Check gradients - only the entry at (1,1) should have gradient 1.0
        expected_grad = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
        assert v.grad is not None
        assert torch.allclose(v.grad, expected_grad)

    def test_native_sparse_indexing_does_not_support_gradients(
        self, simple_sparse_tensor: Tensor
    ) -> None:
        sparse_tensor = simple_sparse_tensor
        sparse_tensor, _ = copy_sparse_tensor_and_set_requires_grad(sparse_tensor)

        selected = sparse_tensor[0]
        loss = selected.to_dense().sum()
        with pytest.raises(
            NotImplementedError, match="Could not run 'aten::select_backward'"
        ):
            loss.backward()

    def test_native_indexing_vs_sparse_select_gradients(self, device: str) -> None:
        """Test that sparse_select supports gradients while native indexing may not."""
        # Create a simple sparse tensor that requires gradients
        i = torch.tensor([[0, 0], [1, 1], [2, 2]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        sparse_tensor = torch.sparse_coo_tensor(i, v, (3, 3)).coalesce()

        # Try native PyTorch indexing again
        try:
            native_selected = sparse_tensor[1]  # Select row 1
            loss = native_selected.to_dense().sum()
            loss.backward()
            native_supports_gradients = True
        except Exception:
            native_supports_gradients = False

        assert not native_supports_gradients

        # Now try our custom sparse_select on a fresh tensor
        v_new = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        sparse_tensor_new = torch.sparse_coo_tensor(i, v_new, (3, 3)).coalesce()

        selected = sparse_select(sparse_tensor_new, 0, 1)  # Select row 1
        loss = selected.to_dense().sum()
        loss.backward()

        # Check our function supports gradients correctly
        expected_grad = torch.tensor([0.0, 1.0, 0.0], device=device)
        assert v_new.grad is not None
        assert torch.allclose(v_new.grad, expected_grad)

    def test_sparse_select_with_mlp(
        self, simple_sparse_tensor: Tensor, device: str
    ) -> None:
        """Test gradients flow correctly when sparse_select values are used in an MLP."""
        sparse_tensor = simple_sparse_tensor
        sparse_tensor, v = copy_sparse_tensor_and_set_requires_grad(sparse_tensor)

        # Create a simple MLP
        mlp = nn.Sequential(
            nn.Linear(3, 5, device=device), nn.ReLU(), nn.Linear(5, 1, device=device)
        )

        # Initialize weights deterministically
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Use sparse_select to get a row and pass through MLP
        selected = sparse_select(sparse_tensor, 0, 1)  # Select row 1
        output = mlp(selected.to_dense().unsqueeze(0))  # add batch dim with unsqueeze

        # Compute loss and gradients
        target = torch.tensor([[1.0]], device=device)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Verify gradients flowed back to the original sparse tensor values
        assert v.grad is not None

        # Check which values have gradients
        # In simple_sparse_tensor, the value at (1,1) is at index 2
        assert v.grad[2] != 0  # (1,1) was selected in row 1
        assert v.grad[0] == 0  # (0,0) was not in row 1
        assert v.grad[1] == 0  # (0,1) was not in row 1
        assert v.grad[3] == 0  # (2,2) was not in row 1

    def test_hybrid_sparse_select_with_mlp(self, device: str) -> None:
        """Test gradient flow through an MLP for hybrid sparse tensor with sparse_select."""
        # Create a hybrid tensor: 2 sparse dims, 1 dense dim
        i = torch.tensor([[0, 0], [1, 0], [1, 1]], device=device).T
        # i = torch.tensor([[0, 1, 1], [0, 0, 1]], device=device).T
        v = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device, requires_grad=True
        )
        shape = (2, 2, 2)  # 2 sparse dims, 1 dense dim of size 2
        hybrid_sparse = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Create a simple MLP for the dense features
        mlp = nn.Sequential(
            nn.Linear(2, 4, device=device),  # Input size is dense dim size
            nn.ReLU(),
            nn.Linear(4, 1, device=device),
        )

        # Initialize weights deterministically
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Use sparse_select
        selected = sparse_select(hybrid_sparse, 0, 1)

        selected_values = selected.values()

        output = mlp(selected_values)
        target = torch.ones_like(output)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients
        assert v.grad is not None

        # Values at positions in row 1 should have gradients
        assert torch.all(v.grad[1] != 0)  # (1,0)
        assert torch.all(v.grad[2] != 0)  # (1,1)
        # Value at (0,0) was not selected, should be zero
        assert torch.all(v.grad[0] == 0)


@pytest.mark.cpu_and_cuda
class TestBatchSparseIndex:
    def test_basic_functionality(
        self, simple_sparse_tensor: Tensor, device: str
    ) -> None:
        """Test basic functionality of batch_sparse_index."""
        sparse_tensor = simple_sparse_tensor

        # Create a batch of indices to look up
        index_tensor = torch.tensor([[0, 0], [1, 1], [2, 2]], device=device)

        # Get selected values and specified mask
        values, mask = batch_sparse_index(sparse_tensor, index_tensor)

        # Expected values: [1.0, 3.0, 4.0] for each row in index_tensor
        expected_values = torch.tensor([1.0, 3.0, 4.0], device=device)
        expected_mask = torch.tensor([True, True, True], device=device)

        assert torch.allclose(values, expected_values)
        assert torch.all(mask == expected_mask)

    def test_with_unspecified_indices(
        self, simple_sparse_tensor: Tensor, device: str
    ) -> None:
        """Test batch_sparse_index with indices not present in sparse tensor."""
        sparse_tensor = simple_sparse_tensor

        # Create indices including ones not in the sparse tensor
        index_tensor = torch.tensor(
            [
                [0, 0],  # present at position 0
                [0, 2],  # not present
                [1, 1],  # present at position 2
                [2, 1],  # not present
            ],
            device=device,
        )

        # Get selected values and specified mask
        values, mask = batch_sparse_index(sparse_tensor, index_tensor)

        # Expected values: wherever mask is True, should have the right value from sparse_tensor
        expected_mask = torch.tensor([True, False, True, False], device=device)
        assert torch.all(mask == expected_mask)

        # For specified positions, check values
        assert values[0] == 1.0  # Value at (0,0)
        assert values[2] == 3.0  # Value at (1,1)

        # For unspecified positions, should have zeros
        assert values[1] == 0.0
        assert values[3] == 0.0

    def test_out_of_bounds_indices(
        self, simple_sparse_tensor: Tensor, device: str
    ) -> None:
        """Test batch_sparse_index with indices outside the tensor dimensions"""
        sparse_tensor = simple_sparse_tensor

        index_tensor = torch.tensor([[0, 0], [0, 3], [-1, 0]], device=device)

        values, mask = batch_sparse_index(sparse_tensor, index_tensor)

        # Expected mask: [True, False, False]
        expected_mask = torch.tensor([True, False, False], device=device)
        assert torch.equal(mask, expected_mask)
        assert values[0] == 1.0

    def test_check_all_specified(
        self, simple_sparse_tensor: Tensor, device: str
    ) -> None:
        """Test batch_sparse_index with check_all_specified=True."""
        # Create a simple 2D sparse tensor
        sparse_tensor = simple_sparse_tensor

        # All specified indices should work fine
        index_tensor = torch.tensor([[0, 0], [1, 1], [2, 2]], device=device)
        values, mask = batch_sparse_index(
            sparse_tensor, index_tensor, check_all_specified=True
        )
        assert torch.all(mask)

        # Including unspecified indices should raise error
        index_tensor = torch.tensor([[0, 0], [0, 2]], device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="not all gathered values were specified",
        ):
            batch_sparse_index(sparse_tensor, index_tensor, check_all_specified=True)

    def test_with_3d_sparse_tensor(self, device: str) -> None:
        """Test batch_sparse_index with a 3D sparse tensor."""
        # Create a 3D sparse tensor
        i = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], device=device).T
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        shape = (2, 2, 2)
        sparse_tensor = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Create indices to look up
        index_tensor = torch.tensor(
            [
                [0, 0, 0],  # exists with value 1.0
                [1, 1, 0],  # exists with value 4.0
                [0, 1, 1],  # exists with value 2.0
            ],
            device=device,
        )

        values, mask = batch_sparse_index(sparse_tensor, index_tensor)

        expected_values = torch.tensor([1.0, 4.0, 2.0], device=device)
        expected_mask = torch.tensor([True, True, True], device=device)

        assert torch.allclose(values, expected_values)
        assert torch.all(mask == expected_mask)

    def test_with_hybrid_tensor(self, device: str) -> None:
        """Test batch_sparse_index with a hybrid tensor."""
        # Create a hybrid tensor: 2 sparse dims, 1 dense dim
        i = torch.tensor([[0, 0], [1, 0], [1, 1]], device=device).T
        v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        shape = (2, 2, 2)  # 2 sparse dims, 1 dense dim of size 2
        hybrid_sparse = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Create indices to look up
        index_tensor = torch.tensor([[0, 0], [1, 1]], device=device)

        values, mask = batch_sparse_index(hybrid_sparse, index_tensor)

        # Expected shape: [batch_size, dense_dim] = [2, 2]
        assert values.shape == (2, 2)

        # Check specific values
        expected_values = torch.tensor(
            [
                [1.0, 2.0],  # Values for (0,0)
                [5.0, 6.0],  # Values for (1,1)
            ],
            device=device,
        )
        expected_mask = torch.tensor([True, True], device=device)

        assert torch.allclose(values, expected_values)
        assert torch.all(mask == expected_mask)

    def test_multiple_dense_dimensions(self, device: str) -> None:
        """Test batch_sparse_index with a tensor having multiple dense dimensions."""
        # Create a sparse tensor with 2 sparse dims, 2 dense dims
        i = torch.tensor([[0, 0], [1, 1], [0, 1]], device=device).T
        v = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],  # Value at (0,0): 2x2 matrix
                [[5.0, 6.0], [7.0, 8.0]],  # Value at (1,1): 2x2 matrix
                [[9.0, 10.0], [11.0, 12.0]],  # Value at (0,1): 2x2 matrix
            ],
            device=device,
        )
        shape = (2, 2, 2, 2)  # 2 sparse dims, 2 dense dims of size 2 each
        hybrid_sparse = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Create indices to look up
        index_tensor = torch.tensor([[0, 0], [1, 1]], device=device)

        values, mask = batch_sparse_index(hybrid_sparse, index_tensor)

        # Expected shape: [batch_size, dense_dim1, dense_dim2] = [2, 2, 2]
        assert values.shape == (2, 2, 2)

        # Check specific values
        expected_values = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],  # Values for (0,0)
                [[5.0, 6.0], [7.0, 8.0]],  # Values for (1,1)
            ],
            device=device,
        )
        expected_mask = torch.tensor([True, True], device=device)

        assert torch.allclose(values, expected_values)
        assert torch.all(mask == expected_mask)

    def test_complex_shape_structure(self, device: str) -> None:
        """Test batch_sparse_index with complex shapes for both sparse and dense dims."""
        # Create a sparse tensor with 3 sparse dims, 3 dense dims
        i = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 1, 0]], device=device).T

        # Create values with 3 dense dimensions
        v = torch.zeros((3, 2, 3, 2), device=device)
        v[0] = 1.0  # Value at (0,0,0)
        v[1] = 2.0  # Value at (1,1,1)
        v[2] = 3.0  # Value at (0,1,0)

        shape = (2, 2, 2, 2, 3, 2)  # 3 sparse dims, 3 dense dims
        hybrid_sparse = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Create a batch of indices with different batch dimensions
        index_tensor = torch.tensor(
            [[[0, 0, 0], [1, 1, 1]], [[0, 1, 0], [0, 0, 0]]], device=device
        )  # Shape: [2, 2, 3]

        values, mask = batch_sparse_index(hybrid_sparse, index_tensor)

        # Expected shape: [2, 2, 2, 3, 2] - batch dims + dense dims
        assert values.shape == (2, 2, 2, 3, 2)

        # Check specific values
        assert torch.all(values[0, 0] == 1.0)
        assert torch.all(values[0, 1] == 2.0)
        assert torch.all(values[1, 0] == 3.0)
        assert torch.all(values[1, 1] == 1.0)

        # Check mask shape and values
        assert mask.shape == (2, 2)
        assert torch.all(mask)

    def test_negative_and_out_of_bounds_indices(
        self, simple_sparse_tensor: Tensor, device: str
    ) -> None:
        """Test batch_sparse_index with negative and out-of-bounds indices."""
        sparse_tensor = simple_sparse_tensor

        # Create indices including negative and out-of-bounds
        index_tensor = torch.tensor(
            [
                [0, 0],  # valid
                [-1, 0],  # negative
                [0, 3],  # out of bounds
                [4, 1],  # out of bounds
            ],
            device=device,
        )

        values, mask = batch_sparse_index(sparse_tensor, index_tensor)

        # Expected mask: only the first index should be valid
        expected_mask = torch.tensor([True, False, False, False], device=device)
        assert torch.all(mask == expected_mask)

        # Valid entries should have correct values
        assert values[0] == 1.0

        # Invalid entries should be 0
        assert values[1] == 0.0
        assert values[2] == 0.0
        assert values[3] == 0.0

    def test_gradient_flow(self, simple_sparse_tensor: Tensor, device: str) -> None:
        """Test that gradients flow correctly through batch_sparse_index."""
        sparse_tensor = simple_sparse_tensor
        # Make copy that requires grad
        sparse_tensor, v = copy_sparse_tensor_and_set_requires_grad(sparse_tensor)

        # Create indices to look up
        index_tensor = torch.tensor([[0, 0], [1, 1]], device=device)

        # Get values and compute loss
        values, mask = batch_sparse_index(sparse_tensor, index_tensor)
        loss = values.sum()
        loss.backward()

        # Check gradients - entries corresponding to (0,0) and (1,1) should have gradient 1.0
        expected_grad = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)
        assert v.grad is not None
        assert torch.allclose(v.grad, expected_grad)

    def test_batched_multi_dense_gradient_flow(self, device: str) -> None:
        """Test gradient flow for sparse tensors with multiple dense dimensions."""
        # Create a sparse tensor with 2 sparse dims, 3 dense dims
        i = torch.tensor([[0, 0], [1, 1], [1, 0]], device=device).T
        v = torch.zeros((3, 2, 3, 2), device=device)
        # Set values with pattern for easy verification
        v[0, 0, 0, 0] = 1.0
        v[0, 0, 0, 1] = 2.0
        v[0, 0, 1, 0] = 3.0
        v[0, 1, 0, 0] = 4.0
        v[1] = 5.0  # All values in the 3D tensor at (1,1) set to 5

        v.requires_grad_(True)

        shape = (2, 2, 2, 3, 2)  # 2 sparse dims, 3 dense dims
        hybrid_sparse = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Create batched indice
        index_tensor = torch.tensor(
            [
                [0, 0],  # This exists (value filled with pattern)
                [1, 1],  # This exists (value filled with 5s)
                [0, 1],  # This doesn't exist
            ],
            device=device,
        )

        # Get selected values and compute loss
        values, mask = batch_sparse_index(hybrid_sparse, index_tensor)

        # Should be [3, 2, 3, 2] (3 batch items, 3 dense dims)
        assert values.shape == (3, 2, 3, 2)

        # Expected mask: [True, True, False]
        expected_mask = torch.tensor([True, True, False], device=device)
        assert torch.equal(mask, expected_mask)

        # Verify the non-zero values
        assert torch.all(values[1] == 5.0)  # All elements from (1,1) should be 5

        # Compute loss and gradients
        loss = values.sum()
        loss.backward()

        # Check gradients - only elements at (0,0) and (1,1) should have gradients
        assert v.grad is not None
        # Elements at (0,0) should have gradient 1.0
        assert torch.all(v.grad[0] == 1.0)
        # Elements at (1,1) should have gradient 1.0
        assert torch.all(v.grad[1] == 1.0)
        # Elements at (0,1) should have gradient 0.0
        assert torch.all(v.grad[2] == 0.0)

    def test_batch_sparse_index_with_mlp(
        self, simple_sparse_tensor: Tensor, device: str
    ) -> None:
        """Test gradients flow correctly when batch_sparse_index values are used
        in an MLP."""
        # Make copy of the tensor that requires grad
        sparse_tensor = simple_sparse_tensor
        sparse_tensor, v = copy_sparse_tensor_and_set_requires_grad(sparse_tensor)

        # Create a batch of indices to look up
        index_tensor = torch.tensor([[0, 0], [1, 1]], device=device)

        # Create a simple MLP for scalar inputs
        mlp = nn.Sequential(
            nn.Linear(1, 5, device=device), nn.ReLU(), nn.Linear(5, 1, device=device)
        )

        # Initialize weights deterministically
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Get selected values and process through MLP
        values, mask = batch_sparse_index(sparse_tensor, index_tensor)

        # Need to reshape for the MLP (add feature dim)
        values_reshaped = values.unsqueeze(-1)

        # Process through MLP
        output = mlp(values_reshaped)

        # Compute loss and gradients
        target = torch.tensor([[0.5], [1.0]], device=device)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Verify gradients flowed back to the original sparse tensor values
        assert v.grad is not None

        # Values used in the computation should have non-zero gradients
        assert v.grad[0] != 0  # (0,0)
        assert v.grad[2] != 0  # (1,1)

        # Values not used should have zero gradients
        assert v.grad[1] == 0  # (0,1)
        assert v.grad[3] == 0  # (2,2)

    def test_hybrid_batch_sparse_index_with_mlp(self, device: str) -> None:
        """Test gradient flow through an MLP for hybrid sparse tensor with
        batch_sparse_index."""
        # Create a hybrid tensor: 2 sparse dims, 1 dense dim
        i = torch.tensor([[0, 0], [1, 0], [1, 1]], device=device).T
        v = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device, requires_grad=True
        )
        shape = (2, 2, 2)  # 2 sparse dims, 1 dense dim of size 2
        hybrid_sparse = torch.sparse_coo_tensor(i, v, shape).coalesce()

        # Create a simple MLP for the dense features
        mlp = nn.Sequential(
            nn.Linear(2, 4, device=device),  # Input size is dense dim size
            nn.ReLU(),
            nn.Linear(4, 1, device=device),
        )

        # Initialize weights deterministically
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Create index tensor to select specific points
        index_tensor = torch.tensor([[1, 0], [1, 1]], device=device)

        # Get selected values
        values, mask = batch_sparse_index(hybrid_sparse, index_tensor)

        # Process through MLP
        output = mlp(values)
        target = torch.ones_like(output)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients
        assert v.grad is not None

        # Values at selected positions should have gradients
        assert torch.all(v.grad[1] != 0)  # (1,0)
        assert torch.all(v.grad[2] != 0)  # (1,1)
        # Value at (0,0) was not selected, should be zero
        assert torch.all(v.grad[0] == 0)

    def test_complex_batch_dims_with_mlp_dense_equivalence(self, device: str) -> None:
        """Test batched sparse indexing with multiple batch dims and processing
        through an MLP."""
        # Create a sparse tensor with 2 sparse dims, 3 dense dims
        i = torch.tensor([[0, 0], [1, 1], [1, 0]], device=device).T
        v = torch.zeros((3, 2, 3, 2), device=device)
        # Set values with pattern for easy verification
        v[0, 0, 0, 0] = 1.0
        v[0, 0, 0, 1] = 2.0
        v[0, 0, 1, 0] = 3.0
        v[0, 1, 0, 0] = 4.0
        v[1] = 5.0  # All values in the 3D tensor at (1,1) set to 5

        v.requires_grad_(True)

        shape = (2, 2, 2, 3, 2)  # 2 sparse dims, 3 dense dims
        sparse_tensor = torch.sparse_coo_tensor(
            i, v, shape, requires_grad=True
        ).coalesce()

        # Create batched indice
        index_tensor = torch.tensor(
            [
                [0, 0],  # This exists (value filled with pattern)
                [1, 1],  # This exists (value filled with 5s)
                [0, 1],  # This doesn't exist
            ],
            device=device,
        )

        # Create dense copy
        dense_tensor = sparse_tensor.clone().detach().to_dense().requires_grad_(True)

        # Create a simple MLP
        mlp = nn.Sequential(
            nn.Linear(2, 4, device=device), nn.ReLU(), nn.Linear(4, 1, device=device)
        )

        # Get selected values
        values_sparse, mask = batch_sparse_index(sparse_tensor, index_tensor)

        # Should be [3, 2, 3, 2] (3 batch items, 3 dense dims)
        assert values_sparse.shape == (3, 2, 3, 2)

        # Expected mask: [True, True, False]
        expected_mask = torch.tensor([True, True, False], device=device)
        assert torch.equal(mask, expected_mask)

        # Get same values from dense tensor
        # values_dense = dense_tensor[tuple(index_tensor.unbind(-1))]
        values_dense = dense_tensor[index_tensor.unbind(-1)]

        # Check values are equal
        assert torch.equal(values_sparse, values_dense)

        # Pass both through mlp
        output_sparse = mlp(values_sparse)
        output_dense = mlp(values_dense)

        # Check outputs equal
        assert torch.equal(output_sparse, output_dense)

        # Create target and loss
        target = torch.ones_like(output_sparse)
        loss_sparse = nn.functional.mse_loss(output_sparse, target)
        loss_dense = nn.functional.mse_loss(output_dense, target)

        # Backprop both losses
        loss_dense.backward()
        loss_sparse.backward()

        # Check the sparse and dense tensors have same gradients
        assert dense_tensor.grad is not None
        assert v.grad is not None
        dense_grads = dense_tensor.grad[index_tensor.unbind(-1)]
        assert torch.equal(v.grad[0], dense_grads[0])
        assert torch.equal(v.grad[1], dense_grads[1])

        assert torch.equal(v.grad[2], torch.zeros_like(v.grad[2]))  # Not accessed


@pytest.mark.cpu_and_cuda
class TestUnionSparseIndices:
    def test_basic_functionality(self, device):
        """Test basic functionality of union_sparse_indices."""
        # Create two sparse tensors with some overlapping indices
        i1 = torch.tensor([[0, 0], [1, 1], [2, 2]], device=device).T
        # i1 = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device).T
        v1 = torch.tensor([1.0, 2.0, 3.0], device=device)
        sparse1 = torch.sparse_coo_tensor(i1, v1, (3, 3)).coalesce()

        i2 = torch.tensor([[0, 1], [1, 1], [2, 0]], device=device).T
        # i2 = torch.tensor([[0, 1, 2], [1, 1, 0]], device=device).T
        v2 = torch.tensor([4.0, 5.0, 6.0], device=device)
        sparse2 = torch.sparse_coo_tensor(i2, v2, (3, 3)).coalesce()

        # Get the union
        union1, union2 = union_sparse_indices(sparse1, sparse2)

        # Check that the indices are the same in both
        assert torch.all(union1.indices() == union2.indices())

        # Check that the union has 5 indices (diagonal + (0,1) and (2,0))
        assert union1.indices().shape[1] == 5

        # Original values should be preserved
        assert union1[0, 0] == 1.0
        assert union1[1, 1] == 2.0
        assert union1[2, 2] == 3.0
        assert union2[0, 1] == 4.0
        assert union2[1, 1] == 5.0
        assert union2[2, 0] == 6.0

        # New positions should be zero
        assert union1[0, 1] == 0.0
        assert union1[2, 0] == 0.0
        assert union2[0, 0] == 0.0
        assert union2[2, 2] == 0.0

    def test_with_no_overlap(self, device):
        """Test union_sparse_indices with tensors having no overlapping indices."""
        # Create two sparse tensors with no overlapping indices
        i1 = torch.tensor([[0, 0], [1, 0]], device=device).T
        # i1 = torch.tensor([[0, 1], [0, 0]], device=device).T
        v1 = torch.tensor([1.0, 2.0], device=device)
        sparse1 = torch.sparse_coo_tensor(i1, v1, (3, 3)).coalesce()

        i2 = torch.tensor([[2, 1], [2, 2]], device=device).T
        # i2 = torch.tensor([[2, 2], [1, 2]], device=device).T
        v2 = torch.tensor([3.0, 4.0], device=device)
        sparse2 = torch.sparse_coo_tensor(i2, v2, (3, 3)).coalesce()

        # Get the union
        union1, union2 = union_sparse_indices(sparse1, sparse2)

        # Check that the indices are the same in both
        assert torch.all(union1.indices() == union2.indices())

        # Check that the union has 4 indices (all indices combined)
        assert union1.indices().shape[1] == 4

        # Original values should be preserved
        assert union1[0, 0] == 1.0
        assert union1[1, 0] == 2.0
        assert union2[2, 1] == 3.0
        assert union2[2, 2] == 4.0

        # New positions should be zero
        assert union1[2, 1] == 0.0
        assert union1[2, 2] == 0.0
        assert union2[0, 0] == 0.0
        assert union2[1, 0] == 0.0

    def test_identical_tensors(self, device):
        """Test union_sparse_indices with sparse tensors with identical indices."""
        # Create two sparse tensors with identical indices
        i = torch.tensor([[0, 0], [1, 1], [2, 2]], device=device).T
        # i = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device).T
        v1 = torch.tensor([1.0, 2.0, 3.0], device=device)
        v2 = torch.tensor([4.0, 5.0, 6.0], device=device)
        sparse1 = torch.sparse_coo_tensor(i, v1, (3, 3)).coalesce()
        sparse2 = torch.sparse_coo_tensor(i, v2, (3, 3)).coalesce()

        # Get the union
        union1, union2 = union_sparse_indices(sparse1, sparse2)

        # Check that the indices are the same in both
        assert torch.all(union1.indices() == union2.indices())

        # Check that the number of indices didn't change (all were already present)
        assert union1.indices().shape[1] == 3

        # Check values are preserved
        assert torch.allclose(union1.values(), v1)
        assert torch.allclose(union2.values(), v2)

    def test_with_hybrid_tensors(self, device):
        """Test union_sparse_indices with hybrid tensors having dense dimensions."""
        # Create two hybrid sparse tensors
        i1 = torch.tensor([[0, 0], [1, 1]], device=device).T
        v1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        hybrid1 = torch.sparse_coo_tensor(i1, v1, (2, 2, 2)).coalesce()

        i2 = torch.tensor([[1, 1], [0, 1]], device=device).T
        v2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
        hybrid2 = torch.sparse_coo_tensor(i2, v2, (2, 2, 2)).coalesce()

        # Get the union
        union1, union2 = union_sparse_indices(hybrid1, hybrid2)

        # Check that the indices are the same in both
        assert torch.all(union1.indices() == union2.indices())

        # Check that the union has 3 indices
        assert union1.indices().shape[1] == 3

        # Convert to dense to check values
        dense1 = union1.to_dense()
        dense2 = union2.to_dense()

        # Original values should be preserved
        assert torch.allclose(dense1[0, 0], torch.tensor([1.0, 2.0], device=device))
        assert torch.allclose(dense1[1, 1], torch.tensor([3.0, 4.0], device=device))
        assert torch.allclose(dense2[1, 1], torch.tensor([5.0, 6.0], device=device))
        assert torch.allclose(dense2[0, 1], torch.tensor([7.0, 8.0], device=device))

        # New positions should have zeros
        assert torch.allclose(dense1[0, 1], torch.tensor([0.0, 0.0], device=device))
        assert torch.allclose(dense2[0, 0], torch.tensor([0.0, 0.0], device=device))

    def test_error_cases(self, device):
        """Test error handling in union_sparse_indices."""
        # Create sparse and dense tensors
        i = torch.tensor([[0, 0], [1, 1]], device=device).T
        # i = torch.tensor([[0, 1], [0, 1]], device=device).T
        v = torch.tensor([1.0, 2.0], device=device)
        sparse = torch.sparse_coo_tensor(i, v, (2, 2)).coalesce()
        dense = torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=device)

        # Test with one dense tensor
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected two sparse tensors",
        ):
            union_sparse_indices(sparse, dense)

        # Test with tensors of different shapes
        sparse2 = torch.sparse_coo_tensor(i, v, (2, 3)).coalesce()
        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="Expected tensors to have same shapes",
        ):
            union_sparse_indices(sparse, sparse2)

        # Test with tensors having different number of sparse dims
        i3 = torch.tensor([[0], [1]], device=device).T
        v3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        # 1 sparse dim, 1 dense dim
        sparse3 = torch.sparse_coo_tensor(i3, v3, (2, 2)).coalesce()

        with pytest.raises(
            (ValueError, torch.jit.Error),  # pyright: ignore[reportArgumentType]
            match="equal numbers of sparse dims",
        ):
            union_sparse_indices(sparse, sparse3)

    def test_with_gradient_tracking(self, device):
        """Test union_sparse_indices with gradient tracking."""
        # Create two sparse tensors with gradient tracking
        i1 = torch.tensor([[0, 1], [0, 1]], device=device).T
        v1 = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
        sparse1 = torch.sparse_coo_tensor(i1, v1, (2, 2)).coalesce()

        i2 = torch.tensor([[1, 0], [0, 1]], device=device).T
        v2 = torch.tensor([3.0, 4.0], device=device, requires_grad=True)
        sparse2 = torch.sparse_coo_tensor(i2, v2, (2, 2)).coalesce()

        # Get the union
        union1, union2 = union_sparse_indices(sparse1, sparse2)

        # Compute loss using values from both
        loss = union1.values().sum() + union2.values().sum()
        loss.backward()

        # Check that gradients flowed back correctly
        assert v1.grad is not None
        assert v2.grad is not None
        assert torch.allclose(v1.grad, torch.tensor([1.0, 1.0], device=device))
        assert torch.allclose(v2.grad, torch.tensor([1.0, 1.0], device=device))
