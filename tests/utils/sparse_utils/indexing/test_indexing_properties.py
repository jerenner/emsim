import pytest
import torch
from torch import nn
import numpy as np

from hypothesis import strategies as st
from hypothesis import given, settings

from emsim.utils.sparse_utils.indexing import batch_sparse_index


@pytest.mark.cuda_if_available
class TestBatchSparseIndexProperties:

    @settings(deadline=None)
    @given(
        sparse_dims=st.lists(
            st.integers(min_value=1, max_value=8),
            min_size=1,  # need at least 1 sparse dim
            max_size=3,
        ),
        dense_dims=st.lists(
            st.integers(min_value=1, max_value=8),
            min_size=0,
            max_size=3,
        ),
        sparsity=st.floats(0.0, 1.0),
        dtype=st.sampled_from([torch.float, torch.double, torch.float16]),
        n_to_select=st.integers(0, 10),
        # Specify what percentage of queried indices should be present in sparse tensor
        presence_ratio=st.floats(min_value=0.2, max_value=0.8),
    )
    def test_batch_sparse_index_gradient_property(
        self,
        sparse_dims: list[int],
        dense_dims: list[int],
        sparsity: float,
        dtype: torch.dtype,
        n_to_select: int,
        presence_ratio: float,
        device: str,
    ) -> None:
        """Property-based test that verifies gradients from batch_sparse_index
        match equivalent dense tensor operations.

        This test:
        1. Creates sparse and dense tensors with identical values
        2. Selects values using batch_sparse_index and normal indexing
        3. Passes both through identical MLPs
        4. Compares gradients to verify they match
        5. Systematically varies tensor dimensions, sparsity, and data types
        """
        # Create a tensor with known sparse elements for easier testing
        tensor_shape = tuple(sparse_dims + dense_dims)
        dense_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)

        # Add non-zero elements at predictable positions
        # We'll put some non-zero values in about 20% of the positions
        n_elements = np.prod(sparse_dims)
        n_nonzero = max(1, int(n_elements * (1 - sparsity)))

        # Generate random indices for non-zero elements
        # Duplicate indices OK since they'll be coalesced.
        indices = torch.stack(
            [
                torch.randint(0, dim_size, (n_nonzero,), device=device)
                for dim_size in sparse_dims
            ],
            dim=0,  # indices are stacked in (D, N) format
        )
        values = torch.randn(
            [n_nonzero] + dense_dims,
            dtype=dtype,
            device=device,
            # requires_grad=True,
        )

        # make sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices,
            values,
            tensor_shape,
            device=device,
            dtype=dtype,
            # requires_grad=True,
        ).coalesce()
        sparse_tensor.requires_grad_(True)

        # make dense copy of tensor
        dense_tensor = sparse_tensor.clone().detach().to_dense().requires_grad_(True)

        # Generate indices to select: mixture of known-specified and random
        n_selected_specified = min(int(n_to_select * presence_ratio), n_nonzero)
        # random indices with replacement
        selected_indices = torch.randint(
            0, n_nonzero, (n_selected_specified,), device=device
        )
        select_indices_specified = indices[:, selected_indices].T

        n_selected_random = n_to_select - n_selected_specified
        random_indices = [
            torch.randint(0, dim_size, (n_selected_random,), device=device)
            for dim_size in sparse_dims
        ]
        selected_random = torch.stack(random_indices, dim=-1)

        # Stacked: (D, N). Transposed: (N, D)
        index_tensor = torch.cat([select_indices_specified, selected_random], 0)

        # Use batch_sparse_index
        sparse_selected, is_specified = batch_sparse_index(sparse_tensor, index_tensor)

        # Use dense indexing
        dense_selected = dense_tensor[index_tensor.unbind(-1)]

        # Double-check the values are the same
        assert torch.equal(sparse_selected, dense_selected)

        # Mask out unspecified part of dense part to stop grads
        dense_selected = dense_selected.masked_fill(
            ~is_specified.view(is_specified.shape + (1,) * len(dense_dims)), 0.0
        )

        # Create a simple mlp to process the selected values
        if len(dense_dims) == 0:
            # No dense dims - the mlp operates on scalar values
            mlp = nn.Sequential(
                nn.Linear(1, 4, device=device, dtype=dtype),
                nn.ReLU(),
                nn.Linear(4, 1, device=device, dtype=dtype),
            )
            sparse_selected = sparse_selected.unsqueeze(-1)
            dense_selected = dense_selected.unsqueeze(-1)
        else:
            # With dense dims - the mlp operates on last dim dimension with others as
            # batch dims
            mlp = nn.Sequential(
                nn.Linear(dense_dims[-1], 4, device=device, dtype=dtype),
                nn.ReLU(),
                nn.Linear(4, 1, device=device, dtype=dtype),
            )

        sparse_output = mlp(sparse_selected)
        dense_output = mlp(dense_selected)

        # check outputs equal
        assert torch.equal(sparse_output, dense_output)

        # Create target and losses
        target = torch.ones_like(sparse_output)
        sparse_loss = nn.functional.mse_loss(sparse_output, target)
        dense_loss = nn.functional.mse_loss(dense_output, target)

        assert torch.equal(sparse_loss, dense_loss) or (
            torch.isnan(sparse_loss) and torch.isnan(dense_loss)
        )

        # backprop both losses
        sparse_loss.backward()
        dense_loss.backward()

        sparse_grad = sparse_tensor.grad.clone().detach()
        sparse_selected_grad, _ = batch_sparse_index(sparse_grad, index_tensor)

        dense_selected_grad = dense_tensor.grad.clone()[index_tensor.unbind(-1)]

        # compare
        if dtype == torch.float16:
            atol = 1e-3
        else:
            atol = 1e-8  # default
        assert torch.allclose(sparse_selected_grad, dense_selected_grad, atol=atol), (
            "max diff: "
            f"{torch.abs(sparse_selected_grad - dense_selected_grad).max()}"
        )
