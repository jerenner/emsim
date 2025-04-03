import pytest
import torch

from emsim.utils.sparse_utils.base import (
    batch_sparse_index_linear,
    batch_sparse_index_subset_attn,
)

from .constants import EMBED_DIM, N_HEADS


@pytest.mark.cuda
@pytest.mark.parametrize(
    "include_bias", [True, False], ids=["include_bias=True", "include_bias=False"]
)
def test_end_to_end_gather_linear(
    setup_sparse_tensor, setup_linear_index_tensor, include_bias, device
):
    """Test end-to-end gather and linear mapping."""
    sparse_tensor = setup_sparse_tensor.to(device)
    index_tensor = setup_linear_index_tensor.to(device)

    # Initialize parameters
    weight = torch.randn(
        EMBED_DIM, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    bias = (
        torch.randn(EMBED_DIM, dtype=torch.double, requires_grad=True, device=device)
        if include_bias
        else None
    )

    # Run the operation
    transformed, is_specified_mask = batch_sparse_index_linear(
        sparse_tensor, index_tensor, weight, bias
    )

    # Check output shape
    assert transformed.shape == (index_tensor.shape[0], EMBED_DIM)
    assert is_specified_mask.shape == (index_tensor.shape[0],)

    # Compute loss and check gradient flow
    loss = transformed.sum()
    loss.backward()

    assert weight.grad is not None
    if include_bias:
        assert bias.grad is not None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "scale_factor", [None, 0.5], ids=["scale_factor=None", "scale_factor=0.5"]
)
def test_end_to_end_subset_attn(
    setup_sparse_tensor, setup_attention_index_tensor, scale_factor, device
):
    """Test end-to-end gather and subset attention."""
    sparse_tensor = setup_sparse_tensor.to(device)
    index_tensor = setup_attention_index_tensor.to(device)

    # Create query vectors
    query_tensor = torch.randn(
        index_tensor.shape[0], EMBED_DIM, dtype=torch.double, requires_grad=True,
        device=device
    )

    # Create attention parameters
    key_weight = torch.randn(
        EMBED_DIM, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    value_weight = torch.randn(
        EMBED_DIM, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    key_bias = torch.randn(EMBED_DIM, dtype=torch.double, requires_grad=True, device=device)
    value_bias = torch.randn(EMBED_DIM, dtype=torch.double, requires_grad=True, device=device)

    # Run the operation
    attended, is_specified_mask = batch_sparse_index_subset_attn(
        sparse_tensor,
        index_tensor,
        query_tensor,
        N_HEADS,
        key_weight,
        value_weight,
        key_bias,
        value_bias,
        scale_factor,
    )

    # Check output shapes
    expected_output_shape = list(query_tensor.shape[:-1])
    expected_output_shape.append(EMBED_DIM)
    assert attended.shape == tuple(expected_output_shape)
    assert is_specified_mask.shape == tuple(index_tensor.shape[:-1])

    # Compute loss and check gradient flow
    loss = attended.sum()
    loss.backward()

    assert query_tensor.grad is not None
    assert key_weight.grad is not None
    assert value_weight.grad is not None
    assert key_bias.grad is not None
    assert value_bias.grad is not None
