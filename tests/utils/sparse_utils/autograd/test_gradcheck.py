import pytest
import torch
from torch.autograd import gradcheck

# Import module functions
from emsim.utils.sparse_utils.base import (
    GatherAndLinearFunction,
    GatherAndSubsetAttentionFunction,
    get_sparse_index_mapping,
)

from .constants import EMBED_DIM, N_HEADS


@pytest.mark.cuda
def test_gather_and_linear_function(
    setup_sparse_tensor, setup_linear_index_tensor, device
):
    """Test gradient computation for gather and linear function."""
    sparse_tensor = setup_sparse_tensor.to(device)
    index_tensor = setup_linear_index_tensor.to(device)

    # Get index mapping
    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = sparse_tensor.values()
    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )

    # Initialize parameters
    weight = torch.randn(
        EMBED_DIM, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    bias = torch.randn(EMBED_DIM, dtype=torch.double, requires_grad=True, device=device)

    # Run gradcheck
    inputs = (sparse_tensor_values, index_search, is_specified_mask, weight, bias)
    assert gradcheck(GatherAndLinearFunction.apply, inputs)


@pytest.mark.cuda
def test_gather_and_subset_attention_function(
    setup_sparse_tensor, setup_attention_index_tensor, device
):
    """Test gradient computation for gather and subset attention function."""
    sparse_tensor = setup_sparse_tensor.to(device)
    index_tensor = setup_attention_index_tensor.to(device)

    # Get index mapping
    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = sparse_tensor.values()
    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )

    # Initialize parameters
    query_tensor = torch.randn(
        index_tensor.shape[0],
        EMBED_DIM,
        dtype=torch.double,
        requires_grad=True,
        device=device,
    )
    key_weight = torch.randn(
        EMBED_DIM, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    value_weight = torch.randn(
        EMBED_DIM, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    key_bias = torch.randn(
        EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    value_bias = torch.randn(
        EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )
    scale_factor = None

    # Run gradcheck
    inputs = (
        query_tensor,
        N_HEADS,
        sparse_tensor_values,
        index_search,
        is_specified_mask,
        key_weight,
        value_weight,
        key_bias,
        value_bias,
        scale_factor,
    )
    assert gradcheck(GatherAndSubsetAttentionFunction.apply, inputs)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "param_name, param_index",
    [
        ("query_tensor", 0),
        ("sparse_tensor_values", 2),
        ("key_weight", 5),
        ("value_weight", 6),
        ("key_bias", 7),
        ("value_bias", 8),
    ],
)
def test_gradients_per_parameter(
    setup_sparse_tensor, setup_attention_index_tensor, param_name, param_index, device
):
    """Test gradients individually for each parameter."""
    sparse_tensor = setup_sparse_tensor.to(device)
    index_tensor = setup_attention_index_tensor.to(device)

    # Get index mapping
    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = (
        sparse_tensor.values().detach().clone().double()
    )  # Detach and no grad by default
    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )

    # Initialize parameters without requiring gradients
    query_tensor = torch.randn(
        index_tensor.shape[0], EMBED_DIM, dtype=torch.double, device=device
    )
    key_weight = torch.randn(EMBED_DIM, EMBED_DIM, dtype=torch.double, device=device)
    value_weight = torch.randn(EMBED_DIM, EMBED_DIM, dtype=torch.double, device=device)
    key_bias = torch.randn(EMBED_DIM, dtype=torch.double, device=device)
    value_bias = torch.randn(EMBED_DIM, dtype=torch.double, device=device)
    scale_factor = None

    # Base inputs
    inputs = [
        query_tensor,
        N_HEADS,
        sparse_tensor_values,
        index_search,
        is_specified_mask,
        key_weight,
        value_weight,
        key_bias,
        value_bias,
        scale_factor,
    ]

    # Enable gradients only for the parameter being tested
    tensor_indices = [
        0,
        2,
        5,
        6,
        7,
        8,
    ]  # Indices of tensors that could require gradients
    for idx in tensor_indices:
        if idx == param_index:
            inputs[idx] = inputs[idx].clone().requires_grad_(True)

    # The parameter we're testing
    param = inputs[param_index]

    # Create a test function that only varies our target parameter
    def check_single_param(x):
        inputs_copy = list(inputs)
        inputs_copy[param_index] = x
        return GatherAndSubsetAttentionFunction.apply(*inputs_copy)

    # Run gradcheck for this parameter only
    try:
        result = gradcheck(check_single_param, (param,))
        assert result, f"Gradient check failed for {param_name}"
    except Exception as e:
        pytest.fail(f"Gradient check error for {param_name}: {str(e)}")
