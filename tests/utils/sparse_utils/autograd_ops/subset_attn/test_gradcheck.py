import pytest
import torch

from emsim.utils.sparse_utils.indexing.script_funcs import get_sparse_index_mapping
from emsim.utils.sparse_utils.ops.subset_attn.autograd import (
    GatherAndSubsetAttentionFunction,
)

from ..constants import (
    EMBED_DIM,
    N_FREQ_GROUPS,
    N_HEADS,
    N_KEYS_PER_QUERY,
    POSITION_DIM,
)


@pytest.mark.parametrize(
    "key_pos_encoding_type",
    ["given", "computed", None],
    ids=[
        "key_pos_encoding_type=given",
        "key_pos_encoding_type=computed",
        "key_pos_encoding_type=None",
    ],
)
@pytest.mark.cuda_if_available
def test_gather_and_subset_attention_function(
    setup_sparse_tensor, setup_attention_index_tensor, device, key_pos_encoding_type
):
    """Test gradient computation for gather and subset attention function."""
    sparse_tensor = setup_sparse_tensor
    index_tensor = setup_attention_index_tensor

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
    key_pos_encoding = (
        torch.randn(
            index_tensor.shape[0],
            N_KEYS_PER_QUERY,
            N_HEADS,
            EMBED_DIM // N_HEADS // 2,
            dtype=torch.double,
            requires_grad=True,
            device=device,
        )
        if key_pos_encoding_type == "given"
        else None
    )
    key_positions = (
        torch.randn(
            index_tensor.shape[0],
            N_KEYS_PER_QUERY,
            POSITION_DIM,
            dtype=torch.double,
            requires_grad=True,
            device=device,
        )
        if key_pos_encoding_type == "computed"
        else None
    )
    rope_freqs = (
        torch.randn(
            POSITION_DIM,
            N_FREQ_GROUPS,
            N_HEADS,
            EMBED_DIM // N_HEADS // 2,
            dtype=torch.double,
            requires_grad=True,
            device=device,
        )
        if key_pos_encoding_type == "computed"
        else None
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
        key_pos_encoding,
        key_positions,
        rope_freqs,
        scale_factor,
    )
    assert torch.autograd.gradcheck(GatherAndSubsetAttentionFunction.apply, inputs)


@pytest.mark.cuda_if_available
@pytest.mark.parametrize(
    "param_name, param_index",
    [
        ("query_tensor", 0),
        ("sparse_tensor_values", 2),
        ("key_weight", 5),
        ("value_weight", 6),
        ("key_bias", 7),
        ("value_bias", 8),
        ("key_pos_encoding", 9),
        ("key_positions", 10),
        ("rope_freqs", 11),
    ],
)
@pytest.mark.parametrize(
    "key_pos_encoding_type",
    ["given", "computed", None],
    ids=[
        "key_pos_encoding_type=given",
        "key_pos_encoding_type=computed",
        "key_pos_encoding_type=None",
    ],
)
def test_gradients_per_parameter(
    setup_sparse_tensor,
    setup_attention_index_tensor,
    device,
    key_pos_encoding_type,
    param_name,
    param_index,
):
    """Test gradients individually for each parameter."""
    sparse_tensor = setup_sparse_tensor
    index_tensor = setup_attention_index_tensor

    if (
        (
            key_pos_encoding_type == "given"
            and param_name
            in (
                "key_positions",
                "rope_freqs",
            )
        )
        or (key_pos_encoding_type == "computed" and param_name == "key_pos_encoding")
        or (
            key_pos_encoding_type is None
            and param_name in ("key_pos_encoding", "key_positions", "rope_freqs")
        )
    ):
        pytest.skip("Invalid param combination")

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
    key_pos_encoding = (
        torch.randn(
            index_tensor.shape[0],
            N_KEYS_PER_QUERY,
            N_HEADS,
            EMBED_DIM // N_HEADS // 2,
            dtype=torch.double,
            device=device,
        )
        if key_pos_encoding_type == "given"
        else None
    )
    key_positions = (
        torch.randn(
            index_tensor.shape[0],
            N_KEYS_PER_QUERY,
            POSITION_DIM,
            dtype=torch.double,
            device=device,
        )
        if key_pos_encoding_type == "computed"
        else None
    )
    rope_freqs = (
        torch.randn(
            POSITION_DIM,
            N_FREQ_GROUPS,
            N_HEADS,
            EMBED_DIM // N_HEADS // 2,
            dtype=torch.double,
            device=device,
        )
        if key_pos_encoding_type == "computed"
        else None
    )
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
        key_pos_encoding,
        key_positions,
        rope_freqs,
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
        9,
        10,
        11,
    ]  # Indices of tensors that could require gradients
    for idx in tensor_indices:
        if idx == param_index and inputs[idx] is not None:
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
        result = torch.autograd.gradcheck(check_single_param, (param,))
        assert result, f"Gradient check failed for {param_name}"
    except Exception as e:
        pytest.fail(f"Gradient check error for {param_name}: {str(e)}")
