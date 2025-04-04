import pytest
import torch

from emsim.utils.sparse_utils.base import (
    GatherAndSubsetAttentionFunction,
    get_sparse_index_mapping,
)

from .traceable_sparse_attn import traceable_sparse_attention
from ..constants import EMBED_DIM, N_HEADS, N_KEYS_PER_QUERY


@pytest.mark.parametrize(
    "use_pos_encoding", [True, False], ids=["pos_encoding=True", "pos_encoding=False"]
)
@pytest.mark.parametrize(
    "use_bias", [True, False], ids=["use_bias=True", "use_bias=False"]
)
@pytest.mark.parametrize(
    "scale_factor", [None, 0.5], ids=["scale_factor=None", "scale_factor=0.5"]
)
@pytest.mark.cuda
def test_subset_attn_against_traceable(
    setup_sparse_tensor,
    setup_attention_index_tensor,
    use_pos_encoding,
    use_bias,
    scale_factor,
    device,
):
    """Test that the custom attention operator produces the same results
    as a traceable implementation using standard PyTorch ops."""
    sparse_tensor = setup_sparse_tensor
    index_tensor = setup_attention_index_tensor

    # Create query tensor and projection matrices
    n_queries = index_tensor.shape[0]
    query_tensor = torch.randn(
        n_queries, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )

    # Create projection matrices
    key_weight = torch.randn(
        EMBED_DIM,
        EMBED_DIM,
        dtype=torch.double,
        requires_grad=True,
        device=device,
    )
    value_weight = torch.randn(
        EMBED_DIM, EMBED_DIM, dtype=torch.double, requires_grad=True, device=device
    )

    # Optional bias and positional encoding
    key_bias = (
        torch.randn(EMBED_DIM, dtype=torch.double, requires_grad=True, device=device)
        if use_bias
        else None
    )
    value_bias = (
        torch.randn(EMBED_DIM, dtype=torch.double, requires_grad=True, device=device)
        if use_bias
        else None
    )

    # RoPE requires even head_dim
    if use_pos_encoding:
        key_pos_encoding = torch.randn(
            n_queries,
            N_KEYS_PER_QUERY,
            EMBED_DIM,
            dtype=torch.double,
            device=device,
            requires_grad=True,
        )
    else:
        key_pos_encoding = None

    # Prepare sparse tensor and index mapping
    sparse_tensor_values = sparse_tensor.values().clone().detach().requires_grad_(True)
    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, index_tensor
    )

    # Run custom implementation
    custom_output = GatherAndSubsetAttentionFunction.apply(
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
        scale_factor,  # scale_factor
    )

    # Run traceable implementation
    traceable_output = traceable_sparse_attention(
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
        scale_factor,  # scale_factor
    )

    # Check that forward outputs match
    assert torch.allclose(custom_output, traceable_output)

    # Create a loss and check that gradients match
    loss = custom_output.sum()
    loss.backward(retain_graph=True)

    # Save gradients from custom implementation
    custom_grads = {
        "query": query_tensor.grad.clone(),
        "sparse_values": sparse_tensor_values.grad.clone(),
        "key_weight": key_weight.grad.clone(),
        "value_weight": value_weight.grad.clone(),
    }

    if key_bias is not None:
        custom_grads["key_bias"] = key_bias.grad.clone()
    if value_bias is not None:
        custom_grads["value_bias"] = value_bias.grad.clone()
    if key_pos_encoding is not None:
        custom_grads["key_pos"] = key_pos_encoding.grad.clone()

    # Zero out gradients
    query_tensor.grad = None
    sparse_tensor_values.grad = None
    key_weight.grad = None
    value_weight.grad = None
    if key_bias is not None:
        key_bias.grad = None
    if value_bias is not None:
        value_bias.grad = None
    if key_pos_encoding is not None:
        key_pos_encoding.grad = None

    # Backward through traceable implementation
    loss = traceable_output.sum()
    loss.backward()

    # Compare gradients
    assert torch.allclose(custom_grads["query"], query_tensor.grad)
    assert torch.allclose(custom_grads["sparse_values"], sparse_tensor_values.grad)
    assert torch.allclose(custom_grads["key_weight"], key_weight.grad)
    assert torch.allclose(custom_grads["value_weight"], value_weight.grad)

    if key_bias is not None:
        assert torch.allclose(custom_grads["key_bias"], key_bias.grad)
    if value_bias is not None:
        assert torch.allclose(custom_grads["value_bias"], value_bias.grad)
    if key_pos_encoding is not None:
        assert torch.allclose(custom_grads["key_pos"], key_pos_encoding.grad)
