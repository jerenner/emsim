import pytest
import torch

from emsim.utils.sparse_utils.base import (
    batch_sparse_index_subset_attn,
)

from ..constants import EMBED_DIM, N_HEADS, N_KEYS_PER_QUERY


@pytest.mark.cuda
@pytest.mark.parametrize(
    "with_key_pos_encoding",
    [True, False],
    ids=["key_pos_encoding=True", "key_pos_encoding=False"],
)
@pytest.mark.parametrize(
    "scale_factor", [None, 0.5], ids=["scale_factor=None", "scale_factor=0.5"]
)
def test_end_to_end_subset_attn(
    setup_sparse_tensor,
    setup_attention_index_tensor,
    with_key_pos_encoding,
    scale_factor,
    device,
):
    """Test end-to-end gather and subset attention."""
    sparse_tensor = setup_sparse_tensor
    index_tensor = setup_attention_index_tensor

    # Create query vectors
    query_tensor = torch.randn(
        index_tensor.shape[0],
        EMBED_DIM,
        dtype=torch.double,
        requires_grad=True,
        device=device,
    )

    # Create attention parameters
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
            EMBED_DIM,
            dtype=torch.double,
            requires_grad=True,
            device=device,
        )
        if with_key_pos_encoding
        else None
    )

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
        key_pos_encoding,
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
