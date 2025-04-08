import pytest
import torch

from emsim.utils.sparse_utils.ops.subset_attn.subset_attn import (
    batch_sparse_index_subset_attn,
)

from ..constants import (
    EMBED_DIM,
    N_FREQ_GROUPS,
    N_HEADS,
    N_KEYS_PER_QUERY,
    POSITION_DIM,
)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "key_pos_encoding_type",
    ["given", "computed", None],
    ids=[
        "key_pos_encoding_type=given",
        "key_pos_encoding_type=computed",
        "key_pos_encoding_type=None",
    ],
)
@pytest.mark.parametrize(
    "scale_factor", [None, 0.5], ids=["scale_factor=None", "scale_factor=0.5"]
)
def test_end_to_end_subset_attn(
    setup_sparse_tensor,
    setup_attention_index_tensor,
    device,
    key_pos_encoding_type,
    scale_factor,
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
        key_positions,
        rope_freqs,
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
