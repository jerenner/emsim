from typing import Any, Union

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from torch import Tensor

from emsim.utils.sparse_utils.ops.subset_attn.autograd import (
    GatherAndSubsetAttentionFunction,
)

from .conftest import DIFFERENTIABLE_TENSOR_NAMES, attention_inputs


@pytest.fixture
def base_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    return attention_inputs()


@pytest.fixture
def query_with_all_keys_unspecified_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    return attention_inputs(unspecified_query_indices=[0, 2])


def ordered_inputs(
    inputs: Union[dict[str, Any], tuple[dict[str, Any], dict[str, Any]]],
) -> tuple:
    if isinstance(inputs, tuple):
        inputs = inputs[0]

    return (
        inputs["query_tensor"],
        inputs["n_heads"],
        inputs["sparse_tensor_values"],
        inputs["index_tensor"],
        inputs["is_specified_mask"],
        inputs["key_weight"],
        inputs["value_weight"],
        inputs["key_bias"],
        inputs["value_bias"],
        inputs["key_rope_encoding"],
        inputs["key_positions"],
        inputs["rope_freqs"],
        inputs["scale_factor"],
    )


def set_requires_grad(inputs: dict[str, Any], tensor_names: Union[str, list[str]]):
    """Sets the requires_grad flag to True for specified tensors in the input dict"""
    modified_inputs = inputs.copy()
    if isinstance(tensor_names, str):
        tensor_names = [tensor_names]
    for name in tensor_names:
        if name in modified_inputs and modified_inputs[name] is not None:
            tensor: Tensor = modified_inputs[name].clone()
            modified_inputs[name] = tensor.requires_grad_(True)
    return modified_inputs


def grad_not_none(inputs: dict[str, Any], name: str, pass_if_none: bool = False):
    if inputs[name] is not None:
        return inputs[name].grad is not None
    return pass_if_none


def grad_same_shape(inputs: dict[str, Any], name: str, pass_if_none: bool = False):
    if inputs[name] is not None and inputs[name].grad is not None:
        return inputs[name].shape == inputs[name].grad.shape
    return pass_if_none


@pytest.mark.cuda_if_available
class TestBasicForwardBackward:
    def test_attention_forward_shape(self, base_inputs):
        """Test that the forward pass produces output with the correct shape."""
        inputs = base_inputs
        metadata = inputs["metadata"]

        output = GatherAndSubsetAttentionFunction.apply(*ordered_inputs(inputs))

        expected_shape = (metadata["n_queries"], metadata["embed_dim"])
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_attention_forward_with_unspecified_keys(
        self,
        query_with_all_keys_unspecified_inputs,
    ):
        """Test forward pass with queries having all keys unspecified."""
        inputs = query_with_all_keys_unspecified_inputs

        output = GatherAndSubsetAttentionFunction.apply(*(ordered_inputs(inputs)))

        # Check that queries with all keys unspecified produce finite values
        unspecified_indices = inputs["metadata"]["unspecified_query_indices"]
        if unspecified_indices is not None:
            assert not torch.isnan(
                output[unspecified_indices]
            ).any(), "Output for queries with all keys unspecified contains NaN values"

    def test_attention_forward_backward(self, base_inputs):
        """Test both forward and backward passes with gradients."""
        inputs = base_inputs
        metadata = inputs["metadata"]

        # Ensure tensors require gradients
        inputs = set_requires_grad(inputs, DIFFERENTIABLE_TENSOR_NAMES)

        # Forward pass
        output = GatherAndSubsetAttentionFunction.apply(*ordered_inputs(inputs))

        # Check output shape
        expected_shape = (metadata["n_queries"], metadata["embed_dim"])
        assert output.shape == expected_shape

        # Create a simple loss and run backward
        loss = output.sum()
        loss.backward()

        # Check that gradients were properly computed
        assert grad_not_none(inputs, "query_tensor")
        assert grad_not_none(inputs, "sparse_tensor_values")
        assert grad_not_none(inputs, "key_weight")
        assert grad_not_none(inputs, "value_weight")

        assert grad_not_none(inputs, "key_bias")
        assert grad_not_none(inputs, "value_bias")

        # Check gradient shapes
        assert grad_same_shape(inputs, "query_tensor")
        assert grad_same_shape(inputs, "sparse_tensor_values")
        assert grad_same_shape(inputs, "key_weight")
        assert grad_same_shape(inputs, "value_weight")

        assert grad_same_shape(inputs, "key_bias")
        assert grad_same_shape(inputs, "value_bias")

    @pytest.mark.parametrize(
        "use_rope",
        ["none", "precomputed", "from_freqs"],
        ids=["rope=none", "rope=precomputed", "rope=from_freqs"],
    )
    @pytest.mark.cuda_if_available
    def test_attention_with_different_rope_settings(
        self, device, use_rope: str
    ) -> None:
        """Test attention with different RoPE settings."""
        inputs = attention_inputs(use_rope=use_rope, device=device)

        # Ensure tensors require gradients
        inputs = set_requires_grad(inputs, DIFFERENTIABLE_TENSOR_NAMES)

        # Forward pass
        output = GatherAndSubsetAttentionFunction.apply(*ordered_inputs(inputs))

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients were properly computed
        assert grad_not_none(inputs, "query_tensor")
        assert grad_not_none(inputs, "sparse_tensor_values")
        assert grad_not_none(inputs, "key_weight")
        assert grad_not_none(inputs, "value_weight")

        assert grad_not_none(inputs, "key_bias")
        assert grad_not_none(inputs, "value_bias")

        assert grad_not_none(inputs, "key_rope_encoding", use_rope != "precomputed")
        assert grad_not_none(inputs, "key_positions", use_rope != "from_freqs")
        assert grad_not_none(inputs, "rope_freqs", use_rope != "from_freqs")

        # Check gradient shapes
        assert grad_same_shape(inputs, "query_tensor")
        assert grad_same_shape(inputs, "sparse_tensor_values")
        assert grad_same_shape(inputs, "key_weight")
        assert grad_same_shape(inputs, "value_weight")

        assert grad_same_shape(inputs, "key_bias")
        assert grad_same_shape(inputs, "value_bias")

        assert grad_same_shape(inputs, "key_rope_encoding", use_rope != "precomputed")
        assert grad_same_shape(inputs, "key_positions", use_rope != "from_freqs")
        assert grad_same_shape(inputs, "rope_freqs", use_rope != "from_freqs")

    @pytest.mark.parametrize("tensor_requiring_grads", DIFFERENTIABLE_TENSOR_NAMES)
    @pytest.mark.cuda_if_available
    def test_individual_tensors_forward_backward(
        self,
        device,
        tensor_requiring_grads,
    ) -> None:
        """Test attention with different RoPE settings."""
        if tensor_requiring_grads == "key_rope_encoding":
            use_rope = "precomputed"
        elif tensor_requiring_grads in ("key_positions", "rope_freqs"):
            use_rope = "from_freqs"
        else:
            use_rope = None

        inputs = attention_inputs(use_rope=use_rope, device=device)
        inputs = set_requires_grad(inputs, tensor_requiring_grads)

        # Forward pass
        output = GatherAndSubsetAttentionFunction.apply(*ordered_inputs(inputs))

        # Backward pass
        loss = output.sum()
        loss.backward()

        assert grad_not_none(inputs, tensor_requiring_grads)
        assert grad_same_shape(inputs, tensor_requiring_grads)

    @settings(deadline=None)
    @given(
        use_rope=st.sampled_from(["none", "precomputed", "from_freqs"]),
        use_biases=st.booleans(),
        tensors_requiring_grads=st.lists(
            st.sampled_from(DIFFERENTIABLE_TENSOR_NAMES),
            min_size=1,
            max_size=len(DIFFERENTIABLE_TENSOR_NAMES),
            unique=True,
        ),
    )
    def test_hypothesis_forward_backward(
        self,
        device,
        use_rope: str,
        use_biases: bool,
        tensors_requiring_grads: list[str],
    ):
        """Hypothesis-based test to try random combinations of inputs"""
        assume(  # filter out invalid rope combinations
            not (
                "key_rope_encoding" in tensors_requiring_grads
                and use_rope != "precomputed"
            )
        )
        assume(
            not (
                (
                    "key_positions" in tensors_requiring_grads
                    or "rope_freqs" in tensors_requiring_grads
                )
                and use_rope in ("none, precomputed")
            )
        )
        assume(  # don't make biases require grads when they aren't being used
            not (
                (
                    "key_bias" in tensors_requiring_grads
                    or "value_bias" in tensors_requiring_grads
                )
                and not use_biases
            )
        )
        inputs = attention_inputs(
            use_biases=use_biases, use_rope=use_rope, device=device
        )

        inputs = set_requires_grad(inputs, tensors_requiring_grads)

        # Forward pass
        output = GatherAndSubsetAttentionFunction.apply(*ordered_inputs(inputs))

        # Backward pass
        loss = output.sum()
        loss.backward()

        for tensor_name in tensors_requiring_grads:
            assert grad_not_none(inputs, tensor_name)
            assert grad_same_shape(inputs, tensor_name)


@pytest.mark.cuda_if_available
class TestGradcheck:
    @pytest.mark.parametrize(
        "use_rope",
        ["none", "precomputed", "from_freqs"],
        ids=["rope=none", "rope=precomputed", "rope=from_freqs"],
    )
    def test_basic_gradcheck(uself, device, use_rope: str) -> None:
        """Test gradcheck with different RoPE settings."""
        inputs = attention_inputs(use_rope=use_rope, device=device, dtype=torch.double)

        tensors_to_diff = [
            name for name in DIFFERENTIABLE_TENSOR_NAMES if inputs[name] is not None
        ]
        inputs = set_requires_grad(inputs, tensors_to_diff)
        inputs = ordered_inputs(inputs)

        assert torch.autograd.gradcheck(GatherAndSubsetAttentionFunction.apply, inputs)

    @settings(deadline=None, max_examples=25)
    @given(
        use_rope=st.sampled_from(["none", "precomputed", "from_freqs"]),
        use_biases=st.booleans(),
        tensors_requiring_grads=st.lists(
            st.sampled_from(DIFFERENTIABLE_TENSOR_NAMES),
            min_size=1,
            max_size=len(DIFFERENTIABLE_TENSOR_NAMES),
            unique=True,
        ),
    )
    def test_hypothesis_gradcheck(
        self,
        device,
        use_rope: str,
        use_biases: bool,
        tensors_requiring_grads: list[str],
    ):
        """Hypothesis-based test to try random combinations of inputs"""
        assume(  # filter out invalid rope combinations
            not (
                "key_rope_encoding" in tensors_requiring_grads
                and use_rope != "precomputed"
            )
        )
        assume(
            not (
                (
                    "key_positions" in tensors_requiring_grads
                    or "rope_freqs" in tensors_requiring_grads
                )
                and use_rope in ("none, precomputed")
            )
        )
        assume(  # don't make biases require grads when they aren't being used
            not (
                (
                    "key_bias" in tensors_requiring_grads
                    or "value_bias" in tensors_requiring_grads
                )
                and not use_biases
            )
        )
        inputs = attention_inputs(
            use_biases=use_biases, use_rope=use_rope, device=device, dtype=torch.double,
        )

        inputs = set_requires_grad(inputs, tensors_requiring_grads)
        inputs = ordered_inputs(inputs)

        # Forward pass
        assert torch.autograd.gradcheck(GatherAndSubsetAttentionFunction.apply, inputs)
