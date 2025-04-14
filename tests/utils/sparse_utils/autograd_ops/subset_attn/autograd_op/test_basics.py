from typing import Any, Union

import pytest
import torch
from hypothesis import given, settings

from emsim.utils.sparse_utils.ops.subset_attn.autograd import (
    GatherAndSubsetAttentionFunction,
)

from ..input_generation import attention_inputs
from .conftest import (
    DIFFERENTIABLE_TENSOR_NAMES,
    ordered_autograd_inputs,
    set_requires_grad,
    simple_attention_input_configs,
)


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
    def test_attention_forward_shape(self, device: str):
        """Test that the forward pass produces output with the correct shape."""
        inputs = attention_inputs(device=device)
        metadata = inputs["metadata"]

        output = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs)
        )

        expected_shape = (sum(metadata["n_queries"]), metadata["embed_dim"])
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_attention_forward_with_unspecified_keys(
        self,
        device: str,
    ):
        """Test forward pass with queries having all keys unspecified."""
        inputs = attention_inputs(
            n_queries=4, unspecified_query_indices=[0, 2], device=device
        )

        output = GatherAndSubsetAttentionFunction.apply(
            *(ordered_autograd_inputs(inputs))
        )

        # Check that queries with all keys unspecified produce finite values
        unspecified_indices = inputs["metadata"]["unspecified_query_indices"]
        if unspecified_indices is not None:
            assert not torch.isnan(
                output[unspecified_indices]
            ).any(), "Output for queries with all keys unspecified contains NaN values"

    def test_attention_forward_backward(self, device):
        """Test both forward and backward passes with gradients."""
        inputs = attention_inputs(device=device)
        metadata = inputs["metadata"]

        # Ensure tensors require gradients
        inputs = set_requires_grad(inputs, DIFFERENTIABLE_TENSOR_NAMES)

        # Forward pass
        output = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs)
        )

        # Check output shape
        expected_shape = (sum(metadata["n_queries"]), metadata["embed_dim"])
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
        output = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs)
        )

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
        output = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs)
        )

        # Backward pass
        loss = output.sum()
        loss.backward()

        assert grad_not_none(inputs, tensor_requiring_grads)
        assert grad_same_shape(inputs, tensor_requiring_grads)

    # Property-based version using Hypothesis

    @settings(deadline=None)
    @given(tensor_configs=simple_attention_input_configs())
    def test_hypothesis_forward_backward(
        self,
        device,
        tensor_configs: dict[str, Union[bool, list[str]]],
    ):
        """Hypothesis-based test to try random combinations of inputs"""
        use_biases = tensor_configs["use_biases"]
        use_rope = tensor_configs["use_rope"]
        tensors_requiring_grads = tensor_configs["tensors_requiring_grads"]

        inputs = attention_inputs(
            use_biases=use_biases, use_rope=use_rope, device=device
        )

        inputs = set_requires_grad(inputs, tensors_requiring_grads)

        # Forward pass
        output = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs)
        )

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
        inputs = ordered_autograd_inputs(inputs)

        assert torch.autograd.gradcheck(GatherAndSubsetAttentionFunction.apply, inputs)

    # Property-based version using hypothesis

    @settings(deadline=None, max_examples=25)
    @given(tensor_configs=simple_attention_input_configs())
    def test_hypothesis_gradcheck(
        self,
        device: str,
        tensor_configs: dict[str, Union[bool, list[str]]],
    ):
        """Hypothesis-based test to try random combinations of inputs"""
        use_biases = tensor_configs["use_biases"]
        use_rope = tensor_configs["use_rope"]
        tensors_requiring_grads = tensor_configs["tensors_requiring_grads"]

        inputs = attention_inputs(
            use_biases=use_biases,
            use_rope=use_rope,
            device=device,
            dtype=torch.double,
        )

        inputs = set_requires_grad(inputs, tensors_requiring_grads)
        inputs = ordered_autograd_inputs(inputs)

        nondet_tol = 1e-5 if device == "cuda" else 0.0

        # Run gradcheck
        assert torch.autograd.gradcheck(
            GatherAndSubsetAttentionFunction.apply, inputs, nondet_tol=nondet_tol
        )
