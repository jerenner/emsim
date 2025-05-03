from typing import Any, Union

import pytest
import torch
from hypothesis import given, settings
from torch import Tensor

from emsim.utils.sparse_utils.ops.subset_attn.autograd import (
    GatherAndSubsetAttentionFunction,
)

from ..input_generation import attention_inputs
from ..traceable_attn import (
    traceable_batched_attention,
    traceable_subset_attention,
    prep_batched_attention,
)
from .conftest import (
    DIFFERENTIABLE_TENSOR_NAMES,
    ordered_autograd_inputs,
    set_requires_grad,
    simple_attention_input_configs,
)
from emsim.utils.sparse_utils.batching import remove_batch_dim_and_concat


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
class TestDropout:
    def test_dropout_effect_on_output(self, device: str):
        """Test that dropout has a measurable effect during training."""
        seed = 1
        # Get inputs without dropout
        inputs_no_dropout = attention_inputs(device=device, dropout_p=0.0, seed=seed)
        # Same inputs with high dropout
        inputs_dropout = attention_inputs(device=device, dropout_p=0.5, seed=seed)

        inputs_no_dropout = ordered_autograd_inputs(inputs_no_dropout)
        inputs_dropout = ordered_autograd_inputs(inputs_dropout)

        # ensure tensor inputs are the same
        for inp_no, inp_with in zip(inputs_no_dropout, inputs_dropout):
            if isinstance(inp_no, Tensor):
                assert torch.equal(inp_no, inp_with)

        # Run forward passes
        output_no_dropout = GatherAndSubsetAttentionFunction.apply(*inputs_no_dropout)
        output_dropout = GatherAndSubsetAttentionFunction.apply(*inputs_dropout)

        # Outputs should be different when dropout is applied
        assert not torch.allclose(
            output_no_dropout, output_dropout, rtol=1e-4, atol=1e-4
        )

    def test_dropout_training_vs_eval(self, device: str):
        """Test that dropout is only applied in training mode."""
        seed = 1

        # Get inputs with dropout in training mode
        inputs_training = attention_inputs(
            device=device, dropout_p=0.5, training=True, seed=seed
        )
        ordered_inputs_training = ordered_autograd_inputs(inputs_training)

        # Get inputs with dropout in eval mode
        # (same seed so input tensors should be the same)
        inputs_eval = attention_inputs(
            device=device, dropout_p=0.5, training=False, seed=seed
        )
        ordered_inputs_eval = ordered_autograd_inputs(inputs_eval)

        # Double check input tensors are the same with the same generation seed
        for inp_train, inp_eval in zip(ordered_inputs_training, ordered_inputs_eval):
            if isinstance(inp_train, Tensor):
                assert torch.equal(inp_train, inp_eval)

        # Run forward passes
        torch.manual_seed(seed)
        output_training_1 = GatherAndSubsetAttentionFunction.apply(
            *ordered_inputs_training
        )

        # If we run again in training mode, should get different results
        torch.manual_seed(seed + 1)  # Different seed
        output_training_2 = GatherAndSubsetAttentionFunction.apply(
            *ordered_inputs_training
        )

        # In eval mode, dropout should be ignored
        output_eval = GatherAndSubsetAttentionFunction.apply(*ordered_inputs_eval)

        # Training outputs should differ from each other
        assert not torch.allclose(
            output_training_1, output_training_2, rtol=1e-4, atol=1e-4
        )

        # Eval mode outputs should be deterministic regardless of dropout_p
        # They should match outputs with dropout_p=0
        inputs_no_dropout = attention_inputs(
            device=device, dropout_p=0.0, training=False, seed=seed
        )
        output_no_dropout = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs_no_dropout)
        )
        assert torch.allclose(output_eval, output_no_dropout, rtol=1e-4, atol=1e-4)

    def test_dropout_reproducibility(self, device: str):
        """Test that dropout is reproducible with the same seed."""
        seed = 42
        dropout_p = 0.3

        # First run with seed
        inputs1 = attention_inputs(
            device=device, dropout_p=dropout_p, training=True, seed=seed
        )
        torch.manual_seed(seed)
        output1 = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs1)
        )

        # Second run with same seed
        inputs2 = attention_inputs(
            device=device, dropout_p=dropout_p, training=True, seed=seed
        )
        torch.manual_seed(seed)
        output2 = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs2)
        )

        # Outputs should be identical with same seed
        assert torch.allclose(output1, output2)

        # Different seed should give different output
        inputs3 = attention_inputs(
            device=device, dropout_p=dropout_p, training=True, seed=seed
        )
        torch.manual_seed(seed + 1)
        output3 = GatherAndSubsetAttentionFunction.apply(
            *ordered_autograd_inputs(inputs3)
        )

        # Outputs should differ with different seed
        assert not torch.allclose(output1, output3)


@pytest.mark.cuda_if_available
class TestGradcheck:
    @pytest.mark.parametrize(
        "use_rope",
        ["none", "precomputed", "from_freqs"],
        ids=["rope=none", "rope=precomputed", "rope=from_freqs"],
    )
    def test_basic_gradcheck(self, device, use_rope: str) -> None:
        """Test gradcheck with different RoPE settings."""
        inputs = attention_inputs(
            use_rope=use_rope, device=device, dtype=torch.double, dropout_p=0.0
        )

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
            dropout_p=0.0,
        )

        inputs = set_requires_grad(inputs, tensors_requiring_grads)
        inputs = ordered_autograd_inputs(inputs)

        nondet_tol = 1e-5 if device == "cuda" else 0.0

        # Run gradcheck
        assert torch.autograd.gradcheck(
            GatherAndSubsetAttentionFunction.apply, inputs, nondet_tol=nondet_tol
        )


@pytest.mark.cuda_if_available
class TestAgainstReference:
    @pytest.mark.parametrize(
        "use_rope",
        ["none", "precomputed", "from_freqs"],
        ids=["rope=none", "rope=precomputed", "rope=from_freqs"],
    )
    def test_forward_against_traceable_stacked(
        self, device: str, use_rope: str
    ) -> None:
        """Basic test of the forward method against a reference implementation
        that doesn't have optimizations."""
        inputs = attention_inputs(use_rope=use_rope, device=device, dropout_p=0.0)
        inputs = ordered_autograd_inputs(inputs)

        optimized_output = GatherAndSubsetAttentionFunction.apply(*inputs)
        reference_output = traceable_subset_attention(
            *inputs, batch_kv_projection=False
        )

        abs_difference = torch.abs(optimized_output - reference_output)
        print(f"Biggest absolute difference: {abs_difference.max().item()}")
        assert torch.allclose(optimized_output, reference_output, rtol=1e-4, atol=1e-5)

        # Test again while letting the subset version use the input projection
        # batching optimization
        reference_output_2 = traceable_subset_attention(*inputs)

        assert torch.allclose(
            optimized_output, reference_output_2, rtol=1e-4, atol=1e-5
        )

    @pytest.mark.parametrize(
        "use_rope",
        ["none", "precomputed", "from_freqs"],
        ids=["rope=none", "rope=precomputed", "rope=from_freqs"],
    )
    def test_forward_against_traceable_batched(
        self, device: str, use_rope: str
    ) -> None:
        """Basic test of the forward method against a reference implementation that
        uses padding instead of stacking the queries, and masking instead of key
        subsets
        """
        inputs = attention_inputs(
            device=device,
            dropout_p=0.0,
        )
        ordered_inputs = ordered_autograd_inputs(inputs)
        batched_inputs = prep_batched_attention(inputs)

        optimized_output = GatherAndSubsetAttentionFunction.apply(*ordered_inputs)
        batched_reference_output = traceable_batched_attention(**batched_inputs)

        stacked_reference_output = remove_batch_dim_and_concat(
            batched_reference_output, inputs["query_padding_mask"]
        )[0]

        assert torch.allclose(optimized_output, stacked_reference_output)

    @pytest.mark.parametrize(
        "use_rope",
        ["none", "precomputed", "from_freqs"],
        ids=["rope=none", "rope=precomputed", "rope=from_freqs"],
    )
    def test_traceables_against_each_other(
        self,
        device: str,
        use_rope: str,
    ) -> None:
        """Test equivalence of the two traceable implementations."""
        inputs = attention_inputs(
            use_rope="none", device=device, dropout_p=0.0, training=False
        )
        ordered_inputs = ordered_autograd_inputs(inputs)
        batched_inputs = prep_batched_attention(inputs)

        subset_output = traceable_subset_attention(
            *ordered_inputs, return_extended_outputs=True
        )
        batched_output = traceable_batched_attention(
            **batched_inputs, return_extended_outputs=True
        )

        subset_attn_out = subset_output["attn_output"]
        batched_attn_out = batched_output["attn_output"]

        batched_attn_out_stacked = remove_batch_dim_and_concat(
            batched_attn_out, inputs["query_padding_mask"]
        )[0]

        assert subset_attn_out.shape == batched_attn_out_stacked.shape

        # check equality of intermediate values
        bsz, sparse_height, sparse_width, n_levels, embed_dim = inputs[
            "sparse_tensor"
        ].shape
        n_heads = inputs["metadata"]["n_heads"]
        n_queries = max(inputs["metadata"]["n_queries"])

        # get indexing tuple for going from all keys to keys per query
        key_b, key_i, key_j, key_l = inputs["index_tensor"].unbind(-1)
        key_q = torch.cat(
            [
                torch.arange(q, device=key_b.device)
                .unsqueeze(1)
                .expand(-1, inputs["metadata"]["n_keys_per_query"])
                for q in inputs["metadata"]["n_queries"]
            ]
        )

        # Gather keys
        batched_keys = batched_output["keys"].view(
            bsz, sparse_height, sparse_width, n_levels, embed_dim
        )
        stacked_keys_from_batched = batched_keys[key_b, key_i, key_j, key_l]
        assert torch.allclose(
            stacked_keys_from_batched,
            subset_output["keys"].reshape_as(stacked_keys_from_batched),
        )

        # same for values...
        batched_values = batched_output["values"].view(
            bsz, sparse_height, sparse_width, n_levels, embed_dim
        )
        stacked_values_from_batched = batched_values[key_b, key_i, key_j, key_l]
        assert torch.allclose(
            stacked_values_from_batched,
            subset_output["values"].reshape_as(stacked_values_from_batched),
        )

        # attention scores
        batched_attn_scores_bqhwlh = (
            batched_output["attn_scores"]
            .permute(0, 2, 3, 1)
            .reshape(bsz, n_queries, sparse_height, sparse_width, n_levels, n_heads)
        )
        # query, key, head
        stacked_attn_scores_from_batched = batched_attn_scores_bqhwlh[
            key_b, key_q, key_i, key_j, key_l
        ]
        subset_attn_scores = subset_output["attn_scores"].transpose(-1, -2)
        assert torch.allclose(
            subset_attn_scores, stacked_attn_scores_from_batched, atol=1e-6
        )

        # masked attention scores
        batched_attn_scores_masked_bqhwlh = (
            batched_output["attn_scores_masked"]
            .permute(0, 2, 3, 1)
            .reshape(bsz, n_queries, sparse_height, sparse_width, n_levels, n_heads)
        )
        # query, key, head
        stacked_attn_scores_masked_from_batched = batched_attn_scores_masked_bqhwlh[
            key_b, key_q, key_i, key_j, key_l
        ]
        subset_attn_scores_masked = subset_output["attn_scores_masked"].transpose(
            -1, -2
        )
        assert torch.allclose(
            subset_attn_scores_masked,
            stacked_attn_scores_masked_from_batched,
            atol=1e-6,
        )

        batched_attn_mask_bqhwl = batched_output["attn_mask"].view(
            bsz, n_queries, sparse_height, sparse_width, n_levels
        )
        nonmask_b, nonmask_q, nonmask_i, nonmask_j, nonmask_l = (
            batched_attn_mask_bqhwl.logical_not().nonzero(as_tuple=True)
        )

        # attention weights
        batched_attn_weights_bqhwlh = (
            batched_output["attn_weights"]
            .permute(0, 2, 3, 1)
            .reshape(bsz, n_queries, sparse_height, sparse_width, n_levels, n_heads)
        )
        # query, key, head
        stacked_attn_weights_from_batched = batched_attn_weights_bqhwlh[
            key_b, key_q, key_i, key_j, key_l
        ]
        subset_attn_weights = subset_output["attn_weights"].transpose(-1, -2)
        assert torch.allclose(
            subset_attn_weights, stacked_attn_weights_from_batched, atol=1e-6
        ), (
            "max attn_weight difference: "
            f"{(subset_attn_weights - stacked_attn_weights_from_batched).abs().max()}"
        )

        abs_difference = torch.abs(subset_attn_out - batched_attn_out_stacked)
        print(f"Biggest absolute difference: {abs_difference.max().item()}")

        assert torch.allclose(subset_attn_out, batched_attn_out_stacked, atol=1e-6)
