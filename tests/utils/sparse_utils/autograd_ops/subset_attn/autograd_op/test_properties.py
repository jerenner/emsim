from typing import Any

import pytest
import torch
from hypothesis import given, settings
from torch import Tensor

from emsim.utils.sparse_utils.ops.subset_attn.autograd import (
    GatherAndSubsetAttentionFunction,
)

from ..input_generation import attention_inputs
from ..traceable_attn import traceable_subset_attention
from .conftest import (
    exhaustive_attention_input_configs,
    ordered_autograd_inputs,
    set_requires_grad,
)


@pytest.mark.cuda_if_available
@settings(deadline=None, max_examples=25)
@given(
    input_params=exhaustive_attention_input_configs(
        dtypes=[torch.double], min_requiring_grads=1
    )
)
def test_gradcheck_exhaustive(device: str, input_params: dict[str, Any]) -> None:
    """Gradcheck test letting Hypothesis really explore the input space.

    Takes a while to run.
    """
    tensors_requiring_grads = input_params["tensors_requiring_grads"]

    inputs = attention_inputs(**input_params, device=device, dropout_p=0.0)

    inputs = set_requires_grad(inputs, tensors_requiring_grads)
    inputs = ordered_autograd_inputs(inputs)

    nondet_tol = 1e-5 if device == "cuda" else 0.0

    assert torch.autograd.gradcheck(
        GatherAndSubsetAttentionFunction.apply, inputs, nondet_tol=nondet_tol
    )


@pytest.mark.cuda_if_available
@settings(deadline=None, max_examples=25)
@given(
    input_params=exhaustive_attention_input_configs(
        dtypes=[torch.float32, torch.float64]  # TODO fully implement 16-bit correctness
    )
)
def test_forward_against_traceable(device: str, input_params: dict[str, Any]):
    """Test the forward method against a reference implementation that doesn't have
    optimizations."""
    inputs = attention_inputs(**input_params, device=device, dropout_p=0.0)
    inputs = ordered_autograd_inputs(inputs)

    optimized_output = GatherAndSubsetAttentionFunction.apply(*inputs)
    reference_output = traceable_subset_attention(*inputs, batch_kv_projection=False)

    abs_difference = torch.abs(optimized_output - reference_output)
    print(f"Biggest absolute difference: {abs_difference.max().item()}")
    assert torch.allclose(optimized_output, reference_output, rtol=1e-4, atol=1e-5)


@pytest.mark.cuda_if_available
@settings(deadline=None, max_examples=25)
@given(
    input_params=exhaustive_attention_input_configs(
        dtypes=[torch.float32, torch.float64],
        min_requiring_grads=1,
    )
)
def test_gradients_against_traceable(device: str, input_params: dict[str, Any]):
    """Test gradients against the reference implementation that uses autograd"""
    tensors_requiring_grads = input_params["tensors_requiring_grads"]

    # set up inputs
    inputs = attention_inputs(**input_params, device=device, dropout_p=0.0)
    inputs = set_requires_grad(inputs, tensors_requiring_grads)
    optimized_inputs = ordered_autograd_inputs(inputs)

    # make a fresh copy of input tensors for reference implementation
    reference_inputs = [
        (
            t.clone().detach().requires_grad_(t.requires_grad)
            if isinstance(t, Tensor)
            else t
        )
        for t in optimized_inputs
    ]

    # get outputs
    optimized_output = GatherAndSubsetAttentionFunction.apply(*optimized_inputs)
    reference_output = traceable_subset_attention(
        *reference_inputs, batch_kv_projection=False
    )

    # check outputs match
    assert torch.allclose(optimized_output, reference_output, rtol=1e-4, atol=1e-5)

    # Create random gradient for backprop
    grad_output = torch.randn_like(optimized_output)
    optimized_output.backward(grad_output)
    reference_output.backward(grad_output.clone())

    # Compare gradients
    for i, (opt_input, ref_input) in enumerate(zip(optimized_inputs, reference_inputs)):
        if isinstance(opt_input, Tensor) and opt_input.requires_grad:
            assert opt_input.grad is not None, f"Optimized grad is None for input {i}"
            assert ref_input.grad is not None, f"Reference grad is None for input {i}"

            diff = torch.abs(opt_input.grad - ref_input.grad)

            assert torch.allclose(
                opt_input.grad, ref_input.grad, rtol=1e-4, atol=1e-4
            ), f"Grad mismatch for input {i}: diff max={diff.max()} mean={diff.mean()}"
