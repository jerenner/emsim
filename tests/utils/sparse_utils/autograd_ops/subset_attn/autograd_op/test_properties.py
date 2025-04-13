from typing import Any

import pytest
import torch
from hypothesis import given, settings, example

from emsim.utils.sparse_utils.ops.subset_attn.autograd import (
    GatherAndSubsetAttentionFunction,
)

from .conftest import (
    exhaustive_attention_input_configs,
    attention_inputs,
    ordered_inputs,
    set_requires_grad,
)


@pytest.mark.cuda_if_available
@settings(deadline=None)
@given(
    input_params=exhaustive_attention_input_configs(
        dtypes=[torch.double], min_requiring_grads=1
    )
)
@example(
    input_params={
        "n_queries": 5,
        "embed_dim": 4,
        "n_heads": 1,
        "n_keys_per_query": 5,
        "num_sparse_values": 5,
        "position_dim": 1,
        "n_freq_groups": 1,
        "unspecified_query_indices": None,
        "unspecified_prob": 0.0,
        "dtype": torch.float64,
        "use_biases": False,
        "use_rope": "precomputed",
        "tensors_requiring_grads": ["query_tensor", "sparse_tensor_values"],
        "seed": 4,
    },
)
def test_gradcheck_exhaustive(device: str, input_params: dict[str, Any]) -> None:
    """Gradcheck test letting Hypothesis really explore the input space.

    Takes a while to run."""
    tensors_requiring_grads = input_params["tensors_requiring_grads"]

    inputs = attention_inputs(**input_params, device=device)

    inputs = set_requires_grad(inputs, tensors_requiring_grads)
    inputs = ordered_inputs(inputs)

    nondet_tol = 1e-5 if device == "cuda" else 0.0

    assert torch.autograd.gradcheck(
        GatherAndSubsetAttentionFunction.apply, inputs, nondet_tol=nondet_tol
    )
