import math

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from emsim.utils.sparse_utils.ops.subset_attn.rotary_embedding import (
    calculate_rope,
    calculate_rope_backward,
    rotate_keys,
    rotate_keys_backward,
)
from .conftest import (
    assert_close,
    valid_dims,
)


@pytest.mark.cuda_if_available
class TestHypothesis:
    """Property-based tests using hypothesis."""

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        position_dim=valid_dims(),
        n_freq_groups=valid_dims(),
        n_heads=valid_dims(),
        half_head_dim=valid_dims(),
    )
    @settings(
        suppress_health_check=[HealthCheck.differing_executors],
        deadline=None,
        max_examples=10,
    )
    def test_calculate_rope_gradient_consistency(
        self,
        n_queries,
        n_keys_per_query,
        position_dim,
        n_freq_groups,
        n_heads,
        half_head_dim,
        device,
    ):
        """Property-based test to verify gradients are consistent with autograd."""
        # Create tensors that require gradients
        key_positions = torch.randn(
            n_queries,
            n_keys_per_query,
            position_dim,
            requires_grad=True,
            device=device,
            dtype=torch.double,
        )
        rope_freqs = torch.randn(
            position_dim,
            n_freq_groups,
            n_heads,
            half_head_dim,
            requires_grad=True,
            device=device,
            dtype=torch.double,
        )

        # Forward pass
        result = calculate_rope(key_positions, rope_freqs)

        # Autograd backward
        grad_output = torch.randn_like(result)
        result.backward(grad_output)

        # Store autograd gradients
        key_positions_grad_autograd = key_positions.grad.clone()
        rope_freqs_grad_autograd = rope_freqs.grad.clone()

        # Reset gradients
        key_positions.grad = None
        rope_freqs.grad = None

        # Test manual backward
        grad_key_positions, grad_rope_freqs = calculate_rope_backward(
            grad_output, key_positions, rope_freqs, True, True
        )

        # Compare gradients
        assert_close(
            grad_key_positions,
            key_positions_grad_autograd,
            msg="Manual grad_key_positions doesn't match autograd",
        )
        assert_close(
            grad_rope_freqs,
            rope_freqs_grad_autograd,
            msg="Manual grad_rope_freqs doesn't match autograd",
        )

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        n_heads=valid_dims(),
        head_dim_half=valid_dims(),
    )
    @settings(
        deadline=None,
        max_examples=10,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_rotate_k_gradient_consistency(
        self, device, n_queries, n_keys_per_query, n_heads, head_dim_half
    ):
        """Test that rotate_k gradients are consistent with autograd."""
        head_dim = head_dim_half * 2  # Ensure head_dim is even

        # Create tensors requiring gradients
        keys = torch.randn(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim,
            requires_grad=True,
            device=device,
            dtype=torch.double,
        )
        rope_encoding = torch.randn(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim_half,
            requires_grad=True,
            device=device,
            dtype=torch.double,
        )

        # Forward pass
        keys_rotated = rotate_keys(keys, rope_encoding)

        # Autograd backward
        grad_output = torch.randn_like(keys_rotated, device=device)
        keys_rotated.backward(grad_output)

        # Store autograd gradients
        keys_grad_autograd = keys.grad.clone()
        rope_encoding_grad_autograd = rope_encoding.grad.clone()

        # Reset gradients
        keys.grad = None
        rope_encoding.grad = None

        # Manual backward pass
        grad_keys, grad_rope_encoding = rotate_keys_backward(
            grad_output, keys, rope_encoding, True, True
        )

        # Compare gradients
        assert_close(
            grad_keys,
            keys_grad_autograd,
            atol=1e-7,
            msg="Manual grad_keys doesn't match autograd",
        )
        assert_close(
            grad_rope_encoding,
            rope_encoding_grad_autograd,
            atol=1e-7,
            msg="Manual grad_rope_encoding doesn't match autograd",
        )

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        position_dim=valid_dims(),
        n_freq_groups=valid_dims(),
        n_heads=valid_dims(),
        half_head_dim=valid_dims(),
    )
    @settings(
        deadline=None,
        max_examples=10,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_numerical_stability(
        self,
        device,
        n_queries,
        n_keys_per_query,
        position_dim,
        n_freq_groups,
        n_heads,
        half_head_dim,
    ):
        """Test numerical stability with large and small values."""
        # Test with very small values
        key_positions_small = (
            torch.rand(
                n_queries,
                n_keys_per_query,
                position_dim,
                device=device,
                dtype=torch.double,
            )
            * 1e-6
        )
        rope_freqs_small = (
            torch.rand(
                position_dim,
                n_freq_groups,
                n_heads,
                half_head_dim,
                device=device,
                dtype=torch.double,
            )
            * 1e-6
        )

        # Test with very large values
        key_positions_large = (
            torch.rand(
                n_queries,
                n_keys_per_query,
                position_dim,
                device=device,
                dtype=torch.double,
            )
            * 1e6
        )
        rope_freqs_large = (
            torch.rand(
                position_dim,
                n_freq_groups,
                n_heads,
                half_head_dim,
                device=device,
                dtype=torch.double,
            )
            * 1e6
        )

        # Forward pass should not produce NaNs or infinities
        result_small = calculate_rope(key_positions_small, rope_freqs_small)
        result_large = calculate_rope(key_positions_large, rope_freqs_large)

        assert not torch.isnan(result_small).any(), "Small values produced NaNs"
        assert not torch.isinf(result_small).any(), "Small values produced infinities"
        assert not torch.isnan(result_large).any(), "Large values produced NaNs"
        assert not torch.isinf(result_large).any(), "Large values produced infinities"

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        n_heads=valid_dims(),
        head_dim_half=valid_dims(),
    )
    @settings(
        deadline=None,
        max_examples=10,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_linearity_of_gradients(
        self, device, n_queries, n_keys_per_query, n_heads, head_dim_half
    ):
        """Test that gradients follow linearity property."""
        head_dim = head_dim_half * 2  # Ensure head_dim is even

        keys = torch.randn(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim,
            requires_grad=True,
            device=device,
            dtype=torch.double,
        )
        rope_encoding = torch.randn(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim_half,
            requires_grad=True,
            device=device,
            dtype=torch.double,
        )

        # Forward pass
        keys_rotated = rotate_keys(keys, rope_encoding)

        # Create two different gradient outputs
        grad_output1 = torch.randn_like(keys_rotated, device=device, dtype=torch.double)
        grad_output2 = torch.randn_like(keys_rotated, device=device, dtype=torch.double)
        alpha = torch.rand(1, device=device, dtype=torch.double).item()

        # Calculate gradients for each output separately
        keys_rotated.backward(grad_output1, retain_graph=True)
        keys_grad1 = keys.grad.clone()
        rope_encoding_grad1 = rope_encoding.grad.clone()

        keys.grad = None
        rope_encoding.grad = None

        keys_rotated.backward(grad_output2, retain_graph=True)
        keys_grad2 = keys.grad.clone()
        rope_encoding_grad2 = rope_encoding.grad.clone()

        keys.grad = None
        rope_encoding.grad = None

        # Calculate gradients for linear combination
        combined_grad_output = alpha * grad_output1 + (1 - alpha) * grad_output2
        keys_rotated.backward(combined_grad_output)
        keys_grad_combined = keys.grad.clone()
        rope_encoding_grad_combined = rope_encoding.grad.clone()

        # Verify linearity: grad(αx + βy) = α*grad(x) + β*grad(y)
        expected_keys_grad = alpha * keys_grad1 + (1 - alpha) * keys_grad2
        expected_rope_grad = (
            alpha * rope_encoding_grad1 + (1 - alpha) * rope_encoding_grad2
        )

        assert_close(
            keys_grad_combined,
            expected_keys_grad,
            rtol=1e-4,
            atol=1e-7,
            msg="Gradients don't satisfy linearity for keys",
        )
        assert_close(
            rope_encoding_grad_combined,
            expected_rope_grad,
            rtol=1e-4,
            atol=1e-7,
            msg="Gradients don't satisfy linearity for rope_encoding",
        )

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        position_dim=valid_dims(),
        n_freq_groups=valid_dims(),
        n_heads=valid_dims(),
        half_head_dim=valid_dims(),
    )
    @settings(
        deadline=None,
        max_examples=5,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_rope_permutation_invariance(
        self,
        device,
        n_queries,
        n_keys_per_query,
        position_dim,
        n_freq_groups,
        n_heads,
        half_head_dim,
    ):
        """Test that permutation of queries doesn't affect batch independence."""
        # Create inputs
        key_positions = torch.randn(
            n_queries, n_keys_per_query, position_dim, device=device, dtype=torch.double
        )
        rope_freqs = torch.randn(
            position_dim,
            n_freq_groups,
            n_heads,
            half_head_dim,
            device=device,
            dtype=torch.double,
        )

        # Get results
        result = calculate_rope(key_positions, rope_freqs)

        # Create a permutation of the queries
        perm_indices = torch.randperm(n_queries, device=device)
        key_positions_perm = key_positions[perm_indices]

        # Get results for permuted input
        result_perm = calculate_rope(key_positions_perm, rope_freqs)

        # The results should match when un-permuted
        assert_close(
            result[perm_indices],
            result_perm,
            msg="calculate_rope is not permutation invariant",
        )

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        n_heads=valid_dims(),
        head_dim_half=valid_dims(),
    )
    @settings(
        deadline=None,
        max_examples=5,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_complex_multiplication_properties(
        self, device, n_queries, n_keys_per_query, n_heads, head_dim_half
    ):
        """Test complex multiplication properties in RoPE implementation."""
        head_dim = head_dim_half * 2  # Ensure head_dim is even

        # Create unit vectors for testing complex arithmetic properties
        keys = torch.zeros(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim,
            device=device,
            dtype=torch.double,
        )
        # Set real parts to 1 (equivalent to complex numbers [1+0j, 1+0j, ...])
        keys[..., 0::2] = 1.0

        # Create rotation vectors (equivalent to e^{iθ})
        theta = torch.rand(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim_half,
            device=device,
            dtype=torch.double,
        ) * (2 * math.pi)
        rope_encoding = torch.zeros(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim_half,
            device=device,
            dtype=torch.double,
        )
        rope_encoding = theta  # phase angle directly

        # Rotation should preserve magnitude (|z| = |e^{iθ}z| = |z|)
        keys_rotated = rotate_keys(keys, rope_encoding)

        # Convert keys to complex for magnitude calculation
        keys_complex_view = keys.view(keys.shape[:-1] + (head_dim_half, 2))
        keys_complex = torch.view_as_complex(keys_complex_view)

        # Convert rotated keys to complex
        k_rotated_complex_view = keys_rotated.view(
            keys_rotated.shape[:-1] + (head_dim_half, 2)
        )
        k_rotated_complex = torch.view_as_complex(k_rotated_complex_view)

        # Compare magnitudes
        original_magnitudes = torch.abs(keys_complex)
        rotated_magnitudes = torch.abs(k_rotated_complex)

        assert_close(
            original_magnitudes,
            rotated_magnitudes,
            rtol=1e-4,
            msg="Complex rotation doesn't preserve magnitude",
        )

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        position_dim=valid_dims(),
        n_freq_groups=valid_dims(),
        n_heads=valid_dims(),
        half_head_dim=valid_dims(),
    )
    @settings(
        deadline=None,
        max_examples=5,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_zeros_ones_edge_cases(
        self,
        device,
        n_queries,
        n_keys_per_query,
        position_dim,
        n_freq_groups,
        n_heads,
        half_head_dim,
    ):
        """Test edge cases with zeros and ones."""
        # All zeros
        key_positions_zeros = torch.zeros(
            n_queries, n_keys_per_query, position_dim, device=device, dtype=torch.double
        )
        rope_freqs_ones = torch.ones(
            position_dim,
            n_freq_groups,
            n_heads,
            half_head_dim,
            device=device,
            dtype=torch.double,
        )

        result_zeros = calculate_rope(key_positions_zeros, rope_freqs_ones)
        assert torch.allclose(
            result_zeros, torch.zeros_like(result_zeros)
        ), "calculate_rope with zero positions should give zero outputs"

        # All ones
        key_positions_ones = torch.ones(
            n_queries, n_keys_per_query, position_dim, device=device, dtype=torch.double
        )
        rope_freqs_ones = torch.ones(
            position_dim,
            n_freq_groups,
            n_heads,
            half_head_dim,
            device=device,
            dtype=torch.double,
        )

        # Result should be sum over position_dim for each frequency group
        expected = torch.ones(
            n_queries,
            n_keys_per_query,
            n_heads,
            half_head_dim,
            device=device,
            dtype=torch.double,
        ) * (position_dim * n_freq_groups)

        result_ones = calculate_rope(key_positions_ones, rope_freqs_ones)
        assert_close(
            result_ones,
            expected,
            msg="calculate_rope with all ones doesn't give expected output",
        )

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        n_heads=valid_dims(),
        head_dim_half=valid_dims(),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        deadline=None,
        max_examples=5,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_determinism(
        self, device, n_queries, n_keys_per_query, n_heads, head_dim_half, seed
    ):
        """Test deterministic behavior with same seed."""
        head_dim = head_dim_half * 2  # Ensure head_dim is even

        # Set seed
        torch.manual_seed(seed)
        keys_1 = torch.randn(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim,
            device=device,
            dtype=torch.double,
        )
        rope_encoding_1 = torch.randn(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim_half,
            device=device,
            dtype=torch.double,
        )
        k_rotated_1 = rotate_keys(keys_1, rope_encoding_1)

        # Reset seed and compute again
        torch.manual_seed(seed)
        keys_2 = torch.randn(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim,
            device=device,
            dtype=torch.double,
        )
        rope_encoding_2 = torch.randn(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim_half,
            device=device,
            dtype=torch.double,
        )
        k_rotated_2 = rotate_keys(keys_2, rope_encoding_2)

        # Results should be identical
        assert torch.all(keys_1 == keys_2), "Random number generation not deterministic"
        assert torch.all(
            rope_encoding_1 == rope_encoding_2
        ), "Random number generation not deterministic"
        assert torch.all(k_rotated_1 == k_rotated_2), "rotate_k is not deterministic"

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        position_dim=valid_dims(),
        n_freq_groups=st.integers(min_value=2, max_value=4),
        n_heads=valid_dims(),
        half_head_dim=valid_dims(),
    )
    @settings(
        deadline=None,
        max_examples=5,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_additive_rope_freq_groups(
        self,
        device,
        n_queries,
        n_keys_per_query,
        position_dim,
        n_freq_groups,
        n_heads,
        half_head_dim,
    ):
        """Test that frequency groups are additive in calculate_rope."""
        key_positions = torch.randn(
            n_queries, n_keys_per_query, position_dim, device=device, dtype=torch.double
        )

        # Create separate frequency groups
        rope_freqs_list = [
            torch.randn(
                position_dim,
                1,
                n_heads,
                half_head_dim,
                device=device,
                dtype=torch.double,
            )
            for _ in range(n_freq_groups)
        ]

        # Combined frequency groups
        rope_freqs_combined = torch.cat([f for f in rope_freqs_list], dim=1)
        result_combined = calculate_rope(key_positions, rope_freqs_combined)

        # Calculate for each group separately and sum
        results_separate = [calculate_rope(key_positions, f) for f in rope_freqs_list]
        result_sum = sum(results_separate)

        # Results should match
        assert_close(
            result_combined,
            result_sum,
            rtol=1e-4,
            msg="Frequency groups aren't correctly additive",
        )

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        n_heads=valid_dims(),
        head_dim_half=valid_dims(),
    )
    @settings(
        deadline=None,
        max_examples=5,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_double_rotation_composition(
        self, device, n_queries, n_keys_per_query, n_heads, head_dim_half
    ):
        """Test that consecutive rotations compose correctly (e^{iθ}*e^{iφ} = e^{i(θ+φ)})."""
        head_dim = head_dim_half * 2  # Ensure head_dim is even

        # Create keys
        keys = torch.randn(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim,
            device=device,
            dtype=torch.double,
        )

        # Create two separate rotation angles
        theta1 = torch.rand(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim_half,
            device=device,
            dtype=torch.double,
        ) * (2 * math.pi)
        theta2 = torch.rand(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim_half,
            device=device,
            dtype=torch.double,
        ) * (2 * math.pi)

        # Apply rotations in sequence
        k_rotated1 = rotate_keys(keys, theta1)
        k_rotated_sequential = rotate_keys(k_rotated1, theta2)

        # Apply combined rotation
        k_rotated_combined = rotate_keys(keys, theta1 + theta2)

        # Results should match
        assert_close(
            k_rotated_sequential,
            k_rotated_combined,
            rtol=1e-4,
            atol=1e-6,
            msg="Consecutive rotations don't compose correctly",
        )

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        n_heads=valid_dims(),
        half_head_dim=valid_dims(),
    )
    @settings(
        deadline=None,
        max_examples=5,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_broadcasting_heads(
        self,
        device,
        n_queries,
        n_keys_per_query,
        n_heads,
        half_head_dim,
    ):
        """Test broadcasting across heads dimension in rotation functions."""
        head_dim = half_head_dim * 2

        # Create a key tensor with multiple heads
        keys = torch.randn(
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim,
            device=device,
            dtype=torch.double,
        )

        # Create rope_encoding with only 1 in the head dimension for broadcasting
        rope_encoding_single_head = torch.randn(
            n_queries,
            n_keys_per_query,
            1,
            half_head_dim,
            device=device,
            dtype=torch.double,
        )

        # Apply rotation
        keys_rotated = rotate_keys(keys, rope_encoding_single_head)

        # Create gradient for backward pass
        grad_k_rotated = torch.randn_like(keys_rotated)

        # Run backward pass
        grad_keys, grad_rope_encoding = rotate_keys_backward(
            grad_k_rotated, keys, rope_encoding_single_head, True, True
        )

        # Verify shape of gradients
        assert grad_keys.shape == keys.shape, "Gradient for keys has wrong shape"
        assert (
            grad_rope_encoding.shape == rope_encoding_single_head.shape
        ), "Gradient for rope_encoding has wrong shape"

        # Alternative calculation to verify correctness
        rope_encoding_expanded = rope_encoding_single_head.expand(-1, -1, n_heads, -1)

        # Forward pass with expanded tensor
        k_rotated_expanded = rotate_keys(keys, rope_encoding_expanded)

        # Results should match
        assert_close(
            keys_rotated,
            k_rotated_expanded,
            rtol=1e-5,
            msg="Broadcasting in rotate_k doesn't match explicit expansion",
        )
