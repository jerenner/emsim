import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from emsim.utils.sparse_utils.ops.subset_attn.rotary_embedding import (
    calculate_rope,
    calculate_rope_backward,
    rotate_k,
    rotate_k_backward,
)

from .conftest import assert_close, even_dims, valid_dims


@pytest.mark.cuda
class TestEndToEnd:
    """End-to-end tests that combine multiple functions."""

    n_heads = 2
    n_queries = 4
    n_keys_per_query = 6
    position_dim = 3
    head_dim = 8
    embed_dim = n_heads * head_dim

    def test_forward_backward_pipeline(self, device):
        """Test the full pipeline: calculate_rope -> rotate_k -> backward passes."""
        # Setup

        # Create inputs that require gradients
        key_positions = torch.randn(
            self.n_queries,
            self.n_keys_per_query,
            self.position_dim,
            requires_grad=True,
            device=device,
        )
        rope_freqs = torch.randn(
            self.position_dim, 1, self.embed_dim, requires_grad=True, device=device
        )
        keys = torch.randn(
            self.n_queries,
            self.n_keys_per_query,
            self.embed_dim,
            requires_grad=True,
            device=device,
        )

        # Forward pass
        rope_encoding = calculate_rope(key_positions, rope_freqs)
        k_rotated, keys_complex, rope_encoding_complex = rotate_k(keys, rope_encoding)

        # Loss and autograd backward
        loss = k_rotated.sum()
        loss.backward()

        # Manual backward pass
        grad_k_rotated = torch.ones_like(k_rotated)  # Gradient from sum() is 1

        # First backward through rotate_k
        grad_k, grad_rope_encoding = rotate_k_backward(
            grad_k_rotated, keys_complex, rope_encoding_complex, True
        )

        # Then backward through calculate_rope
        grad_key_positions, grad_rope_freqs = calculate_rope_backward(
            grad_rope_encoding, key_positions, rope_freqs, True, True
        )

        # Check gradients match autograd
        assert_close(
            grad_k, keys.grad, msg="Manual grad_k doesn't match autograd"
        )
        assert_close(
            grad_key_positions,
            key_positions.grad,
            msg="Manual grad_key_positions doesn't match autograd",
        )
        assert_close(
            grad_rope_freqs,
            rope_freqs.grad,
            msg="Manual grad_rope_freqs doesn't match autograd",
        )

    def test_broadcasting(self, device):
        """Test that rope_encoding can be broadcasted over multiple heads."""
        # Multiple heads in keys, single head in rope_encoding
        keys = torch.tensor(
            [
                # head 1
                [1.0, 0.0, 2.0, 0.0],
                # head 2
                [3.0, 0.0, 4.0, 0.0],
                # head 3
                [5.0, 0.0, 6.0, 0.0],
            ],
            dtype=torch.float,
            device=device,
        ).view(1, 1, 3, 4)

        # Single head rope encoding to be broadcasted
        rope_encoding = torch.tensor(
            [0.5, 0.866, 0.5, 0.866], dtype=torch.float, device=device
        ).view(1, 1, 1, 4)

        # Expected results after broadcasting and complex multiplication:
        # For head 1: (1+0j)*(0.5+0.866j)=0.5+0.866j, (2+0j)*(0.5+0.866j)=1.0+1.732j
        # For head 2: (3+0j)*(0.5+0.866j)=1.5+2.598j, (4+0j)*(0.5+0.866j)=2.0+3.464j
        # For head 3: (5+0j)*(0.5+0.866j)=2.5+4.33j, (6+0j)*(0.5+0.866j)=3.0+5.196j
        expected = torch.tensor(
            [
                [
                    [
                        [0.5, 0.866, 1.0, 1.732],  # head 1
                        [1.5, 2.598, 2.0, 3.464],  # head 2
                        [2.5, 4.33, 3.0, 5.196],  # head 3
                    ]
                ]
            ],
            dtype=torch.float32,
            device=device,
        )

        k_rotated, keys_complex, rope_encoding_complex = rotate_k(keys, rope_encoding)

        # Verify the rotation results
        assert_close(
            k_rotated,
            expected,
            msg="Broadcasting in rotate_k failed",
        )

        # Check that complex representations have the correct shapes
        assert keys_complex.shape == (1, 1, 3, 2), "Keys complex shape incorrect"
        assert rope_encoding_complex.shape == (
            1,
            1,
            1,
            2,
        ), "Rope encoding complex shape incorrect"

        # Check complex representations are correct
        expected_keys_complex = torch.complex(keys[..., 0::2], keys[..., 1::2])
        expected_rope_complex = torch.complex(
            rope_encoding[..., 0::2], rope_encoding[..., 1::2]
        )

        assert_close(
            keys_complex,
            expected_keys_complex,
            msg="Keys complex representation incorrect",
        )
        assert_close(
            rope_encoding_complex,
            expected_rope_complex,
            msg="Rope encoding complex representation incorrect",
        )

        # Test gradient broadcasting - create dummy gradients
        grad_k_rotated = torch.ones_like(k_rotated)
        grad_k, grad_rope_encoding = rotate_k_backward(
            grad_k_rotated, keys_complex, rope_encoding_complex
        )

        # Keys gradient should maintain original shape
        assert grad_k.shape == keys.shape, "Keys gradient has wrong shape"

        # Rope encoding gradient should maintain broadcasting shape
        assert (
            grad_rope_encoding.shape == rope_encoding.shape
        ), "Rope encoding gradient has wrong shape"
        # For broadcasted rope encoding, each head contributes to gradient
        assert (
            grad_rope_encoding[0, 0, 0, 0] == 3.0
        ), "Rope encoding gradient values incorrect"

    def test_frechet_product(self):
        """Test that gradients follow the Fréchet product rule for complex multiplication."""
        # Simple test case
        keys = torch.randn(
            self.n_heads,
            self.n_queries,
            self.n_keys_per_query,
            self.head_dim,
            requires_grad=True,
        )
        rope_encoding = torch.randn(
            self.n_queries, self.n_keys_per_query, self.embed_dim, requires_grad=True
        )

        # Forward pass
        k_rotated, _, _ = rotate_k(keys, rope_encoding)
        loss = k_rotated.sum()

        # Verify gradient chain rule is correctly implemented
        loss.backward()

        # For complex multiplication z = x * y, gradients are:
        # dL/dx = dL/dz * conj(y)
        # View tensors as complex numbers
        keys_complex = torch.view_as_complex(
            keys.view(keys.shape[:-1] + (self.head_dim // 2, 2))
        )
        rope_encoding_permuted = (
            rope_encoding.view(rope_encoding.shape[:-1] + (self.n_heads, self.head_dim))
            .permute(2, 0, 1, 3)
            .contiguous()
        )
        rope_encoding_complex = torch.view_as_complex(
            rope_encoding_permuted.view(
                rope_encoding_permuted.shape[:-1] + (self.head_dim // 2, 2)
            )
        )

        # Create gradient tensors
        dL_dz = keys.new_ones(
            self.n_heads, self.n_queries, self.n_keys_per_query, self.head_dim // 2, 2
        )
        dL_dz_complex = torch.view_as_complex(dL_dz)

        # Expected gradients using the Fréchet product rule
        expected_dL_dx_complex = dL_dz_complex * rope_encoding_complex.conj()
        expected_dL_dy_complex = dL_dz_complex * keys_complex.conj()

        # Convert to real representation to compare with PyTorch gradients
        expected_dL_dx = torch.view_as_real(expected_dL_dx_complex).reshape_as(keys)

        expected_dL_dy = torch.view_as_real(expected_dL_dy_complex).reshape_as(
            rope_encoding
        )

        assert_close(
            keys.grad,
            expected_dL_dx,
            msg="Keys gradient doesn't match Fréchet product rule",
        )
        assert_close(
            rope_encoding.grad,
            expected_dL_dy,
            msg="Rope encoding gradient doesn't match Fréchet product rule",
        )


class TestHypothesis:
    """Property-based tests using hypothesis."""

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        position_dim=valid_dims(),
        n_freq_groups=valid_dims(),
        embed_dim=even_dims(),
    )
    @settings(deadline=None, max_examples=10)
    def test_calculate_rope_gradient_consistency(
        self, n_queries, n_keys_per_query, position_dim, n_freq_groups, embed_dim
    ):
        """Property-based test to verify gradients are consistent with autograd."""
        # Create tensors that require gradients
        key_positions = torch.randn(
            n_queries, n_keys_per_query, position_dim, requires_grad=True
        )
        rope_freqs = torch.randn(
            position_dim, n_freq_groups, embed_dim, requires_grad=True
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
        n_heads=valid_dims(),
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        head_dim_half=valid_dims(),
    )
    @settings(deadline=None, max_examples=10)
    def test_rotate_k_gradient_consistency(
        self, n_heads, n_queries, n_keys_per_query, head_dim_half
    ):
        """Test that rotate_k gradients are consistent with autograd."""
        head_dim = head_dim_half * 2  # Ensure head_dim is even

        # Create tensors requiring gradients
        keys = torch.randn(
            n_heads, n_queries, n_keys_per_query, head_dim, requires_grad=True
        )
        rope_encoding = torch.randn(
            n_queries, n_keys_per_query, head_dim, requires_grad=True
        )

        # Forward pass
        k_rotated, keys_complex, rope_encoding_complex = rotate_k(keys, rope_encoding)

        # Autograd backward
        grad_output = torch.randn_like(k_rotated)
        k_rotated.backward(grad_output)

        # Store autograd gradients
        keys_grad_autograd = keys.grad.clone()
        rope_encoding_grad_autograd = rope_encoding.grad.clone()

        # Reset gradients
        keys.grad = None
        rope_encoding.grad = None

        # Manual backward pass
        grad_k, grad_rope_encoding = rotate_k_backward(
            grad_output, keys_complex, rope_encoding_complex, True
        )

        # Compare gradients
        assert_close(
            grad_k, keys_grad_autograd, msg="Manual grad_k doesn't match autograd"
        )
        assert_close(
            grad_rope_encoding,
            rope_encoding_grad_autograd,
            msg="Manual grad_rope_encoding doesn't match autograd",
        )

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        position_dim=valid_dims(),
        embed_dim=even_dims(),
    )
    @settings(deadline=None, max_examples=10)
    def test_numerical_stability(
        self, n_queries, n_keys_per_query, position_dim, embed_dim
    ):
        """Test numerical stability with large and small values."""
        # Test with very small values
        key_positions_small = (
            torch.rand(n_queries, n_keys_per_query, position_dim) * 1e-6
        )
        rope_freqs_small = torch.rand(position_dim, 1, embed_dim) * 1e-6

        # Test with very large values
        key_positions_large = (
            torch.rand(n_queries, n_keys_per_query, position_dim) * 1e6
        )
        rope_freqs_large = torch.rand(position_dim, 1, embed_dim) * 1e6

        # Forward pass should not produce NaNs or infinities
        result_small = calculate_rope(key_positions_small, rope_freqs_small)
        result_large = calculate_rope(key_positions_large, rope_freqs_large)

        assert not torch.isnan(result_small).any(), "Small values produced NaNs"
        assert not torch.isinf(result_small).any(), "Small values produced infinities"
        assert not torch.isnan(result_large).any(), "Large values produced NaNs"
        assert not torch.isinf(result_large).any(), "Large values produced infinities"

    @given(
        n_heads=valid_dims(),
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        head_dim_half=valid_dims(),
    )
    @settings(deadline=None, max_examples=10)
    def test_linearity_of_gradients(
        self, n_heads, n_queries, n_keys_per_query, head_dim_half
    ):
        """Test that gradients follow linearity property."""
        head_dim = head_dim_half * 2  # Ensure head_dim is even

        keys = torch.randn(
            n_heads, n_queries, n_keys_per_query, head_dim, requires_grad=True
        )
        rope_encoding = torch.randn(
            n_queries, n_keys_per_query, head_dim, requires_grad=True
        )

        # Forward pass
        k_rotated, _, _ = rotate_k(keys, rope_encoding)

        # Create two different gradient outputs
        grad_output1 = torch.randn_like(k_rotated)
        grad_output2 = torch.randn_like(k_rotated)
        alpha = torch.rand(1).item()

        # Calculate gradients for each output separately
        k_rotated.backward(grad_output1, retain_graph=True)
        keys_grad1 = keys.grad.clone()
        rope_encoding_grad1 = rope_encoding.grad.clone()

        keys.grad = None
        rope_encoding.grad = None

        k_rotated.backward(grad_output2, retain_graph=True)
        keys_grad2 = keys.grad.clone()
        rope_encoding_grad2 = rope_encoding.grad.clone()

        keys.grad = None
        rope_encoding.grad = None

        # Calculate gradients for linear combination
        combined_grad_output = alpha * grad_output1 + (1 - alpha) * grad_output2
        k_rotated.backward(combined_grad_output)
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
            msg="Gradients don't satisfy linearity for keys",
        )
        assert_close(
            rope_encoding_grad_combined,
            expected_rope_grad,
            rtol=1e-4,
            msg="Gradients don't satisfy linearity for rope_encoding",
        )

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        position_dim=valid_dims(),
        embed_dim=even_dims(),
    )
    @settings(deadline=None, max_examples=5)
    def test_rope_permutation_invariance(
        self, n_queries, n_keys_per_query, position_dim, embed_dim
    ):
        """Test that permutation of queries doesn't affect batch independence."""
        # Create inputs
        key_positions = torch.randn(n_queries, n_keys_per_query, position_dim)
        rope_freqs = torch.randn(position_dim, 1, embed_dim)

        # Get results
        result = calculate_rope(key_positions, rope_freqs)

        # Create a permutation of the queries
        perm_indices = torch.randperm(n_queries)
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
        n_heads=valid_dims(),
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        head_dim_half=valid_dims(),
    )
    @settings(deadline=None, max_examples=5)
    def test_complex_multiplication_properties(
        self, n_heads, n_queries, n_keys_per_query, head_dim_half
    ):
        """Test complex multiplication properties in RoPE implementation."""
        head_dim = head_dim_half * 2  # Ensure head_dim is even

        # Create unit vectors for testing complex arithmetic properties
        keys = torch.zeros(n_heads, n_queries, n_keys_per_query, head_dim)
        # Set real parts to 1 (equivalent to complex numbers [1+0j, 1+0j, ...])
        keys[..., 0::2] = 1.0

        # Create rotation vectors (equivalent to e^{iθ})
        theta = torch.rand(n_queries, n_keys_per_query, head_dim_half) * (2 * math.pi)
        rope_encoding = torch.zeros(n_queries, n_keys_per_query, head_dim)
        rope_encoding[..., 0::2] = torch.cos(theta)
        rope_encoding[..., 1::2] = torch.sin(theta)

        # Rotation should preserve magnitude (|z| = |e^{iθ}z| = |z|)
        k_rotated, _, _ = rotate_k(keys, rope_encoding)

        # Compute magnitudes (for complex numbers a+bi, magnitude is sqrt(a²+b²))
        original_magnitudes = torch.sqrt(keys[..., 0::2] ** 2 + keys[..., 1::2] ** 2)
        rotated_magnitudes = torch.sqrt(
            k_rotated[..., 0::2] ** 2 + k_rotated[..., 1::2] ** 2
        )

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
        embed_dim=even_dims(),
    )
    @settings(deadline=None, max_examples=5)
    def test_zeros_ones_edge_cases(
        self, n_queries, n_keys_per_query, position_dim, embed_dim
    ):
        """Test edge cases with zeros and ones."""
        # All zeros
        key_positions_zeros = torch.zeros(n_queries, n_keys_per_query, position_dim)
        rope_freqs_ones = torch.ones(position_dim, 1, embed_dim)

        result_zeros = calculate_rope(key_positions_zeros, rope_freqs_ones)
        assert torch.allclose(
            result_zeros, torch.zeros_like(result_zeros)
        ), "calculate_rope with zero positions should give zero outputs"

        # All ones
        key_positions_ones = torch.ones(n_queries, n_keys_per_query, position_dim)
        rope_freqs_ones = torch.ones(position_dim, 1, embed_dim)

        # Result should be sum over position_dim
        expected = torch.ones(n_queries, n_keys_per_query, embed_dim) * position_dim
        result_ones = calculate_rope(key_positions_ones, rope_freqs_ones)
        assert_close(
            result_ones,
            expected,
            msg="calculate_rope with all ones doesn't give expected output",
        )

    @given(
        n_heads=valid_dims(),
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        head_dim_half=valid_dims(),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(deadline=None, max_examples=5)
    def test_determinism(
        self, n_heads, n_queries, n_keys_per_query, head_dim_half, seed
    ):
        """Test deterministic behavior with same seed."""
        head_dim = head_dim_half * 2  # Ensure head_dim is even

        # Set seed
        torch.manual_seed(seed)
        keys1 = torch.randn(n_heads, n_queries, n_keys_per_query, head_dim)
        rope_encoding1 = torch.randn(n_queries, n_keys_per_query, head_dim)
        k_rotated1, _, _ = rotate_k(keys1, rope_encoding1)

        # Reset seed and compute again
        torch.manual_seed(seed)
        keys2 = torch.randn(n_heads, n_queries, n_keys_per_query, head_dim)
        rope_encoding2 = torch.randn(n_queries, n_keys_per_query, head_dim)
        k_rotated2, _, _ = rotate_k(keys2, rope_encoding2)

        # Results should be identical
        assert torch.all(keys1 == keys2), "Random number generation not deterministic"
        assert torch.all(
            rope_encoding1 == rope_encoding2
        ), "Random number generation not deterministic"
        assert torch.all(k_rotated1 == k_rotated2), "rotate_k is not deterministic"

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        position_dim=valid_dims(),
        n_freq_groups=st.integers(min_value=2, max_value=4),
        embed_dim=even_dims(),
    )
    @settings(deadline=None, max_examples=5)
    def test_additive_rope_freq_groups(
        self, n_queries, n_keys_per_query, position_dim, n_freq_groups, embed_dim
    ):
        """Test that frequency groups are additive in calculate_rope."""
        key_positions = torch.randn(n_queries, n_keys_per_query, position_dim)

        # Create separate frequency groups
        rope_freqs_list = [
            torch.randn(position_dim, 1, embed_dim) for _ in range(n_freq_groups)
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
        n_heads=valid_dims(),
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        head_dim_half=valid_dims(),
    )
    @settings(deadline=None, max_examples=5)
    def test_double_rotation_composition(
        self, n_heads, n_queries, n_keys_per_query, head_dim_half
    ):
        """Test that consecutive rotations compose correctly (e^{iθ}*e^{iφ} = e^{i(θ+φ)})."""
        head_dim = head_dim_half * 2  # Ensure head_dim is even

        # Create keys
        keys = torch.randn(n_heads, n_queries, n_keys_per_query, head_dim)

        # Create two separate rotation angles
        theta1 = torch.rand(n_queries, n_keys_per_query, head_dim_half) * (2 * math.pi)
        theta2 = torch.rand(n_queries, n_keys_per_query, head_dim_half) * (2 * math.pi)

        # Create rotation encodings
        rope_encoding1 = torch.zeros(n_queries, n_keys_per_query, head_dim)
        rope_encoding1[..., 0::2] = torch.cos(theta1)
        rope_encoding1[..., 1::2] = torch.sin(theta1)

        rope_encoding2 = torch.zeros(n_queries, n_keys_per_query, head_dim)
        rope_encoding2[..., 0::2] = torch.cos(theta2)
        rope_encoding2[..., 1::2] = torch.sin(theta2)

        # Combined rotation encoding (e^{i(θ+φ)})
        rope_encoding_combined = torch.zeros(n_queries, n_keys_per_query, head_dim)
        rope_encoding_combined[..., 0::2] = torch.cos(theta1 + theta2)
        rope_encoding_combined[..., 1::2] = torch.sin(theta1 + theta2)

        # Apply rotations in sequence
        k_rotated1, _, _ = rotate_k(keys, rope_encoding1)
        k_rotated_sequential, _, _ = rotate_k(k_rotated1, rope_encoding2)

        # Apply combined rotation
        k_rotated_combined, _, _ = rotate_k(keys, rope_encoding_combined)

        # Results should match
        assert_close(
            k_rotated_sequential,
            k_rotated_combined,
            rtol=1e-4,
            msg="Consecutive rotations don't compose correctly",
        )
