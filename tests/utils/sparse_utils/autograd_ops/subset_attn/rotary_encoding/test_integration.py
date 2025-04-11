import pytest
import torch
from hypothesis import given, settings, HealthCheck

from emsim.utils.sparse_utils.ops.subset_attn.rotary_encoding import (
    calculate_rope,
    calculate_rope_backward,
    rotate_embeddings,
    rotate_embeddings_backward,
)

from .conftest import assert_close, valid_dims


@pytest.mark.cuda_if_available
class TestEndToEnd:
    """End-to-end tests that combine multiple functions."""

    n_heads = 2
    n_queries = 4
    n_keys_per_query = 6
    position_dim = 3
    head_dim = 8
    half_head_dim = head_dim // 2

    def test_forward_backward_pipeline(self, device):
        """Test the full pipeline: calculate_rope -> rotate_k -> backward passes."""
        # Setup

        # Create inputs that require gradients
        positions = torch.randn(
            self.n_queries,
            self.n_keys_per_query,
            self.position_dim,
            requires_grad=True,
            device=device,
            dtype=torch.double,
        )
        rope_freqs = torch.randn(
            self.position_dim,
            1,
            self.n_heads,
            self.half_head_dim,
            requires_grad=True,
            device=device,
            dtype=torch.double,
        )
        embeddings = torch.randn(
            self.n_queries,
            self.n_keys_per_query,
            self.n_heads,
            self.head_dim,
            requires_grad=True,
            device=device,
            dtype=torch.double,
        )

        # Forward pass
        rope_encoding = calculate_rope(positions, rope_freqs)
        embeddings_rotated = rotate_embeddings(embeddings, rope_encoding)

        # Loss and autograd backward
        loss = embeddings_rotated.sum()
        loss.backward()

        # Manual backward pass
        grad_embeddings_rotated = torch.ones_like(
            embeddings_rotated
        )  # Gradient from sum() is 1

        # First backward through rotate_k
        grad_embeddings, grad_rope_encoding = rotate_embeddings_backward(
            grad_embeddings_rotated, embeddings, rope_encoding, True, True
        )

        # Then backward through calculate_rope
        grad_positions, grad_rope_freqs = calculate_rope_backward(
            grad_rope_encoding, positions, rope_freqs, True, True
        )

        # Check gradients match autograd
        assert_close(
            grad_embeddings,
            embeddings.grad,
            msg="Manual grad_embeddings doesn't match autograd",
        )
        assert_close(
            grad_positions,
            positions.grad,
            msg="Manual grad_positions doesn't match autograd",
        )
        assert_close(
            grad_rope_freqs,
            rope_freqs.grad,
            msg="Manual grad_rope_freqs doesn't match autograd",
        )

    def test_broadcasting(self, device):
        """Test that rope_encoding can be broadcasted over multiple heads."""
        # Multiple heads in embeddings
        embeddings = torch.tensor(
            [
                # Query 1, Key 1
                [
                    # head 1
                    [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
                    # head 2
                    [3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0],
                    # head 3
                    [5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0],
                ]
            ],
            dtype=torch.float,
            device=device,
        ).view(
            1, 1, 3, 8
        )  # [n_queries, n_keys_per_query, n_heads, head_dim]

        # Single head rope encoding to be broadcasted (phase angles)
        rope_encoding = torch.tensor(
            [[[0.5, 0.5, 0.5, 0.5]]],  # angle values in radians
            dtype=torch.float,
            device=device,
        ).view(
            1, 1, 1, 4
        )  # [n_queries, n_keys_per_query, 1, head_dim/2]

        # Expected results after rotation with angle 0.5 rad:
        # cos(0.5) ≈ 0.8776, sin(0.5) ≈ 0.4794
        # For complex number z = a+bi rotated by θ: z' = a*cos(θ) - b*sin(θ) + i(a*sin(θ) + b*cos(θ))
        # For real input (b=0): z' = a*cos(θ) + i(a*sin(θ))
        expected = torch.zeros(1, 1, 3, 8, dtype=torch.float32, device=device)

        # Compute expected values
        cos_val, sin_val = torch.cos(torch.tensor(0.5)), torch.sin(torch.tensor(0.5))

        # head 1
        expected[0, 0, 0, 0] = 1.0 * cos_val  # real part
        expected[0, 0, 0, 1] = 1.0 * sin_val  # imag part
        expected[0, 0, 0, 2] = 2.0 * cos_val
        expected[0, 0, 0, 3] = 2.0 * sin_val
        expected[0, 0, 0, 4] = 3.0 * cos_val
        expected[0, 0, 0, 5] = 3.0 * sin_val
        expected[0, 0, 0, 6] = 4.0 * cos_val
        expected[0, 0, 0, 7] = 4.0 * sin_val

        # head 2
        expected[0, 0, 1, 0] = 3.0 * cos_val
        expected[0, 0, 1, 1] = 3.0 * sin_val
        expected[0, 0, 1, 2] = 4.0 * cos_val
        expected[0, 0, 1, 3] = 4.0 * sin_val
        expected[0, 0, 1, 4] = 5.0 * cos_val
        expected[0, 0, 1, 5] = 5.0 * sin_val
        expected[0, 0, 1, 6] = 6.0 * cos_val
        expected[0, 0, 1, 7] = 6.0 * sin_val

        # head 3
        expected[0, 0, 2, 0] = 5.0 * cos_val
        expected[0, 0, 2, 1] = 5.0 * sin_val
        expected[0, 0, 2, 2] = 6.0 * cos_val
        expected[0, 0, 2, 3] = 6.0 * sin_val
        expected[0, 0, 2, 4] = 7.0 * cos_val
        expected[0, 0, 2, 5] = 7.0 * sin_val
        expected[0, 0, 2, 6] = 8.0 * cos_val
        expected[0, 0, 2, 7] = 8.0 * sin_val

        embeddings_rotated = rotate_embeddings(embeddings, rope_encoding)

        # Verify the rotation results
        assert_close(
            embeddings_rotated,
            expected,
            rtol=1e-4,
            msg="Broadcasting in rotate_k failed",
        )

        # Test gradient broadcasting - create dummy gradients
        grad_embeddings_rotated = torch.ones_like(embeddings_rotated)
        grad_embeddings, grad_rope_encoding = rotate_embeddings_backward(
            grad_embeddings_rotated, embeddings, rope_encoding, True, True
        )

        # Keys gradient should maintain original shape
        assert (
            grad_embeddings.shape == embeddings.shape
        ), "Embeddings gradient has wrong shape"

        # Rope encoding gradient should maintain broadcasting shape
        assert (
            grad_rope_encoding.shape == rope_encoding.shape
        ), "Rope encoding gradient has wrong shape"

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        n_heads=valid_dims(),
        half_head_dim=valid_dims(),
    )
    @settings(
        deadline=None,
        max_examples=10,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_frechet_product(
        self, device, n_queries, n_keys_per_query, n_heads, half_head_dim
    ):
        """Test that the implementation correctly handles Frechet products in complex space.

        The Frechet derivative of (f·g)(x) is f'(x)·g(x) + f(x)·g'(x).
        For complex multiplication z = x * y, the gradients are:
        dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x).
        """
        head_dim = half_head_dim * 2  # Ensure head_dim is even

        # Create inputs and set requires_grad=True for autograd
        embeddings = torch.randn(
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
            half_head_dim,
            requires_grad=True,
            device=device,
            dtype=torch.double,
        )

        # Forward pass to get complex representations
        embeddings_rotated = rotate_embeddings(embeddings, rope_encoding)

        # Create arbitrary gradients for output
        grad_output = torch.randn_like(embeddings_rotated, device=device)

        # Analytical gradients using the Frechet product formula
        # For complex multiplication z = x * y:
        # dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x)

        # Convert grad_output to complex format
        grad_output_complex_view = grad_output.view(
            grad_output.shape[:-1] + (half_head_dim, 2)
        )
        grad_output_complex = torch.view_as_complex(grad_output_complex_view)

        rope_encoding_complex = torch.polar(
            torch.ones_like(rope_encoding),
            rope_encoding,
        )
        embeddings_complex_shape = embeddings.shape[:-1] + (embeddings.size(-1) // 2, 2)
        embeddings_complex = torch.view_as_complex(
            embeddings.view(embeddings_complex_shape)
        )

        # Compute gradients analytically
        analytical_grad_embeddings_complex = (
            grad_output_complex * rope_encoding_complex.conj()
        )
        analytical_grad_rope_complex = grad_output_complex * embeddings_complex.conj()

        # Convert back to real tensor format
        analytical_grad_embeddings = torch.view_as_real(
            analytical_grad_embeddings_complex
        ).reshape_as(embeddings)

        # For rope_encoding, we need the gradient of the angle
        # Since rope_encoding_complex = exp(i*rope_encoding), the gradient is:
        # dL/d(rope_encoding) = Im(dL/d(rope_encoding_complex) / rope_encoding_complex)
        analytical_grad_rope = (
            analytical_grad_rope_complex / rope_encoding_complex
        ).imag

        # Compare with autograd gradients
        embeddings_rotated.backward(grad_output)

        # Check that analytical and autograd gradients match
        assert_close(
            analytical_grad_embeddings,
            embeddings.grad,
            rtol=1e-4,
            msg="Analytical gradient for embeddings doesn't match autograd",
        )
        assert_close(
            analytical_grad_rope,
            rope_encoding.grad,
            rtol=1e-4,
            msg="Analytical gradient for rope_encoding doesn't match autograd",
        )

        # Now test the backward method directly
        embeddings.grad = None
        rope_encoding.grad = None

        grad_embeddings, grad_rope_encoding = rotate_embeddings_backward(
            grad_output, embeddings, rope_encoding, True, True, True
        )

        # Compare with analytical gradients
        assert_close(
            grad_embeddings,
            analytical_grad_embeddings,
            rtol=1e-4,
            msg=(
                "rotate_embeddings_backward gradient for embeddings doesn't match "
                "analytical gradient",
            ),
        )
        assert_close(
            grad_rope_encoding,
            analytical_grad_rope,
            rtol=1e-4,
            msg=(
                "rotate_embeddings_backward gradient for rope_encoding doesn't match "
                "analytical gradient",
            ),
        )

        # Test broadcasting case (single head in rope_encoding)
        if n_heads > 1:
            # Create rope_encoding with only one head
            rope_encoding_single = torch.randn(
                n_queries,
                n_keys_per_query,
                1,
                half_head_dim,
                device=device,
                dtype=torch.double,
            )
            rope_encoding_single_complex = torch.polar(
                torch.ones_like(rope_encoding_single),
                rope_encoding_single,
            )

            # Forward pass with broadcasting
            embeddings_rotated_broadcast = rotate_embeddings(
                embeddings, rope_encoding_single
            )

            # Create same gradients for both passes
            grad_output_broadcast = torch.randn_like(
                embeddings_rotated_broadcast, device=device
            )

            # Test our backward function directly with broadcasting
            grad_k_broadcast, grad_rope_broadcast = rotate_embeddings_backward(
                grad_output_broadcast,
                embeddings,
                rope_encoding_single,
                True,
                True,
                True,
            )

            # Verify the shape of the gradient for the broadcast tensor
            assert (
                grad_rope_broadcast.shape == rope_encoding_single.shape
            ), f"Expected grad shape {rope_encoding_single.shape}, got {grad_rope_broadcast.shape}"

            # Explicitly calculate what the gradient should be if we were to manually
            # sum contributions from all heads
            grad_output_complex_broadcast = torch.view_as_complex(
                grad_output_broadcast.view(
                    grad_output_broadcast.shape[:-1] + (half_head_dim, 2)
                )
            )

            # Calculate per-head gradients and then sum
            summed_grad_complex = torch.zeros(
                n_queries,
                n_keys_per_query,
                1,
                half_head_dim,
                dtype=torch.complex128,
                device=device,
            )

            # Manually calculate and sum the gradients for each head
            for h in range(n_heads):
                # Extract this head's data
                head_embeddings_complex = embeddings_complex[:, :, h : h + 1, :]
                head_grad_output = grad_output_complex_broadcast[:, :, h : h + 1, :]

                # Calculate gradient for this head
                head_grad_complex = head_grad_output * head_embeddings_complex.conj()

                # Accumulate
                summed_grad_complex += head_grad_complex

            # Convert to angle gradient
            expected_grad_broadcast = (
                summed_grad_complex / rope_encoding_single_complex
            ).imag

            # Verify the summing behavior for broadcasting is correct
            assert_close(
                grad_rope_broadcast,
                expected_grad_broadcast,
                rtol=1e-4,
                msg="Broadcasting in backward pass doesn't correctly sum gradients",
            )
