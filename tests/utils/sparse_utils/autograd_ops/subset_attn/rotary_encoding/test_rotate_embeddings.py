import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from emsim.utils.sparse_utils.ops.subset_attn.rotary_encoding import (
    rotate_embeddings,
    rotate_embeddings_backward,
)

from .conftest import assert_close


@pytest.mark.cuda_if_available
class TestRotateEmbeddings:
    """Tests for the rotate_embeddings function."""

    n_heads = 2
    head_dim = 8

    def test_basic_functionality(self, device):
        """Test basic operation with simple inputs."""
        # Simple case with known values
        embeddings = torch.tensor(
            [1.0, 0.0, 2.0, 0.0], dtype=torch.double, device=device
        ).view(1, 1, 1, 4)

        # π/3 radians = 60 degrees
        angles = torch.tensor(
            [torch.pi / 3, torch.pi / 3], dtype=torch.double, device=device
        ).view(1, 1, 1, 2)

        # For cos(π/3) = 0.5, sin(π/3) = 0.866
        # Complex multiplication: (1+0j)*(cos(π/3)+sin(π/3)j) = 0.5+0.866j
        # Complex multiplication: (2+0j)*(cos(π/3)+sin(π/3)j) = 1.0+1.732j
        expected = torch.tensor(
            [[[[0.5, 0.866, 1.0, 1.7321]]]], dtype=torch.double, device=device
        )

        embeddings_rotated = rotate_embeddings(embeddings, angles)

        assert_close(
            embeddings_rotated, expected, rtol=1e-4, msg="Basic rotate_tensor failed"
        )

    def test_extended_batch_dims(self, device):
        """Test with lots of batch dimensions"""
        batch_dims = [2, 4, 6, 8]

        embeddings = torch.randn(
            *batch_dims, self.n_heads, self.head_dim, device=device
        )
        rope_encoding = torch.randn(
            *batch_dims, self.n_heads, self.head_dim // 2, device=device
        )

        expected_embeddings_rotated_shape = (*batch_dims, self.n_heads, self.head_dim)
        embeddings_rotated = rotate_embeddings(embeddings, rope_encoding)

        assert embeddings_rotated.shape == expected_embeddings_rotated_shape

    def test_broadcasting(self, device):
        """Test that rope_encoding can be broadcasted over multiple heads."""
        # Multiple heads in keys, single head in rope_encoding
        embeddings = torch.tensor(
            [
                # head 1
                [1.0, 0.0, 2.0, 0.0],
                # head 2
                [3.0, 0.0, 4.0, 0.0],
                # head 3
                [5.0, 0.0, 6.0, 0.0],
            ],
            dtype=torch.double,
            device=device,
        ).view(1, 1, 3, 4)

        # Single head rope encoding (angles) to be broadcasted
        # π/3 radians = 60 degrees
        # Note: last dim is head_dim/2 (2 instead of 4)
        angles = torch.tensor(
            [torch.pi / 3, torch.pi / 3], dtype=torch.double, device=device
        ).view(1, 1, 1, 2)

        # Expected results after broadcasting and complex multiplication:
        # For cos(π/3) = 0.5, sin(π/3) = 0.866
        # For head 1: (1+0j)*(cos(π/3)+sin(π/3)j)=0.5+0.866j, (2+0j)*(cos(π/3)+sin(π/3)j)=1.0+1.732j
        # For head 2: (3+0j)*(cos(π/3)+sin(π/3)j)=1.5+2.598j, (4+0j)*(cos(π/3)+sin(π/3)j)=2.0+3.464j
        # For head 3: (5+0j)*(cos(π/3)+sin(π/3)j)=2.5+4.33j, (6+0j)*(cos(π/3)+sin(π/3)j)=3.0+5.196j
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
            dtype=torch.double,
            device=device,
        )

        embeddings_rotated = rotate_embeddings(embeddings, angles)

        # Verify the rotation results
        assert_close(
            embeddings_rotated,
            expected,
            rtol=1e-3,
            atol=1e-3,
            msg="Broadcasting in rotate_k failed",
        )

    def test_error_conditions(self, device):
        """Test that appropriate errors are raised for invalid inputs."""
        # Test embeddings of invalid shape
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected embeddings and rope_encoding to have the same",
        ):
            rotate_embeddings(
                torch.randn(2, 3, 4, device=device),
                torch.randn(2, 3, 4, 5, device=device),
            )
        # Test rope_encoding of invalid shape
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected embeddings and rope_encoding to have the same",
        ):
            rotate_embeddings(
                torch.randn(2, 3, 4, 8, device=device),
                torch.randn(2, 3, 4, device=device),
            )
        # Test odd head_dim
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected rope_encoding to have last dimension",
        ):
            rotate_embeddings(
                torch.randn(2, 3, 4, 5, device=device),  # odd head_dim
                torch.randn(2, 3, 4, 2, device=device),  # not head_dim/2
            )
        # Test mismatch between head_dim and rope_encoding dimension
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected rope_encoding to have last dimension",
        ):
            rotate_embeddings(
                torch.randn(2, 3, 4, 8, device=device),  # head_dim = 8
                torch.randn(2, 3, 4, 3, device=device),  # not head_dim/2 = 4
            )
        # Test complex inputs
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected embeddings to be real",
        ):
            rotate_embeddings(
                torch.randn(2, 3, 4, 8, dtype=torch.complex64, device=device),
                torch.randn(2, 3, 4, 4, device=device),
            )
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected rope_encoding to be real",
        ):
            rotate_embeddings(
                torch.randn(2, 3, 4, 8, device=device),
                torch.randn(2, 3, 4, 4, dtype=torch.complex64, device=device),
            )

        # Test incompatible shapes
        with pytest.raises(
            RuntimeError,
            match="The size of tensor a",
        ):
            # n_queries nonmatching
            rotate_embeddings(
                torch.randn(2, 16, 8, 32, device=device),
                torch.randn(3, 16, 8, 16, device=device),  # 16 = 32/2
            )
        with pytest.raises(
            RuntimeError,
            match="The size of tensor a",
        ):
            # n_keys_per_query nonmatching
            rotate_embeddings(
                torch.randn(2, 16, 8, 32, device=device),
                torch.randn(2, 132, 8, 16, device=device),  # 16 = 32/2
            )
        with pytest.raises(
            RuntimeError,
            match="The size of tensor a",
        ):
            # n_heads for key_rope_encoding nonmatching and not 1
            rotate_embeddings(
                torch.randn(2, 16, 8, 32, device=device),
                torch.randn(2, 16, 4, 16, device=device),  # 16 = 32/2
            )

    def test_needs_autograd(self, device):
        """Test that needs_autograd=False optimizes memory usage while preserving correctness."""
        # Create input tensors
        embeddings = torch.randn(2, 3, 4, 8, device=device)
        rope_encoding = torch.randn(2, 3, 4, 4, device=device)

        # Make copies to ensure we don't accidentally modify the originals
        emb_1 = embeddings.clone()
        emb_2 = embeddings.clone()
        encoding_1 = rope_encoding.clone()
        encoding_2 = rope_encoding.clone()

        # First call with needs_autograd=True (default behavior)
        emb_1_rotated = rotate_embeddings(emb_1, encoding_1, needs_autograd=True)

        # Call with needs_autograd=False (optimization)
        emb_2_rotated = rotate_embeddings(emb_2, encoding_2, needs_autograd=False)

        # Both calls should produce identical results
        assert_close(
            emb_1_rotated,
            emb_2_rotated,
            msg="needs_autograd=False produces different results",
        )

        # Memory optimization test: When needs_autograd=False, we expect the operation
        # to modify the tensor in-place, so the complex view of the input should be modified

        # First, verify we can observe in-place modifications to a complex view:
        emb_test = embeddings.clone().view(
            *embeddings.shape[:-1], embeddings.size(-1) // 2, 2
        )
        emb_complex_test = torch.view_as_complex(emb_test)
        original_emb = emb_complex_test[0, 0, 0, 0].clone()
        emb_complex_test[0, 0, 0, 0] += 1.0  # In-place modification

        # This should affect the original tensor
        modified_emb_test = torch.view_as_real(emb_complex_test).reshape_as(embeddings)
        assert not torch.allclose(
            embeddings, modified_emb_test
        ), "In-place test setup failed"

        # Now for the actual test: verify needs_autograd=False does in-place ops
        # We'll manually do the operations to check if the tensor was modified

        # Create fresh copies for a comparative test
        embeddings_3 = embeddings.clone()
        encoding_3 = rope_encoding.clone()

        # Create a complex view of k3
        emb_3_complex_shape = embeddings_3.shape[:-1] + (embeddings_3.size(-1) // 2, 2)
        emb_3_complex_view = embeddings_3.view(emb_3_complex_shape)
        emb_3_complex = torch.view_as_complex(emb_3_complex_view)

        # Save the original tensor for comparison
        original_emb = emb_3_complex.clone().detach()

        # Apply rotate_k with needs_autograd=False
        _ = rotate_embeddings(embeddings_3, encoding_3, needs_autograd=False)

        # Check if emb_3_complex was modified in-place
        # If it was, the original_value should no longer match
        assert not torch.equal(
            original_emb, emb_3_complex
        ), "needs_autograd=False did not perform in-place operations"

        # Additional test with autograd enabled
        embeddings_4 = embeddings.clone().requires_grad_(True)
        encoding_4 = rope_encoding.clone()

        # Should work fine with needs_autograd=True and requires_grad=True
        embeddings_4_rotated = rotate_embeddings(
            embeddings_4, encoding_4, needs_autograd=True
        )

        # This should be able to compute gradients
        loss = embeddings_4_rotated.sum()
        loss.backward()

        assert (
            embeddings_4.grad is not None
        ), "needs_autograd=True should support autograd"

        # But with needs_autograd=False, we'll get an error with requires_grad=True
        embeddings_5 = embeddings.clone().requires_grad_(True)
        encoding_5 = rope_encoding.clone()

        # This should raise an error because we can't do in-place ops on tensors that require grad
        with pytest.raises(RuntimeError, match="a leaf Variable that requires grad"):
            _ = rotate_embeddings(embeddings_5, encoding_5, needs_autograd=False)

    def test_half_precision(self, device):
        """Tests if the handling of half-precision inputs works correctly"""
        batch_dims = [2]

        embeddings = torch.randn(
            *batch_dims, self.n_heads, self.head_dim, device=device, dtype=torch.float32
        )
        rope_encoding = torch.randn(
            *batch_dims,
            self.n_heads,
            self.head_dim // 2,
            device=device,
            dtype=torch.float32
        )

        # full precision calculation
        rotated_full = rotate_embeddings(embeddings, rope_encoding)
        assert rotated_full.dtype == torch.float32

        # half precision calculation
        embeddings_half = embeddings.half()
        rope_encoding_half = rope_encoding.half()
        rotated_half = rotate_embeddings(embeddings_half, rope_encoding_half)
        assert rotated_half.dtype == torch.float16

        # bfloat16 calculation
        embeddings_bf16 = embeddings.bfloat16()
        rope_encoding_bf16 = rope_encoding.bfloat16()
        rotated_bf16 = rotate_embeddings(embeddings_bf16, rope_encoding_bf16)
        assert rotated_bf16.dtype == torch.bfloat16

        # Determine input precision error
        embeddings_half_error = torch.abs(embeddings - embeddings_half.float()).max()
        rope_half_error = torch.abs(rope_encoding - rope_encoding_half.float()).max()

        embeddings_bf16_error = torch.abs(embeddings - embeddings_bf16.float()).max()
        rope_bf16_error = torch.abs(rope_encoding - rope_encoding_bf16.float()).max()

        # Determine suitable tolerances
        atol_half = max(embeddings_half_error, rope_half_error) * 10
        atol_bf16 = max(embeddings_bf16_error, rope_bf16_error) * 10

        # Verify errors didn't explode during the operation
        assert torch.allclose(
            rotated_full, rotated_half.float(), atol=atol_half, rtol=1e-2
        )
        assert torch.allclose(
            rotated_full, rotated_bf16.float(), atol=atol_bf16, rtol=5e-2
        )


@pytest.mark.cuda_if_available
class TestRotateEmbeddingsProperties:

    @given(
        n_queries=st.integers(1, 10),
        n_keys=st.integers(1, 10),
        n_heads=st.integers(1, 8),
        head_dim=st.integers(2, 16).filter(lambda x: x % 2 == 0),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_shape_preservation(self, n_queries, n_keys, n_heads, head_dim, device):
        """Test that output shape matches input shape."""
        embeddings = torch.randn(n_queries, n_keys, n_heads, head_dim, device=device)
        rope = torch.randn(
            n_queries, n_keys, n_heads, head_dim // 2, device=device
        )  # half size

        rotated = rotate_embeddings(embeddings, rope)

        assert (
            rotated.shape == embeddings.shape
        ), "Output shape should match input shape"

    @given(
        n_queries=st.integers(1, 10),
        n_keys=st.integers(1, 10),
        n_heads=st.integers(1, 8),
        head_dim=st.integers(2, 16).filter(lambda x: x % 2 == 0),
    )
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    def test_broadcasting(self, n_queries, n_keys, n_heads, head_dim, device):
        """Test that broadcasting works the same as explicit expansion."""
        embeddings = torch.randn(n_queries, n_keys, n_heads, head_dim, device=device)

        # Broadcasted version (1 head)
        rope_broadcast = torch.randn(
            n_queries, n_keys, 1, head_dim // 2, device=device
        )  # half size
        # Expanded version (explicit repeated across heads)
        rope_expanded = rope_broadcast.expand(n_queries, n_keys, n_heads, head_dim // 2)

        rotated_broadcast = rotate_embeddings(embeddings, rope_broadcast)
        rotated_expanded = rotate_embeddings(embeddings, rope_expanded)

        assert_close(
            rotated_broadcast,
            rotated_expanded,
            atol=1e-6,
            msg="Broadcasting should produce same result as explicit expansion",
        )

    @given(
        n_queries=st.integers(1, 10),
        n_keys=st.integers(1, 10),
        n_heads=st.integers(1, 8),
        head_dim=st.integers(2, 16).filter(lambda x: x % 2 == 0),
        scale=st.floats(0.1, 10.0),
    )
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    def test_homogeneity(self, n_queries, n_keys, n_heads, head_dim, scale, device):
        """Test that scaling the input scales the output by the same factor."""
        embeddings = torch.randn(
            n_queries, n_keys, n_heads, head_dim, dtype=torch.double, device=device
        )
        # Use angles for rope
        rope = torch.randn(
            n_queries, n_keys, n_heads, head_dim // 2, dtype=torch.double, device=device
        )

        # Get output for original input
        rotated_1 = rotate_embeddings(embeddings, rope)

        # Get output for scaled input
        rotated_2 = rotate_embeddings(embeddings * scale, rope)

        assert_close(
            rotated_2,
            rotated_1 * scale,
            msg="Scaling the input should scale the output",
        )

    @given(
        n_queries=st.integers(1, 10),
        n_keys=st.integers(1, 10),
        n_heads=st.integers(1, 8),
        head_dim=st.integers(2, 16).filter(lambda x: x % 2 == 0),
    )
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    def test_linearity(self, n_queries, n_keys, n_heads, head_dim, device):
        """Test that the function is linear in its first argument."""
        embeddings_1 = torch.randn(
            n_queries, n_keys, n_heads, head_dim, dtype=torch.double, device=device
        )
        embeddings_2 = torch.randn(
            n_queries, n_keys, n_heads, head_dim, dtype=torch.double, device=device
        )
        # Use angles for rope
        rope = torch.randn(
            n_queries, n_keys, n_heads, head_dim // 2, dtype=torch.double, device=device
        )

        # f(a + b) should equal f(a) + f(b)
        rotated_sum = rotate_embeddings(embeddings_1 + embeddings_2, rope)
        rotated_1 = rotate_embeddings(embeddings_1, rope)
        rotated_2 = rotate_embeddings(embeddings_2, rope)

        assert_close(
            rotated_sum,
            rotated_1 + rotated_2,
            msg="Function should be linear in its first argument",
        )

    @given(
        n_queries=st.integers(1, 10),
        n_keys=st.integers(1, 10),
        n_heads=st.integers(1, 8),
        head_dim=st.integers(2, 16).filter(lambda x: x % 2 == 0),
    )
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    def test_identity_rotation(self, n_queries, n_keys, n_heads, head_dim, device):
        """Test that identity rotation preserves the input."""
        embeddings = torch.randn(n_queries, n_keys, n_heads, head_dim, device=device)

        # Create identity rotation - zero angles means no rotation
        identity_rope = torch.zeros(
            n_queries, n_keys, n_heads, head_dim // 2, device=device
        )

        rotated = rotate_embeddings(embeddings, identity_rope)

        assert_close(
            rotated,
            embeddings,
            msg="Identity rotation should preserve the input",
        )

    @given(
        n_queries=st.integers(1, 5),
        n_keys=st.integers(1, 5),
        n_heads=st.integers(1, 4),
        head_dim=st.integers(2, 8).filter(lambda x: x % 2 == 0),
    )
    @settings(
        deadline=None,
        max_examples=10,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_gradient_correctness(self, n_queries, n_keys, n_heads, head_dim, device):
        """Test gradient correctness using PyTorch's gradcheck."""
        # Create small tensors with double precision for better numeric stability
        embeddings = torch.randn(
            n_queries,
            n_keys,
            n_heads,
            head_dim,
            dtype=torch.double,
            requires_grad=True,
            device=device,
        )
        rope = torch.randn(
            n_queries,
            n_keys,
            n_heads,
            head_dim // 2,  # half size
            dtype=torch.double,
            requires_grad=True,
            device=device,
        )

        assert torch.autograd.gradcheck(
            rotate_embeddings,
            (embeddings, rope),
        ), "Gradient check failed"

    @given(
        n_queries=st.integers(1, 5),
        n_keys=st.integers(1, 5),
        n_heads=st.integers(1, 4),
        head_dim=st.integers(2, 8).filter(lambda x: x % 2 == 0),
    )
    @settings(suppress_health_check=[HealthCheck.differing_executors], deadline=None)
    def test_invertibility(self, n_queries, n_keys, n_heads, head_dim, device):
        """Test that applying rotation and then its inverse is identity."""
        embeddings = torch.randn(
            n_queries, n_keys, n_heads, head_dim, dtype=torch.double, device=device
        )

        # Create random rotation angles
        rope = torch.randn(
            n_queries, n_keys, n_heads, head_dim // 2, dtype=torch.double, device=device
        )

        # Apply rotation
        rotated = rotate_embeddings(embeddings, rope)

        # Create inverse rotation angles (negative angles)
        rope_inv = -rope

        # Apply inverse rotation
        rotated_back = rotate_embeddings(rotated, rope_inv)

        # Should get back to original keys (with some numeric tolerance)
        assert_close(
            rotated_back,
            embeddings,
            msg="Applying rotation followed by inverse should recover original",
        )


@pytest.mark.cuda_if_available
class TestRotateEmbeddingsBackward:
    """Tests for the rotate_tensor_backward function."""

    n_heads = 2
    head_dim = 8

    def test_basic_functionality(self, device):
        """Test basic operation with simple inputs."""
        # Setup with simple values for real tensors
        grad_embeddings_rotated = torch.tensor([0.1, 0.2], device=device).view(
            1, 1, 1, 2
        )

        # Create real keys tensor with interleaved real/imaginary components
        embeddings = torch.tensor([1.0, 2.0], device=device).view(1, 1, 1, 2)

        # Use a phase angle for rope encoding (30 degrees)
        angle = torch.tensor([torch.pi / 6], device=device).view(1, 1, 1, 1)

        # For complex multiplication z = x * y, gradients are:
        # dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x)

        # Convert to complex for expected output calculations
        grad_emb_rotated_complex = torch.view_as_complex(
            grad_embeddings_rotated.view(1, 1, 1, 1, 2)
        )
        embddings_complex = torch.view_as_complex(embeddings.view(1, 1, 1, 1, 2))
        rope_encoding_complex = torch.polar(torch.ones_like(angle), angle)

        # Expected gradient for keys
        expected_grad_emb_complex = (
            grad_emb_rotated_complex * rope_encoding_complex.conj()
        )
        expected_grad_emb = torch.view_as_real(expected_grad_emb_complex).reshape_as(
            grad_embeddings_rotated
        )

        # Expected gradient for key_rope_encoding
        grad_rope_encoding_complex = grad_emb_rotated_complex * embddings_complex.conj()
        expected_grad_rope = (
            grad_rope_encoding_complex / rope_encoding_complex
        ).imag.view(1, 1, 1, 1)

        # Call the function with real tensors
        grad_embeddings, grad_rope = rotate_embeddings_backward(
            grad_embeddings_rotated, embeddings, angle, True, True, True
        )

        assert_close(
            grad_embeddings, expected_grad_emb, msg="Gradients for embeddings incorrect"
        )
        assert_close(grad_rope, expected_grad_rope, msg="Gradients for rope incorrect")

    def test_extended_batch_dims(self, device):
        """Test with lots of batch dimensions"""
        batch_dims = [2, 4, 6, 8]

        grad_embeddings_rotated = torch.randn(
            *batch_dims, self.n_heads, self.head_dim, device=device
        )
        embeddings = torch.randn(
            *batch_dims, self.n_heads, self.head_dim, device=device
        )
        rope_encoding = torch.randn(
            *batch_dims, self.n_heads, self.head_dim // 2, device=device
        )

        expected_grad_embeddings_shape = embeddings.shape
        expected_grad_rope_encoding_shape = rope_encoding.shape
        grad_embeddings, grad_rope_encoding = rotate_embeddings_backward(
            grad_embeddings_rotated, embeddings, rope_encoding, True, True
        )

        assert grad_embeddings.shape == expected_grad_embeddings_shape
        assert grad_rope_encoding.shape == expected_grad_rope_encoding_shape

    def test_needs_autograd_optimization(self, device):
        """Test that needs_autograd=False optimizes memory usage."""
        grad_embeddings_rotated = torch.randn(2, 3, 4, 6, device=device)
        embeddings = torch.randn(2, 3, 4, 6, device=device)
        rope_encoding = torch.randn(2, 3, 4, 3, device=device)

        # With autograd tracking
        grad_emb_1, grad_rope_1 = rotate_embeddings_backward(
            grad_embeddings_rotated.clone(), embeddings, rope_encoding, True, True, True
        )

        # Without autograd tracking
        grad_emb_2, grad_rope_2 = rotate_embeddings_backward(
            grad_embeddings_rotated.clone(),
            embeddings,
            rope_encoding,
            True,
            True,
            False,
        )

        # Results should be the same regardless of needs_autograd
        assert_close(
            grad_emb_1,
            grad_emb_2,
            msg="Embeddings gradients differ with needs_autograd=False",
        )
        assert_close(
            grad_rope_1,
            grad_rope_2,
            msg="Rope gradients differ with needs_autograd=False",
        )

    def test_no_grad_rope_encoding(self, device):
        """Test with needs_grad_rope_encoding=False."""
        grad_embeddings_rotated = torch.randn(2, 3, 4, 6, device=device)
        embeddings = torch.randn(2, 3, 4, 6, device=device)
        rope_encoding = torch.randn(2, 3, 4, 3, device=device)

        grad_embeddings, grad_rope = rotate_embeddings_backward(
            grad_embeddings_rotated, embeddings, rope_encoding, True, False
        )

        assert grad_embeddings is not None
        assert grad_rope is None

    def test_no_grad_embeddings(self, device):
        """Test with needs_grad_k=False."""
        grad_embeddings_rotated = torch.randn(2, 3, 4, 6, device=device)
        embeddings = torch.randn(2, 3, 4, 6, device=device)
        rope_encoding = torch.randn(2, 3, 4, 3, device=device)

        grad_embeddings, grad_rope = rotate_embeddings_backward(
            grad_embeddings_rotated, embeddings, rope_encoding, False, True
        )

        assert grad_embeddings is None
        assert grad_rope is not None

    def test_no_grad_both(self, device):
        """Test with both need_grad as False."""
        grad_embeddings_rotated = torch.randn(2, 3, 4, 6, device=device)
        embeddings = torch.randn(2, 3, 4, 6, device=device)
        rope_encoding = torch.randn(2, 3, 4, 3, device=device)

        grad_embeddings, grad_rope = rotate_embeddings_backward(
            grad_embeddings_rotated, embeddings, rope_encoding, False, False
        )

        assert grad_embeddings is None
        assert grad_rope is None

    def test_error_conditions(self, device):
        """Test that appropriate errors are raised for invalid inputs."""
        # Test bad grad_embeddings_rotated - wrong dimensions
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="grad_embeddings_rotated and embeddings to have the same shape",
        ):
            rotate_embeddings_backward(
                torch.randn(2, 4, 6, device=device),  # Not 4D
                torch.randn(2, 4, 6, 8, device=device),
                torch.randn(2, 4, 6, 4, device=device),
            )

        # Test bad embeddings - wrong dimensions
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="grad_embeddings_rotated and embeddings to have the same shape",
        ):
            rotate_embeddings_backward(
                torch.randn(2, 4, 6, 8, device=device),
                torch.randn(2, 4, 6, 6, device=device),  # Wrong head dim
                torch.randn(2, 4, 6, 4, device=device),
            )

        # Test bad grad_embeddings_rotated - wrong dtype (complex)
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="grad_embeddings_rotated to be real",
        ):
            rotate_embeddings_backward(
                torch.randn(
                    2, 4, 6, 8, dtype=torch.complex64, device=device
                ),  # Not real
                torch.randn(2, 4, 6, 8, device=device),
                torch.randn(2, 4, 6, 4, device=device),
            )

        # Test bad embeddings - wrong dtype (complex)
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="embeddings to be real",
        ):
            rotate_embeddings_backward(
                torch.randn(2, 4, 6, 8, device=device),
                torch.randn(
                    2, 4, 6, 8, dtype=torch.complex64, device=device
                ),  # Not real
                torch.randn(2, 4, 6, 4, device=device),
            )

        # Test bad rope_encoding - wrong dimensions
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected embeddings and rope_encoding to have the same number",
        ):
            rotate_embeddings_backward(
                torch.randn(2, 4, 6, 8, device=device),
                torch.randn(2, 4, 6, 8, device=device),
                torch.randn(2, 4, 6, device=device),  # Not 4D
            )

        # Test bad rope_encoding - wrong dtype (complex)
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected rope_encoding to be real",
        ):
            rotate_embeddings_backward(
                torch.randn(2, 4, 6, 8, device=device),
                torch.randn(2, 4, 6, 8, device=device),
                torch.randn(
                    2, 4, 6, 4, dtype=torch.complex64, device=device
                ),  # Not real
            )

        # Test bad shapes - wrong trailing dims for rope
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected rope_encoding to have last dimension",
        ):
            rotate_embeddings_backward(
                torch.randn(2, 4, 6, 8, device=device),
                torch.randn(2, 4, 6, 8, device=device),
                torch.randn(2, 4, 6, 3, device=device),
            )

    def test_half_precision(self, device):
        """Tests if the handling of half-precision inputs works correctly"""
        batch_dims = [2]

        grad_emb_rot_full = torch.randn(
            *batch_dims, self.n_heads, self.head_dim, device=device, dtype=torch.float32
        )
        embeddings = torch.randn(
            *batch_dims, self.n_heads, self.head_dim, device=device, dtype=torch.float32
        )
        rope_encoding = torch.randn(
            *batch_dims,
            self.n_heads,
            self.head_dim // 2,
            device=device,
            dtype=torch.float32
        )

        # full precision calculation
        grad_emb_full, grad_rope_full = rotate_embeddings_backward(
            grad_emb_rot_full, embeddings, rope_encoding
        )
        assert grad_emb_full.dtype == torch.float32
        assert grad_rope_full.dtype == torch.float32

        # half precision calculation
        grad_emb_rot_half = grad_emb_rot_full.half()
        embeddings_half = embeddings.half()
        rope_encoding_half = rope_encoding.half()
        grad_emb_half, grad_rope_half = rotate_embeddings_backward(
            grad_emb_rot_half, embeddings_half, rope_encoding_half
        )
        assert grad_emb_half.dtype == torch.float16
        assert grad_rope_half.dtype == torch.float16

        # bfloat16 calculation
        grad_emb_rot_bf16 = grad_emb_rot_full.bfloat16()
        embeddings_bf16 = embeddings.bfloat16()
        rope_encoding_bf16 = rope_encoding.bfloat16()
        grad_emb_bf16, grad_rope_bf16 = rotate_embeddings_backward(
            grad_emb_rot_bf16, embeddings_bf16, rope_encoding_bf16
        )
        assert grad_emb_bf16.dtype == torch.bfloat16
        assert grad_rope_bf16.dtype == torch.bfloat16

        # Determine input precision error
        grad_emb_rot_half_error = torch.abs(
            grad_emb_rot_full - grad_emb_rot_half.float()
        ).max()
        embeddings_half_error = torch.abs(embeddings - embeddings_half.float()).max()
        rope_half_error = torch.abs(rope_encoding - rope_encoding_half.float()).max()

        grad_emb_rot_bf16_error = torch.abs(
            grad_emb_rot_full - grad_emb_rot_bf16.float()
        ).max()
        embeddings_bf16_error = torch.abs(embeddings - embeddings_bf16.float()).max()
        rope_bf16_error = torch.abs(rope_encoding - rope_encoding_bf16.float()).max()

        # Determine suitable tolerances
        atol_half = (
            max(grad_emb_rot_half_error, embeddings_half_error, rope_half_error) * 10
        )
        atol_bf16 = (
            max(grad_emb_rot_bf16_error, embeddings_bf16_error, rope_bf16_error) * 10
        )

        # Verify errors didn't explode during the operation
        assert torch.allclose(
            grad_emb_full, grad_emb_half.float(), atol=atol_half, rtol=1e-2
        )
        assert torch.allclose(
            grad_rope_full, grad_rope_half.float(), atol=atol_half, rtol=1e-2
        )

        assert torch.allclose(
            grad_emb_full, grad_emb_bf16.float(), atol=atol_bf16, rtol=5e-2
        )
        assert torch.allclose(
            grad_rope_full, grad_rope_bf16.float(), atol=atol_bf16, rtol=5e-2
        )
