import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from emsim.utils.sparse_utils.ops.subset_attn.rotary_embedding import (
    rotate_keys,
    rotate_keys_backward,
)

from .conftest import assert_close


@pytest.mark.cuda
class TestRotateK:
    """Tests for the rotate_k function."""

    n_queries = 4
    n_keys_per_query = 6
    n_heads = 2
    head_dim = 8

    def test_basic_functionality(self, device):
        """Test basic operation with simple inputs."""
        # Simple case with known values
        keys = torch.tensor(
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

        k_rotated = rotate_keys(keys, angles)

        assert_close(k_rotated, expected, rtol=1e-4, msg="Basic rotate_k failed")

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

        k_rotated = rotate_keys(keys, angles)

        # Verify the rotation results
        assert_close(
            k_rotated,
            expected,
            rtol=1e-3,
            atol=1e-3,
            msg="Broadcasting in rotate_k failed",
        )

    def test_error_conditions(self, device):
        """Test that appropriate errors are raised for invalid inputs."""
        # Test keys of invalid shape
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected keys and key_rope_encoding to be 4D",
        ):
            rotate_keys(
                torch.randn(2, 3, 4, device=device),
                torch.randn(2, 3, 4, 5, device=device),
            )
        # Test key_rope_encoding of invalid shape
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected keys and key_rope_encoding to be 4D",
        ):
            rotate_keys(
                torch.randn(2, 3, 4, 8, device=device),
                torch.randn(2, 3, 4, device=device),
            )
        # Test odd head_dim
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected key_rope_encoding to have last dimension",
        ):
            rotate_keys(
                torch.randn(2, 3, 4, 5, device=device),  # odd head_dim
                torch.randn(2, 3, 4, 2, device=device),  # not head_dim/2
            )
        # Test mismatch between head_dim and key_rope_encoding dimension
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected key_rope_encoding to have last dimension",
        ):
            rotate_keys(
                torch.randn(2, 3, 4, 8, device=device),  # head_dim = 8
                torch.randn(2, 3, 4, 3, device=device),  # not head_dim/2 = 4
            )
        # Test complex inputs
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected keys and key_rope_encoding to be real",
        ):
            rotate_keys(
                torch.randn(2, 3, 4, 8, dtype=torch.complex64, device=device),
                torch.randn(2, 3, 4, 4, dtype=torch.float32, device=device),
            )
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected keys and key_rope_encoding to be real",
        ):
            rotate_keys(
                torch.randn(2, 3, 4, 8, dtype=torch.float32, device=device),
                torch.randn(2, 3, 4, 4, dtype=torch.complex64, device=device),
            )

        # Test incompatible shapes
        with pytest.raises(
            RuntimeError,
            match="The size of tensor a",
        ):
            # n_queries nonmatching
            rotate_keys(
                torch.randn(2, 16, 8, 32, device=device),
                torch.randn(3, 16, 8, 16, device=device),  # 16 = 32/2
            )
        with pytest.raises(
            RuntimeError,
            match="The size of tensor a",
        ):
            # n_keys_per_query nonmatching
            rotate_keys(
                torch.randn(2, 16, 8, 32, device=device),
                torch.randn(2, 132, 8, 16, device=device),  # 16 = 32/2
            )
        with pytest.raises(
            RuntimeError,
            match="The size of tensor a",
        ):
            # n_heads for rope_encoding nonmatching and not 1
            rotate_keys(
                torch.randn(2, 16, 8, 32, device=device),
                torch.randn(2, 16, 4, 16, device=device),  # 16 = 32/2
            )

    def test_needs_autograd(self, device):
        """Test that needs_autograd=False optimizes memory usage while preserving correctness."""
        # Create input tensors
        keys = torch.randn(2, 3, 4, 8, device=device)
        key_rope_encoding = torch.randn(2, 3, 4, 4, device=device)

        # Make copies to ensure we don't accidentally modify the originals
        k1 = keys.clone()
        k2 = keys.clone()
        encoding1 = key_rope_encoding.clone()
        encoding2 = key_rope_encoding.clone()

        # First call with needs_autograd=True (default behavior)
        k1_rotated = rotate_keys(k1, encoding1, needs_autograd=True)

        # Call with needs_autograd=False (optimization)
        k2_rotated = rotate_keys(k2, encoding2, needs_autograd=False)

        # Both calls should produce identical results
        assert_close(
            k1_rotated,
            k2_rotated,
            msg="needs_autograd=False produces different results",
        )

        # Memory optimization test: When needs_autograd=False, we expect the operation
        # to modify the tensor in-place, so the complex view of the input should be modified

        # First, verify we can observe in-place modifications to a complex view:
        k_test = keys.clone().view(*keys.shape[:-1], keys.size(-1) // 2, 2)
        k_complex_test = torch.view_as_complex(k_test)
        original_value = k_complex_test[0, 0, 0, 0].clone()
        k_complex_test[0, 0, 0, 0] += 1.0  # In-place modification

        # This should affect the original tensor
        modified_k_test = torch.view_as_real(k_complex_test).reshape_as(keys)
        assert not torch.allclose(keys, modified_k_test), "In-place test setup failed"

        # Now for the actual test: verify needs_autograd=False does in-place ops
        # We'll manually do the operations to check if the tensor was modified

        # Create fresh copies for a comparative test
        k3 = keys.clone()
        encoding3 = key_rope_encoding.clone()

        # Create a complex view of k3
        k3_complex_shape = k3.shape[:-1] + (k3.size(-1) // 2, 2)
        k3_complex_view = k3.view(k3_complex_shape)
        k3_complex = torch.view_as_complex(k3_complex_view)

        # Save the original first value for comparison
        original_value = k3_complex[0, 0, 0, 0].clone()

        # Apply rotate_k with needs_autograd=False
        _ = rotate_keys(k3, encoding3, needs_autograd=False)

        # Check if k3_complex was modified in-place
        # If it was, the original_value should no longer match
        assert (
            original_value != k3_complex[0, 0, 0, 0]
        ), "needs_autograd=False did not perform in-place operations"

        # Additional test with autograd enabled
        k4 = keys.clone().requires_grad_(True)
        encoding4 = key_rope_encoding.clone()

        # Should work fine with needs_autograd=True and requires_grad=True
        k4_rotated = rotate_keys(k4, encoding4, needs_autograd=True)

        # This should be able to compute gradients
        loss = k4_rotated.sum()
        loss.backward()

        assert k4.grad is not None, "needs_autograd=True should support autograd"

        # But with needs_autograd=False, we'll get an error with requires_grad=True
        k5 = keys.clone().requires_grad_(True)
        encoding5 = key_rope_encoding.clone()

        # This should raise an error because we can't do in-place ops on tensors that require grad
        with pytest.raises(RuntimeError, match="a leaf Variable that requires grad"):
            _ = rotate_keys(k5, encoding5, needs_autograd=False)


@pytest.mark.cuda
class TestRotateKProperties:

    @given(
        n_queries=st.integers(1, 10),
        n_keys=st.integers(1, 10),
        n_heads=st.integers(1, 8),
        head_dim=st.integers(2, 16).filter(lambda x: x % 2 == 0),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_shape_preservation(self, n_queries, n_keys, n_heads, head_dim, device):
        """Test that output shape matches input shape."""
        keys = torch.randn(n_queries, n_keys, n_heads, head_dim, device=device)
        rope = torch.randn(
            n_queries, n_keys, n_heads, head_dim // 2, device=device
        )  # half size

        rotated = rotate_keys(keys, rope)

        assert rotated.shape == keys.shape, "Output shape should match input shape"

    @given(
        n_queries=st.integers(1, 10),
        n_keys=st.integers(1, 10),
        n_heads=st.integers(1, 8),
        head_dim=st.integers(2, 16).filter(lambda x: x % 2 == 0),
    )
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    def test_broadcasting(self, n_queries, n_keys, n_heads, head_dim, device):
        """Test that broadcasting works the same as explicit expansion."""
        keys = torch.randn(n_queries, n_keys, n_heads, head_dim, device=device)

        # Broadcasted version (1 head)
        rope_broadcast = torch.randn(
            n_queries, n_keys, 1, head_dim // 2, device=device
        )  # half size
        # Expanded version (explicit repeated across heads)
        rope_expanded = rope_broadcast.expand(n_queries, n_keys, n_heads, head_dim // 2)

        rotated_broadcast = rotate_keys(keys, rope_broadcast)
        rotated_expanded = rotate_keys(keys, rope_expanded)

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
        keys = torch.randn(
            n_queries, n_keys, n_heads, head_dim, dtype=torch.double, device=device
        )
        # Use angles for rope
        rope = torch.randn(
            n_queries, n_keys, n_heads, head_dim // 2, dtype=torch.double, device=device
        )

        # Get output for original input
        rotated1 = rotate_keys(keys, rope)

        # Get output for scaled input
        rotated2 = rotate_keys(keys * scale, rope)

        assert_close(
            rotated2,
            rotated1 * scale,
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
        keys1 = torch.randn(
            n_queries, n_keys, n_heads, head_dim, dtype=torch.double, device=device
        )
        keys2 = torch.randn(
            n_queries, n_keys, n_heads, head_dim, dtype=torch.double, device=device
        )
        # Use angles for rope
        rope = torch.randn(
            n_queries, n_keys, n_heads, head_dim // 2, dtype=torch.double, device=device
        )

        # f(a + b) should equal f(a) + f(b)
        rotated_sum = rotate_keys(keys1 + keys2, rope)
        rotated1 = rotate_keys(keys1, rope)
        rotated2 = rotate_keys(keys2, rope)

        assert_close(
            rotated_sum,
            rotated1 + rotated2,
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
        keys = torch.randn(n_queries, n_keys, n_heads, head_dim, device=device)

        # Create identity rotation - zero angles means no rotation
        identity_rope = torch.zeros(
            n_queries, n_keys, n_heads, head_dim // 2, device=device
        )

        rotated = rotate_keys(keys, identity_rope)

        assert_close(
            rotated,
            keys,
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
        keys = torch.randn(
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
            rotate_keys,
            (keys, rope),
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
        keys = torch.randn(
            n_queries, n_keys, n_heads, head_dim, dtype=torch.double, device=device
        )

        # Create random rotation angles
        rope = torch.randn(
            n_queries, n_keys, n_heads, head_dim // 2, dtype=torch.double, device=device
        )

        # Apply rotation
        rotated = rotate_keys(keys, rope)

        # Create inverse rotation angles (negative angles)
        rope_inv = -rope

        # Apply inverse rotation
        rotated_back = rotate_keys(rotated, rope_inv)

        # Should get back to original keys (with some numeric tolerance)
        assert_close(
            rotated_back,
            keys,
            msg="Applying rotation followed by inverse should recover original",
        )


@pytest.mark.cuda
class TestRotateKBackward:
    """Tests for the rotate_k_backward function."""

    def test_basic_functionality(self, device):
        """Test basic operation with simple inputs."""
        # Setup with simple values for real tensors
        grad_k_rotated = torch.tensor([0.1, 0.2], device=device).view(1, 1, 1, 2)

        # Create real keys tensor with interleaved real/imaginary components
        k_real = torch.tensor([1.0, 2.0], device=device).view(1, 1, 1, 2)

        # Use a phase angle for rope encoding (30 degrees)
        angle = torch.tensor([torch.pi / 6], device=device).view(1, 1, 1, 1)

        # For complex multiplication z = x * y, gradients are:
        # dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x)

        # Convert to complex for expected output calculations
        grad_k_rotated_complex = torch.view_as_complex(
            grad_k_rotated.view(1, 1, 1, 1, 2)
        )
        k_complex = torch.view_as_complex(k_real.view(1, 1, 1, 1, 2))
        rope_encoding_complex = torch.polar(torch.ones_like(angle), angle)

        # Expected gradient for keys
        expected_grad_k_complex = grad_k_rotated_complex * rope_encoding_complex.conj()
        expected_grad_k = torch.view_as_real(expected_grad_k_complex).reshape_as(
            grad_k_rotated
        )

        # Expected gradient for rope_encoding
        grad_rope_encoding_complex = grad_k_rotated_complex * k_complex.conj()
        expected_grad_rope = (
            grad_rope_encoding_complex / rope_encoding_complex
        ).imag.view(1, 1, 1, 1)

        # Call the function with real tensors
        grad_keys, grad_rope = rotate_keys_backward(
            grad_k_rotated, k_real, angle, True, True, True
        )

        assert_close(grad_keys, expected_grad_k, msg="Gradients for keys incorrect")
        assert_close(grad_rope, expected_grad_rope, msg="Gradients for rope incorrect")

    def test_needs_autograd_optimization(self, device):
        """Test that needs_autograd=False optimizes memory usage."""
        grad_k_rotated = torch.randn(2, 3, 4, 6, device=device)
        keys = torch.randn(2, 3, 4, 6, device=device)
        key_rope_encoding = torch.randn(2, 3, 4, 3, device=device)

        # With autograd tracking
        grad_k1, grad_rope1 = rotate_keys_backward(
            grad_k_rotated.clone(), keys, key_rope_encoding, True, True, True
        )

        # Without autograd tracking
        grad_k2, grad_rope2 = rotate_keys_backward(
            grad_k_rotated.clone(), keys, key_rope_encoding, True, True, False
        )

        # Results should be the same regardless of needs_autograd
        assert_close(
            grad_k1, grad_k2, msg="Key gradients differ with needs_autograd=False"
        )
        assert_close(
            grad_rope1,
            grad_rope2,
            msg="Rope gradients differ with needs_autograd=False",
        )

    def test_no_grad_rope_encoding(self, device):
        """Test with needs_grad_rope_encoding=False."""
        grad_k_rotated = torch.randn(2, 3, 4, 6, device=device)
        keys = torch.randn(2, 3, 4, 6, device=device)
        key_rope_encoding = torch.randn(2, 3, 4, 3, device=device)

        grad_keys, grad_rope = rotate_keys_backward(
            grad_k_rotated, keys, key_rope_encoding, True, False
        )

        assert grad_keys is not None
        assert grad_rope is None

    def test_no_grad_k(self, device):
        """Test with needs_grad_k=False."""
        grad_k_rotated = torch.randn(2, 3, 4, 6, device=device)
        keys = torch.randn(2, 3, 4, 6, device=device)
        key_rope_encoding = torch.randn(2, 3, 4, 3, device=device)

        grad_keys, grad_rope = rotate_keys_backward(
            grad_k_rotated, keys, key_rope_encoding, False, True
        )

        assert grad_keys is None
        assert grad_rope is not None

    def test_no_grad_both(self, device):
        """Test with both need_grad as False."""
        grad_k_rotated = torch.randn(2, 3, 4, 6, device=device)
        keys = torch.randn(2, 3, 4, 6, device=device)
        key_rope_encoding = torch.randn(2, 3, 4, 3, device=device)

        grad_keys, grad_rope = rotate_keys_backward(
            grad_k_rotated, keys, key_rope_encoding, False, False
        )

        assert grad_keys is None
        assert grad_rope is None

    def test_error_conditions(self, device):
        """Test that appropriate errors are raised for invalid inputs."""
        # Test bad grad_k_rotated - wrong dimensions
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected grad_k_rotated to be a 4D real tensor",
        ):
            rotate_keys_backward(
                torch.randn(2, 4, 6, device=device),  # Not 4D
                torch.randn(2, 4, 6, 8, device=device),  # 4D real tensor
                torch.randn(2, 4, 6, 4, device=device),  # 4D real tensor
            )

        # Test bad grad_k_rotated - wrong dtype (complex)
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected grad_k_rotated to be a 4D real tensor",
        ):
            rotate_keys_backward(
                torch.randn(
                    2, 4, 6, 8, dtype=torch.complex64, device=device
                ),  # Not real
                torch.randn(2, 4, 6, 8, device=device),  # 4D real tensor
                torch.randn(2, 4, 6, 4, device=device),  # 4D real tensor
            )

        # Test bad keys - wrong dimensions
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected keys to be a 4D real tensor",
        ):
            rotate_keys_backward(
                torch.randn(2, 4, 6, 8, device=device),  # 4D real tensor
                torch.randn(2, 4, 6, device=device),  # Not 4D
                torch.randn(2, 4, 6, 4, device=device),  # 4D real tensor
            )

        # Test bad keys - wrong dtype (complex)
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected keys to be a 4D real tensor",
        ):
            rotate_keys_backward(
                torch.randn(2, 4, 6, 8, device=device),  # 4D real tensor
                torch.randn(
                    2, 4, 6, 4, dtype=torch.complex64, device=device
                ),  # Not real
                torch.randn(2, 4, 6, 4, device=device),  # 4D real tensor
            )

        # Test bad key_rope_encoding - wrong dimensions
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected rope_encoding to be a 4D real tensor",
        ):
            rotate_keys_backward(
                torch.randn(2, 4, 6, 8, device=device),  # 4D real tensor
                torch.randn(2, 4, 6, 8, device=device),  # 4D real tensor
                torch.randn(2, 4, 6, device=device),  # Not 4D
            )

        # Test bad key_rope_encoding - wrong dtype (complex)
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected rope_encoding to be a 4D real tensor",
        ):
            rotate_keys_backward(
                torch.randn(2, 4, 6, 8, device=device),  # 4D real tensor
                torch.randn(2, 4, 6, 8, device=device),  # 4D real tensor
                torch.randn(
                    2, 4, 6, 4, dtype=torch.complex64, device=device
                ),  # Not real
            )
