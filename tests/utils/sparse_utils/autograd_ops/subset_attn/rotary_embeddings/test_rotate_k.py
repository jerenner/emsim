import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from emsim.utils.sparse_utils.ops.subset_attn.rotary_embedding import (
    rotate_k,
    rotate_k_backward,
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

        k_rotated, keys_complex, rope_encoding_complex = rotate_k(keys, angles)

        assert_close(k_rotated, expected, rtol=1e-4, msg="Basic rotate_k failed")

        # Check complex representations are correct
        expected_keys_complex = torch.complex(keys[..., 0::2], keys[..., 1::2])

        # Now rope_encoding_complex should be exp(i*angles) = cos(angles) + i*sin(angles)
        # For the angles [π/3, π/3], we expect complex numbers [1+0j, cos(π/3)+sin(π/3)j]
        expected_rope_complex = torch.polar(torch.ones_like(angles), angles)

        assert_close(keys_complex.real, expected_keys_complex.real)
        assert_close(keys_complex.imag, expected_keys_complex.imag)
        assert_close(rope_encoding_complex.real, expected_rope_complex.real)
        assert_close(rope_encoding_complex.imag, expected_rope_complex.imag)

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

        k_rotated, keys_complex, rope_encoding_complex = rotate_k(keys, angles)

        # Verify the rotation results
        assert_close(
            k_rotated,
            expected,
            rtol=1e-3,
            atol=1e-3,
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
        expected_rope_complex = torch.polar(torch.ones_like(angles), angles)

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

    def test_error_conditions(self, device):
        """Test that appropriate errors are raised for invalid inputs."""
        # Test keys of invalid shape
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected k and rope_encoding to be 4D"
        ):
            rotate_k(
                torch.randn(2, 3, 4, device=device),
                torch.randn(2, 3, 4, 5, device=device),
            )
        # Test rope_encoding of invalid shape
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected k and rope_encoding to be 4D"
        ):
            rotate_k(
                torch.randn(2, 3, 4, 8, device=device),
                torch.randn(2, 3, 4, device=device),
            )
        # Test odd head_dim
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected rope_encoding to have last dimension",
        ):
            rotate_k(
                torch.randn(2, 3, 4, 5, device=device),  # odd head_dim
                torch.randn(2, 3, 4, 2, device=device),  # not head_dim/2
            )
        # Test mismatch between head_dim and rope_encoding dimension
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected rope_encoding to have last dimension",
        ):
            rotate_k(
                torch.randn(2, 3, 4, 8, device=device),  # head_dim = 8
                torch.randn(2, 3, 4, 3, device=device),  # not head_dim/2 = 4
            )
        # Test complex inputs
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected keys and rope_encoding to be real",
        ):
            rotate_k(
                torch.randn(2, 3, 4, 8, dtype=torch.complex64, device=device),
                torch.randn(2, 3, 4, 4, dtype=torch.float32, device=device),
            )
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected keys and rope_encoding to be real",
        ):
            rotate_k(
                torch.randn(2, 3, 4, 8, dtype=torch.float32, device=device),
                torch.randn(2, 3, 4, 4, dtype=torch.complex64, device=device),
            )

        # Test incompatible shapes
        with pytest.raises(
            RuntimeError,
            match="The size of tensor a",
        ):
            # n_queries nonmatching
            rotate_k(
                torch.randn(2, 16, 8, 32, device=device),
                torch.randn(3, 16, 8, 16, device=device),  # 16 = 32/2
            )
        with pytest.raises(
            RuntimeError,
            match="The size of tensor a",
        ):
            # n_keys_per_query nonmatching
            rotate_k(
                torch.randn(2, 16, 8, 32, device=device),
                torch.randn(2, 132, 8, 16, device=device),  # 16 = 32/2
            )
        with pytest.raises(
            RuntimeError,
            match="The size of tensor a",
        ):
            # n_heads for rope_encoding nonmatching and not 1
            rotate_k(
                torch.randn(2, 16, 8, 32, device=device),
                torch.randn(2, 16, 4, 16, device=device),  # 16 = 32/2
            )


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

        rotated, _, _ = rotate_k(keys, rope)

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

        rotated_broadcast, _, _ = rotate_k(keys, rope_broadcast)
        rotated_expanded, _, _ = rotate_k(keys, rope_expanded)

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
        rotated1, _, _ = rotate_k(keys, rope)

        # Get output for scaled input
        rotated2, _, _ = rotate_k(keys * scale, rope)

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
        rotated_sum, _, _ = rotate_k(keys1 + keys2, rope)
        rotated1, _, _ = rotate_k(keys1, rope)
        rotated2, _, _ = rotate_k(keys2, rope)

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

        rotated, _, _ = rotate_k(keys, identity_rope)

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

        def rotate_wrapper(k, r):
            """Wrapper that returns only the rotated keys to check gradients."""
            rotated, _, _ = rotate_k(k, r)
            return rotated

        assert torch.autograd.gradcheck(
            rotate_wrapper,
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
        rotated, _, _ = rotate_k(keys, rope)

        # Create inverse rotation angles (negative angles)
        rope_inv = -rope

        # Apply inverse rotation
        rotated_back, _, _ = rotate_k(rotated, rope_inv)

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
        # Setup with simple values
        grad_k_rotated = torch.tensor([0.1, 0.2], device=device).view(1, 1, 1, 2)
        k_complex = (
            torch.complex(torch.tensor(1.0), torch.tensor(2.0))
            .view(1, 1, 1, 1)
            .to(device)
        )

        # Use a unit complex number (magnitude 1)
        angle = torch.tensor(torch.pi / 6)  # 30 degrees
        rope_encoding_complex = torch.polar(
            torch.ones(1, device=device), torch.tensor([angle], device=device)
        ).view(1, 1, 1, 1)

        # For complex multiplication z = x * y, gradients are:
        # dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x)
        grad_k_rotated_complex = torch.complex(
            grad_k_rotated[..., 0], grad_k_rotated[..., 1]
        )

        # Gradient for k is straightforward: grad_z * conj(rope_encoding_complex)
        expected_grad_k_complex = grad_k_rotated_complex * rope_encoding_complex.conj()
        expected_grad_k = torch.cat(
            [expected_grad_k_complex.real, expected_grad_k_complex.imag], dim=-1
        ).reshape_as(grad_k_rotated)

        # Gradient for rope_encoding (the angle) is now different:
        # For z = x * e^(iθ), dz/dθ = ix * e^(iθ) = iz
        # So dL/dθ = Im(dL/dz * z* / |z|^2) = Im(dL/dz / e^(iθ))
        grad_rope_encoding_complex = grad_k_rotated_complex * k_complex.conj()
        expected_grad_rope = (
            grad_rope_encoding_complex / rope_encoding_complex
        ).imag.view_as(grad_k_rotated[..., :1])

        grad_k, grad_rope = rotate_k_backward(
            grad_k_rotated, k_complex, rope_encoding_complex, True, True
        )

        assert_close(grad_k, expected_grad_k, msg="Gradients for keys incorrect")
        assert_close(grad_rope, expected_grad_rope, msg="Gradients for rope incorrect")

    def test_no_grad_rope_encoding(self, device):
        """Test with needs_grad_rope_encoding=False."""
        grad_k_rotated = torch.randn(2, 3, 4, 6, device=device)
        k_complex = torch.randn(2, 3, 4, 3, dtype=torch.complex64, device=device)
        key_pos_complex = torch.randn(2, 3, 4, 3, dtype=torch.complex64, device=device)

        grad_k, grad_rope = rotate_k_backward(
            grad_k_rotated, k_complex, key_pos_complex, True, False
        )

        assert grad_k is not None
        assert grad_rope is None

    def test_no_grad_k(self, device):
        """Test with needs_grad_k=False."""
        grad_k_rotated = torch.randn(2, 3, 4, 6, device=device)
        k_complex = torch.randn(2, 3, 4, 3, dtype=torch.complex64, device=device)
        key_pos_complex = torch.randn(2, 3, 4, 3, dtype=torch.complex64, device=device)

        grad_k, grad_rope = rotate_k_backward(
            grad_k_rotated, k_complex, key_pos_complex, False, True
        )

        assert grad_k is None
        assert grad_rope is not None

    def test_no_grad_both(self, device):
        """Test with both need_grad as False."""
        grad_k_rotated = torch.randn(2, 3, 4, 6, device=device)
        k_complex = torch.randn(2, 3, 4, 3, dtype=torch.complex64, device=device)
        key_pos_complex = torch.randn(2, 3, 4, 3, dtype=torch.complex64, device=device)

        grad_k, grad_rope = rotate_k_backward(
            grad_k_rotated, k_complex, key_pos_complex, False, False
        )

        assert grad_k is None
        assert grad_rope is None

    def test_error_conditions(self, device):
        """Test that appropriate errors are raised for invalid inputs."""
        # Test bad grad_k_rotated - wrong dimensions
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected grad_k_rotated to be a 4D real tensor",
        ):
            rotate_k_backward(
                torch.randn(2, 4, 6, device=device),  # Not 4D
                torch.randn(
                    2, 4, 6, 4, dtype=torch.complex64, device=device
                ),  # head_dim/2 = 4
                torch.randn(
                    2, 4, 6, 4, dtype=torch.complex64, device=device
                ),  # head_dim/2 = 4
            )

        # Test bad grad_k_rotated - wrong dtype (complex)
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected grad_k_rotated to be a 4D real tensor",
        ):
            rotate_k_backward(
                torch.randn(
                    2, 4, 6, 8, dtype=torch.complex64, device=device
                ),  # Not real
                torch.randn(
                    2, 4, 6, 4, dtype=torch.complex64, device=device
                ),  # head_dim/2 = 4
                torch.randn(
                    2, 4, 6, 4, dtype=torch.complex64, device=device
                ),  # head_dim/2 = 4
            )

        # Test bad k_complex - wrong dimensions
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected k_complex to be a 4D complex tensor",
        ):
            rotate_k_backward(
                torch.randn(2, 4, 6, 8, device=device),  # head_dim = 8
                torch.randn(2, 4, 6, dtype=torch.complex64, device=device),  # Not 4D
                torch.randn(
                    2, 4, 6, 4, dtype=torch.complex64, device=device
                ),  # head_dim/2 = 4
            )

        # Test bad k_complex - wrong dtype (not complex)
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected k_complex to be a 4D complex tensor",
        ):
            rotate_k_backward(
                torch.randn(2, 4, 6, 8, device=device),  # head_dim = 8
                torch.randn(2, 4, 6, 4, device=device),  # Not complex
                torch.randn(
                    2, 4, 6, 4, dtype=torch.complex64, device=device
                ),  # head_dim/2 = 4
            )

        # Test bad rope_encoding_complex - wrong dimensions
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected rope_encoding_complex to be a 4D complex tensor",
        ):
            rotate_k_backward(
                torch.randn(2, 4, 6, 8, device=device),  # head_dim = 8
                torch.randn(
                    2, 4, 6, 4, dtype=torch.complex64, device=device
                ),  # head_dim/2 = 4
                torch.randn(2, 4, 6, dtype=torch.complex64, device=device),  # Not 4D
            )

        # Test bad rope_encoding_complex - wrong dtype (not complex)
        with pytest.raises(
            (ValueError, torch.jit.Error),
            match="Expected rope_encoding_complex to be a 4D complex tensor",
        ):
            rotate_k_backward(
                torch.randn(2, 4, 6, 8, device=device),  # head_dim = 8
                torch.randn(
                    2, 4, 6, 4, dtype=torch.complex64, device=device
                ),  # head_dim/2 = 4
                torch.randn(2, 4, 6, 4, device=device),  # Not complex
            )

        # Test odd head_dim
        with pytest.raises(
            (ValueError, torch.jit.Error), match="k_complex's last dimension"
        ):
            rotate_k_backward(
                torch.randn(2, 3, 4, 5, device=device),  # Odd head_dim
                torch.randn(
                    2, 3, 4, 2, dtype=torch.complex64, device=device
                ),  # head_dim/2
                torch.randn(
                    2, 3, 4, 2, dtype=torch.complex64, device=device
                ),  # head_dim/2
            )
