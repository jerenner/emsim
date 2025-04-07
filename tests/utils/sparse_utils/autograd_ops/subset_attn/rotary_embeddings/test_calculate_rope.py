import pytest
import torch
from hypothesis import HealthCheck, given, settings

from emsim.utils.sparse_utils.ops.subset_attn.rotary_embedding import (
    calculate_rope,
    calculate_rope_backward,
)

from .conftest import assert_close, even_dims, valid_dims


@pytest.mark.cuda
class TestCalculateRope:
    """Tests for the calculate_rope function."""

    def test_basic_functionality(self, device):
        """Test basic operation with simple inputs."""
        key_positions = torch.tensor(
            [[[1.0, 2.0]]], dtype=torch.float32, device=device
        )  # [1, 1, 2]

        # Now with position_dim=2 to match key_positions
        rope_freqs = torch.tensor(
            [[[[1.0, 2.0]]], [[[5.0, 6.0]]]],  # position_dim=0  # position_dim=1
            dtype=torch.float32,
            device=device,
        )  # [2, 1, 1, 2] -> [position_dim=2, n_freq_groups=1, n_heads=1, head_dim=2]

        # Expected: matrix multiplication of key_positions and rope_freqs
        # 1.0 * [1.0, 2.0] + 2.0 * [5.0, 6.0] = [11.0, 14.0]
        expected = torch.tensor(
            [[[[11.0, 14.0]]]], dtype=torch.float32, device=device
        )  # [1, 1, 1, 2]
        result = calculate_rope(key_positions, rope_freqs)

        assert_close(result, expected, msg="Basic calculate_rope failed")

    def test_multi_freq_groups(self, device):
        """Test with multiple frequency groups."""
        key_positions = torch.tensor(
            [[[1.0, 2.0]]], dtype=torch.float32, device=device
        )  # [1, 1, 2]
        rope_freqs = torch.tensor(
            [
                [  # position_dim=2
                    [  # n_freq_groups=2
                        [[1.0, 2.0]],  # n_heads=1, head_dim=2
                        [[3.0, 4.0]],
                    ],
                    [
                        [[5.0, 6.0]],
                        [[7.0, 8.0]],
                    ],
                ]
            ],
            dtype=torch.float32,
            device=device,
        ).squeeze(
            0
        )  # [2, 2, 1, 2]

        # Expected: sum over frequency groups after matrix multiplication
        expected = torch.tensor(
            [[[[11.0 + 17.0, 14.0 + 20.0]]]], dtype=torch.float32, device=device
        ).squeeze(
            0
        )  # [1, 1, 1, 2]
        result = calculate_rope(key_positions, rope_freqs)

        assert_close(result, expected, msg="Multi-group calculate_rope failed")

    def test_multi_heads(self, device):
        """Test with multiple heads."""
        key_positions = torch.tensor(
            [[[1.0, 2.0]]], dtype=torch.float32, device=device
        )  # [1, 1, 2]

        rope_freqs = torch.tensor(
            [
                # position_dim=0
                [
                    # n_freq_groups=1 (explicit dimension)
                    [
                        [1.0, 2.0],  # head 0, head_dim=2
                        [3.0, 4.0],  # head 1, head_dim=2
                    ]
                ],
                # position_dim=1
                [
                    # n_freq_groups=1 (explicit dimension)
                    [
                        [5.0, 6.0],  # head 0, head_dim=2
                        [7.0, 8.0],  # head 1, head_dim=2
                    ]
                ],
            ],
            dtype=torch.float32,
            device=device,
        )  # [2, 1, 2, 2]

        # Expected calculation for each head:
        # Head 0: 1.0 * [1.0, 2.0] + 2.0 * [5.0, 6.0] = [11.0, 14.0]
        # Head 1: 1.0 * [3.0, 4.0] + 2.0 * [7.0, 8.0] = [17.0, 20.0]
        expected = torch.tensor(
            [
                [
                    [
                        [11.0, 14.0],  # head 0 result
                        [17.0, 20.0],  # head 1 result
                    ]
                ]
            ],
            dtype=torch.float32,
            device=device,
        )  # [1, 1, 2, 2]

        result = calculate_rope(key_positions, rope_freqs)
        assert_close(result, expected, msg="Multi-head calculate_rope failed")

    def test_error_conditions(self, device):
        """Test that appropriate errors are raised for invalid inputs."""
        # Test 2D key_positions (should be 3D)
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected 3 dimensions"
        ):
            calculate_rope(
                torch.randn(2, 3, device=device), torch.randn(3, 1, 1, 4, device=device)
            )

        # Test 3D rope_freqs (should be 4D)
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected 4 dimnensions"
        ):
            calculate_rope(
                torch.randn(2, 3, 4, device=device),
                torch.randn(4, 2, 6, device=device),
            )

        # Test dimension mismatch
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected first dimension"
        ):
            calculate_rope(
                torch.randn(2, 3, 4, device=device),
                torch.randn(3, 1, 1, 6, device=device),
            )

        # Test odd head_dim
        with pytest.raises(
            (ValueError, torch.jit.Error), match="head_dim must be even"
        ):
            calculate_rope(
                torch.randn(2, 3, 2, device=device),
                torch.randn(2, 1, 1, 3, device=device),
            )

    @given(
        n_queries=valid_dims(),
        n_keys_per_query=valid_dims(),
        position_dim=valid_dims(),
        n_freq_groups=valid_dims(),
        n_heads=valid_dims(),
        head_dim=even_dims(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_property_shapes(
        self,
        n_queries,
        n_keys_per_query,
        position_dim,
        n_freq_groups,
        n_heads,
        head_dim,
        device,
    ):
        """Property-based test to ensure output shapes are correct."""
        # Test with 4D rope_freqs (position_dim, n_freq_groups, n_heads, head_dim)
        key_positions = torch.randn(
            n_queries, n_keys_per_query, position_dim, device=device
        )
        rope_freqs = torch.randn(
            position_dim, n_freq_groups, n_heads, head_dim, device=device
        )

        result = calculate_rope(key_positions, rope_freqs)
        assert result.shape == (n_queries, n_keys_per_query, n_heads, head_dim)

        # Test with broadcasting dimensions
        rope_freqs_broadcast = torch.randn(
            position_dim, 1, n_heads, head_dim, device=device
        )
        result_broadcast = calculate_rope(key_positions, rope_freqs_broadcast)
        assert result_broadcast.shape == (
            n_queries,
            n_keys_per_query,
            n_heads,
            head_dim,
        )


@pytest.mark.cuda
class TestCalculateRopeBackward:
    """Tests for the calculate_rope_backward function."""

    def test_basic_functionality(self, device):
        """Test basic operation with simple inputs."""
        # [n_queries=1, n_keys_per_query=1, n_heads=2, head_dim=2]
        grad_key_pos_encoding = torch.tensor(
            [[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32, device=device
        )
        key_positions = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32, device=device)

        # [position_dim=2, n_freq_groups=1, n_heads=2, head_dim=2]
        rope_freqs = torch.tensor(
            [
                [  # position_dim=0
                    [  # n_freq_groups=1
                        [1.0, 2.0],  # head 0, head_dim=2
                        [3.0, 4.0],  # head 1, head_dim=2
                    ]
                ],
                [  # position_dim=1
                    [  # n_freq_groups=1
                        [5.0, 6.0],  # head 0, head_dim=2
                        [7.0, 8.0],  # head 1, head_dim=2
                    ]
                ],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Gradient for key_positions
        # Head 0: 0.1 * [1.0, 5.0] + 0.2 * [2.0, 6.0] = [0.5, 1.7]
        # Head 1: 0.3 * [3.0, 7.0] + 0.4 * [4.0, 8.0] = [2.5, 5.3]
        # Sum over heads: [0.5 + 2.5, 1.7 + 5.3] = [3.0, 7.0]
        expected_grad_key_positions = torch.tensor(
            [[[3.0, 7.0]]],
            dtype=torch.float32,
            device=device,
        )

        # Gradient for rope_freqs: [position_dim=2, n_freq_groups=1, n_heads=2, head_dim=2]
        # position_dim=0, head=0: 0.1 * 1.0, 0.2 * 1.0 = [0.1, 0.2]
        # position_dim=0, head=1: 0.3 * 1.0, 0.4 * 1.0 = [0.3, 0.4]
        # position_dim=1, head=0: 0.1 * 2.0, 0.2 * 2.0 = [0.2, 0.4]
        # position_dim=1, head=1: 0.3 * 2.0, 0.4 * 2.0 = [0.6, 0.8]
        expected_grad_rope_freqs = torch.tensor(
            [
                [  # position_dim=0
                    [  # n_freq_groups=1
                        [0.1, 0.2],  # head 0
                        [0.3, 0.4],  # head 1
                    ]
                ],
                [  # position_dim=1
                    [  # n_freq_groups=1
                        [0.2, 0.4],  # head 0
                        [0.6, 0.8],  # head 1
                    ]
                ],
            ],
            dtype=torch.float32,
            device=device,
        )

        grad_key_positions, grad_rope_freqs = calculate_rope_backward(
            grad_key_pos_encoding, key_positions, rope_freqs, True, True
        )

        assert_close(
            grad_key_positions,
            expected_grad_key_positions,
            msg="Gradients for key_positions incorrect",
        )
        assert_close(
            grad_rope_freqs,
            expected_grad_rope_freqs,
            msg="Gradients for rope_freqs incorrect",
        )

    def test_head_broadcasting(self, device):
        """Test with broadcasting in the n_heads dimension."""
        # [n_queries=1, n_keys_per_query=1, n_heads=2, head_dim=2]
        grad_key_pos_encoding = torch.tensor(
            [[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32, device=device
        )
        key_positions = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32, device=device)

        # [position_dim=2, n_freq_groups=1, n_heads=1, head_dim=2
        rope_freqs = torch.tensor(
            [
                [  # position_dim=0
                    [  # n_freq_groups=1
                        [1.0, 2.0],  # head 0 (broadcast to all heads), head_dim=2
                    ]
                ],
                [  # position_dim=1
                    [  # n_freq_groups=1
                        [5.0, 6.0],  # head 0 (broadcast to all heads), head_dim=2
                    ]
                ],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Gradient for key_positions - fixed calculation
        # Head 0: 0.1 * [1.0, 5.0] + 0.2 * [2.0, 6.0] = [0.5, 1.7]
        # Head 1: 0.3 * [1.0, 5.0] + 0.4 * [2.0, 6.0] = [0.3 + 0.8, 1.5 + 2.4] = [1.1, 3.9]
        # Sum over heads: [0.5 + 1.1, 1.7 + 3.9] = [1.6, 5.6]
        expected_grad_key_positions = torch.tensor(
            [[[1.6, 5.6]]],
            dtype=torch.float32,
            device=device,
        )

        # Gradient for rope_freqs - should sum across the broadcast dimension
        # position_dim=0: (0.1 + 0.3) * 1.0, (0.2 + 0.4) * 1.0 = [0.4, 0.6]
        # position_dim=1: (0.1 + 0.3) * 2.0, (0.2 + 0.4) * 2.0 = [0.8, 1.2]
        expected_grad_rope_freqs = torch.tensor(
            [
                [  # position_dim=0
                    [  # n_freq_groups=1
                        [0.4, 0.6],  # head 0 (sum of both head gradients)
                    ]
                ],
                [  # position_dim=1
                    [  # n_freq_groups=1
                        [0.8, 1.2],  # head 0 (sum of both head gradients)
                    ]
                ],
            ],
            dtype=torch.float32,
            device=device,
        )

        grad_key_positions, grad_rope_freqs = calculate_rope_backward(
            grad_key_pos_encoding, key_positions, rope_freqs, True, True
        )

        assert_close(
            grad_key_positions,
            expected_grad_key_positions,
            msg="Gradients for key_positions with head broadcasting incorrect",
        )
        assert_close(
            grad_rope_freqs,
            expected_grad_rope_freqs,
            msg="Gradients for rope_freqs with head broadcasting incorrect",
        )

    def test_freq_group_broadcasting(self, device):
        """Test with broadcasting in the n_freq_groups dimension."""
        # [n_queries=1, n_keys_per_query=1, n_heads=1, head_dim=2]
        grad_key_pos_encoding = torch.tensor(
            [[[[0.1, 0.2]]]], dtype=torch.float32, device=device
        )
        key_positions = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32, device=device)

        # [position_dim=2, n_freq_groups=2, n_heads=1, head_dim=2]
        rope_freqs = torch.tensor(
            [
                [  # position_dim=0
                    [  # n_freq_groups=0
                        [1.0, 2.0],  # n_heads=1, head_dim=2
                    ],
                    [  # n_freq_groups=1
                        [3.0, 4.0],  # n_heads=1, head_dim=2
                    ],
                ],
                [  # position_dim=1
                    [  # n_freq_groups=0
                        [5.0, 6.0],  # n_heads=1, head_dim=2
                    ],
                    [  # n_freq_groups=1
                        [7.0, 8.0],  # n_heads=1, head_dim=2
                    ],
                ],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Gradient for key_positions - sum of all freq groups
        # Freq 0: 0.1 * [1.0, 5.0] + 0.2 * [2.0, 6.0] = [0.5, 1.7]
        # Freq 1: 0.1 * [3.0, 7.0] + 0.2 * [4.0, 8.0] = [1.1, 2.3]
        # Sum over freq groups: [0.5 + 1.1, 1.7 + 2.3] = [1.6, 4.0]
        expected_grad_key_positions = torch.tensor(
            [[[1.6, 4.0]]],
            dtype=torch.float32,
            device=device,
        )

        # Gradient for rope_freqs
        # position_dim=0, freq_group=0: 0.1 * 1.0, 0.2 * 1.0 = [0.1, 0.2]
        # position_dim=0, freq_group=1: 0.1 * 1.0, 0.2 * 1.0 = [0.1, 0.2]
        # position_dim=1, freq_group=0: 0.1 * 2.0, 0.2 * 2.0 = [0.2, 0.4]
        # position_dim=1, freq_group=1: 0.1 * 2.0, 0.2 * 2.0 = [0.2, 0.4]
        expected_grad_rope_freqs = torch.tensor(
            [
                [  # position_dim=0
                    [  # n_freq_groups=0
                        [0.1, 0.2],  # n_heads=1, head_dim=2
                    ],
                    [  # n_freq_groups=1
                        [0.1, 0.2],  # n_heads=1, head_dim=2
                    ],
                ],
                [  # position_dim=1
                    [  # n_freq_groups=0
                        [0.2, 0.4],  # n_heads=1, head_dim=2
                    ],
                    [  # n_freq_groups=1
                        [0.2, 0.4],  # n_heads=1, head_dim=2
                    ],
                ],
            ],
            dtype=torch.float32,
            device=device,
        )

        grad_key_positions, grad_rope_freqs = calculate_rope_backward(
            grad_key_pos_encoding, key_positions, rope_freqs, True, True
        )

        assert_close(
            grad_key_positions,
            expected_grad_key_positions,
            msg="Gradients for key_positions with freq group broadcasting incorrect",
        )
        assert_close(
            grad_rope_freqs,
            expected_grad_rope_freqs,
            msg="Gradients for rope_freqs with freq group broadcasting incorrect",
        )

    def test_selective_gradient_computation(self, device):
        """Test that only requested gradients are computed."""
        # Updated shapes
        grad_key_pos_encoding = torch.randn(
            3, 4, 2, 4, device=device
        )  # [n_queries, n_keys_per_query, n_heads, head_dim]
        key_positions = torch.randn(
            3, 4, 2, device=device
        )  # [n_queries, n_keys_per_query, position_dim]
        rope_freqs = torch.randn(
            2, 1, 2, 4, device=device
        )  # [position_dim, n_freq_groups, n_heads, head_dim]

        # Only key_positions gradient
        grad_key_positions, grad_rope_freqs = calculate_rope_backward(
            grad_key_pos_encoding, key_positions, rope_freqs, True, False
        )
        assert grad_key_positions is not None
        assert grad_rope_freqs is None

        # Only rope_freqs gradient
        grad_key_positions, grad_rope_freqs = calculate_rope_backward(
            grad_key_pos_encoding, key_positions, rope_freqs, False, True
        )
        assert grad_key_positions is None
        assert grad_rope_freqs is not None

    def test_error_conditions(self, device):
        """Test that appropriate errors are raised for invalid inputs."""
        grad_key_pos_encoding = torch.randn(
            3, 4, 2, 6, device=device
        )  # [n_queries, n_keys_per_query, n_heads, head_dim]

        # Test 2D key_positions (should be 3D)
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected 3 dimensions"
        ):
            calculate_rope_backward(
                grad_key_pos_encoding,
                torch.randn(3, 2, device=device),
                torch.randn(2, 1, 2, 6, device=device),
                True,
                True,
            )

        # Test 3D rope_freqs (should be 4D)
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected 4 dimensions"
        ):
            calculate_rope_backward(
                grad_key_pos_encoding,
                torch.randn(3, 4, 2, device=device),
                torch.randn(2, 1, 6, device=device),
                True,
                True,
            )

        # Test dimension mismatch between key_positions and rope_freqs
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected first dimension"
        ):
            calculate_rope_backward(
                grad_key_pos_encoding,
                torch.randn(3, 4, 2, device=device),
                torch.randn(3, 1, 2, 6, device=device),  # position_dim=3 doesn't match
                True,
                True,
            )

        # Test odd head_dim
        with pytest.raises(
            (ValueError, torch.jit.Error), match="head_dim must be even"
        ):
            calculate_rope_backward(
                torch.randn(3, 4, 2, 5, device=device),  # odd head_dim=5
                torch.randn(3, 4, 2, device=device),
                torch.randn(2, 1, 2, 5, device=device),
                True,
                True,
            )
