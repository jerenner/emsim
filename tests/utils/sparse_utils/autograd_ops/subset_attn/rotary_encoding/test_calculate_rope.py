import pytest
import torch
from hypothesis import HealthCheck, given, settings

from emsim.utils.sparse_utils.ops.subset_attn.rotary_encoding import (
    calculate_rope,
    calculate_rope_backward,
)

from .conftest import assert_close, even_dims, valid_dims, batch_dims_strategy


@pytest.mark.cuda_if_available
class TestCalculateRope:
    """Tests for the calculate_rope function."""

    def test_basic_functionality(self, device):
        """Test basic operation with simple inputs."""
        positions = torch.tensor(
            [[[1.0, 2.0]]], dtype=torch.float32, device=device
        )  # [1, 1, 2]

        # Now with position_dim=2 to match positions
        rope_freqs = torch.tensor(
            [[[[1.0, 2.0]]], [[[5.0, 6.0]]]],  # position_dim=0  # position_dim=1
            dtype=torch.float32,
            device=device,
        )  # [2, 1, 1, 2] -> [position_dim=2, n_freq_groups=1, n_heads=1, head_dim/2=2]

        # Expected: matrix multiplication of positions and rope_freqs
        # 1.0 * [1.0, 2.0] + 2.0 * [5.0, 6.0] = [11.0, 14.0]
        expected = torch.tensor(
            [[[[11.0, 14.0]]]], dtype=torch.float32, device=device
        )  # [1, 1, 1, 2]
        result = calculate_rope(positions, rope_freqs)

        assert_close(result, expected, msg="Basic calculate_rope failed")

    def test_multi_freq_groups(self, device):
        """Test with multiple frequency groups."""
        positions = torch.tensor(
            [[[1.0, 2.0]]], dtype=torch.float32, device=device
        )  # [1, 1, 2]
        rope_freqs = torch.tensor(
            [
                [  # position_dim=2
                    [  # n_freq_groups=2
                        [[1.0, 2.0]],  # n_heads=1, head_dim/2=2
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
        result = calculate_rope(positions, rope_freqs)

        assert_close(result, expected, msg="Multi-group calculate_rope failed")

    def test_multi_heads(self, device):
        """Test with multiple heads."""
        positions = torch.tensor(
            [[[1.0, 2.0]]], dtype=torch.float32, device=device
        )  # [1, 1, 2]

        rope_freqs = torch.tensor(
            [
                # position_dim=0
                [
                    # n_freq_groups=1 (explicit dimension)
                    [
                        [1.0, 2.0],  # head 0, head_dim/2=2
                        [3.0, 4.0],  # head 1, head_dim/2=2
                    ]
                ],
                # position_dim=1
                [
                    # n_freq_groups=1 (explicit dimension)
                    [
                        [5.0, 6.0],  # head 0, head_dim/2=2
                        [7.0, 8.0],  # head 1, head_dim/2=2
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

        result = calculate_rope(positions, rope_freqs)
        assert_close(result, expected, msg="Multi-head calculate_rope failed")

    def test_error_conditions(self, device):
        """Test that appropriate errors are raised for invalid inputs."""
        # Test 1D positions (should be at least 2D)
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected at least 2 dimensions"
        ):
            calculate_rope(
                torch.randn(2, device=device), torch.randn(3, 1, 1, 6, device=device)
            )

        # Test 3D rope_freqs (should be 4D)
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected 4 dimensions"
        ):
            calculate_rope(
                torch.randn(2, 3, 4, device=device),
                torch.randn(4, 2, 6, device=device),
            )
        # Test head dimension mismatch
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected first dimension"
        ):
            calculate_rope(
                torch.randn(2, 3, 4, device=device),
                torch.randn(3, 1, 1, 6, device=device),
            )

    def test_extended_batch_dims(self, device):
        """Test inputs with lots of batch dimensions"""
        batch_dims = [2, 4, 6, 8]
        positions = torch.randn(*batch_dims, 3, device=device)  # position_dim = 3

        rope_freqs = torch.randn(
            3,  # position_dim
            2,
            2,  # n_heads
            4,  # n_heads/2
            device=device,
        )

        expected_shape = (*batch_dims, 2, 4)
        result = calculate_rope(positions, rope_freqs)

        assert result.shape == expected_shape

    @given(
        batch_dims=batch_dims_strategy(),
        position_dim=valid_dims(),
        n_freq_groups=valid_dims(),
        n_heads=valid_dims(),
        head_dim=even_dims(),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    def test_property_shapes(
        self,
        batch_dims,
        position_dim,
        n_freq_groups,
        n_heads,
        head_dim,
        device,
    ):
        """Property-based test to ensure output shapes are correct."""
        # Test with 4D rope_freqs (position_dim, n_freq_groups, n_heads, head_dim/2)
        positions = torch.randn(
            *batch_dims, position_dim, device=device
        )
        rope_freqs = torch.randn(
            position_dim, n_freq_groups, n_heads, head_dim, device=device
        )

        result = calculate_rope(positions, rope_freqs)
        assert result.shape == (*batch_dims, n_heads, head_dim)

        # Test with broadcasting dimensions
        rope_freqs_broadcast = torch.randn(
            position_dim, 1, n_heads, head_dim, device=device
        )
        result_broadcast = calculate_rope(positions, rope_freqs_broadcast)
        assert result_broadcast.shape == (
            *batch_dims,
            n_heads,
            head_dim,
        )


@pytest.mark.cuda_if_available
class TestCalculateRopeBackward:
    """Tests for the calculate_rope_backward function."""

    def test_basic_functionality(self, device):
        """Test basic operation with simple inputs."""
        # [n_queries=1, n_keys_per_query=1, n_heads=2, head_dim/2=2]
        grad_rope_encoding = torch.tensor(
            [[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32, device=device
        )
        positions = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32, device=device)

        # [position_dim=2, n_freq_groups=1, n_heads=2, head_dim/2=2]
        rope_freqs = torch.tensor(
            [
                [  # position_dim=0
                    [  # n_freq_groups=1
                        [1.0, 2.0],  # head 0, head_dim/2=2
                        [3.0, 4.0],  # head 1, head_dim/2=2
                    ]
                ],
                [  # position_dim=1
                    [  # n_freq_groups=1
                        [5.0, 6.0],  # head 0, head_dim/2=2
                        [7.0, 8.0],  # head 1, head_dim/2=2
                    ]
                ],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Gradient for positions
        # Head 0: 0.1 * [1.0, 5.0] + 0.2 * [2.0, 6.0] = [0.5, 1.7]
        # Head 1: 0.3 * [3.0, 7.0] + 0.4 * [4.0, 8.0] = [2.5, 5.3]
        # Sum over heads: [0.5 + 2.5, 1.7 + 5.3] = [3.0, 7.0]
        expected_grad_positions = torch.tensor(
            [[[3.0, 7.0]]],
            dtype=torch.float32,
            device=device,
        )

        # Gradient for rope_freqs:
        #   [position_dim=2, n_freq_groups=1, n_heads=2, head_dim/2=2]
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

        grad_positions, grad_rope_freqs = calculate_rope_backward(
            grad_rope_encoding, positions, rope_freqs, True, True
        )

        assert_close(
            grad_positions,
            expected_grad_positions,
            msg="Gradients for positions incorrect",
        )
        assert_close(
            grad_rope_freqs,
            expected_grad_rope_freqs,
            msg="Gradients for rope_freqs incorrect",
        )

    def test_extended_batch_dims(self, device):
        """Test inputs with lots of batch dimensions"""
        batch_dims = [2, 4, 6, 8]
        n_heads = 2
        half_head_dim = 4
        position_dim = 3
        n_freq_groups = 2
        grad_rope_encoding = torch.randn(*batch_dims, n_heads, half_head_dim)
        positions = torch.randn(*batch_dims, position_dim)
        rope_freqs = torch.randn(position_dim, n_freq_groups, n_heads, half_head_dim)

        expected_shape_grad_positions = (*batch_dims, position_dim)
        expected_shape_grad_rope_freqs = rope_freqs.shape

        grad_positions, grad_rope_freqs = calculate_rope_backward(
            grad_rope_encoding, positions, rope_freqs, True, True
        )

        assert grad_positions.shape == expected_shape_grad_positions
        assert grad_rope_freqs.shape == expected_shape_grad_rope_freqs

    def test_head_broadcasting(self, device):
        """Test with broadcasting in the n_heads dimension."""
        # [n_queries=1, n_keys_per_query=1, n_heads=2, head_dim=2]
        grad_rope_encoding = torch.tensor(
            [[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32, device=device
        )
        positions = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32, device=device)

        # [position_dim=2, n_freq_groups=1, n_heads=1, head_dim=2
        rope_freqs = torch.tensor(
            [
                [  # position_dim=0
                    [  # n_freq_groups=1
                        [1.0, 2.0],  # head 0 (broadcast to all heads), head_dim/2=2
                    ]
                ],
                [  # position_dim=1
                    [  # n_freq_groups=1
                        [5.0, 6.0],  # head 0 (broadcast to all heads), head_dim/2=2
                    ]
                ],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Gradient for positions
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

        grad_positions, grad_rope_freqs = calculate_rope_backward(
            grad_rope_encoding, positions, rope_freqs, True, True
        )

        assert_close(
            grad_positions,
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
        # [n_queries=1, n_keys_per_query=1, n_heads=1, head_dim/2=2]
        grad_rope_encoding = torch.tensor(
            [[[[0.1, 0.2]]]], dtype=torch.float32, device=device
        )
        positions = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32, device=device)

        # [position_dim=2, n_freq_groups=2, n_heads=1, head_dim/2=2]
        rope_freqs = torch.tensor(
            [
                [  # position_dim=0
                    [  # n_freq_groups=0
                        [1.0, 2.0],  # n_heads=1, head_dim/2=2
                    ],
                    [  # n_freq_groups=1
                        [3.0, 4.0],  # n_heads=1, head_dim/2=2
                    ],
                ],
                [  # position_dim=1
                    [  # n_freq_groups=0
                        [5.0, 6.0],  # n_heads=1, head_dim/2=2
                    ],
                    [  # n_freq_groups=1
                        [7.0, 8.0],  # n_heads=1, head_dim/2=2
                    ],
                ],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Gradient for positions - sum of all freq groups
        # Freq 0: 0.1 * [1.0, 5.0] + 0.2 * [2.0, 6.0] = [0.5, 1.7]
        # Freq 1: 0.1 * [3.0, 7.0] + 0.2 * [4.0, 8.0] = [1.1, 2.3]
        # Sum over freq groups: [0.5 + 1.1, 1.7 + 2.3] = [1.6, 4.0]
        expected_grad_positions = torch.tensor(
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
                        [0.1, 0.2],  # n_heads=1, head_dim/2=2
                    ],
                    [  # n_freq_groups=1
                        [0.1, 0.2],  # n_heads=1, head_dim/2=2
                    ],
                ],
                [  # position_dim=1
                    [  # n_freq_groups=0
                        [0.2, 0.4],  # n_heads=1, head_dim/2=2
                    ],
                    [  # n_freq_groups=1
                        [0.2, 0.4],  # n_heads=1, head_dim/2=2
                    ],
                ],
            ],
            dtype=torch.float32,
            device=device,
        )

        grad_positions, grad_rope_freqs = calculate_rope_backward(
            grad_rope_encoding, positions, rope_freqs, True, True
        )

        assert_close(
            grad_positions,
            expected_grad_positions,
            msg="Gradients for positions with freq group broadcasting incorrect",
        )
        assert_close(
            grad_rope_freqs,
            expected_grad_rope_freqs,
            msg="Gradients for rope_freqs with freq group broadcasting incorrect",
        )

    def test_selective_gradient_computation(self, device):
        """Test that only requested gradients are computed."""
        # Updated shapes
        grad_rope_encoding = torch.randn(
            3, 4, 2, 4, device=device
        )  # [n_queries, n_keys_per_query, n_heads, head_dim/2]
        positions = torch.randn(
            3, 4, 2, device=device
        )  # [n_queries, n_keys_per_query, position_dim]
        rope_freqs = torch.randn(
            2, 1, 2, 4, device=device
        )  # [position_dim, n_freq_groups, n_heads, head_dim/2]

        # Only positions gradient
        grad_positions, grad_rope_freqs = calculate_rope_backward(
            grad_rope_encoding, positions, rope_freqs, True, False
        )
        assert grad_positions is not None
        assert grad_rope_freqs is None

        # Only rope_freqs gradient
        grad_positions, grad_rope_freqs = calculate_rope_backward(
            grad_rope_encoding, positions, rope_freqs, False, True
        )
        assert grad_positions is None
        assert grad_rope_freqs is not None

    def test_error_conditions(self, device):
        """Test that appropriate errors are raised for invalid inputs."""
        batch_dims = [2, 4, 6]
        n_heads = 4
        head_dim = 8
        position_dim = 3
        n_freq_groups = 2
        grad_rope_encoding = torch.randn(
            *batch_dims, n_heads, head_dim // 2, device=device
        )  # [2, 4, 6, n_heads, head_dim/2]
        positions = torch.randn(*batch_dims, position_dim, device=device)
        rope_freqs = torch.randn(position_dim, n_freq_groups, n_heads, head_dim // 2)

        # Test 1D positions (should be at least 2D)
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected at least 2 dimensions"
        ):
            calculate_rope_backward(
                grad_rope_encoding,
                torch.randn(position_dim, device=device),
                rope_freqs,
                True,
                True,
            )

        # Test 3D rope_freqs (should be 4D)
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected 4 dimensions"
        ):
            calculate_rope_backward(
                grad_rope_encoding,
                positions,
                torch.randn(2, 1, 6, device=device),
                True,
                True,
            )

        # Test 2D grad_rope_encoding (should be at least 3D)
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected at least 3 dimensions"
        ):
            calculate_rope_backward(
                torch.randn(n_heads, head_dim // 2, device=device),
                positions,
                rope_freqs,
                True,
                True,
            )

        # Test position_dim mismatch between positions and rope_freqs
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected first dimension"
        ):
            calculate_rope_backward(
                grad_rope_encoding,
                positions,
                torch.randn(
                    position_dim - 1, n_freq_groups, n_heads, head_dim, device=device
                ),
                True,
                True,
            )

        # Test mismatched batch dims between grad_rope_encoding and positions
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Expected matching batch dims"
        ):
            calculate_rope_backward(
                grad_rope_encoding,
                positions[0],
                rope_freqs,
                True,
                True,
            )
