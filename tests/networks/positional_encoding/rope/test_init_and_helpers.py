from typing import Any, Optional

import pytest
import torch

from emsim.networks.positional_encoding.rope import (
    init_2d_freqs_rope_mixed,
    init_nd_freqs,
    prep_multilevel_positions,
    get_multilevel_freq_group_pattern,
)


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.float32


@pytest.mark.cuda_if_available
class TestInit2DFreqsRopeMixed:
    def test_init_2d_freqs_shape(self, device: str, dtype: torch.dtype):
        head_dim = 64
        num_heads = 8
        freqs = init_2d_freqs_rope_mixed(
            head_dim, num_heads, dtype=dtype, device=device
        )

        assert freqs.shape == (2, num_heads, head_dim // 2)
        assert freqs.dtype == dtype
        assert freqs.device.type == device

    def test_init_2d_freqs_rotate_param(self, device: str, dtype: torch.dtype):
        head_dim = 64
        num_heads = 8

        torch.manual_seed(0)
        freqs_rotate = init_2d_freqs_rope_mixed(
            head_dim, num_heads, rotate=True, dtype=dtype, device=device
        )

        torch.manual_seed(0)
        freqs_no_rotate = init_2d_freqs_rope_mixed(
            head_dim, num_heads, rotate=False, dtype=dtype, device=device
        )

        assert not torch.allclose(freqs_rotate, freqs_no_rotate)

        # With rotate=False, all heads should have the same pattern
        for i in range(1, num_heads):
            assert torch.allclose(freqs_no_rotate[:, 0, :], freqs_no_rotate[:, i, :])

        # With rotate=True, heads should have different patterns
        assert not torch.allclose(freqs_rotate[:, 0, :], freqs_rotate[:, 1, :])

    def test_init_2d_freqs_theta_param(self, device: str, dtype: torch.dtype):
        head_dim = 64
        num_heads = 8

        torch.manual_seed(0)
        freqs1 = init_2d_freqs_rope_mixed(
            head_dim, num_heads, theta=10.0, rotate=False, dtype=dtype, device=device
        )

        torch.manual_seed(0)
        freqs2 = init_2d_freqs_rope_mixed(
            head_dim, num_heads, theta=100.0, rotate=False, dtype=dtype, device=device
        )

        # Different theta should produce different magnitude scaling
        assert not torch.allclose(freqs1, freqs2)

        # Check decay pattern - higher indices should have smaller values for smaller theta
        assert torch.mean(torch.abs(freqs1)) < torch.mean(torch.abs(freqs2))

    def test_init_2d_freqs_odd_head_dim(self, device: str, dtype: torch.dtype):
        head_dim = 65  # Odd head dimension
        num_heads = 8

        with pytest.raises(ValueError, match="head_dim must be even for RoPE"):
            init_2d_freqs_rope_mixed(head_dim, num_heads, dtype=dtype, device=device)


@pytest.mark.cuda_if_available
class TestInitNDFreqs:
    def _assert_freqs_shape(
        self, freqs, group_counts, num_heads, encoding_dims, dtype, device_type
    ):
        """Helper to validate frequency tensor shapes and properties"""
        assert len(freqs) == len(group_counts)
        for i, (freq, pos_count, enc_dim) in enumerate(
            zip(freqs, group_counts, encoding_dims)
        ):
            assert freq.shape == (pos_count, num_heads, enc_dim)
            assert freq.dtype == dtype
            assert freq.device.type == device_type

    def _assert_encoding_ranges(self, encoding_ranges, expected_ranges, device_type):
        """Helper to validate encoding ranges tensor"""
        assert encoding_ranges.shape == (len(expected_ranges), 2)
        assert encoding_ranges.dtype == torch.long
        assert encoding_ranges.device.type == device_type

        # Check against expected ranges
        expected_tensor = torch.tensor(expected_ranges, dtype=torch.long)
        assert torch.equal(encoding_ranges, expected_tensor.to(encoding_ranges.device))

        # Verify ranges are contiguous and non-overlapping
        assert torch.all(encoding_ranges[:, 0] <= encoding_ranges[:, 1])
        if len(expected_ranges) > 1:
            assert torch.all(encoding_ranges[:-1, 1] == encoding_ranges[1:, 0])

    @pytest.mark.parametrize("position_dim", [1, 2, 3, 4], ids=["1D", "2D", "3D", "4D"])
    def test_dimensions(self, position_dim: int, device: str, dtype: torch.dtype):
        """Test initialization with different dimensions"""
        head_dim = 64
        num_heads = 8
        freq_group_pattern = torch.ones(
            1, position_dim, dtype=torch.bool, device=device
        )

        freqs, encoding_ranges = init_nd_freqs(
            position_dim,
            head_dim,
            num_heads,
            freq_group_pattern,
            dtype=dtype,
            device=device,
        )

        self._assert_freqs_shape(
            freqs, [position_dim], num_heads, [head_dim // 2], dtype, device
        )
        self._assert_encoding_ranges(encoding_ranges, [[0, head_dim // 2]], device)

    @pytest.mark.parametrize(
        "rotate", [True, False], ids=["rotate=True", "rotate=False"]
    )
    def test_rotation(self, rotate: bool, device: str, dtype: torch.dtype):
        """Test effect of rotation parameter"""
        position_dim = 2
        head_dim = 64
        num_heads = 8
        freq_group_pattern = torch.ones(
            1, position_dim, dtype=torch.bool, device=device
        )

        torch.manual_seed(0)
        freqs, _ = init_nd_freqs(
            position_dim,
            head_dim,
            num_heads,
            freq_group_pattern,
            rotate=rotate,
            dtype=dtype,
            device=device,
        )

        if not rotate:
            # With rotate=False, all heads should have same pattern
            for i in range(1, num_heads):
                assert torch.allclose(freqs[0][:, 0, :], freqs[0][:, i, :])
        else:
            # With rotate=True and sufficient heads, patterns should differ
            # Note: This could theoretically fail with extremely low probability
            # if random rotations happened to be very similar
            if num_heads > 1:
                any_different = False
                for i in range(1, num_heads):
                    if not torch.allclose(freqs[0][:, 0, :], freqs[0][:, i, :]):
                        any_different = True
                        break
                assert (
                    any_different
                ), "Expected at least some heads to be different with rotation"

    @pytest.mark.parametrize(
        "pattern_case",
        [
            {
                "name": "equal_dims_per_group",
                "pattern": lambda device: torch.tensor(
                    [
                        [1, 1, 0],  # Group 1: dims 0,1
                        [0, 1, 1],  # Group 2: dims 1,2
                    ],
                    device=device,
                    dtype=torch.bool,
                ),
                "position_dim": 3,
                "expected_dims": [2, 2],
                "expected_range_fraction": [1 / 2, 1 / 2],
            },
            {
                "name": "heterogeneous_dims",
                "pattern": lambda device: torch.tensor(
                    [
                        [1, 1, 0],  # Group 1: dims 0,1
                        [0, 0, 1],  # Group 2: dim 2
                    ],
                    device=device,
                    dtype=torch.bool,
                ),
                "position_dim": 3,
                "expected_dims": [2, 1],
                "expected_range_fraction": [1 / 2, 1 / 2],
            },
            {
                "name": "complex_pattern",
                "pattern": lambda device: torch.tensor(
                    [
                        [1, 0, 1, 0],  # Group 1: dims 0,2
                        [0, 1, 0, 1],  # Group 2: dims 1,3
                        [1, 1, 0, 1],  # Group 3: dims 0,1,3
                        [0, 0, 1, 0],  # Group 4: dim 2
                    ],
                    device=device,
                    dtype=torch.bool,
                ),
                "position_dim": 4,
                "expected_dims": [2, 2, 3, 1],
                "expected_range_fraction": [1 / 4, 1 / 4, 1 / 4, 1 / 4],
            },
            {
                "name": "empty_group",
                "pattern": lambda device: torch.tensor(
                    [
                        [1, 1, 1],  # Group 1: all dims
                        [0, 0, 0],  # Group 2: no dims (empty)
                    ],
                    device=device,
                    dtype=torch.bool,
                ),
                "position_dim": 3,
                "expected_dims": [3, 0],
                "expected_range_fraction": [1 / 2, 1 / 2],
            },
        ],
        ids=lambda x: x["name"],
    )
    def test_frequency_group_patterns(
        self, pattern_case: dict[str, Any], device: str, dtype: torch.dtype
    ):
        """Test various frequency group patterns"""
        head_dim = 64
        num_heads = 8
        position_dim = pattern_case["position_dim"]

        # Get pattern tensor using the lambda
        freq_group_pattern = pattern_case["pattern"](device)

        freqs, encoding_ranges = init_nd_freqs(
            position_dim,
            head_dim,
            num_heads,
            freq_group_pattern,
            dtype=dtype,
            device=device,
        )

        # Calculate expected encoding dimensions and ranges
        expected_enc_dims = [
            int(head_dim // 2 * fraction)
            for fraction in pattern_case["expected_range_fraction"]
        ]

        # Calculate expected ranges
        expected_ranges = []
        start = 0
        for enc_dim in expected_enc_dims:
            end = start + enc_dim
            expected_ranges.append([start, end])
            start = end

        self._assert_freqs_shape(
            freqs,
            pattern_case["expected_dims"],
            num_heads,
            expected_enc_dims,
            dtype,
            device,
        )
        self._assert_encoding_ranges(encoding_ranges, expected_ranges, device)

    @pytest.mark.parametrize(
        "head_dim,num_groups,enforce_equal,expected_enc_dims,expected_error",
        [
            # Even distribution cases
            (64, 2, True, [16, 16], None),
            (64, 4, True, [8, 8, 8, 8], None),
            # Uneven distribution cases (not enforcing equality)
            (62, 3, False, [11, 10, 10], None),  # 31 total with 1 remainder
            (70, 6, False, [6, 6, 6, 6, 6, 5], None),  # 35 total with 5 remainders
            # Error cases - enforcing equality with non-divisible dimensions
            (62, 3, True, None, ValueError),
            (70, 6, True, None, ValueError),
        ],
        ids=[
            "dim64_grp2_equal",
            "dim64_grp4_equal",
            "dim62_grp3_uneven",
            "dim70_grp6_uneven",
            "dim62_grp3_equal_error",
            "dim70_grp6_equal_error",
        ],
    )
    def test_dimension_distribution(
        self,
        head_dim: int,
        num_groups: int,
        enforce_equal: bool,
        expected_enc_dims: Optional[list[int]],
        expected_error: Optional[Exception],
        device: str,
        dtype: torch.dtype,
    ):
        """Test distribution of RoPE dimensions across frequency groups"""
        position_dim = 3
        num_heads = 8

        # Create a pattern with one position dimension per group
        pattern_data = torch.zeros(num_groups, position_dim, dtype=torch.bool)
        for g in range(num_groups):
            dim_idx = g % position_dim  # Cycle through available dimensions
            pattern_data[g, dim_idx] = True

        freq_group_pattern = pattern_data.to(device)

        if expected_error:
            with pytest.raises(expected_error):
                init_nd_freqs(
                    position_dim,
                    head_dim,
                    num_heads,
                    freq_group_pattern,
                    enforce_freq_groups_equal=enforce_equal,
                    dtype=dtype,
                    device=device,
                )
        else:
            freqs, encoding_ranges = init_nd_freqs(
                position_dim,
                head_dim,
                num_heads,
                freq_group_pattern,
                enforce_freq_groups_equal=enforce_equal,
                dtype=dtype,
                device=device,
            )

            # Check that each group has one position dimension
            expected_dims = [1] * num_groups

            # Calculate expected ranges
            expected_ranges = []
            start = 0
            for enc_dim in expected_enc_dims:
                end = start + enc_dim
                expected_ranges.append([start, end])
                start = end

            self._assert_freqs_shape(
                freqs, expected_dims, num_heads, expected_enc_dims, dtype, device
            )
            self._assert_encoding_ranges(encoding_ranges, expected_ranges, device)

            # Verify total dimensions match half_head_dim
            total_encodings = sum(freq.shape[2] for freq in freqs)
            assert total_encodings == head_dim // 2

    @pytest.mark.parametrize(
        "theta_case",
        [
            {
                "name": "scalar",
                "theta": lambda device, dtype: 10.0,
            },
            {
                "name": "full_tensor",
                "theta": lambda device, dtype: torch.full(
                    (1, 2), 10.0, device=device, dtype=dtype
                ),
            },
            {
                "name": "broadcasted",
                "theta": lambda device, dtype: torch.tensor(
                    [[10.0]], device=device, dtype=dtype
                ),
            },
        ],
        ids=lambda x: x["name"],
    )
    def test_theta_formats(
        self, theta_case: dict[str, Any], device: str, dtype: torch.dtype
    ):
        """Test different formats for theta parameter"""
        position_dim = 2
        head_dim = 64
        num_heads = 4
        freq_group_pattern = torch.ones(
            1, position_dim, dtype=torch.bool, device=device
        )

        # Get theta using the lambda
        theta = theta_case["theta"](device, dtype)

        torch.manual_seed(0)  # For reproducibility
        freqs, encoding_ranges = init_nd_freqs(
            position_dim,
            head_dim,
            num_heads,
            freq_group_pattern,
            thetas=theta,
            rotate=False,
            dtype=dtype,
            device=device,
        )

        # Store as reference result
        expected_freqs = freqs
        expected_ranges = encoding_ranges

        # Compare with scalar theta (should be identical)
        torch.manual_seed(0)
        freqs_scalar, ranges_scalar = init_nd_freqs(
            position_dim,
            head_dim,
            num_heads,
            freq_group_pattern,
            thetas=10.0,
            rotate=False,
            dtype=dtype,
            device=device,
        )

        # Results should match the reference
        assert torch.allclose(freqs_scalar[0], expected_freqs[0])
        assert torch.equal(ranges_scalar, expected_ranges)

    @pytest.mark.parametrize(
        "error_case",
        [
            {
                "name": "odd_head_dim",
                "args": lambda device: {
                    "position_dim": 2,
                    "head_dim": 65,  # Odd
                    "num_heads": 8,
                    "freq_group_pattern": torch.ones(
                        1, 2, dtype=torch.bool, device=device
                    ),
                },
                "error": ValueError,
                "match": "head_dim must be even for RoPE",
            },
            {
                "name": "1d_tensor_thetas",
                "args": lambda device: {
                    "position_dim": 2,
                    "head_dim": 64,
                    "num_heads": 8,
                    "freq_group_pattern": torch.ones(
                        1, 2, dtype=torch.bool, device=device
                    ),
                    "thetas": torch.tensor([10.0, 20.0], device=device),
                },
                "error": ValueError,
                "match": "Expected thetas to either be a scalar or a 2D tensor",
            },
            {
                "name": "wrong_shape_thetas",
                "args": lambda device: {
                    "position_dim": 2,
                    "head_dim": 64,
                    "num_heads": 8,
                    "freq_group_pattern": torch.ones(
                        2, 2, dtype=torch.bool, device=device
                    ),
                    "thetas": torch.ones(
                        3, 2, device=device
                    ),  # 3 groups when pattern has 2
                },
                "error": ValueError,
                "match": "Expected thetas to be broadcastable to",
            },
        ],
        ids=lambda x: x["name"],
    )
    def test_error_cases(
        self, error_case: dict[str, Any], device: str, dtype: torch.dtype
    ):
        """Test various error conditions"""
        # Get args using the lambda
        args = error_case["args"](device)

        with pytest.raises(error_case["error"], match=error_case["match"]):
            init_nd_freqs(**args, dtype=dtype, device=device)

    def test_theta_magnitude_effect(self, device: str, dtype: torch.dtype):
        """Test that different theta values affect frequency magnitudes"""
        position_dim = 2
        head_dim = 64
        num_heads = 4

        freq_group_pattern = torch.tensor(
            [
                [1, 0],  # Group 1: dim 0
                [0, 1],  # Group 2: dim 1
            ],
            device=device,
            dtype=torch.bool,
        )

        # Different theta for each group
        thetas_multi = torch.tensor(
            [
                [10.0, 20.0],  # Group 1: small theta for dim 0
                [30.0, 40.0],  # Group 2: large theta for dim 1
            ],
            device=device,
            dtype=dtype,
        )

        freqs_multi, _ = init_nd_freqs(
            position_dim,
            head_dim,
            num_heads,
            freq_group_pattern,
            thetas=thetas_multi,
            rotate=False,
            dtype=dtype,
            device=device,
        )

        # Group 2 (larger theta) should have larger magnitude frequencies
        assert torch.mean(torch.abs(freqs_multi[0])) < torch.mean(
            torch.abs(freqs_multi[1])
        )


# Tests for prep_multilevel_positions
@pytest.mark.cuda_if_available
class TestPrepMultilevelPositions:
    def test_prep_multilevel_positions(self, device):
        # Sample batch of indices (i, j)
        spatial_indices = torch.tensor(
            [
                [10, 20],  # batch 0, level 0
                [5, 15],  # batch 0, level 1
                [8, 12],  # batch 1, level 0
                [3, 7],  # batch 1, level 1
            ],
            dtype=torch.long,
            device=device,
        )
        batch_level_indices = torch.tensor(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=torch.long,
            device=device,
        )

        # Spatial shapes (level, 2) - height and width for each level
        spatial_shapes = torch.tensor(
            [
                [100, 100],  # level 0: 100x100
                [50, 50],  # level 1: 50x50
            ],
            dtype=torch.float,
            device=device,
        )

        positions = prep_multilevel_positions(
            spatial_indices, batch_level_indices, spatial_shapes
        )

        assert positions.shape == (spatial_indices.size(0), spatial_indices.size(1) + 1)
        assert torch.is_floating_point(positions)

        # Verify scaling for a level 1 position (should be scaled up relative to level 0)
        # Since level 1 is half the size, its coordinates get doubled in the common space
        scale_factor = 100 / 50  # max_shape / level_shape
        expected_i = (5 + 0.5) * scale_factor  # +0.5 for pixel center
        assert torch.isclose(
            positions[1, 0], torch.tensor(expected_i, dtype=torch.float, device=device)
        )

    def test_prep_multilevel_positions_batched_shapes(self, device):
        # Sample indices
        spatial_indices = torch.tensor(
            [
                [10, 20],  # batch 0, level 0
                [5, 15],  # batch 1, level 1
            ],
            dtype=torch.long,
            device=device,
        )
        batch_level_indices = torch.tensor(
            [
                [0, 0],
                [1, 1],
            ],
            dtype=torch.long,
            device=device
        )

        # Batched spatial shapes (batch, level, 2)
        spatial_shapes = torch.tensor(
            [
                [  # batch 0
                    [100, 100],  # level 0
                    [50, 50],  # level 1
                ],
                [  # batch 1
                    [200, 200],  # level 0
                    [100, 100],  # level 1
                ],
            ],
            dtype=torch.float,
            device=device,
        )

        positions = prep_multilevel_positions(
            spatial_indices, batch_level_indices, spatial_shapes
        )
        assert positions.shape == (spatial_indices.shape[0], spatial_indices.shape[1] + 1)


@pytest.mark.cuda_if_available
class TestMultilevelFreqGroupPattern:
    @pytest.mark.parametrize(
        "position_dim,pattern_name,expected_shape,expected_values",
        [
            (2, "single", (1, 3), [[True, True, True]]),
            (2, "partition", (2, 3), [[True, True, False], [False, False, True]]),
            (
                2,
                "closure",
                (3, 3),
                [[True, True, False], [False, False, True], [True, True, True]],
            ),
        ],
        ids=["single", "partition", "closure"],
    )
    def test_valid_patterns(
        self,
        position_dim: int,
        pattern_name: str,
        expected_shape: tuple,
        expected_values: list,
        device: str,
    ):
        """Test that valid pattern names return the correct tensors."""
        result = get_multilevel_freq_group_pattern(
            position_dim, pattern_name, device=device
        )

        # Check shape
        assert result.shape == expected_shape

        # Check values
        expected_tensor = torch.tensor(expected_values, device=device)
        assert torch.equal(result, expected_tensor)

        # Check device
        assert result.device.type == device

    def test_invalid_pattern(self, device: str):
        """Test that invalid pattern names raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognized pattern_name"):
            get_multilevel_freq_group_pattern(2, "invalid_pattern", device=device)
