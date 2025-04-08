import pytest
import torch

from emsim.networks.positional_encoding.rope import (
    init_2d_freqs,
    init_nd_freqs,
    RoPEEncodingND,
    RoPEEncodingNDGroupedFreqs,
    prep_multilevel_positions,
)


@pytest.fixture
def dtype():
    return torch.float32


@pytest.mark.cuda_if_available
class TestInit2DFreqs:
    def test_init_2d_freqs_shape(self, device, dtype):
        head_dim = 64
        num_heads = 8
        freqs = init_2d_freqs(head_dim, num_heads, dtype=dtype, device=device)

        assert freqs.shape == (num_heads, head_dim // 2, 2)
        assert freqs.dtype == dtype
        assert freqs.device.type == device


    def test_init_2d_freqs_rotate_param(self, device, dtype):
        head_dim = 64
        num_heads = 8

        # With and without rotation should produce different results
        freqs_rotate = init_2d_freqs(
            head_dim, num_heads, rotate=True, dtype=dtype, device=device
        )
        freqs_no_rotate = init_2d_freqs(
            head_dim, num_heads, rotate=False, dtype=dtype, device=device
        )

        assert not torch.allclose(freqs_rotate, freqs_no_rotate)


@pytest.mark.cuda_if_available
class TestInitNDFreqs:
    @pytest.mark.cuda_if_available
    def test_init_nd_freqs_shape(self, device, dtype):
        position_dim = 3
        head_dim = 64
        num_heads = 8
        freqs = init_nd_freqs(position_dim, head_dim, num_heads, dtype=dtype, device=device)

        assert freqs.shape == (num_heads, head_dim // 2, position_dim)
        assert freqs.dtype == dtype
        assert freqs.device.type == device


    @pytest.mark.cuda_if_available
    def test_init_nd_freqs_multiple_thetas(self, device, dtype):
        position_dim = 3
        head_dim = 64
        num_heads = 8

        # Test with varied thetas per dimension
        thetas = [10.0, 20.0, 30.0]
        freqs = init_nd_freqs(
            position_dim, head_dim, num_heads, thetas=thetas, dtype=dtype, device=device
        )

        assert freqs.shape == (num_heads, head_dim // 2, position_dim)


@pytest.mark.cuda_if_available
class TestRopeEncodingND:
    def test_rope_encoding_nd_init(self, device, dtype):
        position_dim = 2
        embed_dim = 256
        n_heads = 8

        rope = RoPEEncodingND(position_dim, embed_dim, n_heads, dtype=dtype).to(device)

        assert rope.pos_dim == position_dim
        assert rope.embed_dim == embed_dim
        assert rope.n_heads == n_heads
        assert rope.head_dim == embed_dim // n_heads
        assert rope.freqs.shape == (n_heads, rope.head_dim // 2, position_dim)


    def test_rope_encoding_nd_forward(self, device, dtype):
        position_dim = 2
        embed_dim = 256
        n_heads = 8
        batch_size = 2
        seq_len = 10

        rope = RoPEEncodingND(position_dim, embed_dim, n_heads, dtype=dtype).to(device)

        # Test with query only
        query = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
        query_pos = (
            torch.rand(batch_size, seq_len, position_dim, dtype=dtype, device=device) * 10
        )

        query_rotated = rope(query, query_pos)
        assert query_rotated.shape == query.shape

        # Test with query and key
        key = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
        key_pos = (
            torch.rand(batch_size, seq_len, position_dim, dtype=dtype, device=device) * 10
        )

        query_rotated, key_rotated = rope(query, query_pos, key, key_pos)
        assert query_rotated.shape == query.shape
        assert key_rotated.shape == key.shape


    def test_rope_encoding_nd_normalized_warning(self, device):
        position_dim = 2
        embed_dim = 256
        n_heads = 8
        batch_size = 2
        seq_len = 10

        rope = RoPEEncodingND(position_dim, embed_dim, n_heads).to(device)

        # Create normalized positions (between 0 and 1)
        query = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # Values between 0 and 1
        query_pos = torch.rand(batch_size, seq_len, position_dim, device=device)

        # Verify warning is raised
        with pytest.warns(UserWarning, match="potentially normalized coordinates"):
            rope(query, query_pos)


    def test_rope_encoding_nd_validation_errors(self, device):
        position_dim = 2
        embed_dim = 256
        n_heads = 8

        rope = RoPEEncodingND(position_dim, embed_dim, n_heads).to(device)

        # Test wrong embed_dim
        query_wrong_dim = torch.randn(2, 10, embed_dim + 32, device=device)
        query_pos = torch.randn(2, 10, position_dim, device=device) * 10

        with pytest.raises(
            ValueError, match="Expected query_or_key to have last dim equal to embed_dim"
        ):
            rope(query_wrong_dim, query_pos)

        # Test wrong position_dim
        query = torch.randn(2, 10, embed_dim, device=device)
        query_pos_wrong_dim = torch.randn(2, 10, position_dim + 1, device=device) * 10

        with pytest.raises(
            ValueError, match="Expected query_or_key_pos to have last dim equal to pos_dim"
        ):
            rope(query, query_pos_wrong_dim)

        # Test mismatched shapes
        query = torch.randn(2, 10, embed_dim, device=device)
        query_pos_wrong_shape = (
            torch.randn(2, 15, position_dim, device=device) * 10
        )  # Different sequence length

        with pytest.raises(
            ValueError,
            match="Expected query_or_key and query_or_key_pos to have matching leading dims",
        ):
            rope(query, query_pos_wrong_shape)


# Tests for RoPEEncodingNDGroupedFreqs
@pytest.mark.cuda_if_available
class TestRoPEEncodingNDGroupedFreqs:
    def test_rope_encoding_nd_grouped_freqs_init(self):
        position_dim = 3
        embed_dim = 256
        n_heads = 8
        pos_dim_to_rope_group = [0, 1, 0]  # 2 groups

        rope = RoPEEncodingNDGroupedFreqs(
            position_dim, embed_dim, n_heads, pos_dim_to_rope_group
        )

        assert rope.pos_dim == position_dim
        assert rope.embed_dim == embed_dim
        assert rope.n_heads == n_heads
        assert rope.n_freq_groups == 2

        # Check freqs shape (adjusted for grouped frequencies)
        expected_shape = (n_heads, rope.head_dim // 2 // rope.n_freq_groups, position_dim)
        assert rope.freqs.shape == expected_shape


    @pytest.mark.cuda_if_available
    def test_rope_encoding_nd_grouped_freqs_forward(self, device, dtype):
        position_dim = 3
        embed_dim = 256
        n_heads = 8
        batch_size = 2
        seq_len = 10
        pos_dim_to_rope_group = [0, 1, 0]  # 2 groups

        rope = RoPEEncodingNDGroupedFreqs(
            position_dim, embed_dim, n_heads, pos_dim_to_rope_group, dtype=dtype
        ).to(device)

        query = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
        query_pos = (
            torch.randn(batch_size, seq_len, position_dim, dtype=dtype, device=device) * 10
        )

        query_rotated = rope(query, query_pos)
        assert query_rotated.shape == query.shape


# Tests for prep_multilevel_positions
@pytest.mark.cuda_if_available
class TestPrepMultilevelPositions:
    def test_prep_multilevel_positions(self, device):
        # Sample batch of indices (batch, i, j, level)
        bijl_indices = torch.tensor(
            [
                [0, 10, 20, 0],  # batch 0, level 0
                [0, 5, 15, 1],  # batch 0, level 1
                [1, 8, 12, 0],  # batch 1, level 0
                [1, 3, 7, 1],  # batch 1, level 1
            ],
            dtype=torch.long,
            device=device
        )

        # Spatial shapes (level, 2) - height and width for each level
        spatial_shapes = torch.tensor(
            [
                [100, 100],  # level 0: 100x100
                [50, 50],  # level 1: 50x50
            ],
            dtype=torch.float,
            device=device
        )

        positions = prep_multilevel_positions(bijl_indices, spatial_shapes)

        assert positions.shape == bijl_indices.shape
        assert torch.is_floating_point(positions)

        # Verify scaling for a level 1 position (should be scaled up relative to level 0)
        # Since level 1 is half the size, its coordinates get doubled in the common space
        scale_factor = 100 / 50  # max_shape / level_shape
        expected_i = (5 + 0.5) * scale_factor  # +0.5 for pixel center
        assert torch.isclose(positions[1, 1], torch.tensor(expected_i, dtype=torch.float, device=device))


    def test_prep_multilevel_positions_batched_shapes(self, device):
        # Sample indices
        bijl_indices = torch.tensor(
            [
                [0, 10, 20, 0],  # batch 0, level 0
                [1, 5, 15, 1],  # batch 1, level 1
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
            device=device
        )

        positions = prep_multilevel_positions(bijl_indices, spatial_shapes)
        assert positions.shape == bijl_indices.shape
