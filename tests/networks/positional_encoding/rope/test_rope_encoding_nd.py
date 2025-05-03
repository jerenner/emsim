from typing import Any, Union, Optional

import pytest
import torch
from torch import Tensor
import numpy as np
from math import isclose
from hypothesis import given, settings, example, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, array_shapes

from emsim.networks.positional_encoding.rope import (
    RoPEEncodingND,
    get_multilevel_freq_group_pattern,
)

from emsim.utils.sparse_utils.ops.subset_attn.rotary_encoding import (
    calculate_rope,
    rotate_embeddings,
)
from tests.networks.conftest import positions_strategy


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Base configuration for RoPEEncodingND tests."""
    return {
        "position_dim": 2,
        "embed_dim": 256,
        "n_heads": 8,
        "dtype": torch.float32,
    }


@pytest.fixture
def sample_data(base_config: dict[str, Any], device: str):
    """Sample input data for testing forward passes."""
    batch_size = 2
    seq_len = 10

    return {
        "query": torch.randn(
            batch_size,
            seq_len,
            base_config["embed_dim"],
            dtype=base_config["dtype"],
            device=device,
        ),
        "query_pos": torch.rand(
            batch_size,
            seq_len,
            base_config["position_dim"],
            dtype=base_config["dtype"],
            device=device,
        )
        * 10,  # Unnormalized positions
        "key": torch.randn(
            batch_size,
            seq_len,
            base_config["embed_dim"],
            dtype=base_config["dtype"],
            device=device,
        ),
        "key_pos": torch.rand(
            batch_size,
            seq_len,
            base_config["position_dim"],
            dtype=base_config["dtype"],
            device=device,
        )
        * 10,  # Unnormalized positions
    }


@st.composite
def rope_config_strategy(draw):
    position_dim = draw(st.integers(1, 4))
    n_heads = draw(st.integers(1, 8))
    embed_dim = draw(st.integers(1, 8)) * 2 * n_heads
    share_heads = draw(st.booleans())
    n_freq_groups = draw(st.integers(1, 4))
    freq_group_pattern = draw(
        arrays(
            np.bool,
            shape=(n_freq_groups, position_dim),
            elements=st.booleans(),
            fill=None,
        ).filter(lambda x: x.any())
    )
    enforce_freq_groups_equal = (embed_dim // n_heads) % (n_freq_groups * 2) == 0
    rope_base_theta = draw(
        st.one_of(
            st.floats(min_value=1.0, max_value=1e10, exclude_min=True),
            arrays(
                np.float32,
                shape=(
                    draw(st.sampled_from([1, n_freq_groups])),
                    draw(st.sampled_from([1, position_dim])),
                ),
                elements=st.floats(min_value=1.0, max_value=1e10, exclude_min=True),
                fill=None,
            ),
        )
    )
    dtype = draw(st.just(torch.float32))

    # Tensor creation config
    # Embedding value range
    embedding_min_value = draw(st.floats(-1e15, 1e15, exclude_max=True))
    embedding_max_value = draw(st.floats(embedding_min_value, 1e15))

    # shared helper function for position grid shape
    def _position_grid_shape() -> tuple[int, ...]:
        return draw(
            array_shapes(
                min_dims=position_dim, max_dims=position_dim, min_side=2, max_side=8
            )
        )

    # Query positions: Random or use grid
    use_position_grid_query = draw(st.booleans())
    if use_position_grid_query:
        position_dtype_query = draw(st.none())
        position_min_query = draw(st.none())
        position_max_query = draw(st.none())
        # Need to have at least position_dim dims before embed_dim when using
        # position grid
        position_shape_query = _position_grid_shape()
        batch_shape_query = draw(array_shapes(min_side=0)) + position_shape_query
    else:
        position_dtype_query, position_min_query, position_max_query = draw(
            positions_strategy().filter(lambda drawn: drawn[1] <= 0 or drawn[2] >= 1)
        )
        batch_shape_query = draw(array_shapes(min_side=0))

    # Decide whether to include separate key embeddings
    include_key = draw(st.booleans())
    if include_key:
        # Use separate position tensor for key or share query positions
        include_key_pos = draw(st.booleans())
        if include_key_pos:
            # For generality, allow key tensor to have different shape from query
            # Key positions: random or use grid
            use_position_grid_key = draw(st.booleans())
            if use_position_grid_key:
                position_dtype_key = draw(st.none())
                position_min_key = draw(st.none())
                position_max_key = draw(st.none())
                # Key tensor needs position_dim dims before embed_dim, like query
                position_shape_key = _position_grid_shape()
                batch_shape_key = draw(array_shapes(min_side=0)) + position_shape_key
            else:  # non-grid positions
                position_dtype_key, position_min_key, position_max_key = draw(
                    positions_strategy().filter(
                        lambda drawn: drawn[1] <= 0 or drawn[2] >= 1
                    )
                )
                batch_shape_key = draw(array_shapes(min_side=0))
        else:  # key_pos=None (reuse query positions)
            batch_shape_key = draw(st.just(batch_shape_query))
            use_position_grid_key = draw(st.just(False))
            position_dtype_key = draw(st.none())
            position_min_key = draw(st.none())
            position_max_key = draw(st.none())
    else:
        include_key_pos = draw(st.just(False))
        use_position_grid_key = draw(st.just(False))
        batch_shape_key = draw(st.none())
        position_dtype_key = draw(st.none())
        position_min_key = draw(st.none())
        position_max_key = draw(st.none())

    seed = draw(st.integers(min_value=0, max_value=1e10))

    return {
        "config": {
            "position_dim": position_dim,
            "embed_dim": embed_dim,
            "n_heads": n_heads,
            "share_heads": share_heads,
            "freq_group_pattern": freq_group_pattern,
            "enforce_freq_groups_equal": enforce_freq_groups_equal,
            "rope_base_theta": rope_base_theta,
            "dtype": dtype,
        },
        "tensor_config": {
            "embed_dim": embed_dim,
            "embedding_min_value": embedding_min_value,
            "embedding_max_value": embedding_max_value,
            "position_dim": position_dim,
            "batch_shape_query": batch_shape_query,
            "use_position_grid_query": use_position_grid_query,
            "position_dtype_query": position_dtype_query,
            "position_min_query": position_min_query,
            "position_max_query": position_max_query,
            "include_key": include_key,
            "batch_shape_key": batch_shape_key,
            "include_key_pos": include_key_pos,
            "use_position_grid_key": use_position_grid_key,
            "position_dtype_key": position_dtype_key,
            "position_min_key": position_min_key,
            "position_max_key": position_max_key,
            "seed": seed,
        },
    }


def rope_input_tensors(
    embed_dim: int,
    embedding_min_value: float,
    embedding_max_value: float,
    position_dim: int,
    batch_shape_query: tuple[int, ...],
    use_position_grid_query: bool,
    position_dtype_query: Optional[torch.dtype],
    position_min_query: Optional[Union[int, float]],
    position_max_query: Optional[Union[int, float]],
    include_key: bool,
    batch_shape_key: Optional[tuple[int, ...]],
    include_key_pos: bool,
    use_position_grid_key: Optional[bool],
    position_dtype_key: Optional[torch.dtype],
    position_min_key: Optional[Union[int, float]],
    position_max_key: Optional[Union[int, float]],
    seed: int,
    device: str,
    force_unnormalized_positions: bool = True,
):
    device = torch.device(device)

    # save rng state and set seed
    if device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(device)
    else:
        rng_state = torch.get_rng_state(device)
    torch.manual_seed(seed)

    # Helper function for creating non-grid position tensor
    def _random_positions(
        batch_shape: tuple[int, ...],
        dtype: torch.dtype,
        pos_min: Union[int, float],
        pos_max: Union[int, float],
    ):
        pos_tensor = torch.empty(
            batch_shape + (position_dim,),
            device=device,
            dtype=dtype,
        )
        if dtype == torch.float32:
            pos_tensor.uniform_(pos_min, pos_max)
        else:
            pos_tensor.random_(pos_min, pos_max + 1)
        if force_unnormalized_positions and pos_tensor.numel() > 0:
            while pos_tensor.min() > 0.0 and pos_tensor.max() <= 1.0:
                pos_tensor = pos_tensor + 1
        return pos_tensor

    # Create query embeddings and position tensor
    query = torch.empty(batch_shape_query + (embed_dim,), device=device).uniform_(
        embedding_min_value, embedding_max_value
    )
    if use_position_grid_query:
        query_pos = RoPEEncodingND.position_grid(
            query.shape, query.ndim - position_dim - 1, device=device
        )
        assert query_pos.size(-1) == position_dim
    else:
        query_pos = _random_positions(
            batch_shape_query,
            position_dtype_query,
            position_min_query,
            position_max_query,
        )

    # Create key embeddings and position tensor if needed
    if include_key:
        key = torch.empty(batch_shape_key + (embed_dim,), device=device).uniform_(
            embedding_min_value,
            embedding_max_value,
        )
        if include_key_pos:
            if use_position_grid_key:
                key_pos = RoPEEncodingND.position_grid(
                    key.shape, key.ndim - position_dim - 1, device=device
                )
                assert key_pos.size(-1) == position_dim
            else:
                key_pos = _random_positions(
                    batch_shape_key,
                    position_dtype_key,
                    position_min_key,
                    position_max_key,
                )
        else:
            key_pos = None
    else:
        key = None
        key_pos = None

    # reset rng state
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng_state, device)
    else:
        torch.set_rng_state(rng_state)

    return {
        "query": query,
        "query_pos": query_pos,
        "key": key,
        "key_pos": key_pos,
    }


def allclose_zero(tensor: Tensor):
    return torch.allclose(tensor, torch.zeros_like(tensor))


@pytest.mark.cuda_if_available
class TestRoPEEncodingNDInitialization:
    """Tests for RoPEEncodingND initialization with various configurations."""

    def test_basic_initialization(self, base_config: dict[str, Any], device: str):
        """Test basic initialization with default parameters."""
        rope = RoPEEncodingND(**base_config).to(device)

        assert rope.position_dim == base_config["position_dim"]
        assert rope.embed_dim == base_config["embed_dim"]
        assert rope.n_heads == base_config["n_heads"]
        assert rope.head_dim == base_config["embed_dim"] // base_config["n_heads"]
        assert rope.dtype == base_config["dtype"]
        assert len(rope.freqs) == 1  # Default is 1 frequency group
        assert rope.freqs[0].shape == (
            base_config["position_dim"],
            base_config["n_heads"],
            rope.head_dim // 2,
        )
        assert rope.freq_group_pattern.shape == (1, base_config["position_dim"])
        assert torch.all(rope.freq_group_pattern)  # All True

    @pytest.mark.parametrize("position_dim", [1, 2, 3, 4])
    def test_different_dimensions(
        self, position_dim, base_config: dict[str, Any], device: str
    ):
        """Test initialization with different position dimensions."""
        rope = RoPEEncodingND(
            position_dim=position_dim,
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        assert rope.position_dim == position_dim
        assert rope.freq_group_pattern.shape == (1, position_dim)
        assert rope.freqs[0].shape == (
            position_dim,
            base_config["n_heads"],
            rope.head_dim // 2,
        )

    @pytest.mark.parametrize("rope_base_theta", [10.0, 100.0, 1000.0])
    def test_different_theta(
        self, rope_base_theta, base_config: dict[str, Any], device: str
    ):
        """Test initialization with different theta values."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            rope_base_theta=rope_base_theta,
            dtype=base_config["dtype"],
        ).to(device)

        # Check that the base theta is stored
        assert isclose(float(rope._base_theta), rope_base_theta)

        # Generate two models with different theta
        if rope_base_theta == 10.0:  # Only need to compare once
            rope2 = RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["embed_dim"],
                n_heads=base_config["n_heads"],
                rope_base_theta=rope_base_theta * 10,  # Different theta
                dtype=base_config["dtype"],
            ).to(device)

            # Compare frequencies - they should be different
            assert not torch.allclose(rope.freqs[0], rope2.freqs[0])

    def test_custom_frequency_pattern(self, base_config: dict[str, Any], device: str):
        """Test initialization with custom frequency group patterns."""
        position_dim = 3  # Use 3 dimensions for this test

        # Create a custom pattern with 2 frequency groups
        freq_group_pattern = torch.tensor(
            [
                [True, True, False],  # Group 1: dims 0,1
                [False, False, True],  # Group 2: dim 2
            ],
            device=device,
        )

        rope = RoPEEncodingND(
            position_dim=position_dim,
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            freq_group_pattern=freq_group_pattern,
            dtype=base_config["dtype"],
        ).to(device)

        # Verify the pattern was stored correctly
        assert torch.equal(rope.freq_group_pattern, freq_group_pattern)

        # Verify we have 2 frequency groups
        assert len(rope.freqs) == 2

        # First group should have 2 dimensions, second group should have 1
        assert rope.freqs[0].shape[0] == 2
        assert rope.freqs[1].shape[0] == 1

        # Verify encoding ranges
        head_dim_half = rope.head_dim // 2
        assert torch.equal(
            rope.encoding_ranges,
            torch.tensor(
                [[0, head_dim_half // 2], [head_dim_half // 2, head_dim_half]],
                device=device,
            ),
        )

    def test_initialization_errors(self, base_config: dict[str, Any], device: str):
        """Test initialization with invalid parameters triggers appropriate errors."""
        # Test odd embed_dim
        with pytest.raises(
            ValueError, match="Expected embed_dim to be divisible by n_heads"
        ):
            RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["embed_dim"] + 1,  # Odd number
                n_heads=base_config["n_heads"],
            ).to(device)

        # Test odd head_dim (embed_dim / n_heads)
        with pytest.raises(ValueError, match="Expected head_dim to be divisible by 2"):
            RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["n_heads"] * 3,  # Makes head_dim odd
                n_heads=base_config["n_heads"],
            ).to(device)

        # Test invalid frequency group pattern
        with pytest.raises(
            ValueError, match="Expected 2D tensor for freq_group_pattern"
        ):
            RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["embed_dim"],
                n_heads=base_config["n_heads"],
                freq_group_pattern=torch.ones(3, device=device),  # 1D tensor
            ).to(device)

        # Test mismatched position_dim and freq_group_pattern
        with pytest.raises(
            ValueError, match="Expected second dimension of freq_group_pattern"
        ):
            RoPEEncodingND(
                position_dim=base_config["position_dim"],
                embed_dim=base_config["embed_dim"],
                n_heads=base_config["n_heads"],
                freq_group_pattern=torch.ones(
                    1, base_config["position_dim"] + 1, device=device
                ),
            ).to(device)

    @settings(deadline=None)
    @given(inputs=rope_config_strategy())
    def test_initialization_hypothesis(self, inputs: dict[str, Any], device: str):
        config = inputs["config"]
        rope = RoPEEncodingND(**config).to(device)

        assert rope.position_dim == config["position_dim"]
        assert rope.embed_dim == config["embed_dim"]
        assert rope.n_heads == config["n_heads"]
        assert rope.head_dim == config["embed_dim"] // config["n_heads"]
        assert rope.dtype == config["dtype"]
        n_freq_groups = config["freq_group_pattern"].shape[0]
        assert config["freq_group_pattern"].shape[1] == config["position_dim"]
        assert len(rope.freqs) == n_freq_groups
        for i, freq in enumerate(rope.freqs):
            if config["enforce_freq_groups_equal"]:
                assert freq.shape == (
                    config["freq_group_pattern"][i].sum(),
                    config["n_heads"] if not config["share_heads"] else 1,
                    rope.head_dim // n_freq_groups // 2,
                )
            else:
                assert freq.shape[:-1] == (
                    config["freq_group_pattern"][i].sum(),
                    config["n_heads"] if not config["share_heads"] else 1,
                )


@pytest.mark.cuda_if_available
class TestRoPEEncodingNDForward:
    """Tests for the forward method with different input configurations."""

    def test_forward_query_only(
        self, base_config: dict[str, Any], sample_data, device: str
    ):
        """Test forward pass with only query tensor."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        query_rotated = rope(sample_data["query"], sample_data["query_pos"])

        # Verify shape is preserved
        assert query_rotated.shape == sample_data["query"].shape

        # Verify output is different from input
        assert not torch.allclose(query_rotated, sample_data["query"])

    @settings(deadline=None)
    @given(inputs=rope_config_strategy())
    def test_forward_hypothesis(self, inputs: dict[str, Any], device: str):
        """Test forward pass with Hypothesis-generated configs"""
        rope = RoPEEncodingND(**inputs["config"]).to(device)
        input_tensors = rope_input_tensors(**inputs["tensor_config"], device=device)

        output = rope(**input_tensors)
        if input_tensors["key"] is not None:
            query_rotated, key_rotated = output
        else:
            query_rotated = output

        # Test that rotation happened
        query = input_tensors["query"]
        query_pos = input_tensors["query_pos"]
        key = input_tensors["key"]
        if (
            not allclose_zero(query)
            and not allclose_zero(query_pos)
            and rope.freq_group_pattern.any()
        ):
            assert not torch.allclose(query, query_rotated)
        else:
            assert torch.allclose(query, query_rotated)
        if key is not None:
            key_pos = (
                input_tensors["key_pos"]
                if input_tensors["key_pos"] is not None
                else query_pos
            )
            if (
                not allclose_zero(key)
                and not allclose_zero(key_pos)
                and rope.freq_group_pattern.any()
            ):
                assert not torch.allclose(key, key_rotated)
            else:
                assert torch.allclose(key, key_rotated)

    def test_forward_query_and_key(
        self, base_config: dict[str, Any], sample_data, device: str
    ):
        """Test forward pass with both query and key tensors."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        query_rotated, key_rotated = rope(
            sample_data["query"],
            sample_data["query_pos"],
            sample_data["key"],
            sample_data["key_pos"],
        )

        # Verify shapes are preserved
        assert query_rotated.shape == sample_data["query"].shape
        assert key_rotated.shape == sample_data["key"].shape

        # Verify outputs are different from inputs
        assert not torch.allclose(query_rotated, sample_data["query"])
        assert not torch.allclose(key_rotated, sample_data["key"])

    @pytest.mark.parametrize("share_heads", [True, False], ids=["shared", "not_shared"])
    def test_share_heads(
        self, share_heads: bool, base_config: dict[str, Any], device: str
    ):
        """Test initialization with head sharing enabled and disabled."""
        n_heads = base_config["n_heads"]
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=n_heads,
            share_heads=share_heads,
            dtype=base_config["dtype"],
        ).to(device)

        assert rope.share_heads == share_heads

        expected_n_heads = 1 if share_heads else n_heads
        assert rope.freqs[0].shape[1] == expected_n_heads

        # Create query with same embedding for each head
        head_dim = base_config["embed_dim"] // n_heads
        query = torch.randn(1, 1, 1, head_dim, device=device)
        query = query.expand(-1, -1, n_heads, -1).reshape(1, 1, -1)
        for i in range(n_heads):
            assert torch.equal(
                query[..., :head_dim], query[..., i * head_dim : (i + 1) * head_dim]
            )

        query_pos = torch.ones(1, 1, base_config["position_dim"], device=device) * 5

        # Process the query
        query_rotated = rope(query, query_pos)

        # Reshape to separate heads
        query_rotated_heads = query_rotated.reshape(1, 1, n_heads, head_dim)

        # Check if all heads have the same rotation if heads shared
        for i in range(1, n_heads):
            allclose = torch.allclose(
                query_rotated_heads[0, 0, 0], query_rotated_heads[0, 0, i]
            )
            if share_heads:
                assert allclose
            else:
                assert not allclose

    def test_forward_key_without_pos(
        self, base_config: dict[str, Any], sample_data, device: str
    ):
        """Test forward pass with key but no key positions (should use query positions)."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        # Call without key_pos
        query_rotated, key_rotated_1 = rope(
            sample_data["query"], sample_data["query_pos"], sample_data["key"]
        )

        # Call with key_pos = query_pos explicitly
        query_rotated_2, key_rotated_2 = rope(
            sample_data["query"],
            sample_data["query_pos"],
            sample_data["key"],
            sample_data["query_pos"],
        )

        # Results should be identical
        assert torch.allclose(key_rotated_1, key_rotated_2)

    def test_warning_for_normalized_positions(
        self, base_config: dict[str, Any], device: str
    ):
        """Test warning is raised for potentially normalized coordinates."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        batch_size, seq_len = 2, 10
        query = torch.randn(
            batch_size,
            seq_len,
            base_config["embed_dim"],
            dtype=base_config["dtype"],
            device=device,
        )

        # Create positions in [0, 1] range (normalized)
        query_pos = torch.rand(
            batch_size,
            seq_len,
            base_config["position_dim"],
            dtype=base_config["dtype"],
            device=device,
        )

        with pytest.warns(UserWarning, match="potentially normalized coordinates"):
            rope(query, query_pos)

    @pytest.mark.parametrize(
        "batch_shape",
        [
            (2,),  # Batch dim only
            (2, 10),  # Batch and sequence dims
            (2, 3, 4),  # Batch, width, height
            (2, 3, 4, 5),  # 4D batch shape
        ],
    )
    def test_different_batch_shapes(
        self, batch_shape, base_config: dict[str, Any], device: str
    ):
        """Test forward pass with different batch shapes."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        # Create query and positions with the specified batch shape
        query = torch.randn(
            *batch_shape,
            base_config["embed_dim"],
            dtype=base_config["dtype"],
            device=device,
        )
        query_pos = (
            torch.rand(
                *batch_shape,
                base_config["position_dim"],
                dtype=base_config["dtype"],
                device=device,
            )
            * 10
        )

        # Process
        query_rotated = rope(query, query_pos)

        # Verify shape is preserved
        assert query_rotated.shape == query.shape


@pytest.mark.cuda_if_available
class TestRoPEEncodingNDFrequencyGroups:
    """Tests specifically for frequency group functionality."""

    @pytest.mark.parametrize(
        "n_freq_groups,enforce_equal",
        [
            (1, True),  # Single group (default)
            (2, True),  # Two equal groups
            (4, True),  # Four equal groups
            (3, False),  # Three groups, may not be equal
        ],
    )
    def test_freq_group_dimension_distribution(
        self, n_freq_groups, enforce_equal, base_config: dict[str, Any], device: str
    ):
        """Test different frequency group configurations and dimension distribution."""
        # Create pattern with one position dimension per group, cycling if needed
        position_dim = max(base_config["position_dim"], n_freq_groups)
        freq_group_pattern = torch.zeros(
            n_freq_groups, position_dim, dtype=torch.bool, device=device
        )
        for g in range(n_freq_groups):
            freq_group_pattern[g, g % position_dim] = True

        # Adjust embed_dim to ensure it's cleanly divisible when enforce_equal=True
        embed_dim = base_config["embed_dim"]
        head_dim = embed_dim // base_config["n_heads"]

        if enforce_equal and (head_dim // 2) % n_freq_groups != 0:
            # Adjust embed_dim to make head_dim/2 divisible by n_freq_groups
            head_dim = ((head_dim // 2) // n_freq_groups) * n_freq_groups * 2
            if head_dim == 0:
                head_dim = n_freq_groups * 2  # minimum valid head_dim
            embed_dim = head_dim * base_config["n_heads"]

        # Initialize RoPE
        rope = RoPEEncodingND(
            position_dim=position_dim,
            embed_dim=embed_dim,
            n_heads=base_config["n_heads"],
            freq_group_pattern=freq_group_pattern,
            enforce_freq_groups_equal=enforce_equal,
            dtype=base_config["dtype"],
        ).to(device)

        # Verify the number of frequency groups
        assert len(rope.freqs) == n_freq_groups

        # Check encoding dimensions
        half_head_dim = rope.head_dim // 2

        if enforce_equal:
            # All groups should have the same number of encoding dimensions
            expected_dims_per_group = half_head_dim // n_freq_groups
            for freq in rope.freqs:
                assert freq.shape[2] == expected_dims_per_group
        else:
            # Sum of dimensions should equal half_head_dim
            total_dims = sum(freq.shape[2] for freq in rope.freqs)
            assert total_dims == half_head_dim

            # If not enforcing equality, earlier groups may have more dimensions
            if n_freq_groups > 1 and half_head_dim % n_freq_groups != 0:
                # First group should have more dimensions than last
                assert rope.freqs[0].shape[2] >= rope.freqs[-1].shape[2]

        # Verify encoding ranges
        assert rope.encoding_ranges.shape == (n_freq_groups, 2)
        assert (
            rope.encoding_ranges[-1, 1] == half_head_dim
        )  # Last end should be half_head_dim

        # Check that ranges are contiguous
        for i in range(n_freq_groups - 1):
            assert rope.encoding_ranges[i, 1] == rope.encoding_ranges[i + 1, 0]

    def test_grouped_rope_freqs_tensor(self, base_config: dict[str, Any], device: str):
        """Test the grouped_rope_freqs_tensor method."""
        position_dim = 3
        n_freq_groups = 2

        # Create a custom pattern
        freq_group_pattern = torch.tensor(
            [
                [True, True, False],  # Group 1: dims 0,1
                [False, False, True],  # Group 2: dim 2
            ],
            device=device,
        )

        rope = RoPEEncodingND(
            position_dim=position_dim,
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            freq_group_pattern=freq_group_pattern,
            dtype=base_config["dtype"],
        ).to(device)

        # Get the frequency tensor
        freq_tensor = rope.grouped_rope_freqs_tensor(rope.freqs)

        # Check shape
        expected_shape = (
            position_dim,
            n_freq_groups,
            base_config["n_heads"],
            rope.head_dim // 2,
        )
        assert freq_tensor.shape == expected_shape

        # Check zeros in appropriate places (where freq_group_pattern is False)
        for dim in range(position_dim):
            for group in range(n_freq_groups):
                if not freq_group_pattern[group, dim]:
                    assert torch.all(freq_tensor[dim, group] == 0)

    def test_grouped_rope_freqs_tensor_implementation_equivalence(
        self, base_config: dict[str, Any], device: str
    ):
        """Test that current grouped_rope_freqs_tensor implementation matches the old
        version that uses direct indexing."""
        # Setup test parameters with mixed frequency groups for a thorough test
        position_dim = 3

        # Create a custom pattern where each group handles different position dimensions
        freq_group_pattern = torch.tensor(
            [
                [True, True, False],  # Group 1: dims 0,1
                [False, False, True],  # Group 2: dim 2
            ],
            device=device,
        )

        rope = RoPEEncodingND(
            position_dim=position_dim,
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            freq_group_pattern=freq_group_pattern,
            dtype=base_config["dtype"],
        ).to(device)

        # Define the old implementation
        def old_grouped_rope_freqs_tensor(rope_instance, grouped_rope_freqs):
            if isinstance(grouped_rope_freqs, torch.Tensor):
                grouped_rope_freqs = [grouped_rope_freqs]

            n_heads = rope_instance.n_heads if not rope_instance.share_heads else 1
            rope_freqs = grouped_rope_freqs[0].new_zeros(
                rope_instance.n_freq_groups,
                rope_instance.position_dim,
                n_heads,
                rope_instance.head_dim // 2,
            )

            freq_group_pattern = rope_instance.freq_group_pattern
            for g, (freqs_g, range_g) in enumerate(
                zip(grouped_rope_freqs, rope_instance.encoding_ranges)
            ):
                range_start, range_end = range_g
                rope_freqs[g, freq_group_pattern[g], :, range_start:range_end] = freqs_g

            # Transpose to output shape
            rope_freqs = rope_freqs.transpose(0, 1).contiguous()
            return rope_freqs

        # Get results from both implementations
        current_result = rope.grouped_rope_freqs_tensor(rope.freqs)
        old_result = old_grouped_rope_freqs_tensor(rope, rope.freqs)

        # Compare results
        assert current_result.shape == old_result.shape, (
            f"Shape mismatch: current {current_result.shape}, "
            f"old {old_result.shape}"
        )

        assert torch.allclose(current_result, old_result), (
            "Values differ between implementations. Max diff: "
            f"{(current_result - old_result).abs().max()}"
        )

        # Test with both standard and shared heads configurations
        rope_shared = RoPEEncodingND(
            position_dim=position_dim,
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            freq_group_pattern=freq_group_pattern,
            share_heads=True,
            dtype=base_config["dtype"],
        ).to(device)

        current_result_shared = rope_shared.grouped_rope_freqs_tensor(rope_shared.freqs)
        old_result_shared = old_grouped_rope_freqs_tensor(
            rope_shared, rope_shared.freqs
        )

        assert torch.allclose(
            current_result_shared, old_result_shared
        ), "Implementations produce different results with shared heads"

    @settings(deadline=None)
    @given(inputs=rope_config_strategy())
    def test_grouped_rope_freqs_tensor_hypothesis(
        self, inputs: dict[str, Any], device: str
    ):
        """Test structural properties of the grouped_rope_freqs_tensor method."""
        rope = RoPEEncodingND(**inputs["config"]).to(device)

        # Get the full frequency tensor
        freq_tensor = rope.grouped_rope_freqs_tensor(rope.freqs)

        # Verify shape
        n_heads = 1 if rope.share_heads else rope.n_heads
        expected_shape = (
            rope.position_dim,
            rope.n_freq_groups,
            n_heads,
            rope.head_dim // 2,
        )
        assert freq_tensor.shape == expected_shape

        # Verify placement of frequencies
        for g, freq_g in enumerate(rope.freqs):
            range_start, range_end = rope.encoding_ranges[g]
            pos_dims = torch.nonzero(rope.freq_group_pattern[g], as_tuple=True)[0]

            for i, pos_dim in enumerate(pos_dims):
                extracted = freq_tensor[pos_dim, g, :, range_start:range_end]
                expected = freq_g[i, :, :]
                assert torch.allclose(extracted, expected)

        # Verify zeros where expected
        for g in range(rope.n_freq_groups):
            for pos_dim in range(rope.position_dim):
                if not rope.freq_group_pattern[g, pos_dim]:
                    # This position dimension should be zero in this frequency group
                    assert torch.all(freq_tensor[pos_dim, g] == 0)

        # Verify encoding ranges are contiguous and complete
        assert rope.encoding_ranges[0, 0] == 0
        assert rope.encoding_ranges[-1, 1] == rope.head_dim // 2
        for i in range(len(rope.encoding_ranges) - 1):
            assert rope.encoding_ranges[i, 1] == rope.encoding_ranges[i + 1, 0]

        # Verify encoding dimension distribution
        if rope.n_freq_groups > 1:
            encoding_sizes = [
                (end - start).item() for start, end in rope.encoding_ranges
            ]
            if rope.enforce_freq_groups_equal:
                # All groups should have equal encoding dimensions
                assert len(set(encoding_sizes)) == 1
            else:
                # For unequal distribution, the difference should be at most 1
                assert max(encoding_sizes) - min(encoding_sizes) <= 1

        # Verify all values are finite
        assert torch.all(torch.isfinite(freq_tensor))

        # Verify dtype and device match input
        assert freq_tensor.dtype == rope.freqs[0].dtype
        assert freq_tensor.device == rope.freqs[0].device

    @settings(deadline=None)
    @given(inputs=rope_config_strategy())
    def test_grouped_rope_freqs_tensor_hypothesis_math(
        self, inputs: dict[str, Any], device: str
    ):
        """Test mathematical properties of the grouped_rope_freqs_tensor method."""
        rope = RoPEEncodingND(**inputs["config"]).to(device)
        freq_tensor = rope.grouped_rope_freqs_tensor(rope.freqs)
        n_heads = 1 if rope.share_heads else rope.n_heads

        # Check magnitudes are monotonically decreasing
        if rope.freqs[0].numel() > 0:
            for g, ranges in enumerate(rope.encoding_ranges):
                range_size = ranges[1] - ranges[0]
                if range_size <= 1:
                    continue  # Skip if not enough dimensions to compare

                # Find any non-zero position dimension for this group
                pos_dims = torch.nonzero(rope.freq_group_pattern[g], as_tuple=True)[0]
                if len(pos_dims) == 0:
                    continue

                for pos_dim in pos_dims:
                    for head_idx in range(n_heads):
                        pos_dim = pos_dims[0]
                        head_idx = 0

                        # Extract frequencies and take absolute values
                        freqs = freq_tensor[pos_dim, g, head_idx, ranges[0] : ranges[1]]
                        magnitudes = torch.abs(freqs)

                        if magnitudes.numel() >= 3:
                            diffs = magnitudes[1:] - magnitudes[:-1]

                            # Should be monotonic
                            assert torch.all(diffs <= 0), (
                                "Expected consistent geometric progression "
                                "in frequency magnitudes"
                            )


@pytest.mark.cuda_if_available
class TestRoPEEncodingNDHelperMethods:
    """Tests for the helper methods of RoPEEncodingND."""

    def test_position_grid(self, device: str):
        """Test the position_grid static method."""
        # Test 2D grid
        shape_2d = (1, 3, 4, 256)  # batch, height, width, features
        grid_2d = RoPEEncodingND.position_grid(shape_2d, device=device)

        # Check shape: should be [height, width, 2]
        assert grid_2d.shape == (3, 4, 2)

        # Check values: grid should contain coordinates
        for h in range(3):
            for w in range(4):
                assert torch.equal(grid_2d[h, w], torch.tensor([h, w], device=device))

        # Test 3D grid
        shape_3d = (1, 2, 3, 4, 256)  # batch, depth, height, width, features
        grid_3d = RoPEEncodingND.position_grid(shape_3d, device=device)

        # Check shape: should be [depth, height, width, 3]
        assert grid_3d.shape == (2, 3, 4, 3)

        # Check values: grid should contain coordinates
        for d in range(2):
            for h in range(3):
                for w in range(4):
                    assert torch.equal(
                        grid_3d[d, h, w], torch.tensor([d, h, w], device=device)
                    )

    def test_calculate_rope(self, base_config: dict[str, Any], device: str):
        """Test the calculate_rope static method."""
        position_dim = base_config["position_dim"]
        n_heads = base_config["n_heads"]
        head_dim = base_config["embed_dim"] // n_heads

        # Create positions and frequencies
        positions = torch.rand(2, 10, position_dim, device=device) * 10
        rope_freqs = torch.rand(position_dim, 1, n_heads, head_dim // 2, device=device)

        # Calculate rope encodings
        rope_encodings = RoPEEncodingND.calculate_rope(positions, rope_freqs)

        # Check shape
        assert rope_encodings.shape == (2, 10, n_heads, head_dim // 2)

        # Verify it matches the imported function
        expected = calculate_rope(positions, rope_freqs)
        assert torch.allclose(rope_encodings, expected)

    def test_rotate_embeddings(self, base_config: dict[str, Any], device: str):
        """Test the rotate_embeddings static method."""
        n_heads = base_config["n_heads"]
        head_dim = base_config["embed_dim"] // n_heads

        # Create embeddings and encodings
        embeddings = torch.randn(2, 10, n_heads, head_dim, device=device)
        rope_encodings = torch.rand(2, 10, n_heads, head_dim // 2, device=device)

        # Rotate embeddings
        rotated = RoPEEncodingND.rotate_embeddings(embeddings, rope_encodings)

        # Check shape
        assert rotated.shape == embeddings.shape

        # Verify it matches the imported function
        expected = rotate_embeddings(embeddings, rope_encodings)
        assert torch.allclose(rotated, expected)

    def test_reset_parameters(self, base_config: dict[str, Any], device: str):
        """Test the reset_parameters method."""
        rope = RoPEEncodingND(
            position_dim=base_config["position_dim"],
            embed_dim=base_config["embed_dim"],
            n_heads=base_config["n_heads"],
            dtype=base_config["dtype"],
        ).to(device)

        # Store initial parameters
        initial_params = [param.clone() for param in rope.freqs]

        # Reset parameters
        rope.reset_parameters()

        # Check if parameters changed
        any_changed = False
        for old_param, new_param in zip(initial_params, rope.freqs):
            if not torch.allclose(old_param, new_param):
                any_changed = True
                break

        # Parameters should have changed (very unlikely they'd be identical)
        assert any_changed, "Expected parameters to change after reset"
