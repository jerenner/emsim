from typing import Optional, Union, Any

import random
import torch
from torch import Tensor, nn
import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given, settings, example, HealthCheck

from emsim.networks.sparse_ms_deform_attn.utils import (
    sparse_split_heads,
    multilevel_sparse_bilinear_grid_sample,
    _make_index_and_weight_tensors,
)
from emsim.networks.sparse_ms_deform_attn.layer import SparseMSDeformableAttention

from emsim.utils.sparse_utils.batching.batch_utils import (
    batch_offsets_to_indices,
    seq_lengths_to_indices,
)
from emsim.utils.sparse_utils.indexing.indexing import batch_sparse_index


from .conftest import (
    simple_sparse_input_tensors,
    random_multilevel_sparse_tensor_indices,
)


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Base configuration for SparseMSDeformableAttention"""
    return {
        "embed_dim": 64,
        "n_heads": 4,
        "n_levels": 3,
        "n_points": 4,
    }


@st.composite
def grid_sample_strategy(draw, require_grads: bool = False):
    # Draw shape parameters
    n_heads = draw(st.integers(1, 8))
    head_dim = draw(st.integers(1, 16)) * 2
    embed_dim = n_heads * head_dim
    position_dim = draw(st.just(2))
    n_levels = draw(st.integers(1, 4))

    extra_batch_dims = draw(st.lists(st.integers(0, 10), min_size=0, max_size=3))

    # Draw data size parameters
    seq_lengths = draw(st.lists(st.integers(0, 32), min_size=1, max_size=4))
    batch_size = len(seq_lengths)
    query_batch_offsets = np.zeros((batch_size + 1,), dtype=int)
    query_batch_offsets[1:] = np.cumsum(seq_lengths)

    # Level shapes
    level_spatial_shapes = []
    last_level = [1] * position_dim
    for level in range(n_levels):
        shape = []
        for pos_dim in range(position_dim):
            shape.append(draw(st.integers(last_level[pos_dim], 1000 * (level + 1))))
        last_level = shape
        level_spatial_shapes.append(shape)

    level_spatial_shapes = np.array(level_spatial_shapes)
    assert np.array_equal(np.sort(level_spatial_shapes, 0), level_spatial_shapes)

    sparsity = draw(st.floats(0.4, 1.0, exclude_max=True))
    make_level_indices = draw(st.booleans())
    make_head_indices = draw(st.booleans())
    make_background_embedding = draw(st.booleans())

    seed = draw(st.integers(0, int(1e8)))

    return {
        "n_heads": n_heads,
        "head_dim": head_dim,
        "embed_dim": embed_dim,
        "n_levels": n_levels,
        "extra_batch_dims": extra_batch_dims,
        "seq_lengths": seq_lengths,
        "level_spatial_shapes": level_spatial_shapes,
        "sparsity": sparsity,
        "require_grads": require_grads,
        "make_level_indices": make_level_indices,
        "make_head_indices": make_head_indices,
        "make_background_embedding": make_background_embedding,
        "seed": seed,
    }


def grid_sample_tensors(
    n_heads: int,
    head_dim: int,
    n_levels: int,
    extra_batch_dims: list[int],
    seq_lengths: list[int],
    level_spatial_shapes: Union[np.ndarray, Tensor],
    sparsity: float,
    require_grads: bool,
    make_level_indices: bool,
    make_head_indices: bool,
    make_background_embedding: bool,
    seed: int,
    device: Union[str, torch.device],
    **kwargs,
) -> dict[str, Optional[Tensor]]:
    if isinstance(device, str):
        device = torch.device(device, 0)

    # save rng state and set seed
    if device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(device)
    else:
        rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    batch_size = len(seq_lengths)
    sum_seq_lens = sum(seq_lengths)

    batch_indices = (
        seq_lengths_to_indices(torch.as_tensor(seq_lengths, device=device))
        .view([sum_seq_lens] + [1] * len(extra_batch_dims))
        .expand([sum_seq_lens] + extra_batch_dims)
    )
    if make_level_indices:
        level_indices = torch.arange(n_levels, device=device)
    else:
        level_indices = None
    if make_head_indices:
        head_indices = torch.arange(n_heads, device=device)
    else:
        head_indices = None

    level_spatial_shapes = torch.as_tensor(level_spatial_shapes, device=device)

    spatial_positions = torch.rand(
        [sum_seq_lens] + extra_batch_dims + [n_levels, n_heads, 2], device=device
    )
    if level_indices is not None:
        spatial_positions *= level_spatial_shapes[level_indices].unsqueeze(1)
    else:
        spatial_positions *= level_spatial_shapes.unsqueeze(1)

    if make_background_embedding:
        background_embedding = torch.randn(
            batch_size, n_levels, n_heads, head_dim, device=device
        )
    else:
        background_embedding = None

    # find max spatial shape
    max_spatial_shape = level_spatial_shapes.max(-2)[0]
    # if different spatial shapes per batch, find max among batch images
    if max_spatial_shape.ndim == 2:
        max_spatial_shape = max_spatial_shape.max(0)[0]
    assert max_spatial_shape.numel() == 2

    # make sparse tensor
    sparse_tensor_indices = random_multilevel_sparse_tensor_indices(
        level_spatial_shapes, sparsity, batch_size, 1000, device
    )
    sparse_tensor = torch.sparse_coo_tensor(
        sparse_tensor_indices,
        torch.randn(sparse_tensor_indices.size(1), head_dim * n_heads, device=device),
        size=[batch_size] + max_spatial_shape.tolist() + [n_levels, head_dim * n_heads],
        device=device,
    ).coalesce()

    # Check validity of the sparse tensor
    assert (sparse_tensor.indices() >= 0).all()
    for level in range(n_levels):
        level_mask = sparse_tensor.indices()[-1] == level
        sparse_level_indices = sparse_tensor.indices().T[level_mask, 1:-1]
        assert torch.all(sparse_level_indices < level_spatial_shapes[level])

    sparse_tensor: Tensor = sparse_split_heads(sparse_tensor, n_heads)

    if require_grads:
        sparse_tensor.requires_grad_(True)

    if device.type == "cuda":
        torch.cuda.set_rng_state(rng_state)
    else:
        torch.set_rng_state(rng_state)

    return {
        "sparse_tensor": sparse_tensor,
        "spatial_positions": spatial_positions,
        "batch_indices": batch_indices,
        "level_indices": level_indices,
        "level_spatial_shapes": level_spatial_shapes,
        "head_indices": head_indices,
        "background_embedding": background_embedding,
    }


@pytest.mark.cuda_if_available
class TestSparseMSDeformAttention:
    def test_basics(self, base_config, device: str):
        msdeform_attn = SparseMSDeformableAttention(**base_config).to(device)

        data = simple_sparse_input_tensors(
            device=device, embed_dim=base_config["embed_dim"]
        )

        out = msdeform_attn(**data)

        assert out is not None
        assert isinstance(out, Tensor)


@pytest.mark.cpu_and_cuda
class TestMultilevelSparseBilinearGridSample:
    @given(inputs=grid_sample_strategy())
    @settings(suppress_health_check=[HealthCheck.differing_executors], deadline=None)
    def test_hypothesis(self, inputs, device: Union[str, torch.device]):
        data = grid_sample_tensors(**inputs, device=device)
        assert data["sparse_tensor"] is not None

        out = multilevel_sparse_bilinear_grid_sample(
            **data  #  pyright: ignore[reportArgumentType]
        )

        assert out is not None

        if data["sparse_tensor"].requires_grad:
            out.sum().backward()
            assert data["sparse_tensor"].grad is not None


@pytest.mark.cpu_and_cuda
class TestMakeIndexAndWeightTensors:
    def test_basic(self, device: Union[str, torch.device]):
        device = torch.device(device)
        data = simple_sparse_input_tensors(device=device, random_seed=0)

        sparse_tensor = data["stacked_feature_maps"]
        level_spatial_shapes = data["level_spatial_shapes"]
        n_levels = level_spatial_shapes.size(0)

        n_heads = 4

        sparse_tensor_split = sparse_split_heads(sparse_tensor, n_heads=n_heads)

        # generate sampling points
        n_pts = 12
        torch.manual_seed(0)
        level_indices = torch.arange(n_levels, device=device)
        spatial_positions = torch.rand(
            (n_pts, n_levels, n_heads, 2), device=device
        ) * level_spatial_shapes[level_indices].unsqueeze(1)
        batch_indices = level_indices.new_zeros(n_pts)
        head_indices = torch.arange(n_heads, device=device)

        index_tensor, weight_tensor = _make_index_and_weight_tensors(
            spatial_positions,
            batch_indices,
            level_indices,
            level_spatial_shapes,
            head_indices,
        )

        indexed_values, _ = batch_sparse_index(sparse_tensor_split, index_tensor)
        assert indexed_values is not None
        assert isinstance(indexed_values, Tensor)

        out = (weight_tensor.unsqueeze(-2) @ indexed_values).squeeze(-2)
        assert out is not None
        assert out.shape == (n_pts, n_levels, n_heads, sparse_tensor_split.shape[-1])


@pytest.mark.cpu_and_cuda
class TestSparseSplitHeads:
    def test_basics(self, device: str):
        inputs = simple_sparse_input_tensors(device=device)

        sparse_tensor = inputs["stacked_feature_maps"]
        n_heads = 4

        split = sparse_split_heads(sparse_tensor, n_heads=4)

        assert split.shape == (
            *sparse_tensor.shape[:-1],
            n_heads,
            sparse_tensor.shape[-1] // n_heads,
        )

        # test some random indices to make sure the embeddings are correctly split
        for _ in range(5):
            i = random.randint(0, sparse_tensor._nnz() - 1)
            spatial_index = sparse_tensor.indices()[:, i]

            embedding = sparse_tensor[*spatial_index]

            split_embedding = []
            for h in range(n_heads):
                split_embedding.append(split[*spatial_index, h])
            stacked_embedding = torch.cat(split_embedding)

            assert torch.equal(embedding, stacked_embedding)
