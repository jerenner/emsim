from typing import Any, Optional, Sequence, Union
import math

import numpy as np
import torch
from hypothesis import given, settings, HealthCheck, assume, example
from hypothesis import strategies as st
from torch import Tensor
import pytest


from emsim.utils.sparse_utils.batching.batch_ops.lexsort_nd import (
    _lexsort_nd_float,
    _lexsort_nd_int,
    _lexsort_nd_robust,
    lexsort_nd,
    _permute_dims,
)

_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


@st.composite
def lexsort_nd_inputs(
    draw: st.DrawFn, dtypes: Sequence[torch.dtype] = _DTYPES
) -> dict[str, Any]:
    n_dims = draw(st.integers(2, 4))
    shape = draw(st.lists(st.integers(0, 5), min_size=n_dims, max_size=n_dims))

    def _different_normalized_dims(dims: list[int]):
        normalized_dims = [dim + n_dims if dim < 0 else dim for dim in dims]
        return not normalized_dims[0] == normalized_dims[1]

    vector_dim, sort_dim = draw(
        st.lists(
            st.integers(-n_dims, n_dims - 1), min_size=2, max_size=2, unique=True
        ).filter(_different_normalized_dims)
    )
    vector_len = draw(st.integers(0, 32))
    sort_len = draw(st.integers(0, 64))

    shape[vector_dim] = vector_len
    shape[sort_dim] = sort_len

    descending = draw(st.booleans())
    stable = draw(st.booleans())
    force_robust = draw(st.booleans())

    dtype = draw(st.sampled_from(dtypes))
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        finfo = torch.finfo(dtype)
        min_value = draw(st.floats(finfo.min, finfo.max, exclude_max=True))
        max_value = draw(st.floats(min_value, finfo.max))
        max_value = min(max_value, min_value + finfo.max - finfo.eps)
    else:
        iinfo = torch.iinfo(dtype)
        min_value = draw(st.integers(iinfo.min, iinfo.max - 1))
        max_value = draw(st.integers(min_value + 1, iinfo.max))
    seed = draw(st.integers(0, 2**32 - 1))

    return {
        "inputs": {
            "vector_dim": vector_dim,
            "sort_dim": sort_dim,
            "descending": descending,
            "stable": stable,
            "force_robust": force_robust,
        },
        "tensor_config": {
            "shape": shape,
            "dtype": dtype,
            "min_value": min_value,
            "max_value": max_value,
            "seed": seed,
        },
    }


def make_tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    min_value: Union[int, float],
    max_value: Union[int, float],
    seed: int,
    device: Optional[Union[str, torch.device]] = None,
) -> Tensor:
    generator = torch.Generator(device).manual_seed(seed)
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        tensor = torch.empty(shape, dtype=torch.float64, device=device)
        tensor.uniform_(min_value, max_value)
        tensor = tensor.to(dtype)
    else:
        tensor = torch.empty(shape, dtype=dtype, device=device)
        assert isinstance(min_value, int) and isinstance(max_value, int)
        tensor.random_(min_value, max_value, generator=generator)

    return tensor


def _lex_compare(row1: Tensor, row2: Tensor, descending: bool = False) -> bool:
    """Compares two 1D tensors lexicographically."""
    diff_positions = (row1 != row2).nonzero()

    if diff_positions.numel() == 0:
        return True

    first_diff_pos = int(diff_positions[0].item())

    if descending:
        return bool((row1[first_diff_pos] >= row2[first_diff_pos]).item())
    else:
        return bool((row1[first_diff_pos] <= row2[first_diff_pos]).item())


def is_lexsorted(tensor: Tensor, descending: bool = False) -> Tensor:
    """
    Lexicographic sorting check for batches of 2D tensors.

    Args:
        tensor: 3D tensor of shape (batch_size, sort_dim, vector_dim)
        descending: Whether to check for descending lexsort

    Returns:
        Tensor: Boolean tensor of shape (batch_size,) indicating sorted status
    """
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError("Expected 2D or 3D tensor")
    batch_size, sort_len, vector_len = tensor.shape
    device = tensor.device

    if sort_len <= 1 or vector_len <= 1 or tensor.numel() == 0:
        return torch.ones(batch_size, dtype=torch.bool, device=tensor.device)

    batch_size, n_rows, _ = tensor.shape

    if n_rows <= 1:
        return tensor.new_ones(batch_size, dtype=torch.bool)

    tensor = tensor.cpu()
    results = tensor.new_empty(batch_size, dtype=torch.bool)

    for b in range(batch_size):
        batch_sorted = True
        for r in range(n_rows - 1):
            curr_row = tensor[b, r]
            next_row = tensor[b, r + 1]
            if not _lex_compare(curr_row, next_row, descending=descending):
                batch_sorted = False
                break

        results[b] = batch_sorted

    return results.to(device)


def lexsort_nd_numpy(
    tensor: Tensor,
    vector_dim: int,
    sort_dim: int,
    descending: bool = False,
    stable: bool = False,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper around numpy.lexsort for testing lexsort_nd.

    Args:
        tensor: Input PyTorch tensor
        vector_dim: Dimension defining vectors (components to compare)
        sort_dim: Dimension along which to sort
        descending: If True, sort in descending order
        stable: Ignored (np.lexsort is always stable)

    Returns:
        (sorted_tensor, sort_indices) matching lexsort_nd's behavior
    """
    device = tensor.device

    # Normalize dimensions
    ndim = tensor.ndim
    vector_dim = vector_dim + ndim if vector_dim < 0 else vector_dim
    sort_dim = sort_dim + ndim if sort_dim < 0 else sort_dim

    # Validate inputs
    if sort_dim == vector_dim:
        raise ValueError("sort_dim and vector_dim must be different")
    if tensor.isnan().any():
        raise ValueError("Tensor has nan values")
    if tensor.isinf().any():
        raise ValueError("Tensor has infinite values")

    vector_len = tensor.shape[vector_dim]

    # Handle edge cases
    if tensor.numel() == 0:
        indices_shape = list(tensor.shape)
        indices_shape.pop(vector_dim)
        return tensor, torch.zeros(
            indices_shape, device=tensor.device, dtype=torch.long
        )
    if vector_len == 1:
        tensor, sort_indices = torch.sort(
            tensor,
            dim=sort_dim,
            descending=descending,
            stable=stable,
        )
        sort_indices = sort_indices.squeeze(vector_dim)
        return tensor, sort_indices

    # Convert to numpy
    if tensor.dtype == torch.bfloat16:
        # numpy doesn't support bfloat
        array = tensor.detach().float().cpu().numpy()
        stable = True  # lexsort_nd forces stable for bfloat
    else:
        array = tensor.detach().cpu().numpy()

    # Permute to batch-first order
    perm = list(range(ndim))
    perm.remove(sort_dim)
    perm.remove(vector_dim)
    perm = perm + [sort_dim, vector_dim]

    array_perm = array.transpose(perm)

    # Reshape for batch processing: (batch_size, sort_len, vector_len)
    *batch_dims, sort_len, vector_len = array_perm.shape
    batch_size = int(np.prod(batch_dims)) if batch_dims else 1

    array_batched = array_perm.reshape(batch_size, sort_len, vector_len)

    # Process each batch
    indices = np.zeros((batch_size, sort_len), dtype=np.int64)

    for b in range(batch_size):
        batch = array_batched[b]  # (sort_len, vector_len)

        keys: list[np.ndarray] = [batch[:, i] for i in range(vector_len - 1, -1, -1)]

        if descending:  # np.lexsort only works with ascending, need to negate keys

            def _flip_keys(keys: list[np.ndarray]) -> list[np.ndarray]:
                if np.issubdtype(keys[0].dtype, np.unsignedinteger) or (
                    # can't represent negative min in signed dtype
                    np.issubdtype(keys[0].dtype, np.signedinteger)
                    and ((batch == np.iinfo(batch.dtype).min).any())
                ):
                    if keys[0].dtype == np.int64:
                        # special case for np.int64 with min value
                        return [~(k.astype(np.uint64)) for k in keys]
                    # have to promote to negate
                    keys = [k.astype(np.int64) for k in keys]
                return [-k for k in keys]

            keys = _flip_keys(keys)

        indices[b] = np.lexsort(keys)

        assert is_lexsorted(
            torch.from_numpy(batch[indices[b]]).unsqueeze(0), descending=descending
        )

    # Unflatten indices back to original batch dims
    indices = indices.reshape(batch_dims + [sort_len])

    # Expand indices for gather
    indices_expanded = np.expand_dims(indices, axis=-1)

    # Gather sorted values
    sorted_perm = np.take_along_axis(
        array_perm, indices_expanded.repeat(vector_len, -1), axis=-2
    )

    # Permute back to original layout
    inv_perm = np.argsort(perm)
    sorted_np = sorted_perm.transpose(inv_perm)

    # Permute indices
    indices = indices_expanded.transpose(inv_perm).squeeze(vector_dim)

    # Convert back to PyTorch
    sorted_tensor = torch.from_numpy(sorted_np).to(device)

    indices_torch = torch.from_numpy(indices).to(device, dtype=torch.long)

    return sorted_tensor, indices_torch


def _permute_dims_to_batched(tensor: Tensor, vector_dim: int, sort_dim: int) -> Tensor:
    ndims = tensor.ndim
    vector_dim = vector_dim + ndims if vector_dim < 0 else vector_dim
    sort_dim = sort_dim + ndims if sort_dim < 0 else sort_dim

    vector_len = tensor.shape[vector_dim]
    sort_len = tensor.shape[sort_dim]

    perm = list(range(tensor.ndim))
    perm.remove(vector_dim)
    perm.remove(sort_dim)
    perm = perm + [sort_dim, vector_dim]

    tensor = tensor.permute(perm)

    batch_len = math.prod(tensor.shape[:-2])

    return tensor.reshape(batch_len, sort_len, vector_len)


@pytest.mark.cpu_and_cuda
class TestUnit:
    def test_unit(self, device):
        tensor = make_tensor(
            [4, 16, 5],
            dtype=torch.float32,
            min_value=-100.0,
            max_value=100.0,
            seed=0,
            device=device,
        )

        vector_dim = 2
        sort_dim = 1
        # (batch dim = 0)

        # Make some duplicates
        tensor[:, 1] = tensor[:, 3]
        tensor[:, 4] = tensor[:, 2]
        tensor[:, ::6] = tensor[:, -1].unsqueeze(1)

        tensor_copy = tensor.clone()

        tensor_sorted, _ = lexsort_nd(tensor, vector_dim, sort_dim)
        tensor_sorted_stable, sorted_idx_stable = lexsort_nd(
            tensor, vector_dim, sort_dim, stable=True
        )
        assert torch.equal(tensor_sorted, tensor_sorted_stable)
        tensor_sorted_robust, sorted_idx_robust = lexsort_nd(
            tensor_copy.clone(), vector_dim, sort_dim, force_robust=True
        )
        assert torch.equal(sorted_idx_stable, sorted_idx_robust)
        tensor_sorted_int, _ = lexsort_nd(tensor_copy.long(), vector_dim, sort_dim)
        numpy_sorted, numpy_idx = lexsort_nd_numpy(tensor_copy, vector_dim, sort_dim)

        assert is_lexsorted(numpy_sorted, False).all()
        assert is_lexsorted(tensor_sorted_robust, False).all()
        assert is_lexsorted(tensor_sorted_int, False).all()
        assert is_lexsorted(tensor_sorted, False).all()

        assert torch.allclose(tensor_sorted, numpy_sorted)
        assert torch.equal(sorted_idx_robust, numpy_idx)

    def test_long_vector(self, device):
        tensor = make_tensor(
            [2, 4, 100],
            dtype=torch.float32,
            min_value=-100.0,
            max_value=100.0,
            seed=0,
            device=device,
        )

        tensor_copy = tensor.clone()

        tensor_sorted, sorted_indices = lexsort_nd(tensor, -1, -2)
        tensor_sorted_robust, _ = lexsort_nd(
            tensor_copy.clone(), -1, -2, force_robust=True
        )
        tensor_sorted_int, _ = lexsort_nd(tensor_copy.long(), -1, -2)
        numpy_sorted, numpy_indices = lexsort_nd_numpy(tensor_copy, -1, -2)

        assert is_lexsorted(numpy_sorted, False).all()
        assert is_lexsorted(tensor_sorted_robust, False).all()
        assert is_lexsorted(tensor_sorted_int, False).all()
        assert is_lexsorted(tensor_sorted, False).all()

        assert torch.allclose(tensor_sorted, numpy_sorted)
        assert torch.equal(sorted_indices, numpy_indices)

    def test_epsilon_float(self, device):
        tensor = torch.tensor([[1.0, 1.0000001], [1.0, 1.0]], device=device)
        assert not torch.equal(tensor[0], tensor[1])

        _, sorted_indices = lexsort_nd(tensor, -1, -2)

        # Assert the epsilon-small difference was sorted correctly
        assert torch.equal(sorted_indices, torch.tensor([1, 0], device=tensor.device))

    def test_descending(self, device):
        tensor = make_tensor([3, 4, 5], torch.float32, -100.0, 100.0, 0, device)

        vector_dim = -1
        sort_dim = -2

        sorted_asc, indices_asc = lexsort_nd(
            tensor, vector_dim, sort_dim, descending=False
        )
        sorted_desc, indices_desc = lexsort_nd(
            tensor, vector_dim, sort_dim, descending=True
        )

        assert torch.equal(sorted_asc, sorted_desc.flip(sort_dim))
        assert torch.equal(indices_asc, indices_desc.flip(vector_dim))

    def test_overflow_range(self, device):
        tensor = torch.randint(
            -10000, 10000, [2, 4, 5], device=device, dtype=torch.int16
        )
        tensor[0, 0, 1] = torch.iinfo(torch.int16).min
        tensor[0, 0, 2] = torch.iinfo(torch.int16).max

        tensor_sorted, _ = lexsort_nd(tensor, -1, -2)
        assert is_lexsorted(tensor_sorted).all()

    def test_tiny_float(self, device):
        tiny_value = 4.9407e-324
        tensor = torch.tensor(
            [
                [tiny_value, 0.0],
                [0.0, tiny_value],
                [tiny_value, tiny_value],
                [tiny_value, tiny_value],
                [tiny_value, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        )

        sorted_tensor, indices = lexsort_nd(tensor, vector_dim=1, sort_dim=0)

        expected_sorted = tensor.new_tensor(
            [
                [0.0, tiny_value],
                [tiny_value, 0.0],
                [tiny_value, 0.0],
                [tiny_value, tiny_value],
                [tiny_value, tiny_value],
            ]
        )
        expected_indices = indices.new_tensor([1, 0, 4, 2, 3])
        assert torch.equal(sorted_tensor, expected_sorted)
        assert torch.equal(indices, expected_indices)

    def test_large_int(self, device):
        tensor = torch.tensor(
            [
                [241342945717553440, 1969971363771739734, 8666018656948951979],
                [6959987956964269583, 3145603029896912342, -564571699210388018],
                [7079198638595629862, 8492600540221939246, -389147068868319639],
                [3648643930887760505, 3983299825984528153, -251015285358538311],
            ],
            dtype=torch.int64,
            device=device,
        )

        sorted_tensor, indices = lexsort_nd(tensor, vector_dim=1, sort_dim=0)

        assert is_lexsorted(sorted_tensor).all()

        expected_sorted = tensor.new_tensor(
            [
                [241342945717553440, 1969971363771739734, 8666018656948951979],
                [3648643930887760505, 3983299825984528153, -251015285358538311],
                [6959987956964269583, 3145603029896912342, -564571699210388018],
                [7079198638595629862, 8492600540221939246, -389147068868319639],
            ],
        )
        expected_indices = indices.new_tensor([0, 3, 1, 2])

        assert torch.equal(sorted_tensor, expected_sorted)
        assert torch.equal(indices, expected_indices)


@pytest.mark.cpu_and_cuda
class TestAdvanced:
    def test_return_int_unique_inverse(self, device: str):
        tensor = make_tensor(
            [16, 4, 5],
            dtype=torch.int64,
            min_value=-100,
            max_value=100,
            seed=0,
            device=device,
        )

        # add duplicates
        tensor[1] = tensor[3]
        tensor[2] = tensor[5]
        tensor[::6] = tensor[-1]

        descending = False

        sort_indices, sorted_inverse, has_duplicates = _lexsort_nd_int(
            tensor, descending, False, return_unique_inverse=True
        )
        assert sorted_inverse is not None

        sorted_tensor = tensor.gather(0, sort_indices.unsqueeze(-1).expand_as(tensor))

        for b in range(tensor.size(1)):
            sorted_b = sorted_tensor[:, b]
            assert not descending  # calling unique to get the gt doesn't work otherwise
            _, inverse_b = sorted_b.unique(dim=0, return_inverse=True)
            assert torch.equal(inverse_b, sorted_inverse[:, b])


@pytest.mark.cpu_and_cuda
class TestProperties:
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.differing_executors],
        max_examples=1000,
    )
    @given(inputs=lexsort_nd_inputs())
    def test_hypothesis(self, inputs: dict[str, Any], device: str):

        is_bfloat = inputs["tensor_config"]["dtype"] == torch.bfloat16
        # cpu doesn't support bfloat
        assume(not (is_bfloat and device == "cpu"))

        vector_dim = inputs["inputs"]["vector_dim"]
        sort_dim = inputs["inputs"]["sort_dim"]
        descending = inputs["inputs"]["descending"]

        tensor = make_tensor(**inputs["tensor_config"], device=device)

        tensor_copy = tensor.clone()

        tensor_sorted, sorted_indices = lexsort_nd(tensor, **inputs["inputs"])
        numpy_sorted, numpy_indices = lexsort_nd_numpy(tensor_copy, **inputs["inputs"])

        tensor_batched = _permute_dims_to_batched(tensor_sorted, vector_dim, sort_dim)
        numpy_batched = _permute_dims_to_batched(numpy_sorted, vector_dim, sort_dim)

        tensor_lexsorted = is_lexsorted(tensor_batched, descending)
        numpy_lexsorted = is_lexsorted(numpy_batched, descending)

        assert numpy_lexsorted.all(), "Numpy not sorting"
        assert (
            tensor_lexsorted.all()
        ), f"lexsort_nd not sorted: {tensor_batched[~tensor_lexsorted]}"

        assert torch.allclose(tensor_sorted, numpy_sorted.to(tensor_sorted))
        if inputs["inputs"]["stable"]:
            assert torch.equal(sorted_indices, numpy_indices)

    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.differing_executors],
        max_examples=1000,
    )
    @given(
        inputs=lexsort_nd_inputs(
            dtypes=(torch.int64, torch.uint8, torch.int8, torch.int16, torch.int32)
        )
    )
    def test_return_inverse_hypothesis(self, inputs: dict[str, Any], device: str):
        assume(math.prod(inputs["tensor_config"]["shape"]) > 0)
        assume(
            inputs["tensor_config"]["max_value"] - inputs["tensor_config"]["min_value"]
            > 1
        )  # don't make everything equal
        vector_dim = inputs["inputs"]["vector_dim"]
        sort_dim = inputs["inputs"]["sort_dim"]
        descending = inputs["inputs"]["descending"]
        stable = inputs["inputs"]["stable"]

        tensor = make_tensor(**inputs["tensor_config"], device=device)

        sort_dim = sort_dim + tensor.ndim if sort_dim < 0 else sort_dim
        vector_dim = vector_dim + tensor.ndim if vector_dim < 0 else vector_dim

        sort_len = tensor.shape[sort_dim]
        vector_len = tensor.shape[vector_dim]

        tensor, _ = _permute_dims(tensor, vector_dim, sort_dim)

        sort_indices, sorted_inverse, has_duplicates = _lexsort_nd_int(
            tensor, descending=descending, stable=stable, return_unique_inverse=True
        )
        assert sorted_inverse is not None
        assert has_duplicates is not None

        sorted_tensor = tensor.gather(0, sort_indices.unsqueeze(-1).expand_as(tensor))

        sorted_tensor = sorted_tensor.view(sort_len, -1, vector_len).transpose(0, 1)
        sorted_tensor = sorted_tensor.contiguous()
        sorted_inverse = sorted_inverse.view(sort_len, -1).transpose(0, 1).contiguous()
        has_duplicates = has_duplicates.view(-1)

        for b in range(sorted_tensor.size(0)):
            sorted_b = sorted_tensor[b]  # [sort_len, vector_len]
            inverse_b = sorted_inverse[b]  # [sort_len]
            unique_ids, counts = inverse_b.unique(return_counts=True)
            if (counts > 1).any():
                assert has_duplicates[b]
            else:
                assert not has_duplicates[b]
            for unique, count in zip(unique_ids, counts):
                idx = inverse_b == unique
                assert idx.sum() == count
                if count > 1:
                    # check all marked as duplicates are duplicates
                    duplicates = sorted_b[idx]
                    assert (duplicates[0] == duplicates).all()
