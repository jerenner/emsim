from typing import Any, Sequence, Union

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch import Tensor

from emsim.utils.sparse_utils.batching.batch_ops import (
    batch_topk,
    BatchTopK,
    unpack_batch_topk,
)
from emsim.utils.sparse_utils.batching.batch_utils import (
    seq_lengths_to_batch_offsets,
)


# Helper utils
def random_tensor(
    seq_lens: Sequence[int], extra_dims: Sequence[int], seed: int, device: str
) -> tuple[Tensor, Tensor]:
    """Generate a concatenated tensor together with its batch offsets."""
    gen = torch.Generator(device=device).manual_seed(seed)
    total = int(sum(seq_lens))
    tensor = torch.randn((total, *extra_dims), generator=gen, device=device)
    batch_offsets: Tensor = seq_lengths_to_batch_offsets(
        torch.tensor(seq_lens, dtype=torch.long, device=device)
    )
    return tensor, batch_offsets


def topk_reference(
    tensor: Tensor,
    batch_offsets: Tensor,
    k: Sequence[int],
    dim: int,
    largest: bool,
    sorted_: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Pure-Python reference that applies torch.topk separately on every subsequence.
    Returned indices are mapped back to the global (concatenated-tensor) index space
    as in `batch_topk`.
    """
    all_idx: list[Tensor] = []
    all_val: list[Tensor] = []
    lengths: list[int] = []

    for b, k_b in enumerate(k):
        start, end = map(int, (batch_offsets[b], batch_offsets[b + 1]))
        subseq = tensor[start:end]

        # normalize the topk dim
        norm_dim = dim if dim >= 0 else subseq.ndim + dim
        k_eff = min(k_b, subseq.shape[norm_dim])  # clamp
        if k_eff == 0:
            lengths.append(0)
            continue

        values, idx = torch.topk(
            subseq, k_eff, dim=norm_dim, largest=largest, sorted=sorted_
        )
        norm_dim = dim if dim >= 0 else subseq.ndim + dim
        if norm_dim == 0:
            idx = idx + start
        all_idx.append(idx.flatten())
        all_val.append(values.flatten())
        lengths.append(idx.numel())

    if all_idx:
        idx_cat = torch.cat(all_idx)
        val_cat = torch.cat(all_val)
    else:  # empty batch
        idx_cat = torch.empty(0, dtype=torch.long, device=tensor.device)
        val_cat = tensor[0:0].flatten()
    offsets_out: Tensor = seq_lengths_to_batch_offsets(
        torch.tensor(lengths, dtype=torch.long, device=tensor.device)
    )
    return idx_cat, offsets_out, val_cat


@st.composite
def batch_topk_inputs(draw) -> dict[str, Any]:
    # Sample sequence lengths
    seq_lens = draw(st.lists(st.integers(0, 7), min_size=0, max_size=6))
    # force at least one element overall so that tensor shape is valid
    if sum(seq_lens) == 0:
        seq_lens = [0, 1]

    # Sample tensor shape
    extra_dims = draw(st.lists(st.integers(1, 4), min_size=0, max_size=3))
    tensor_ndim = 1 + len(extra_dims)

    # Sample topk dim
    dim = draw(st.integers(-tensor_ndim, tensor_ndim - 1))

    # Sample k
    norm_dim = dim if dim >= 0 else dim + (1 + len(extra_dims))
    if norm_dim == 0:  # sequence dimension
        max_len_along_dim = max(seq_lens) if seq_lens else 0
    else:  # one of the extra dims
        max_len_along_dim = extra_dims[norm_dim - 1]

    k_scalar = draw(st.integers(0, max(0, max_len_along_dim)))
    k_kind = draw(st.sampled_from(["scalar", "list", "tensor"]))
    if k_kind == "scalar":
        k = k_scalar
    else:
        ks = [
            draw(st.integers(0, max(0, len_ if dim == 0 else max_len_along_dim)))
            for len_ in seq_lens
        ]
        k = ks if k_kind == "list" else torch.tensor(ks)

    # Sample flags
    largest = draw(st.booleans())
    sorted_ = draw(st.booleans())
    seed = draw(st.integers(0, 2**32 - 1))

    return dict(
        seq_lens=seq_lens,
        extra_dims=extra_dims,
        dim=dim,
        k=k,
        largest=largest,
        sorted_=sorted_,
        seed=seed,
    )


@pytest.mark.cpu_and_cuda
class TestBatchTopK:
    # Basic tests
    @pytest.mark.parametrize(
        "seq_lens,k,dim,largest,sorted_",
        [
            ([4, 4, 4], 2, 0, True, True),  # uniform, scalar k
            ([2, 3, 4], 2, 0, True, True),  # variable, scalar k
            ([2, 3, 4], [1, 2, 3], 0, True, True),  # per-sequence k
            ([2, 3, 4], 10, 0, True, True),  # k larger than any len
            ([4, 4, 4], 2, 0, False, True),  # smallest values
            ([4, 4, 4], 2, 0, True, False),  # unsorted
            ([2, 2, 2], 1, -1, True, True),  # negative dim
            ([4], 2, 0, True, True),  # single batch
            ([], 2, 0, True, True),  # empty batch
        ],
    )
    def test_examples(
        self,
        seq_lens: list[int],
        k: Union[int, list[int]],
        dim: int,
        largest: bool,
        sorted_: bool,
        device,
    ):
        tensor, offsets = random_tensor(seq_lens, extra_dims=(), seed=1, device=device)

        out = batch_topk(tensor, offsets, k, dim=dim, largest=largest, sorted=sorted_)
        idx, off, val = out

        # shape / basic invariants
        assert off[-1] == idx.numel(), "offsets must mark the end of the last slice"
        if isinstance(k, int):
            k_as_list = [k] * len(seq_lens)
        else:
            k_as_list = list(k)

        # Compare to reference
        ref_idx, ref_off, _ = topk_reference(
            tensor, offsets, k_as_list, dim, largest, sorted_
        )
        assert torch.equal(idx, ref_idx)
        assert torch.equal(off, ref_off)
        # With return_values=False we do not get the values tensor
        assert val is None

    def test_return_values(self, device):
        seq_lens = [3, 1, 5]
        tensor, offsets = random_tensor(seq_lens, (), seed=1, device=device)
        k = torch.tensor([2, 1, 4], device=device)

        out = batch_topk(tensor, offsets, k, return_values=True)
        idx, off, val = out
        assert val is not None and val.shape == idx.shape

        # Check that each val == tensor[idx]
        assert torch.allclose(val, tensor[idx])

        # Offsets consistency
        assert off[-1] == idx.numel()

    def test_gradients(self, device):
        """Tests equality of gradients for returned values against reference."""
        seq_lens = [3, 5, 2]
        extra_dims = [4]
        k_per_batch = [2, 1, 2]
        dim = -1
        largest, srt = True, True

        tensor, offsets = random_tensor(seq_lens, extra_dims, seed=999, device=device)

        # batch_topk gradients
        tensor_batch = tensor.detach().clone().requires_grad_(True)
        out_batch = batch_topk(
            tensor_batch,
            offsets,
            k_per_batch,
            dim=dim,
            largest=largest,
            sorted=srt,
            return_values=True,
        )
        assert out_batch.values is not None
        out_batch.values.sum().backward()

        # reference gradients
        tensor_ref = tensor.detach().clone().requires_grad_(True)
        _, _, values_ref = topk_reference(
            tensor_ref, offsets, k_per_batch, dim, largest, srt
        )
        values_ref.sum().backward()

        assert tensor_batch.grad is not None
        assert tensor_ref.grad is not None

        assert torch.equal(tensor_batch.grad, tensor_ref.grad)

    # Error tests
    @pytest.mark.parametrize("bad_dim", [-10, 5])
    def test_invalid_dim_raises(self, device: str, bad_dim: int):
        t = torch.randn(5, 4, device=device)
        off = torch.tensor([0, 5], device=device)
        with pytest.raises((ValueError, torch.jit.Error)):  # type: ignore
            batch_topk(t, off, 1, dim=bad_dim)

    def test_negative_k_raises(self, device):
        t = torch.randn(3, device=device)
        off = torch.tensor([0, 3], device=device)
        with pytest.raises((ValueError, torch.jit.Error)):  # type: ignore
            batch_topk(t, off, k=-1)

    # Property-based test
    @settings(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
    @given(params=batch_topk_inputs())
    def test_property(self, params, device):
        tensor, offsets = random_tensor(
            params["seq_lens"], params["extra_dims"], params["seed"], device
        )

        torch.autograd.set_detect_anomaly(True)

        tensor_batch = tensor.clone().detach().requires_grad_(True)
        tensor_ref = tensor.clone().detach().requires_grad_(True)

        out: BatchTopK = batch_topk(
            tensor_batch,
            offsets,
            params["k"].to(device) if isinstance(params["k"], Tensor) else params["k"],
            dim=params["dim"],
            largest=params["largest"],
            sorted=params["sorted_"],
            return_values=True,
        )
        assert out.values is not None
        assert out.offsets[-1] == out.indices.numel()

        if isinstance(params["k"], (int, list)):
            k_per_batch = (
                [params["k"]] * len(params["seq_lens"])
                if isinstance(params["k"], int)
                else params["k"]
            )
        else:  # tensor
            k_per_batch = params["k"].tolist()

        ref_idx, ref_off, ref_vals = topk_reference(
            tensor_ref,
            offsets,
            k_per_batch,
            params["dim"],
            params["largest"],
            params["sorted_"],
        )
        assert torch.equal(out.indices, ref_idx)
        assert torch.equal(out.offsets, ref_off)
        assert torch.equal(out.values, ref_vals.to(out.values))

        indices_unpacked, values_unpacked = unpack_batch_topk(
            out, offsets, tensor.shape, params["dim"]
        )
        assert values_unpacked is not None

        dim = params["dim"] if params["dim"] >= 0 else params["dim"] + tensor.ndim
        for b in range(len(offsets) - 1):
            b_start, b_end = offsets[b], offsets[b + 1]
            if b_end == b_start:
                continue
            subseq_b = tensor[b_start:b_end]
            if dim == 0:
                k_b = min(k_per_batch[b], int(b_end - b_start))
            else:
                k_b = min(k_per_batch[b], tensor.shape[dim])

            out_shape = list(subseq_b.shape)
            out_shape[params["dim"]] = k_b

            idx_b = out.indices[out.offsets[b] : out.offsets[b + 1]]
            idx_b = idx_b.view(out_shape)
            if dim == 0:
                idx_b = idx_b - offsets[b]

            assert torch.equal(idx_b, indices_unpacked[b])

            val_b = out.values[out.offsets[b] : out.offsets[b + 1]].view(out_shape)
            val_b_topk = torch.take_along_dim(subseq_b, idx_b, params["dim"])
            assert torch.equal(val_b, val_b_topk)

            assert torch.equal(val_b, values_unpacked[b])

        # Check gradients
        assert out.values.requires_grad
        assert ref_vals.requires_grad

        out.values.sum().backward()
        ref_vals.sum().backward()

        assert tensor_batch.grad is not None
        assert tensor_ref.grad is not None
        assert torch.equal(tensor_batch.grad, tensor_ref.grad)
