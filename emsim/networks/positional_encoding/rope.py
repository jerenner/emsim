from typing import Optional, Union
import warnings

import torch
from torch import Tensor, nn


# Based on code from
# https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py


def init_2d_freqs(
    head_dim: int,
    num_heads: int,
    theta: float = 10.0,
    rotate: bool = True,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    freqs_x = []
    freqs_y = []
    mag = 1 / (
        theta ** (torch.arange(0, head_dim, 4, dtype=dtype, device=device) / head_dim)
    )
    for _ in range(num_heads):
        angles = (
            torch.rand(1, device=device) * 2 * torch.pi
            if rotate
            else torch.zeros(1, device=device)
        )
        fx = torch.cat(
            [mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1
        )
        fy = torch.cat(
            [mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=-1)
    return freqs  # n_head, head_dim/2, 2


def init_nd_freqs(
    position_dim: int,
    head_dim: int,
    num_heads: int,
    thetas: Union[Tensor, list[float], float] = 10.0,
    rotate: bool = True,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
):
    thetas: Tensor = torch.as_tensor(thetas, dtype=dtype, device=device)
    if thetas.numel() == 1:
        thetas = thetas.expand(position_dim)
    mag = 1 / (
        thetas.view(-1, 1)
        ** (
            torch.arange(0, head_dim, 4, dtype=dtype, device=device).view(1, -1)
            / head_dim
        )
    )
    freqs = [[] for _ in range(position_dim)]
    for _ in range(num_heads):
        angles = (
            torch.rand(1, device=device) * 2 * torch.pi
            if rotate
            else torch.zeros(1, device=device)
        )
        for i, dim_freqs in enumerate(freqs):
            f = torch.cat(
                [
                    mag[i] * torch.cos(angles + torch.pi * 2 * i / (2 * position_dim)),
                    mag[i]
                    * torch.cos(angles + torch.pi * ((2 * i) + 1) / (2 * position_dim)),
                ],
                dim=-1,
            )
            dim_freqs.append(f)
    freqs = [torch.stack(dim_freqs, dim=0) for dim_freqs in freqs]
    freqs = torch.stack(freqs, dim=-1)
    return freqs  # n_head, head_dim/(2 * n_groups), pos_dim


class RoPEEncodingND(nn.Module):
    def __init__(
        self,
        position_dim: int,
        d_model: int,
        n_heads: int,
        rope_base_theta: float = 10.0,
        dtype=torch.float,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "Expected d_model to be divisible by n_heads, got "
                f"{d_model} and {n_heads}"
            )
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"Expected head dim to be divisible by 2, got {self.head_dim}"
            )
        self.pos_dim = position_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self._base_theta = torch.as_tensor(rope_base_theta)
        self.dtype = dtype
        self._init_freq_param()

    def _init_freq_param(self):
        freqs = init_nd_freqs(
            self.pos_dim,
            self.head_dim,
            self.n_heads,
            self._base_theta,
            dtype=self.dtype,
        )
        assert freqs.shape == (self.n_heads, self.head_dim // 2, self.pos_dim)
        self.freqs = nn.Parameter(freqs)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        self.shape_check(query, query_pos)
        if query_pos.numel() > 0 and query_pos.max() <= 1.0:
            warnings.warn(
                "Expected un-normalized (i.e., not inside [0,1]) coordinates "
                "for position but found potentially normalized coordinates. "
                "Did you accidentally pass in normalized coordinates?\n"
                f"(Your coord range: [{query_pos.min(), query_pos.max()}])"
            )
        if key_pos is not None:
            self.shape_check(key, key_pos)
        query_rot_vec = self._make_rot_vec(query_pos)
        query_rotated = self._apply_rot_vec(query, query_rot_vec)

        if key is None:
            return query_rotated

        if key_pos is not None:
            key_rot_vec = self._make_rot_vec(key_pos)
        else:
            key_rot_vec = query_rot_vec
        key_rotated = self._apply_rot_vec(key, key_rot_vec)

        return query_rotated, key_rotated

    def _make_rot_vec(self, positions: Tensor) -> Tensor:
        leading_dims = positions.shape[:-1]
        rot_vec = torch.mm(
            positions.view(-1, self.pos_dim).to(self.freqs),
            self.freqs.view(-1, self.pos_dim).T,
            # ).view([bsz, tgt_len, self.n_heads, self.head_dim // self.pos_dim])
        ).view(leading_dims + (self.n_heads, self.head_dim // 2))
        out = torch.polar(torch.ones_like(rot_vec), rot_vec)
        assert out.shape == leading_dims + (self.n_heads, self.head_dim // 2)
        assert out.is_complex()
        return out

    def _apply_rot_vec(self, query_or_key: Tensor, rot_vec: Tensor) -> Tensor:
        leading_dims = query_or_key.shape[:-1]
        query_or_key = query_or_key.view(
            leading_dims + (self.n_heads, self.head_dim // 2, 2)
        )
        query_or_key = torch.view_as_complex(query_or_key)
        return torch.view_as_real(query_or_key * rot_vec).flatten(-2)

    def shape_check(self, query_or_key: Tensor, query_or_key_pos: Tensor):
        if query_or_key.ndim != query_or_key_pos.ndim:  # ..., seq_len, d_model
            raise ValueError(
                "Expected query_or_key and query_or_key_pos to have same number "
                f"of dimensions, got {query_or_key.ndim} and {query_or_key_pos.ndim}"
            )
        if query_or_key.shape[-1] != self.d_model:
            raise ValueError(
                "Expected query_or_key to have last dim equal to d_model "
                f"(={self.d_model}), got {query_or_key.shape[-1]}"
            )
        if query_or_key_pos.shape[-1] != self.pos_dim:
            raise ValueError(
                "Expected query_or_key_pos to have last dim equal to pos_dim "
                f"(={self.pos_dim}), got {query_or_key_pos.shape[-1]}"
            )
        if query_or_key.shape[:-1] != query_or_key_pos.shape[:-1]:
            raise ValueError(
                "Expected query_or_key and query_or_key_pos to have matching leading dims,"
                f" got {query_or_key.shape[:-1]} and {query_or_key_pos.shape[:-1]}"
            )

    def reset_parameters(self):
        freqs = init_nd_freqs(
            self.pos_dim,
            self.head_dim,
            self.n_heads,
            self._base_theta,
            dtype=self.freqs.dtype,
            device=self.freqs.device,
        )
        with torch.no_grad():
            self.freqs.copy_(freqs)


class RoPEEncodingNDGroupedFreqs(RoPEEncodingND):
    def __init__(
        self,
        position_dim: int,
        d_model: int,
        n_heads: int,
        pos_dim_to_rope_group: Union[Tensor, list[int]],
        rope_base_theta: float = 10.0,
        dtype=torch.float,
    ):
        self.pos_dim_to_rope_group = torch.as_tensor(pos_dim_to_rope_group)
        if self.pos_dim_to_rope_group.ndim != 1:
            raise ValueError(
                f"Expected 1D pos_dim_to_rope_group, got {pos_dim_to_rope_group.ndim}"
            )
        if len(self.pos_dim_to_rope_group) != position_dim:
            raise ValueError(
                "Expected pos_dim_to_rope_group to have length equal to position_dim,"
                f" got {len(self.pos_dim_to_rope_group)} and {position_dim}"
            )
        self.n_freq_groups = len(self.pos_dim_to_rope_group.unique())
        super().__init__(position_dim, d_model, n_heads, rope_base_theta, dtype)
        if self.head_dim % self.n_freq_groups != 0:
            raise ValueError(
                "head_dim must be divisible by number of freq groups, got "
                f"{self.head_dim} and {self.n_freq_groups}"
            )

    def _init_freq_param(self):
        freqs = init_nd_freqs(
            self.pos_dim,
            self.head_dim // self.n_freq_groups,
            self.n_heads,
            self._base_theta,
            dtype=self.dtype,
        )
        assert freqs.shape == (
            self.n_heads,
            self.head_dim // 2 // self.n_freq_groups,
            self.pos_dim,
        )
        self.freqs = nn.Parameter(freqs)

    def _make_rot_vec(self, positions: Tensor):
        leading_dims = positions.shape[:-1]
        assert positions.shape[-1] == self.pos_dim
        unique_indices, index_counts = torch.unique(
            self.pos_dim_to_rope_group, return_counts=True
        )
        split_positions = [
            positions[..., self.pos_dim_to_rope_group == i] for i in unique_indices
        ]
        split_freqs = [
            self.freqs[..., self.pos_dim_to_rope_group == i] for i in unique_indices
        ]
        rot_subvecs = [
            torch.mm(pos.view(-1, count).to(freq), freq.view(-1, count).T).view(
                leading_dims + (self.n_heads, self.head_dim // 2 // self.n_freq_groups)
            )
            for pos, freq, count in zip(split_positions, split_freqs, index_counts)
        ]
        rot_subvecs = [
            torch.polar(torch.ones_like(subvec), subvec) for subvec in rot_subvecs
        ]
        out = torch.cat(rot_subvecs, -1)
        assert out.shape == leading_dims + (self.n_heads, self.head_dim // 2)
        assert out.is_complex()
        return out

    def reset_parameters(self):
        freqs = init_nd_freqs(
            self.pos_dim,
            self.head_dim // self.n_freq_groups,
            self.n_heads,
            self._base_theta,
            dtype=self.freqs.dtype,
            device=self.freqs.device,
        )
        with torch.no_grad():
            self.freqs.copy_(freqs)


def prep_multilevel_positions(bijl_indices: Tensor, spatial_shapes: Tensor):
    assert bijl_indices.ndim == 2
    ij = bijl_indices[:, 1:-1]
    if not torch.is_floating_point(ij):
        # convert from indices to coordinates of pixel centers
        ij = ij + 0.5
    batch_level = torch.stack([bijl_indices[:, 0], bijl_indices[:, -1]], -1)
    assert ij.shape[-1] == spatial_shapes.shape[-1]
    assert spatial_shapes.ndim in (2, 3)  # batch, level, 2 or level, 2

    if spatial_shapes.ndim == 2:
        spatial_shapes = spatial_shapes.unsqueeze(0).expand(
            torch.unique(batch_level[:, 0]).shape[0], -1, -1
        )

    max_spatial_shape = spatial_shapes.max(-2)[0][batch_level[:, 0]]
    spatial_shapes = spatial_shapes[batch_level.unbind(-1)]

    rescaled_positions = ij * spatial_shapes / max_spatial_shape

    positions = bijl_indices.clone().to(rescaled_positions)
    positions[:, 1:3] = rescaled_positions
    return positions
