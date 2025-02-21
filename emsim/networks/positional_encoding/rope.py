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
    return freqs  # n_head, head_dim/2, pos_dim


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
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0
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
        key: Tensor,
        key_pos: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        self.shape_check(query, query_pos)
        if query_pos.numel() > 0 and query_pos.max() <= 1.0:
            warnings.warn(
                "Expected un-normalized (i.e., not inside [0,1]) coordinates "
                "for position but found normalized coordinates. Did you accidentally "
                "pass in normalized coordinates?\n"
                f"Coord range: [{query_pos.min(), query_pos.max()}]"
            )
        if key_pos is not None:
            self.shape_check(key, key_pos)
        query_rot_vec = self._make_rot_vec(query_pos)
        query_rotated = self._apply_rot_vec(query, query_rot_vec)

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
        assert query_or_key.ndim == query_or_key_pos.ndim  # ..., seq_len, d_model
        # assert query_or_key_pos.ndim == 3  # batch, seq_len, pos_dim
        assert query_or_key.shape[-1] == self.d_model
        assert query_or_key_pos.shape[-1] == self.pos_dim
        assert query_or_key.shape[:-1] == query_or_key_pos.shape[:-1]

    def reset_parameters(self):
        freqs = init_nd_freqs(
            self.pos_dim, self.head_dim, self.n_heads, self._base_theta
        )
        with torch.no_grad():
            self.freqs.copy_(freqs)


class RoPEEncodingNDGroupedFreqs(RoPEEncodingND):
    def __init__(
        self,
        position_dim: int,
        d_model: int,
        n_heads: int,
        pos_dim_to_rope_dim: Union[Tensor, list[int]],
        rope_base_theta: float = 10.0,
        dtype=torch.float,
    ):
        self.pos_dim_to_rope_group = torch.as_tensor(pos_dim_to_rope_dim)
        assert self.pos_dim_to_rope_group.ndim == 1
        assert len(self.pos_dim_to_rope_group) == position_dim
        self.n_freq_groups = len(self.pos_dim_to_rope_group.unique())
        super().__init__(position_dim, d_model, n_heads, rope_base_theta, dtype)
        assert self.head_dim % self.n_freq_groups == 0

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
            self.pos_dim, self.head_dim // self.n_freq_groups, self.n_heads, self._base_theta
        )
        with torch.no_grad():
            self.freqs.copy_(freqs)


def rescale_multilevel_positions_to_finest(
    multilevel_positions: list[Tensor], spatial_shapes: Tensor
) -> Tensor:
    assert spatial_shapes.ndim == 2
    max_spatial_shape = spatial_shapes.max(dim=0)[0]
    rescaled_positions = [
        pos / shape * max_spatial_shape
        for pos, shape in zip(multilevel_positions, spatial_shapes)
    ]
    return rescaled_positions


def prep_multilevel_positions(indices: Tensor, spatial_shapes: Tensor):
    assert indices.ndim == 2
    ij = indices[:, -3:-1] + 0.5
    assert ij.shape[-1] == 2
    level_indices = torch.unique(indices[..., -1])
    level_to_position_indices = [indices[..., -1] == index for index in level_indices]
    level_split_positions = [
        ij[level_indices] for level_indices in level_to_position_indices
    ]
    rescaled_split_positions = rescale_multilevel_positions_to_finest(
        level_split_positions, spatial_shapes
    )
    rescaled_positions = torch.cat(rescaled_split_positions, 0)
    positions = indices.clone().to(rescaled_positions)
    positions[:, 1:3] = rescaled_positions
    return positions
