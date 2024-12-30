from typing import Optional
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
):
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


class RoPEEncoding2D(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope_base_theta: float = 10.0,
        dtype=torch.float,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self._base_theta = rope_base_theta
        self.head_dim = d_model // n_heads
        self.pos_dim = 2
        assert self.head_dim % self.pos_dim == 0

        freqs = init_2d_freqs(self.head_dim, n_heads, rope_base_theta, dtype=dtype)
        assert freqs.shape == (n_heads, self.head_dim // 2, 2)
        self.freqs = nn.Parameter(freqs)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key: Tensor,
        key_pos: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        # if query.ndim == 2:
        #     assert query_pos.ndim == 2
        #     assert key.ndim == 2
        #     if key_pos is not None:
        #         assert key_pos.ndim == 2
        #     unbatched = True
        #     query = query.unsqueeze(0)
        #     query_pos = query_pos.unsqueeze(0)
        #     key = key.unsqueeze(0)
        #     key_pos = key_pos.unsqueeze(0) if key_pos else key_pos
        # else:
        #     unbatched = False
        self.shape_check(query, query_pos)
        if query_pos.max() <= 1.0:
            warnings.warn(
                "Expected un-normalized (i.e., not inside [0,1]) coordinates"
                "for position but found normalized coordinates. Did you accidentally"
                "pass in normalized coordinates?"
            )
        # bsz = query.shape[0]
        # tgt_len = query.shape[1]
        query_leading_dims = query.shape[:-1]
        if key_pos is not None:
            self.shape_check(key, key_pos)
        # src_len = key.shape[1]
        key_leading_dims = key.shape[:-1]

        # query_rot_vec = torch.einsum("blt,nht->blnh", query_pos, self.freqs)
        query_rot_vec = self._make_rot_vec(query_pos)

        # query = query.view(
        #     # bsz, tgt_len, self.n_heads, self.head_dim // self.pos_dim, self.pos_dim
        #     query_leading_dims
        #     + (self.n_heads, self.head_dim // self.pos_dim, self.pos_dim)
        # )
        # query = torch.view_as_complex(query)
        # query_rotated = torch.view_as_real(query * query_rot_vec).flatten(-2)
        query_rotated = self._apply_rot_vec(query, query_rot_vec)

        if key_pos is not None:
            # key_rot_vec = torch.einsum("blt,nht->blnh", key_pos, self.freqs)
            # key_rot_vec = torch.mm(
            #     key_pos.view(-1, self.pos_dim),
            #     self.freqs.view(-1, self.pos_dim).T,
            #     # ).view([bsz, src_len, self.n_heads, self.head_dim // 2])
            # ).view(key_leading_dims + (self.n_heads, self.head_dim // 2))
            # key_rot_vec = torch.polar(torch.ones_like(key_rot_vec), key_rot_vec)
            key_rot_vec = self._make_rot_vec(key_pos)
        else:
            key_rot_vec = query_rot_vec
        # key = key.view(
        #     # bsz, src_len, self.n_heads, self.head_dim // self.pos_dim, self.pos_dim
        #     key_leading_dims
        #     + (self.n_heads, self.head_dim // 2, 2)
        # )
        # key = torch.view_as_complex(key)
        # key_rotated = torch.view_as_real(key * key_rot_vec).flatten(-2)
        key_rotated = self._apply_rot_vec(key, key_rot_vec)

        #  out dim: batch x seq_len x n_heads x head_dim
        # if unbatched:
        #     # out dim: seq_len x n_heads x head_dim
        #     query_rotated = query_rotated.squeeze(0)
        #     key_rotated = key_rotated.squeeze(0)
        return query_rotated, key_rotated

    def _make_rot_vec(self, positions: Tensor) -> Tensor:
        leading_dims = positions.shape[:-1]
        rot_vec = torch.mm(
            positions.view(-1, self.pos_dim).to(self.freqs),
            self.freqs.view(-1, self.pos_dim).T,
            # ).view([bsz, tgt_len, self.n_heads, self.head_dim // self.pos_dim])
        ).view(leading_dims + (self.n_heads, self.head_dim // 2))
        return torch.polar(torch.ones_like(rot_vec), rot_vec)

    def _apply_rot_vec(self, query_or_key: Tensor, rot_vec: Tensor):
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
        freqs = init_2d_freqs(self.head_dim, self.n_heads, self._base_theta)
        with torch.no_grad():
            self.freqs.copy_(freqs)
