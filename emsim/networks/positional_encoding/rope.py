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
        self, d_model: int, n_heads: int, rope_theta: float = 10.0, dtype=torch.float
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self._base_theta = rope_theta
        self.head_dim = d_model // n_heads
        self.pos_dim = 2
        assert self.head_dim % self.pos_dim == 0

        freqs = init_2d_freqs(self.head_dim, n_heads, rope_theta, dtype=dtype)
        assert freqs.shape == (n_heads, self.head_dim, 2)
        self.freqs = nn.Parameter(freqs)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key: Tensor,
        key_pos: Optional[Tensor] = None,
    ):
        self.shape_check(query, query_pos)
        if query_pos.max() <= 1.0:
            warnings.warn(
                "Expected un-normalized (i.e., not inside [0,1]) coordinates"
                "for position but found normalized coordinates. Did you accidentally"
                "pass in normalized coordinates?"
            )
        bsz = query.shape[0]
        target_len = query.shape[1]
        if key_pos is not None:
            self.shape_check(key, key_pos)
            assert key.shape[0] == key_pos.shape[0] == bsz
        source_len = key.shape[1]

        # query_rot_vec = torch.einsum("blt,nht->blnh", query_pos, self.freqs)
        query_rot_vec = torch.mm(query_pos.view(-1, 2), self.freqs.view(-1, 2).T).view(
            [bsz, target_len, self.n_heads, self.head_dim // 2]
        )
        query_rot_vec = torch.polar(torch.ones_like(query_rot_vec), query_rot_vec)

        query = torch.view_as_complex(self.split_head_by_dim(query))
        query_rotated = torch.view_as_real(query * query_rot_vec).flatten(-2)

        if key_pos is not None:
            # key_rot_vec = torch.einsum("blt,nht->blnh", key_pos, self.freqs)
            key_rot_vec = torch.mm(key_pos.view(-1, 2), self.freqs.view(-1, 2).T).view(
                [bsz, source_len, self.n_heads, self.head_dim // 2]
            )
            key_rot_vec = torch.polar(torch.ones_like(key_rot_vec), key_rot_vec)
        else:
            key_rot_vec = query_rot_vec
        key = torch.view_as_complex(self.split_head_by_dim(key))
        key_rotated = torch.view_as_real(key * key_rot_vec).flatten(-2)

        return query_rotated, key_rotated

    def shape_check(self, query_or_key: Tensor, query_or_key_pos: Tensor):
        assert query_or_key.ndim == 3  # batch, seq_len, d_model
        assert query_or_key_pos.ndim == 3  # batch, seq_len, 2
        assert query_or_key.shape[2] == self.d_model
        assert query_or_key_pos.shape[2] == self.pos_dim
        assert query_or_key.shape[0] == query_or_key_pos.shape[0]
        assert query_or_key_pos.shape[1] == query_or_key_pos.shape[1]

    def split_head_by_dim(self, tensor: Tensor):
        assert tensor.shape[-1] % self.pos_dim == 0
        out_shape = tensor.shape[:-1]
        out_shape.extend([tensor.shape // self.pos_dim, self.pos_dim])
        return tensor.view(out_shape)

    def reset_parameters(self):
        freqs = init_2d_freqs(self.head_dim, self.n_heads, self._base_theta)
        with torch.no_grad():
            self.freqs.copy_(freqs)
