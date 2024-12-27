from typing import Optional

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


def init_t_xy(end_x: int, end_y: int, device: Optional[torch.device] = None):
    # t = torch.arange(end_x * end_y, dtype=torch.float, device=device)
    # t_x = (t % end_x).float()
    # t_y = torch.div(t, end_x, rounding_mode="floor").float()
    # return t_x, t_y
    t_x, t_y = torch.meshgrid(
        torch.arange(end_x, dtype=torch.float, device=device),
        torch.arange(end_y, dtype=torch.float, device=device),
        indexing="xy",
    )
    return t_x.flatten(), t_y.flatten()


def compute_mixed_cis(freqs: Tensor, t_x: Tensor, t_y: Tensor, num_heads: int):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.amp.autocast("cuda", enabled=False):
        freqs_x = (
            (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
            .view(N, num_heads, -1)
            .permute(1, 0, 2)
        )
        freqs_y = (
            (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
            .view(N, num_heads, -1)
            .permute(1, 0, 2)
        )
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    new_xq_shape = xq.shape[:-1]
    new_xq_shape.extend([-1, 2])
    xq_ = torch.view_as_complex(xq.float().reshape(new_xq_shape))
    new_xk_shape = xk.shape[:-1]
    new_xk_shape.extend([-1, 2])
    xk_ = torch.view_as_complex(xk.float().reshape(new_xk_shape))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


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
        key: Optional[Tensor],
        key_pos: Optional[Tensor],
    ):
        self.shape_check(query, query_pos)
        bsz = query.shape[0]
        target_len = query.shape[1]
        if key is None:
            assert key_pos is None
        else:
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

        if key is not None:
            # key_rot_vec = torch.einsum("blt,nht->blnh", key_pos, self.freqs)
            key_rot_vec = torch.mm(key_pos.view(-1, 2), self.freqs.view(-1, 2).T).view(
                [bsz, source_len, self.n_heads, self.head_dim // 2]
            )
            key_rot_vec = torch.polar(torch.ones_like(key_rot_vec), key_rot_vec)

            key = torch.view_as_complex(self.split_head_by_dim(key))
            key_rotated = torch.view_as_real(key * key_rot_vec).flatten(-2)
        else:
            key_rotated = query_rotated

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
