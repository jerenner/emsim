from typing import Optional

import torch
from torch import Tensor, nn

from emsim.networks.positional_encoding.rope import (
    RoPEEncodingND,
    RoPEEncodingNDGroupedFreqs,
    prep_multilevel_positions,
)


class MultilevelIndependentRoPE(nn.Module):
    def __init__(
        self,
        n_levels: int,
        d_model: int,
        n_heads: int,
        rope_base_theta: int,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope_base_theta = rope_base_theta
        self.ropes = nn.ModuleList(
            RoPEEncodingND(
                position_dim=2,
                embed_dim=d_model,
                n_heads=n_heads,
                rope_base_theta=rope_base_theta,
                dtype=dtype,
            )
            for _ in range(n_levels)
        )

    def forward(
        self,
        q: Tensor,
        q_positions: Tensor,
        k: Tensor,
        k_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        query_leading_dims = q.shape[:-1]
        key_leading_dims = k.shape[:-1]
        split_head_dims = (self.n_heads, self.d_model // self.n_heads)
        level_arange = torch.arange(
            self.n_levels, device=q_positions.device, dtype=q_positions.dtype
        )
        assert torch.all(torch.isin(q_positions[..., -1], level_arange))
        # q_positions has last dim of [batch, i, j, level]
        split_position_indices_q = [q_positions[..., -1] == i for i in level_arange]
        if k_positions is not None:
            assert torch.all(torch.isin(k_positions[..., -1], level_arange))
            split_position_indices_k = [k_positions[..., -1] == i for i in level_arange]
        out_q = torch.zeros_like(q).view(query_leading_dims + split_head_dims)
        out_k = torch.zeros_like(k).view(key_leading_dims + split_head_dims)
        for i, rope in enumerate(self.ropes):
            level_q_indices = split_position_indices_q[i]
            level_q = q[level_q_indices]
            level_q_pos = q_positions[level_q_indices][..., -3:-1] + 0.5
            level_k_indices = (
                split_position_indices_k[i]
                if k_positions is not None
                else level_q_indices
            )
            level_k = k[level_k_indices]
            level_k_pos = (
                k_positions[level_k_indices][..., -3:-1] + 0.5
                if k_positions is not None
                else None
            )
            # if level_q.numel() > 0:
            if True:
                level_out_q, level_out_k = rope(
                    level_q, level_q_pos, level_k, level_k_pos
                )
                out_q[level_q_indices] = level_out_q.to(out_q)
                out_k[level_k_indices] = level_out_k.to(out_k)
        return out_q, out_k

    def reset_parameters(self):
        for module in self.ropes:
            module.reset_parameters()
