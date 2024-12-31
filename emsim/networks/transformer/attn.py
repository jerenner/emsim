from emsim.networks.positional_encoding import RoPEEncodingND
from emsim.utils.batching_utils import deconcat_add_batch_dim, remove_batch_dim_and_concat


import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SelfAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.pos_encoding = RoPEEncodingND
        (n_heads, d_model // n_heads)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, queries: Tensor, query_positions: Tensor, query_pad_mask: Tensor):
        q, k, v = self.in_proj(queries).chunk(3, -1)

        q, k = self.pos_encoding(q, k, query_positions)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, query_pad_mask, dropout_p)

        out = self.out_proj(out)
        return out

    def reset_parameters(self):
        self.in_proj.reset_parameters()
        self.pos_encoding.reset_parameters()
        self.out_proj.reset_parameters()
