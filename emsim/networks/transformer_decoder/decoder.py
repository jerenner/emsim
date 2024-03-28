import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..positional_encoding import (
    RelativePositionalEncodingTableInterpolate2D,
    SubpixelPositionalEncoding,
    PixelPositionalEncoding,
)
from ...utils.sparse_utils import gather_from_sparse_tensor
from ...utils.batching_utils import deconcat_add_batch_dim, remove_batch_dim_and_concat


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        key_patch_size: int = 5,
        dropout: float = 0.1,
        activation_fn: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()

        # self.pixel_pos_encoding = PixelPositionalEncoding(d_model)
        # self.subpixel_pos_encoding = SubpixelPositionalEncoding(d_model)

        self.delta_pos_decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)
        )

        self.layers = nn.ModuleList(
            TransformerDecoderLayer(
                d_model,
                n_heads,
                dim_feedforward,
                key_patch_size,
                dropout,
                activation_fn,
                norm_first,
            )
            for _ in range(n_layers)
        )

    def forward(self, query_dict: dict[str, Tensor], image_feature_tensor: Tensor):
        for i, layer in enumerate(self.layers):
            x = layer(query_dict, image_feature_tensor)

            query_dict["queries"] = x

            delta_position = self.delta_pos_decoder(x)
            new_positions = (
                query_dict["indices"][..., 1:]
                + query_dict["subpixel_coordinates"]
                + delta_position
            )
            new_indices = torch.floor(new_positions).to(query_dict["indices"])
            new_subpixel_coordinates = torch.frac(new_positions)

            query_dict["indices"] = torch.cat(
                [query_dict["indices"][:, :1], new_indices], -1
            )
            query_dict["subpixel_coordinates"] = new_subpixel_coordinates

        return query_dict


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        key_patch_size: int = 5,
        dropout: float = 0.1,
        activation_fn="relu",
        norm_first=True,
    ):
        super().__init__()

        self.pixel_pos_encoding = PixelPositionalEncoding(d_model)
        self.subpixel_pos_encoding = SubpixelPositionalEncoding(d_model)

        self.self_attn = SelfAttentionBlock(
            d_model, n_heads, dropout, norm_first=norm_first
        )
        self.cross_attn = CrossAttentionBlock(
            d_model, n_heads, key_patch_size, dropout, norm_first=norm_first
        )
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation_fn=activation_fn)

    def forward(self, query_dict: dict[str, Tensor], image_feature_tensor: Tensor):
        image_size = query_dict["indices"].new_tensor(image_feature_tensor.shape[1:-1])
        pixel_encoding = self.pixel_pos_encoding(
            query_dict["indices"][..., -2:], image_size
        )
        subpixel_encoding = self.subpixel_pos_encoding(
            query_dict["subpixel_coordinates"]
        )

        x = query_dict["queries"] + pixel_encoding + subpixel_encoding

        t0 = time.time()
        x = self.self_attn(x, query_dict["batch_offsets"])
        torch.cuda.synchronize()
        print(f"SA block time = {time.time() - t0}")
        t0 = time.time()
        x = self.cross_attn(
            x,
            query_dict["indices"],
            query_dict["subpixel_coordinates"],
            image_feature_tensor,
        )
        torch.cuda.synchronize()
        print(f"CA block time = {time.time() - t0}")
        t0 = time.time()
        x = self.ffn(x)
        torch.cuda.synchronize()
        print(f"FFN block time = {time.time() - t0}")
        return x


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        norm_first: bool = True,
    ):
        assert activation_fn in ["gelu", "relu"]
        super().__init__()
        self.norm_first = norm_first
        if activation_fn == "relu":
            activation = nn.ReLU
        elif activation_fn == "gelu":
            activation = nn.GELU

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor):
        if self.norm_first:
            x = x + self.mlp(self.norm(x))
        else:
            x = self.norm(x + self.mlp(x))
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm_first = norm_first

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, bias=bias, batch_first=True
        )
        self.in_proj = nn.Linear(d_model, d_model * 3, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, batch_offsets: Tensor):
        # Add batch dimension for self-attention computation
        x, pad_mask = deconcat_add_batch_dim(x, batch_offsets)

        if self.norm_first:
            residual = x
            x = self.norm(x)
            x = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)[0]
            x = self.dropout(x)
            x = residual + x
        else:
            x = x + self.dropout(self.attn(x, x, x, need_weights=False)[0])
            x = self.norm(x)

        # remove batch dimension and return to stacked format
        x, batch_offsets_2 = remove_batch_dim_and_concat(x, pad_mask)
        assert torch.equal(batch_offsets, batch_offsets_2)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        key_patch_size=5,
        dropout: float = 0.0,
        bias: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm_first = norm_first

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.rel_pos_encoding = RelativePositionalEncodingTableInterpolate2D(
            d_model, key_patch_size, key_patch_size
        )

        self.key_patch_size = key_patch_size

    def forward(
        self,
        queries: Tensor,
        query_indices: Tensor,
        query_subpixel_coordinates: Tensor,
        image_feature_tensor: Tensor,
    ):
        if self.norm_first:
            residual = queries
            x = self.norm(queries)
            x = residual + self.dropout(
                self.attn(
                    queries,
                    query_indices,
                    query_subpixel_coordinates,
                    image_feature_tensor,
                )
            )
        else:
            x = x + self.dropout(
                self.attn(
                    queries,
                    query_indices,
                    query_subpixel_coordinates,
                    image_feature_tensor,
                )
            )
            x = self.norm(x)
        return x

    def attn(
        self,
        queries: Tensor,
        query_indices: Tensor,
        query_subpixel_coordinates: Tensor,
        image_feature_tensor: Tensor,
    ):
        n_queries = queries.shape[0]
        key_offsets = key_offset_grid(
            self.key_patch_size, self.key_patch_size, query_indices.device
        )
        qk_rel_posn_encoding = self.rel_pos_encoding(
            query_subpixel_coordinates, key_offsets
        )  # query x H x W x feat

        key_pixels = key_index_grid(query_indices, key_offsets)
        key_pixels[..., 1] = torch.clamp(
            key_pixels[..., 1], 0, image_feature_tensor.shape[1]
        )
        key_pixels[..., 2] = torch.clamp(
            key_pixels[..., 2], 0, image_feature_tensor.shape[2]
        )

        # queries x H x W x feat
        keys = gather_from_sparse_tensor(image_feature_tensor, key_pixels)

        # queries x HW x feat
        keys = keys.view(
            n_queries, self.key_patch_size * self.key_patch_size, self.d_model
        )
        qk_rel_posn_encoding = qk_rel_posn_encoding.view(
            n_queries, self.key_patch_size * self.key_patch_size, self.d_model
        )

        # compute input projections
        q = self.q_proj(queries)  # queries x feat
        k = self.k_proj(keys + qk_rel_posn_encoding)
        v = self.v_proj(keys)

        q = q.reshape(n_queries, self.n_heads, 1, self.head_dim)
        k = k.reshape(
            n_queries,
            self.key_patch_size * self.key_patch_size,
            self.n_heads,
            self.head_dim,
        )
        v = v.reshape(
            n_queries,
            self.key_patch_size * self.key_patch_size,
            self.n_heads,
            self.head_dim,
        )

        k = k.permute(0, 2, 1, 3)  # query x head x key x head dim
        v = v.permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
        attn = attn.permute(0, 2, 1, 3).reshape(-1, self.n_heads * self.head_dim)

        out = self.out_proj(attn)
        return out


def key_offset_grid(window_height: int, window_width: int, device=None):
    assert window_height % 2 == 1
    assert window_width % 2 == 1
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-(window_height // 2), window_height // 2 + 1, device=device),
        torch.arange(-(window_width // 2), window_width // 2 + 1, device=device),
        indexing="ij",
    )
    offsets = torch.stack([offsets_y, offsets_x], -1)
    return offsets


def key_index_grid(query_indices: Tensor, key_offsets: int):
    # add batch dim with 0 batch offset
    key_offsets = torch.cat(
        [
            key_offsets.new_zeros([key_offsets.shape[0], key_offsets.shape[1], 1]),
            key_offsets,
        ],
        -1,
    )
    return query_indices.unsqueeze(1).unsqueeze(1) + key_offsets
