import torch
import torch.nn.functional as F
from torch import Tensor, nn
from timm.models.layers import DropPath

from emsim.networks.transformer.blocks import FFNBlock

from ...utils.window_utils import windowed_keys_for_queries

from ...utils.batching_utils import deconcat_add_batch_dim, remove_batch_dim_and_concat
from ..positional_encoding import (
    PixelPositionalEncoding,
    RelativePositionalEncodingTableInterpolate2D,
    SubpixelPositionalEncoding,
)
from ..ms_deform_attn import SparseMSDeformableAttention


class EMTransformerDecoder(nn.Module):
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
        return_intermediate_outputs: bool = False,
    ):
        super().__init__()

        # self.pixel_pos_encoding = PixelPositionalEncoding(d_model)
        # self.subpixel_pos_encoding = SubpixelPositionalEncoding(d_model)

        self.delta_pos_decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2), nn.Tanh()
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
        self.return_intermediate_outputs = return_intermediate_outputs

    def forward(self, in_query_dict: dict[str, Tensor], image_feature_tensor: Tensor):
        if self.return_intermediate_outputs:
            intermediates = []

        for i, layer in enumerate(self.layers):
            if self.return_intermediate_outputs:
                intermediates.append(in_query_dict)
            x = layer(in_query_dict, image_feature_tensor)

            out_query_dict = {}
            out_query_dict["occupancy_logits"] = in_query_dict["occupancy_logits"]
            out_query_dict["batch_offsets"] = in_query_dict["batch_offsets"]

            out_query_dict["queries"] = x

            delta_position = self.delta_pos_decoder(x)
            new_positions = in_query_dict["positions"] + delta_position
            new_positions = torch.clamp(
                new_positions,
                new_positions.new_tensor(0.0),
                new_positions.new_tensor(
                    [[image_feature_tensor.shape[1], image_feature_tensor.shape[2]]]
                ),
            )
            new_indices = torch.floor(new_positions).to(in_query_dict["indices"])
            new_subpixel_coordinates = torch.frac(new_positions)

            out_query_dict["positions"] = new_positions
            out_query_dict["indices"] = torch.cat(
                [in_query_dict["indices"][:, :1], new_indices], -1
            )
            out_query_dict["subpixel_coordinates"] = new_subpixel_coordinates

            in_query_dict = out_query_dict

        if self.return_intermediate_outputs:
            return out_query_dict, intermediates
        else:
            return out_query_dict


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        key_patch_size: int = 5,
        dropout: float = 0.1,
        activation_fn: str = "relu",
        norm_first: bool = True,
        attn_proj_bias: bool = False,
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
        self.ffn = FFNBlock(d_model, dim_feedforward, dropout, activation_fn=activation_fn)

    def forward(self, query_dict: dict[str, Tensor], image_feature_tensor: Tensor):
        image_size = query_dict["indices"].new_tensor(image_feature_tensor.shape[1:-1])
        pixel_encoding = self.pixel_pos_encoding(
            query_dict["indices"][..., -2:], image_size
        )
        subpixel_encoding = self.subpixel_pos_encoding(
            query_dict["subpixel_coordinates"]
        )

        x = query_dict["queries"] + pixel_encoding + subpixel_encoding

        x = self.self_attn(x, query_dict["batch_offsets"])
        x = self.cross_attn(
            x,
            query_dict["indices"],
            query_dict["subpixel_coordinates"],
            image_feature_tensor,
        )
        x = self.ffn(x)
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

        keys, _, key_offsets, key_pad_mask, _ = windowed_keys_for_queries(
            query_indices,
            image_feature_tensor,
            self.key_patch_size,
            self.key_patch_size,
        )
        qk_rel_posn_encoding = self.rel_pos_encoding(
            query_subpixel_coordinates, key_offsets
        )  # query x H x W x feat

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
        key_pad_mask = key_pad_mask.reshape(
            n_queries, 1, 1, self.key_patch_size * self.key_patch_size
        ).expand(-1, self.n_heads, -1, -1)
        key_pad_mask = key_pad_mask.logical_not()

        k = k.permute(0, 2, 1, 3)  # query x head x key x head dim
        v = v.permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=key_pad_mask, dropout_p=self.dropout.p
        )
        if torch.any(attn.isinf()):
            raise ValueError("Got inf attention output")
        attn = attn.permute(0, 2, 1, 3).reshape(-1, self.n_heads * self.head_dim)

        out = self.out_proj(attn)
        return out
