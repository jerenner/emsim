import torch
import torch.nn.functional as F
from torch import Tensor, nn
import copy
from timm.models.layers import DropPath

from typing import Optional
from emsim.networks.transformer.blocks import FFNBlock

from ...utils.window_utils import windowed_keys_for_queries
from ...utils.misc_utils import inverse_sigmoid

from ...utils.batching_utils import deconcat_add_batch_dim, remove_batch_dim_and_concat
from ..positional_encoding import (
    PixelPositionalEncoding,
    RelativePositionalEncodingTableInterpolate2D,
    SubpixelPositionalEncoding,
    FourierEncoding,
)
from ..ms_deform_attn import SparseMSDeformableAttention
from .blocks import SelfAttentionBlock, SparseDeformableAttentionBlock, FFNBlock


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        n_deformable_value_levels: int = 4,
        n_deformable_points: int = 4,
        dropout: float = 0.1,
        activation_fn: str = "gelu",
        norm_first: bool = True,
        attn_proj_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward

        self.self_attn = SelfAttentionBlock(
            d_model, n_heads, dropout, attn_proj_bias, norm_first=norm_first
        )
        self.msdeform_attn = SparseDeformableAttentionBlock(
            d_model,
            n_heads,
            n_deformable_value_levels,
            n_deformable_points,
            dropout,
            norm_first,
        )
        self.ffn = FFNBlock(
            d_model, dim_feedforward, dropout, activation_fn, norm_first
        )

    def forward(
        self,
        queries: Tensor,
        query_pos_encoding: Tensor,
        query_ij_indices: Tensor,
        query_normalized_xy_positions: Tensor,
        batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        spatial_shapes: Tensor,
    ):
        queries_batched, pad_mask = deconcat_add_batch_dim(queries, batch_offsets)
        pos_encoding_batched, pad_mask_2 = deconcat_add_batch_dim(
            query_pos_encoding, batch_offsets
        )
        assert torch.equal(pad_mask, pad_mask_2)

        x = self.self_attn(queries_batched, pos_encoding_batched, pad_mask)
        x, batch_offsets_2 = remove_batch_dim_and_concat(x, pad_mask)
        assert torch.equal(batch_offsets, batch_offsets_2)

        x = self.msdeform_attn(
            x,
            query_pos_encoding,
            query_normalized_xy_positions,
            batch_offsets,
            stacked_feature_maps,
            spatial_shapes,
        )
        x = self.ffn(x)
        return x

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        self.msdeform_attn.reset_parameters()
        self.ffn.reset_parameters()


class EMTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int = 6,
        layers_share_heads: bool = True,
        class_head: Optional[nn.Module] = None,
        position_offset_head: Optional[nn.Module] = None,
        look_forward_twice: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.d_model = decoder_layer.d_model
        self.look_forward_twice = look_forward_twice
        self.query_pos_encoding = FourierEncoding(
            2, decoder_layer.d_model, dtype=torch.double
        )

        self.layers_share_heads = layers_share_heads
        if self.layers_share_heads:
            if class_head is None or position_offset_head is None:
                raise ValueError(
                    "Expected `class_head` and `position_offset_head` to be specified "
                    "when layers_share_heads is True; got "
                    f"{class_head} and {position_offset_head}"
                )
            self.class_head = class_head
            self.position_offset_head = position_offset_head
        else:
            self.class_heads = nn.ModuleList(
                [nn.Linear(self.d_model, 1) for _ in range(num_layers)]
            )
            self.position_offset_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.d_model, self.d_model, dtype=torch.double),
                        nn.ReLU(),
                        nn.Linear(self.d_model, self.d_model, dtype=torch.double),
                        nn.ReLU(),
                        nn.Linear(self.d_model, 2, dtype=torch.double),
                    )
                    for _ in range(num_layers)
                ]
            )

        # self.ref_point_head = nn.Sequential(
        #     nn.Linear(2 * self.d_model, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, self.d_model),
        # )

        self.norm = nn.LayerNorm(self.d_model)

        self.reset_parameters()

    def forward(
        self,
        queries: Tensor,
        query_reference_points: Tensor,
        query_batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        spatial_shapes: Tensor,
    ):
        layer_output_logits = []
        layer_output_positions = []
        for i, layer in enumerate(self.layers):
            query_pos_encoding = self.query_pos_encoding(query_reference_points)
            queries = layer(
                queries,
                query_pos_encoding,
                None,
                query_reference_points.detach(),
                query_batch_offsets,
                stacked_feature_maps,
                spatial_shapes,
            )

            queries_normed = self.norm(queries)

            if self.layers_share_heads:
                class_head = self.class_head
                delta_pos_head = self.position_offset_head
            else:
                class_head = self.class_heads[i]
                delta_pos_head = self.position_offset_heads[i]

            query_logits = class_head(queries_normed)
            query_delta_pos = delta_pos_head(queries_normed.double())

            new_reference_points = torch.sigmoid(
                query_delta_pos + inverse_sigmoid(query_reference_points)
            )

            layer_output_logits.append(query_logits)
            if self.look_forward_twice:
                layer_output_positions.append(new_reference_points)
            else:
                layer_output_positions.append(new_reference_points.detach())

            query_reference_points = new_reference_points.detach()

        stacked_query_logits = torch.stack(layer_output_logits)
        stacked_query_positions = torch.stack(layer_output_positions)
        return stacked_query_logits, stacked_query_positions
