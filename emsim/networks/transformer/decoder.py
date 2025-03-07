import torch
import torch.nn.functional as F
from torch import Tensor, nn
import copy
from timm.models.layers import DropPath

from typing import Optional, Union

from ...utils.misc_utils import inverse_sigmoid

from ...utils.batching_utils import deconcat_add_batch_dim, remove_batch_dim_and_concat
from ...utils.sparse_utils import (
    batch_offsets_from_sparse_tensor_indices,
    sparse_tensor_to_batched,
    multilevel_normalized_xy,
)
from ..positional_encoding import (
    PixelPositionalEncoding,
    RelativePositionalEncodingTableInterpolate2D,
    SubpixelPositionalEncoding,
    FourierEncoding,
)
from .std_dev_head import StdDevHead
from .blocks import (
    SelfAttentionBlock,
    SparseDeformableAttentionBlock,
    FFNBlock,
    MultilevelCrossAttentionBlockWithRoPE,
    MultilevelSelfAttentionBlockWithRoPE,
)
from ..segmentation_map import PatchedSegmentationMapPredictor


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        cross_attn_type: str = "ms_deform_attn",
        n_deformable_value_levels: int = 4,
        n_deformable_points: int = 4,
        predict_box: bool = False,
        dropout: float = 0.1,
        activation_fn: Union[str, nn.Module] = "gelu",
        norm_first: bool = True,
        attn_proj_bias: bool = False,
        self_attn_use_rope: bool = True,
        rope_base_theta: float = 10.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.self_attn_rope = self_attn_use_rope
        self.cross_attn_type = cross_attn_type
        self.use_rope = self_attn_use_rope
        self.predict_box = predict_box

        if self_attn_use_rope:
            self.self_attn = MultilevelSelfAttentionBlockWithRoPE(
                d_model,
                n_heads,
                n_deformable_value_levels,
                6 if predict_box else 2,
                dropout,
                attn_proj_bias,
                norm_first,
                rope_theta=rope_base_theta,
            )
        else:
            self.self_attn = SelfAttentionBlock(
                d_model, n_heads, dropout, attn_proj_bias, norm_first=norm_first
            )
        if cross_attn_type == "ms_deform_attn":
            self.cross_attn = SparseDeformableAttentionBlock(
                d_model,
                n_heads,
                n_deformable_value_levels,
                n_deformable_points,
                dropout,
                norm_first,
            )
        elif cross_attn_type == "multi_head_attn":
            self.cross_attn = MultilevelCrossAttentionBlockWithRoPE(
                d_model,
                n_heads,
                n_deformable_value_levels,
                dropout,
                attn_proj_bias,
                norm_first=norm_first,
            )
        self.ffn = FFNBlock(
            d_model, dim_feedforward, dropout, activation_fn, norm_first
        )
        self.pos_encoding = None

    def forward(
        self,
        queries: Tensor,
        query_pos_encoding: Tensor,
        query_normalized_xy_positions: Tensor,
        batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        spatial_shapes: Tensor,
        attn_mask: Optional[Tensor] = None,
    ):
        if self.self_attn_rope:
            max_spatial_shape, max_level_index = spatial_shapes.max(dim=0)
            max_level_index = torch.unique(max_level_index)
            assert len(max_level_index) == 1
            if self.predict_box:
                raise NotImplementedError("Box prediction not finished yet")
                spatial_scaler = max_spatial_shape.repeat(3)
                query_positions_ij = (
                    torch.cat(
                        [
                            query_normalized_xy_positions[..., :2].flip(-1),
                            query_normalized_xy_positions[..., 2:],
                        ],
                        -1,
                    )
                    * spatial_scaler
                )
            else:
                query_positions_ij = (
                    query_normalized_xy_positions.flip(-1) * max_spatial_shape
                )
            x = self.self_attn(
                queries,
                query_positions_ij,
                max_level_index.expand(query_positions_ij.shape[0]),
                batch_offsets,
                attn_mask=attn_mask,
            )
        else:
            x = self.self_attn(
                queries,
                query_pos_encoding,
                attn_mask=attn_mask,
                batch_offsets=batch_offsets,
            )

        if self.cross_attn_type == "ms_deform_attn":
            x = self.cross_attn(
                x,
                query_pos_encoding,
                query_normalized_xy_positions,
                batch_offsets,
                stacked_feature_maps,
                spatial_shapes,
            )
        elif self.cross_attn_type == "multi_head_attn":
            # batched_values, value_indices, value_pad_mask = sparse_tensor_to_batched(
            #     stacked_feature_maps
            # )
            # value_pos_normalized_xy = multilevel_normalized_xy(
            #     stacked_feature_maps, spatial_shapes
            # )
            x = self.cross_attn(
                query=x,
                query_normalized_xy_positions=query_normalized_xy_positions,
                query_batch_offsets=batch_offsets,
                stacked_feature_maps=stacked_feature_maps,
                level_spatial_shapes=spatial_shapes,
            )
        x = self.ffn(x)
        return x

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        self.cross_attn.reset_parameters()
        self.ffn.reset_parameters()


class EMTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int = 6,
        predict_box: bool = False,
        layers_share_heads: bool = True,
        class_head: Optional[nn.Module] = None,
        position_offset_head: Optional[nn.Module] = None,
        std_head: Optional[nn.Module] = None,
        segmentation_head: Optional[nn.Module] = None,
        look_forward_twice: bool = True,
        detach_updated_positions: bool = True,
    ):
        super().__init__()
        self.layers: list[TransformerDecoderLayer] = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.d_model = decoder_layer.d_model
        self.predict_box = predict_box
        self.look_forward_twice = look_forward_twice
        self.detach_updated_positions = detach_updated_positions
        self.use_rope = self.layers[0].use_rope
        if self.use_rope:
            self.query_pos_encoding = nn.Identity()
        else:
            assert not predict_box, "Box prediction only implemented for RoPE"
            self.query_pos_encoding = FourierEncoding(
                2, decoder_layer.d_model, dtype=torch.double
            )

        self.layers_share_heads = layers_share_heads
        if self.layers_share_heads:
            if (
                class_head is None
                or position_offset_head is None
                or std_head is None
                or segmentation_head is None
            ):
                raise ValueError(
                    "Expected `class_head`, `position_offset_head`, "
                    "`std_head`, and `segmentation_head` to be specified when "
                    "layers_share_heads is True; got "
                    f"{class_head}, {position_offset_head}, {std_head}, "
                    f"and {segmentation_head}"
                )
            self.class_head = class_head
            self.position_offset_head = position_offset_head
            self.std_head = std_head
            self.segmentation_head = segmentation_head
            self.per_layer_class_heads = None
            self.per_layer_position_heads = None
            self.per_layer_std_heads = None
            self.per_layer_segmentation_heads = None
        else:
            self.class_head = None
            self.position_offset_head = None
            self.std_head = None
            self.segmentation_head = None
            self.per_layer_class_heads = nn.ModuleList(
                [nn.Linear(self.d_model, 1) for _ in range(num_layers)]
            )
            self.per_layer_position_heads = nn.ModuleList(
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
            self.per_layer_std_heads = nn.ModuleList(
                [StdDevHead(self.d_model) for _ in range(num_layers)]
            )
            self.per_layer_segmentation_heads = nn.ModuleList(
                [
                    PatchedSegmentationMapPredictor(self.d_model)
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
        attn_mask: Optional[Tensor] = None,
    ):
        layer_output_logits = []
        layer_output_positions = []
        layer_output_queries = []
        layer_output_std = []
        layer_output_segmentation = []
        for i, layer in enumerate(self.layers):
            if not self.use_rope:
                query_pos_encoding = self.query_pos_encoding(query_reference_points)
                query_pos_encoding = query_pos_encoding.to(queries)
            else:
                query_pos_encoding = None
            queries = layer(
                queries,
                query_pos_encoding,
                query_reference_points.detach(),
                query_batch_offsets,
                stacked_feature_maps,
                spatial_shapes,
                attn_mask,
            )

            queries_normed = self.norm(queries)

            class_head = self._get_class_head(i)
            delta_pos_head = self._get_position_head(i)
            std_head = self._get_std_head(i)
            segmentation_head = self._get_segmentation_head(i)

            query_logits = class_head(queries_normed)
            query_delta_pos = delta_pos_head(queries_normed.to(delta_pos_head.dtype))
            query_std = std_head(queries_normed)

            new_reference_points = torch.sigmoid(
                query_delta_pos + inverse_sigmoid(query_reference_points)
            )

            query_segmentation = segmentation_head(
                stacked_feature_maps,
                queries_normed,
                query_batch_offsets,
                new_reference_points,
            )

            layer_output_logits.append(query_logits)
            layer_output_std.append(query_std)
            if self.look_forward_twice:
                layer_output_positions.append(new_reference_points)
            else:
                layer_output_positions.append(new_reference_points.detach())
            layer_output_queries.append(queries)
            layer_output_segmentation.append(query_segmentation)

            if self.detach_updated_positions:
                query_reference_points = new_reference_points.detach()
            else:
                query_reference_points = new_reference_points

        stacked_query_logits = torch.stack(layer_output_logits)
        stacked_query_positions = torch.stack(layer_output_positions)
        stacked_queries = torch.stack(layer_output_queries)
        stacked_std = torch.stack(layer_output_std)
        return {
            "logits": stacked_query_logits,
            "positions": stacked_query_positions,
            "queries": stacked_queries,
            "std": stacked_std,
            "segmentation_logits": layer_output_segmentation,
        }

    def _get_class_head(self, layer_index):
        if self.layers_share_heads:
            return self.class_head
        else:
            return self.per_layer_class_heads[layer_index]

    def _get_position_head(self, layer_index):
        if self.layers_share_heads:
            return self.position_offset_head
        else:
            return self.per_layer_position_heads[layer_index]

    def _get_std_head(self, layer_index):
        if self.layers_share_heads:
            return self.std_head
        else:
            return self.per_layer_std_heads[layer_index]

    def _get_segmentation_head(self, layer_index):
        if self.layers_share_heads:
            return self.segmentation_head
        else:
            return self.per_layer_segmentation_heads[layer_index]

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.per_layer_class_heads is not None:
            for head in self.per_layer_class_heads:
                head.reset_parameters()
        if self.per_layer_position_heads is not None:
            for head in self.per_layer_position_heads:
                for layer in head:
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
        if self.per_layer_std_heads is not None:
            for head in self.per_layer_std_heads:
                head.reset_parameters()
        if self.per_layer_segmentation_heads is not None:
            for head in self.per_layer_segmentation_heads:
                head.reset_parameters()
