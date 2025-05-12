import copy
from typing import Optional, Union

import torch
from torch import Tensor, nn

from ...utils.misc_utils import (
    inverse_sigmoid,
)
from ..positional_encoding import (
    FourierEncoding,
)
from ..segmentation_map import PatchedSegmentationMapPredictor
from .blocks import (
    FFNBlock,
    MultilevelCrossAttentionBlockWithRoPE,
    MultilevelSelfAttentionBlockWithRoPE,
    SelfAttentionBlock,
    SparseDeformableAttentionBlock,
    SparseNeighborhoodAttentionBlock,
)
from .std_dev_head import StdDevHead
from emsim.config.transformer import RoPEConfig, TransformerDecoderConfig


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        n_feature_levels: int = 4,
        use_ms_deform_attn: bool = True,
        n_deformable_points: int = 4,
        use_neighborhood_attn: bool = True,
        neighborhood_sizes: list[int] = [3, 5, 7, 9],
        use_full_cross_attn: bool = False,
        use_rope: bool = True,
        rope_config: Optional[RoPEConfig] = None,
        dropout: float = 0.1,
        activation_fn: Union[str, nn.Module] = "gelu",
        norm_first: bool = True,
        attn_proj_bias: bool = False,
        predict_box: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.use_ms_deform_attn = use_ms_deform_attn
        self.use_neighborhood_attn = use_neighborhood_attn
        self.use_full_cross_attn = use_full_cross_attn
        self.use_rope = use_rope
        self.predict_box = predict_box

        if use_rope:
            self.self_attn = MultilevelSelfAttentionBlockWithRoPE(
                d_model,
                n_heads,
                (
                    rope_config.spatial_dimension + 4
                    if predict_box
                    else rope_config.spatial_dimension
                ),
                dropout,
                attn_proj_bias,
                norm_first,
                rope_spatial_base_theta=rope_config.spatial_base_theta,
                rope_level_base_theta=rope_config.level_base_theta,
                rope_share_heads=rope_config.share_heads,
                rope_freq_group_pattern=rope_config.freq_group_pattern,
                rope_enforce_freq_groups_equal=rope_config.enforce_freq_groups_equal,
            )
        else:
            self.self_attn = SelfAttentionBlock(
                d_model, n_heads, dropout, attn_proj_bias, norm_first=norm_first
            )
        if use_ms_deform_attn:
            self.ms_deform_attn = SparseDeformableAttentionBlock(
                d_model,
                n_heads,
                n_feature_levels,
                n_deformable_points,
                dropout,
                norm_first,
            )
        else:
            self.ms_deform_attn = None
        if use_neighborhood_attn:
            assert rope_config is not None
            assert neighborhood_sizes is not None
            self.neighborhood_attn = SparseNeighborhoodAttentionBlock(
                d_model,
                n_heads,
                n_feature_levels,
                neighborhood_sizes=neighborhood_sizes,
                position_dim=rope_config.spatial_dimension,
                dropout=dropout,
                bias=attn_proj_bias,
                norm_first=norm_first,
                rope_spatial_base_theta=rope_config.spatial_base_theta,
                rope_level_base_theta=rope_config.level_base_theta,
                rope_share_heads=rope_config.share_heads,
                rope_freq_group_pattern=rope_config.freq_group_pattern,
                rope_enforce_freq_groups_equal=rope_config.enforce_freq_groups_equal,
            )
        else:
            self.neighborhood_attn = None
        if use_full_cross_attn:
            self.full_cross_attn = MultilevelCrossAttentionBlockWithRoPE(
                d_model,
                n_heads,
                n_feature_levels,
                dropout,
                attn_proj_bias,
                norm_first=norm_first,
            )
        else:
            self.full_cross_attn = None
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
        if self.use_ms_deform_attn:
            x = self.ms_deform_attn(
                x,
                query_pos_encoding,
                query_normalized_xy_positions,
                batch_offsets,
                stacked_feature_maps,
                spatial_shapes,
            )
        if self.use_neighborhood_attn:
            x = self.neighborhood_attn(
                x,
                query_positions_ij,
                batch_offsets,
                stacked_feature_maps,
                spatial_shapes,
            )
        if self.use_full_cross_attn:
            x = self.full_cross_attn(
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
        if hasattr(self.ms_deform_attn, "reset_parameters"):
            self.ms_deform_attn.reset_parameters()
        if hasattr(self.neighborhood_attn, "reset_parameters"):
            self.neighborhood_attn.reset_parameters()
        if hasattr(self.full_cross_attn, "reset_parameters"):
            self.full_cross_attn.reset_parameters()
        self.ffn.reset_parameters()


class EMTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        config: TransformerDecoderConfig,
        class_head: Optional[nn.Module] = None,
        position_offset_head: Optional[nn.Module] = None,
        std_head: Optional[nn.Module] = None,
        segmentation_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers: list[TransformerDecoderLayer] = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(config.n_layers)]
        )
        self.n_layers = config.n_layers
        self.d_model = decoder_layer.d_model
        self.predict_box = config.predict_box
        self.look_forward_twice = config.look_forward_twice
        self.detach_updated_positions = config.detach_updated_positions
        self.use_rope = config.use_rope
        if self.use_rope:
            self.query_pos_encoding = nn.Identity()
        else:
            assert not config.predict_box, "Box prediction only implemented for RoPE"
            self.query_pos_encoding = FourierEncoding(
                2, decoder_layer.d_model, dtype=torch.double
            )

        self.layers_share_heads = config.layers_share_heads
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
                [nn.Linear(self.d_model, 1) for _ in range(config.n_layers)]
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
                    for _ in range(config.n_layers)
                ]
            )
            self.per_layer_std_heads = nn.ModuleList(
                [StdDevHead(self.d_model) for _ in range(config.n_layers)]
            )
            self.per_layer_segmentation_heads = nn.ModuleList(
                [
                    PatchedSegmentationMapPredictor(self.d_model)
                    for _ in range(config.n_layers)
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
