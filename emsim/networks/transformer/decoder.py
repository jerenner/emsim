import copy
from typing import Optional, Union

import torch
from torch import Tensor, nn

from ..positional_encoding import (
    FourierEncoding,
)
from ..segmentation_map import PatchedSegmentationMapPredictor
from .blocks import (
    FFNBlock,
    MultilevelCrossAttentionBlockWithRoPE,
    MultilevelSelfAttentionBlockWithRoPE,
    SelfAttentionBlock,
    SparseMSDeformableAttentionBlock,
    SparseNeighborhoodAttentionBlock,
)
from .position_head import PositionOffsetHead
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
        activation_fn: Union[str, type[nn.Module]] = "gelu",
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
        self.dropout = dropout
        self.attn_proj_bias = attn_proj_bias
        self.predict_box = predict_box

        if use_rope:
            assert rope_config is not None
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
            self.ms_deform_attn = SparseMSDeformableAttentionBlock(
                d_model,
                n_heads,
                n_feature_levels,
                n_deformable_points,
                dropout,
                bias=attn_proj_bias,
                norm_first=norm_first,
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
        queries: Tensor,  # shape: [query_batch_offsets[-1]]
        query_batch_offsets: Tensor,
        query_spatial_positions: Tensor,
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
        attn_mask: Optional[Tensor] = None,
        background_embedding: Optional[Tensor] = None,
    ):
        max_level_index = level_spatial_shapes.argmax(dim=0)
        max_level_index = torch.unique(max_level_index)
        assert len(max_level_index) == 1
        query_level_indices = max_level_index.expand(query_spatial_positions.shape[0])
        if self.use_rope:
            if self.predict_box:
                raise NotImplementedError("Box prediction not finished yet")
            x = self.self_attn(
                queries,
                spatial_positions=query_spatial_positions,
                level_indices=query_level_indices,
                level_spatial_shapes=level_spatial_shapes,
                batch_offsets=query_batch_offsets,
                attn_mask=attn_mask,
            )
        else:
            raise ValueError("Non-RoPE not supported.")
        if self.ms_deform_attn is not None:
            x = self.ms_deform_attn(
                x,
                query_spatial_positions,
                query_batch_offsets,
                stacked_feature_maps,
                level_spatial_shapes,
                background_embedding=background_embedding,
                query_level_indices=query_level_indices,
            )
        if self.neighborhood_attn is not None:
            x = self.neighborhood_attn(
                x,
                query_spatial_positions=query_spatial_positions,
                query_batch_offsets=query_batch_offsets,
                stacked_feature_maps=stacked_feature_maps,
                level_spatial_shapes=level_spatial_shapes,
                background_embedding=background_embedding,
                query_level_indices=query_level_indices,
            )
        if self.use_full_cross_attn:
            raise ValueError("Full cross attn no updated yet")
            x = self.full_cross_attn(
                query=x,
                query_normalized_xy_positions=query_spatial_positions,
                query_batch_offsets=query_batch_offsets,
                stacked_feature_maps=stacked_feature_maps,
                level_spatial_shapes=level_spatial_shapes,
            )
        x = self.ffn(x)
        return x

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        if self.ms_deform_attn is not None:
            self.ms_deform_attn.reset_parameters()
        if self.neighborhood_attn is not None:
            self.neighborhood_attn.reset_parameters()
        if self.full_cross_attn is not None:
            self.full_cross_attn.reset_parameters()
        self.ffn.reset_parameters()


class EMTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        config: TransformerDecoderConfig,
        class_head: Optional[nn.Module] = None,
        position_offset_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
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
            if class_head is None or position_offset_head is None:
                raise ValueError(
                    "Expected `class_head` and `position_offset_head` "
                    "to be specified when layers_share_heads is True; got "
                    f"{class_head} and {position_offset_head}"
                )
            self.class_head = class_head
            self.position_offset_head = position_offset_head
            self.std_head = self._make_std_head(config)
            self.segmentation_head = self._make_segmentation_head(config)
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
                    PositionOffsetHead(
                        self.d_model,
                        config.position_head.hidden_dim,
                        config.position_head.n_layers,
                        predict_box=config.predict_box,
                        activation_fn=config.position_head.activation_fn,
                    )
                    for _ in range(config.n_layers)
                ]
            )
            self.per_layer_std_heads = nn.ModuleList(
                [self._make_std_head(config) for _ in range(config.n_layers)]
            )

            self.per_layer_segmentation_heads = nn.ModuleList(
                [self._make_segmentation_head(config) for _ in range(config.n_layers)]
            )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(self.n_layers)]
        )

        self.reset_parameters()

    def _make_std_head(self, config: TransformerDecoderConfig) -> nn.Module:
        return StdDevHead.from_config(config.std_dev_head)

    def _make_segmentation_head(self, config: TransformerDecoderConfig) -> nn.Module:
        return PatchedSegmentationMapPredictor.from_config(config.segmentation_head)

    def forward(
        self,
        queries: Tensor,
        query_reference_points: Tensor,
        query_batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
        background_embedding: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        dn_info_dict: Optional[dict] = None,
    ):
        layer_output_logits = []
        layer_output_positions = []
        layer_output_queries = []
        layer_output_std = []
        layer_output_segmentation = []
        denoising_segmentation = []
        for i, layer in enumerate(self.layers):
            if not self.use_rope:
                query_pos_encoding = self.query_pos_encoding(query_reference_points)
                query_pos_encoding = query_pos_encoding.to(queries)
            else:
                query_pos_encoding = None
            queries = layer(
                queries=queries,
                query_batch_offsets=query_batch_offsets,
                query_spatial_positions=query_reference_points.detach(),
                stacked_feature_maps=stacked_feature_maps,
                level_spatial_shapes=level_spatial_shapes,
                attn_mask=attn_mask,
                background_embedding=background_embedding,
            )

            queries_normed: Tensor = self.norms[i](queries)

            class_head = self._get_class_head(i)
            delta_pos_head = self._get_position_head(i)
            std_head = self._get_std_head(i)
            segmentation_head = self._get_segmentation_head(i)

            query_logits: Tensor = class_head(queries_normed)
            query_delta_pos: Tensor = delta_pos_head(
                queries_normed.to(delta_pos_head.dtype)
            )
            query_std: Tensor = std_head(queries_normed)

            new_reference_points = query_reference_points + query_delta_pos

            query_segmentation, dn_segmentation = segmentation_head(
                queries_normed,
                query_batch_offsets,
                new_reference_points,
                stacked_feature_maps,
                level_spatial_shapes,
                background_embedding=background_embedding,
                attn_mask=attn_mask,
                dn_info_dict=dn_info_dict,
            )

            layer_output_logits.append(query_logits)
            layer_output_std.append(query_std)
            if self.look_forward_twice:
                layer_output_positions.append(new_reference_points)
            else:
                layer_output_positions.append(new_reference_points.detach())
            layer_output_queries.append(queries)
            layer_output_segmentation.append(query_segmentation)
            denoising_segmentation.append(dn_segmentation)

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
            "dn_segmentation_logits": denoising_segmentation,
        }

    def _get_class_head(self, layer_index) -> nn.Module:
        if self.layers_share_heads:
            assert self.class_head is not None
            return self.class_head
        else:
            assert self.per_layer_class_heads is not None
            return self.per_layer_class_heads[layer_index]

    def _get_position_head(self, layer_index) -> nn.Module:
        if self.layers_share_heads:
            assert self.position_offset_head is not None
            return self.position_offset_head
        else:
            assert self.per_layer_position_heads is not None
            return self.per_layer_position_heads[layer_index]

    def _get_std_head(self, layer_index) -> nn.Module:
        if self.layers_share_heads:
            assert self.std_head is not None
            return self.std_head
        else:
            assert self.per_layer_std_heads is not None
            return self.per_layer_std_heads[layer_index]

    def _get_segmentation_head(self, layer_index):
        if self.layers_share_heads:
            assert self.segmentation_head is not None
            return self.segmentation_head
        else:
            assert self.per_layer_segmentation_heads is not None
            return self.per_layer_segmentation_heads[layer_index]

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.per_layer_class_heads is not None:
            for head in self.per_layer_class_heads:
                head.reset_parameters()
        if self.per_layer_position_heads is not None:
            for head in self.per_layer_position_heads:
                if hasattr(head, "reset_parameters"):
                    layer.reset_parameters()
        if self.per_layer_std_heads is not None:
            for head in self.per_layer_std_heads:
                head.reset_parameters()
        if self.per_layer_segmentation_heads is not None:
            for head in self.per_layer_segmentation_heads:
                head.reset_parameters()
