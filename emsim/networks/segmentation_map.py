from typing import Union

import torch
from torch import Tensor, nn

from emsim.networks.positional_encoding.rope import RoPEEncodingND, FreqGroupPattern
from emsim.networks.transformer.blocks.neighborhood_attn import (
    SparseNeighborhoodAttentionBlock,
    get_multilevel_neighborhoods,
)
from emsim.networks.transformer.blocks import FFNBlock
from emsim.utils.sparse_utils.batching import (
    batch_offsets_from_sparse_tensor_indices,
    split_batch_concatted_tensor,
    batch_offsets_to_indices,
    batch_offsets_to_seq_lengths,
    seq_lengths_to_indices,
)
from emsim.utils.sparse_utils.indexing.indexing import batch_sparse_index, sparse_select


class SegmentationMapPredictor(nn.Module):
    def __init__(self, d_model: int, mask_head_hidden_layers: int = 3):
        super().__init__()
        layers = []
        for _ in range(mask_head_hidden_layers):
            layers.extend([nn.Linear(d_model, d_model), nn.ReLU()])
        layers.append(nn.Linear(d_model, d_model))
        self.mask_embed = nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.mask_embed:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(
        self, stacked_feature_map: Tensor, queries: Tensor, query_batch_offsets: Tensor
    ) -> Tensor:
        queries = self.mask_embed(queries)
        # unbind over the level dimension
        fullscale_feature_map = sparse_select(stacked_feature_map, 3, 3)
        assert fullscale_feature_map.ndim == 4  # (batch, height, width, feature)

        split_queries = split_batch_concatted_tensor(queries, query_batch_offsets)
        feature_map_batch_offsets = batch_offsets_from_sparse_tensor_indices(
            fullscale_feature_map.indices()
        )
        split_feature_values = split_batch_concatted_tensor(
            fullscale_feature_map.values(), feature_map_batch_offsets
        )
        split_feature_indices = split_batch_concatted_tensor(
            fullscale_feature_map.indices().T, feature_map_batch_offsets
        )

        split_segmentation_logits = []
        for im_feats, im_queries in zip(split_feature_values, split_queries):
            split_segmentation_logits.append(torch.mm(im_feats, im_queries.T))

        split_segmentation_logit_indices = []
        for segmentation_logits, feature_indices in zip(
            split_segmentation_logits, split_feature_indices
        ):
            query_index = torch.arange(
                segmentation_logits.shape[-1], device=segmentation_logits.device
            )
            segmentation_logit_indices = torch.cat(
                [
                    feature_indices.unsqueeze(-2).expand(-1, len(query_index), -1),
                    query_index.expand(*segmentation_logits.shape[:-1], -1).unsqueeze(
                        -1
                    ),
                ],
                -1,
            )
            split_segmentation_logit_indices.append(segmentation_logit_indices)

        return torch.sparse_coo_tensor(
            torch.cat(
                [
                    indices.view(-1, indices.shape[-1])
                    for indices in split_segmentation_logit_indices
                ]
            ).T,
            torch.cat([logits.flatten() for logits in split_segmentation_logits]),
            (*fullscale_feature_map.shape[:-1], max(len(q) for q in split_queries)),
        ).coalesce()


class SegmentationMapLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        attn_proj_bias: bool,
        activation_fn: Union[str, type[nn.Module]],
        norm_first: bool,
        rope_share_heads: bool,
        rope_spatial_base_theta: float,
        rope_level_base_theta: float,
        rope_freq_group_pattern: Union[str, FreqGroupPattern],
    ):
        super().__init__()
        self.attn = SparseNeighborhoodAttentionBlock(
            d_model,
            n_heads,
            n_levels=4,
            dropout=dropout,
            bias=attn_proj_bias,
            norm_first=norm_first,
            rope_spatial_base_theta=rope_spatial_base_theta,
            rope_level_base_theta=rope_level_base_theta,
            rope_share_heads=rope_share_heads,
            rope_freq_group_pattern=rope_freq_group_pattern,
        )
        self.ffn = FFNBlock(
            d_model, dim_feedforward, dropout, activation_fn, norm_first
        )

    def forward(
        self,
        queries: Tensor,
        query_batch_offsets: Tensor,
        query_positions: Tensor,
        stacked_feature_map: Tensor,
        level_spatial_shapes: Tensor,
    ) -> Tensor:
        max_level_index = level_spatial_shapes.argmax(dim=0)
        max_level_index = torch.unique(max_level_index)
        assert len(max_level_index) == 1
        query_level_indices = max_level_index.expand(query_positions.shape[0])
        queries = self.attn(
            queries,
            query_positions,
            query_batch_offsets,
            stacked_feature_map,
            level_spatial_shapes,
            query_level_indices,
        )
        queries = self.ffn(queries)
        return queries

    def reset_parameters(self):
        self.attn.reset_parameters()
        self.ffn.reset_parameters()


class PatchedSegmentationMapPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
        attn_proj_bias: bool = False,
        activation_fn: Union[str, type[nn.Module]] = "gelu",
        norm_first: bool = True,
        rope_share_heads: bool = False,
        rope_spatial_base_theta: float = 10.0,
        rope_level_base_theta: float = 10.0,
        rope_freq_group_pattern: Union[
            str, FreqGroupPattern
        ] = FreqGroupPattern.PARTITION,
        query_patch_diameter: int = 7,
    ):
        super().__init__()
        layers = []
        for _ in range(n_transformer_layers):
            layers.append(
                SegmentationMapLayer(
                    d_model,
                    n_heads,
                    dim_feedforward,
                    dropout,
                    attn_proj_bias,
                    activation_fn,
                    norm_first,
                    rope_share_heads,
                    rope_spatial_base_theta,
                    rope_level_base_theta,
                    rope_freq_group_pattern,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.query_patch_diameter = query_patch_diameter
        freq_group_pattern = torch.tensor([[True, True]])  # 1 freq group, no level dim
        self.pos_encoding = RoPEEncodingND(
            2,
            d_model,
            n_heads,
            rope_share_heads,
            freq_group_pattern=freq_group_pattern,
            rope_base_theta=rope_spatial_base_theta,
        )

    def forward(
        self,
        queries: Tensor,
        query_batch_offsets: Tensor,
        query_positions: Tensor,
        stacked_feature_map: Tensor,
        level_spatial_shapes: Tensor,
    ) -> Tensor:
        # Run the queries through the transformer layers
        for layer in self.layers:
            queries = layer(
                queries,
                query_batch_offsets,
                query_positions,
                stacked_feature_map,
                level_spatial_shapes,
            )

        max_level_index = level_spatial_shapes.argmax(dim=0)
        max_level_index = torch.unique(max_level_index)
        assert len(max_level_index) == 1  # same level for all batches
        max_level_index = int(max_level_index.item())

        # get the full-scale feature map level
        fullscale_feature_map = sparse_select(stacked_feature_map, 3, max_level_index)
        assert isinstance(fullscale_feature_map, Tensor)
        assert fullscale_feature_map.ndim == 4  # (batch, height, width, feature)

        # Get the patch indices
        if level_spatial_shapes.ndim == 3:
            max_level_shape = level_spatial_shapes[0, max_level_index].view(1, 2)
        else:
            assert level_spatial_shapes.ndim == 2
            max_level_shape = level_spatial_shapes[max_level_index].view(1, 2)
        patch_indices, patch_oob, _ = get_multilevel_neighborhoods(
            query_positions, max_level_shape, [self.query_patch_diameter]
        )

        # Indices: [n_total_query x n_patch_pixels x (i, j)]
        # -> [n_total_query x n_patch_pixels x (batch, i, j, query)]
        indices = patch_indices.new_empty(
            patch_indices.shape[:-1] + (patch_indices.shape[-1] + 2,)
        )
        query_seq_lengths: Tensor = batch_offsets_to_seq_lengths(query_batch_offsets)
        indices[..., 0] = seq_lengths_to_indices(query_seq_lengths).unsqueeze(-1)
        for i in range(query_batch_offsets.size(0) - 1):
            batch_start, batch_end = int(query_batch_offsets[i]), int(
                query_batch_offsets[i + 1]
            )
            indices[batch_start:batch_end, :, -1] = torch.arange(
                batch_end - batch_start, device=indices.device
            ).unsqueeze(-1)

        # Extract the patches
        patch_embeddings, patch_is_specified_mask = batch_sparse_index(
            fullscale_feature_map, indices[..., :-1]  # index (batch, i, j)
        )
        assert isinstance(patch_embeddings, Tensor)
        assert torch.all(
            (patch_oob & (~patch_is_specified_mask)) == patch_oob
        )  # sanity check

        # dot product each query vector with its patch's embeddings
        # [n_total_query x embed_dim] @ [n_total_query x patch_pixels x embed_dim] -> [n_total_query x patch_pixels]
        patch_segmentation_logits = torch.bmm(
            queries.unsqueeze(1), patch_embeddings.transpose(-1, -2)
        ).squeeze(1)

        # now put the segmentation logits into a sparse tensor
        nonzero_mask = patch_segmentation_logits != 0.0
        nonzero_indices = indices[nonzero_mask].T

        max_query_index = int(query_seq_lengths.amax().item())

        patch_segmap = torch.sparse_coo_tensor(
            nonzero_indices,
            patch_segmentation_logits[nonzero_mask],
            size=fullscale_feature_map.shape[:-1] + (max_query_index,),
        ).coalesce()

        return patch_segmap

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


def sparse_binary_segmentation_map(segmentation_map: Tensor):
    assert segmentation_map.is_sparse
    return torch.sparse_coo_tensor(
        segmentation_map.indices(),
        segmentation_map.values() > 0.0,
        segmentation_map.shape,
        device=segmentation_map.device,
    )
