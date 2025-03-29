import copy
from typing import Optional, Union

import MinkowskiEngine as ME
import numpy as np
import torch
from torch import Tensor, nn

from emsim.networks.positional_encoding import (
    FourierEncoding,
    ij_indices_to_normalized_xy,
)
from emsim.networks.positional_encoding.rope import prep_multilevel_positions
from emsim.networks.transformer.blocks import (
    FFNBlock,
    MultilevelSelfAttentionBlockWithRoPE,
    SelfAttentionBlock,
    SparseDeformableAttentionBlock,
    SparseNeighborhoodAttentionBlock,
)
from emsim.utils.batching_utils import (
    deconcat_add_batch_dim,
    remove_batch_dim_and_concat,
)
from emsim.utils.sparse_utils import (
    batch_sparse_index,
    linearize_sparse_and_index_tensors,
    minkowski_to_torch_sparse,
    scatter_to_sparse_tensor,
)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        use_msdeform_attn: bool = True,
        n_deformable_value_levels: int = 4,
        n_deformable_points: int = 4,
        use_neighborhood_attn: bool = True,
        neighborhood_sizes: list[int] = [3, 5, 7, 9],
        dropout: float = 0.1,
        activation_fn: Union[str, nn.Module] = "gelu",
        norm_first: bool = True,
        attn_proj_bias: bool = False,
        topk_sa: int = 1000,
        use_rope: bool = False,
        rope_base_theta: float = 10.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.topk_sa = topk_sa
        self.use_msdeform_attn = use_msdeform_attn
        self.use_neighborhood_attn = use_neighborhood_attn
        self.use_rope = use_rope

        if not use_rope:
            self.self_attn = SelfAttentionBlock(
                d_model, n_heads, dropout, attn_proj_bias, norm_first
            )
        else:
            self.self_attn = MultilevelSelfAttentionBlockWithRoPE(
                d_model,
                n_heads,
                n_deformable_value_levels,
                2,
                dropout,
                attn_proj_bias,
                norm_first,
                rope_theta=rope_base_theta,
            )
        if use_msdeform_attn:
            self.msdeform_attn = SparseDeformableAttentionBlock(
                d_model,
                n_heads,
                n_deformable_value_levels,
                n_deformable_points,
                dropout,
                norm_first,
            )
        else:
            self.msdeform_attn = None
        if use_neighborhood_attn:
            self.neighborhood_attn = SparseNeighborhoodAttentionBlock(
                d_model,
                n_heads,
                n_deformable_value_levels,
                neighborhood_sizes=neighborhood_sizes,
                dropout=dropout,
                bias=attn_proj_bias,
                norm_first=norm_first,
                rope_theta=rope_base_theta,
            )
        else:
            self.neighborhood_attn = None
        self.ffn = FFNBlock(
            d_model, dim_feedforward, dropout, activation_fn, norm_first
        )

    def forward(
        self,
        queries: Tensor,
        query_pos_encoding: Tensor,
        query_bijl_indices: Tensor,
        query_normalized_xy_positions: Tensor,
        batch_offsets: Tensor,
        stacked_feature_maps: Tensor,
        spatial_shapes: Tensor,
        token_predicted_salience: Tensor,
        token_electron_probs: Tensor,
    ):
        token_scores = token_electron_probs * token_predicted_salience.sigmoid()
        token_scores_batched, pad_mask = deconcat_add_batch_dim(
            token_scores.unsqueeze(-1), batch_offsets, -torch.inf
        )
        token_scores_batched = token_scores_batched.squeeze(-1)
        queries_batched, pad_mask_2 = deconcat_add_batch_dim(queries, batch_offsets)
        ij_indices_batched, pad_mask_4 = deconcat_add_batch_dim(
            query_bijl_indices, batch_offsets
        )
        assert torch.equal(pad_mask, pad_mask_2)
        assert torch.equal(pad_mask, pad_mask_4)

        k = min(self.topk_sa, token_scores_batched.shape[1])
        indices = torch.topk(token_scores_batched, k, dim=1)[1]
        selected_pad_mask = torch.gather(pad_mask, 1, indices)
        selected_ij_indices = torch.gather(
            ij_indices_batched,
            1,
            indices.unsqueeze(-1).expand(-1, -1, ij_indices_batched.shape[-1]),
        )
        indices_unsq = indices.unsqueeze(-1).expand(-1, -1, queries_batched.shape[-1])
        selected_queries = torch.gather(queries_batched, 1, indices_unsq)
        # selected_queries = queries_batched
        # selected_pos_encoding = pos_encoding_batched
        # selected_pad_mask = pad_mask

        ### unbatched gather
        indices_flat = torch.flatten(
            indices + batch_offsets.unsqueeze(-1).expand_as(indices)
        )[selected_pad_mask.flatten().logical_not()]
        batch_offsets_flat = torch.arange(
            0,
            indices_flat.numel(),
            indices.shape[-1],
            device=indices_flat.device,
            dtype=torch.int32,
        ) - torch.cat([indices.new_zeros([1]), selected_pad_mask.sum(-1)[:-1]], 0)
        # selected_queries_flat = queries[indices_flat]
        selected_bijl_indices_flat = torch.gather(
            query_bijl_indices,
            0,
            indices_flat.unsqueeze(-1).expand(-1, query_bijl_indices.shape[-1]),
        )
        indices_flat_unsq = indices_flat.unsqueeze(-1).expand(-1, queries.shape[-1])
        selected_queries_flat = torch.gather(queries, 0, indices_flat_unsq)
        assert torch.equal(selected_queries_flat, selected_queries[~selected_pad_mask])
        ###

        if not self.use_rope:
            pos_encoding_batched, pad_mask_3 = deconcat_add_batch_dim(
                query_pos_encoding, batch_offsets
            )
            selected_pos_encoding = torch.gather(pos_encoding_batched, 1, indices_unsq)
            assert torch.equal(pad_mask, pad_mask_3)
            selected_pos_encoding_flat = torch.gather(
                query_pos_encoding, 0, indices_flat_unsq
            )
            assert torch.equal(
                selected_pos_encoding_flat, selected_pos_encoding[~selected_pad_mask]
            )
            self_attn_out = self.self_attn(
                selected_queries_flat,
                selected_pos_encoding_flat,
                batch_offsets=batch_offsets_flat,
            )
        else:
            positions = prep_multilevel_positions(
                selected_bijl_indices_flat, spatial_shapes
            )
            self_attn_out = self.self_attn(
                selected_queries_flat,
                positions[:, 1:3],
                positions[:, -1],
                batch_offsets_flat,
            )
        # queries_batched = queries_batched.scatter(1, indices_unsq, self_attn_out)
        # queries_batched = self_attn_out
        # queries_2, batch_offsets_2 = remove_batch_dim_and_concat(
        #     queries_batched, pad_mask
        # )
        # assert torch.equal(batch_offsets, batch_offsets_2)
        queries = queries.scatter(0, indices_flat_unsq, self_attn_out)

        if self.use_msdeform_attn:
            queries = self.msdeform_attn(
                queries,
                query_pos_encoding,
                query_normalized_xy_positions,
                batch_offsets,
                stacked_feature_maps,
                spatial_shapes,
            )
        if self.use_neighborhood_attn:
            queries = self.neighborhood_attn(
                queries,
                query_bijl_indices,
                batch_offsets,
                stacked_feature_maps,
                spatial_shapes,
            )
        queries = self.ffn(queries)
        return queries

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        if hasattr(self.msdeform_attn, "reset_parameters"):
            self.msdeform_attn.reset_parameters()
        if hasattr(self.neighborhood_attn, "reset_parameters"):
            self.neighborhood_attn.reset_parameters()
        self.ffn.reset_parameters()


class EMTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int = 6,
        score_predictor: nn.Module = None,
    ):
        super().__init__()
        self.layers: list[TransformerEncoderLayer] = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.d_model = encoder_layer.d_model
        self.use_rope = self.layers[0].use_rope

        self.reset_parameters()

        self.enhance_score_predictor = score_predictor
        self.background_embedding = FourierEncoding(3, self.d_model, dtype=torch.double)

    def forward(
        self,
        feature_maps: list[ME.SparseTensor],
        position_encoding: list[ME.SparseTensor],
        spatial_shapes: Tensor,
        token_salience_scores: list[Tensor],  # foreground scores
        token_bijl_indices: list[Tensor],
        token_normalized_xy_positions: list[Tensor],
        token_layer_subset_indices: list[Tensor],
    ):
        # stack the MinkowskiEngine tensors over the batch dimension for faster indexing
        stacked_feature_maps = self.stack_sparse_tensors(
            feature_maps, spatial_shapes[-1]
        )
        stacked_pos_encodings = (
            self.stack_sparse_tensors(position_encoding, spatial_shapes[-1])
            if not self.use_rope
            else None
        )
        for layer_index, layer in enumerate(self.layers):
            indices_for_layer = [
                indices[layer_index] for indices in token_layer_subset_indices
            ]
            ij_indices_for_layer = [
                ij_indices[indices]
                for ij_indices, indices in zip(
                    token_bijl_indices,
                    indices_for_layer,
                )
            ]
            xy_positions_for_layer = [
                xy_positions[indices]
                for xy_positions, indices in zip(
                    token_normalized_xy_positions, indices_for_layer
                )
            ]
            tokens_per_batch = [indices.shape[0] for indices in ij_indices_for_layer]
            batch_offsets = torch.tensor(
                [0, *tokens_per_batch],
                device=stacked_feature_maps.device,
                dtype=torch.int32,
            ).cumsum(0)[:-1]
            stacked_ij_indices_for_layer = torch.cat(ij_indices_for_layer, 0)
            stacked_xy_positions_for_layer = torch.cat(xy_positions_for_layer, 0)
            query_for_layer = batch_sparse_index(
                stacked_feature_maps, stacked_ij_indices_for_layer, True
            )[0]
            # query_for_layer = torch.nested.as_nested_tensor(
            #     list(
            #         torch.tensor_split(
            #             query_for_layer, np.cumsum(tokens_per_batch)[:-1].tolist()
            #         )
            #     )
            # )
            pos_encoding_for_layer = (
                batch_sparse_index(
                    stacked_pos_encodings, stacked_ij_indices_for_layer, True
                )[0]
                if not self.use_rope
                else None
            )
            # pos_encoding_for_layer = torch.nested.as_nested_tensor(
            #     list(
            #         torch.tensor_split(
            #             pos_encoding_for_layer,
            #             np.cumsum(tokens_per_batch)[:-1].tolist(),
            #         )
            #     )
            # )
            token_scores_for_layer = [
                scores[indices]
                for scores, indices in zip(token_salience_scores, indices_for_layer)
            ]
            token_scores_for_layer = torch.cat(token_scores_for_layer)
            electron_prob = (
                self.enhance_score_predictor(query_for_layer).squeeze(-1).sigmoid()
            )
            query = layer(
                query_for_layer,
                pos_encoding_for_layer,
                stacked_ij_indices_for_layer,
                stacked_xy_positions_for_layer,
                batch_offsets,
                stacked_feature_maps,
                spatial_shapes,
                token_scores_for_layer,
                electron_prob,  # score_tgt
            )

            stacked_feature_maps = scatter_to_sparse_tensor(
                stacked_feature_maps, stacked_ij_indices_for_layer, query
            )

        # learned background embedding
        background_indices = self.get_background_indices(
            stacked_feature_maps, stacked_ij_indices_for_layer
        )

        background_ij = background_indices[:, 1:3]
        background_level = background_indices[:, 3]
        background_xy = ij_indices_to_normalized_xy(
            background_ij, spatial_shapes[background_level]
        )
        background_xy_level = torch.cat(
            [
                background_xy,
                background_level.unsqueeze(-1).to(background_xy)
                / (len(feature_maps) - 1),
            ],
            -1,
        )

        background_pos_encoding = self.background_embedding(background_xy_level)
        background_pos_encoding = background_pos_encoding.float()
        stacked_feature_maps = scatter_to_sparse_tensor(
            stacked_feature_maps, background_indices, background_pos_encoding
        )

        return stacked_feature_maps

    @staticmethod
    def stack_sparse_tensors(
        tensor_list: list[ME.SparseTensor], full_scale_spatial_shape: Tensor
    ):
        converted_tensors = [
            (
                minkowski_to_torch_sparse(tensor, full_scale_spatial_shape)
                if isinstance(tensor, ME.SparseTensor)
                else tensor
            )
            for tensor in tensor_list
        ]
        max_size = (
            np.stack([tensor.shape for tensor in converted_tensors], 0).max(0).tolist()
        )
        indices = []
        values = []
        for level_index, level_tensor in enumerate(converted_tensors):
            values.append(level_tensor.values())
            level_indices = level_tensor.indices()
            indices.append(
                torch.cat(
                    [
                        level_indices,
                        level_indices.new_tensor([[level_index]]).expand(
                            -1, level_indices.shape[-1]
                        ),
                    ],
                    0,
                )
            )
        return torch.sparse_coo_tensor(
            torch.cat(indices, -1),
            torch.cat(values, 0),
            list(max_size[:-1]) + [len(tensor_list)] + [max_size[-1]],
        ).coalesce()

    @staticmethod
    def get_background_indices(stacked_feature_maps, foreground_indices):
        (
            linear_sparse_indices,
            _,
            index_tensor_linearized,
            _,
            _,
        ) = linearize_sparse_and_index_tensors(stacked_feature_maps, foreground_indices)
        background_token_indices = ~torch.isin(
            linear_sparse_indices, index_tensor_linearized
        )
        background_indices = stacked_feature_maps.indices()[
            :, background_token_indices
        ].transpose(0, 1)
        return background_indices

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
