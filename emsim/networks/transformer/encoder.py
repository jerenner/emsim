import copy
from typing import Union, Optional
import numpy as np

from torch import Tensor, nn
import torch
import MinkowskiEngine as ME

from emsim.networks.transformer.blocks import (
    FFNBlock,
    SelfAttentionBlockWithRoPE,
    SparseDeformableAttentionBlock,
    SelfAttentionBlock,
)
from emsim.networks.positional_encoding import FourierEncoding
from emsim.networks.positional_encoding import ij_indices_to_normalized_xy
from emsim.utils.sparse_utils import (
    minkowski_to_torch_sparse,
    gather_from_sparse_tensor,
    scatter_to_sparse_tensor,
    linearize_sparse_and_index_tensors,
)
from emsim.utils.batching_utils import (
    deconcat_add_batch_dim,
    remove_batch_dim_and_concat,
)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        n_deformable_value_levels: int = 4,
        n_deformable_points: int = 4,
        dropout: float = 0.1,
        activation_fn="gelu",
        norm_first: bool = True,
        attn_proj_bias: bool = False,
        topk_sa: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.topk_sa = topk_sa

        self.self_attn = SelfAttentionBlock(
            d_model, n_heads, dropout, attn_proj_bias, norm_first
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
        token_predicted_salience: Tensor,
        token_electron_probs: Tensor,
    ):
        token_scores = token_electron_probs * token_predicted_salience.sigmoid()
        token_scores_batched, pad_mask = deconcat_add_batch_dim(
            token_scores.unsqueeze(-1), batch_offsets
        )
        token_scores_batched = token_scores_batched.squeeze(-1)
        queries_batched, pad_mask_2 = deconcat_add_batch_dim(queries, batch_offsets)
        pos_encoding_batched, pad_mask_3 = deconcat_add_batch_dim(
            query_pos_encoding, batch_offsets
        )
        assert torch.equal(pad_mask, pad_mask_2)
        assert torch.equal(pad_mask, pad_mask_3)

        k = min(self.topk_sa, token_scores_batched.shape[1])
        indices = torch.topk(token_scores_batched, k, dim=1)[1]
        selected_pad_mask = torch.gather(pad_mask, 1, indices)
        indices = indices.unsqueeze(-1).expand(-1, -1, queries_batched.shape[-1])
        selected_queries = torch.gather(queries_batched, 1, indices)
        selected_pos_encoding = torch.gather(pos_encoding_batched, 1, indices)

        self_attn_out = self.self_attn(
            selected_queries, selected_pos_encoding, selected_pad_mask
        )
        queries_batched = queries_batched.scatter(1, indices, self_attn_out)

        queries_2, batch_offsets_2 = remove_batch_dim_and_concat(
            queries_batched, pad_mask
        )
        assert torch.equal(batch_offsets, batch_offsets_2)

        queries_3 = self.msdeform_attn(
            queries_2,
            query_pos_encoding,
            query_normalized_xy_positions,
            batch_offsets,
            stacked_feature_maps,
            spatial_shapes,
        )
        queries_4 = self.ffn(queries_3)
        return queries_4

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        self.msdeform_attn.reset_parameters()
        self.ffn.reset_parameters()


class EMTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int = 6,
        score_predictor: nn.Module = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.d_model = encoder_layer.d_model

        self.reset_parameters()

        self.enhance_score_predictor = score_predictor
        self.background_embedding = FourierEncoding(3, self.d_model, dtype=torch.double)

    def forward(
        self,
        feature_maps: list[ME.SparseTensor],
        position_encoding: list[ME.SparseTensor],
        spatial_shapes: Tensor,
        token_salience_scores: list[Tensor],  # foreground scores
        token_ij_level_indices: list[Tensor],
        token_normalized_xy_positions: list[Tensor],
        token_layer_subset_indices: list[Tensor],
    ):
        # stack the spconv tensors over the batch dimension for faster indexing
        stacked_feature_maps = self.stack_sparse_tensors(feature_maps, spatial_shapes[-1])
        stacked_pos_encodings = self.stack_sparse_tensors(position_encoding, spatial_shapes[-1])
        for layer_index, layer in enumerate(self.layers):
            indices_for_layer = [
                indices[layer_index] for indices in token_layer_subset_indices
            ]
            ij_indices_for_layer = [
                ij_indices[indices]
                for ij_indices, indices in zip(
                    token_ij_level_indices,
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
                np.cumsum(np.concatenate([[0], tokens_per_batch]))[:-1]
            )
            stacked_ij_indices_for_layer = torch.cat(ij_indices_for_layer, 0)
            stacked_xy_positions_for_layer = torch.cat(xy_positions_for_layer, 0)
            query_for_layer = gather_from_sparse_tensor(
                stacked_feature_maps, stacked_ij_indices_for_layer, True
            )[0]
            # query_for_layer = torch.nested.as_nested_tensor(
            #     list(
            #         torch.tensor_split(
            #             query_for_layer, np.cumsum(tokens_per_batch)[:-1].tolist()
            #         )
            #     )
            # )
            pos_encoding_for_layer = gather_from_sparse_tensor(
                stacked_pos_encodings, stacked_ij_indices_for_layer, True
            )[0]
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
    def stack_sparse_tensors(tensor_list: list[ME.SparseTensor], full_scale_spatial_shape: Tensor):
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
