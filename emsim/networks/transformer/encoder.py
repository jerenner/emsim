import copy
from typing import Union, Optional

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
from emsim.utils.sparse_utils.batching.batching import (
    deconcat_add_batch_dim,
    seq_lengths_to_batch_offsets,
    batch_offsets_to_seq_lengths,
)
from emsim.utils.sparse_utils.conversion import minkowski_to_torch_sparse
from emsim.utils.sparse_utils.indexing.indexing import batch_sparse_index
from emsim.utils.sparse_utils.indexing.scatter import (
    scatter_to_sparse_tensor,
)
from emsim.utils.sparse_utils.indexing.script_funcs import (
    linearize_sparse_and_index_tensors,
    flatten_nd_indices,
)
from emsim.config.transformer import RoPEConfig, TransformerEncoderConfig


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        n_feature_levels: int = 4,
        use_msdeform_attn: bool = True,
        n_deformable_points: Optional[int] = 4,
        use_neighborhood_attn: bool = True,
        neighborhood_sizes: Optional[list[int]] = None,
        use_rope: bool = True,
        rope_config: Optional[RoPEConfig] = None,
        dropout: float = 0.1,
        activation_fn: Union[str, nn.Module] = "gelu",
        norm_first: bool = True,
        attn_proj_bias: bool = False,
        max_tokens_sa: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.max_tokens_sa = max_tokens_sa
        self.use_msdeform_attn = use_msdeform_attn
        self.use_neighborhood_attn = use_neighborhood_attn
        self.use_rope = use_rope

        if not use_rope:
            self.self_attn = SelfAttentionBlock(
                d_model, n_heads, dropout, attn_proj_bias, norm_first
            )
        else:
            assert rope_config is not None
            self.self_attn = MultilevelSelfAttentionBlockWithRoPE(
                d_model,
                n_heads,
                rope_config.spatial_dimension,
                dropout,
                attn_proj_bias,
                norm_first,
                rope_spatial_base_theta=rope_config.spatial_base_theta,
                rope_level_base_theta=rope_config.level_base_theta,
                rope_share_heads=rope_config.share_heads,
                rope_freq_group_pattern=rope_config.freq_group_pattern,
                rope_enforce_freq_groups_equal=rope_config.enforce_freq_groups_equal,
            )
        if use_msdeform_attn:
            self.msdeform_attn = SparseDeformableAttentionBlock(
                d_model,
                n_heads,
                n_feature_levels,
                n_deformable_points,
                dropout,
                norm_first,
            )
        else:
            self.msdeform_attn = None
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
        self.ffn = FFNBlock(
            d_model, dim_feedforward, dropout, activation_fn, norm_first
        )

    def forward(
        self,
        queries: Tensor,  # [query_batch_offsets[-1]]
        query_batch_offsets: Tensor,  # [bsz+1]
        token_predicted_salience_score: Tensor,  # [query_batch_offsets[-1]]
        query_spatial_indices: Tensor,  # [spatial_dim+2, query_batch_offsets[-1]]
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
        token_electron_scores: Tensor,
        query_pos_encoding: Optional[Tensor] = None,
    ):
        token_scores = token_electron_scores + token_predicted_salience_score

        # Find the max_tokens_sa topk for self attention
        token_seq_lens: Tensor = batch_offsets_to_seq_lengths(query_batch_offsets)
        min_len, max_len = torch.aminmax(token_seq_lens)

        if min_len == max_len:  # sequences same length, batchify for efficiency
            k = min_len.clamp_max(self.max_tokens_sa)
            _, topk_indices = token_scores.reshape(-1, min_len).topk(k)
            topk_indices += query_batch_offsets.unsqueeze(1)
            topk_indices = topk_indices.flatten()
        else:  # sequences different length, run topk for each
            batch_ks = token_seq_lens.clamp_max(self.max_tokens_sa)
            topk_offsets = seq_lengths_to_batch_offsets(batch_ks)
            topk_indices = torch.empty(
                topk_offsets[-1], dtype=torch.long, device=queries.device
            )

            # holder for topk's first (values) output
            scratch_values = token_scores.new_empty(batch_ks.max())

            # per-batch topk
            for b, k in enumerate(batch_ks):
                k = int(k)
                batch_start, batch_end = query_batch_offsets[b : b + 2]
                slice_topk = slice(*topk_offsets[b : b + 2])

                torch.topk(
                    token_scores[batch_start:batch_end].detach(),
                    k,
                    out=(scratch_values[:k], topk_indices[slice_topk]),
                )
                topk_indices[slice_topk] += batch_start
            del scratch_values

        queries_sa = queries[topk_indices]
        spatial_indices_sa = query_spatial_indices[:, topk_indices]

        if not self.use_rope:
            raise NotImplementedError("Non-RoPE SA not implemented")
        else:
            self_attn_out = self.self_attn(
                x=queries_sa,
                spatial_positions=spatial_indices_sa[1:-1].T,
                level_indices=spatial_indices_sa[-1],
                level_spatial_shapes=level_spatial_shapes,
                batch_offsets=topk_offsets,
            )
        queries = queries.index_copy(0, topk_indices, self_attn_out)

        if self.use_msdeform_attn:
            raise NotImplementedError("SparseMSDeformAttention not updated yet")
            # queries = self.msdeform_attn(
            #     queries,
            #     query_pos_encoding,
            #     query_normalized_xy_positions,
            #     query_batch_offsets,
            #     stacked_feature_maps,
            #     level_spatial_shapes,
            # )
        if self.use_neighborhood_attn:
            query_positions = prep_multilevel_positions(
                query_spatial_indices[1:-1].T,
                query_spatial_indices[0],
                query_spatial_indices[-1],
                level_spatial_shapes,
            )
            queries = self.neighborhood_attn(
                query=queries,
                query_spatial_positions=query_positions[:, :-1],
                query_batch_offsets=query_batch_offsets,
                stacked_feature_maps=stacked_feature_maps,
                level_spatial_shapes=level_spatial_shapes,
                query_level_indices=query_spatial_indices[-1],
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
        config: TransformerEncoderConfig,
        score_predictor: nn.Module = None,
    ):
        super().__init__()
        self.layers: list[TransformerEncoderLayer] = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(config.n_layers)]
        )
        self.n_layers = config.n_layers
        self.d_model = encoder_layer.d_model
        self.use_rope = config.use_rope
        self.max_tokens_sa = config.max_tokens_sa
        self.max_tokens_non_sa = config.max_tokens_non_sa

        self.register_buffer(
            "layer_filter_ratio", torch.as_tensor(config.layer_filter_ratio)
        )

        self.reset_parameters()

        self.enhance_score_predictor = score_predictor

    def forward(
        self,
        sparse_feature_maps: list[ME.SparseTensor],
        level_spatial_shapes: Tensor,
        token_salience_scores: list[Tensor],  # foreground scores
        token_spatial_indices: list[Tensor],
        token_level_indices: list[Tensor],
        position_encoding: Optional[list[ME.SparseTensor]] = None,
    ):
        # stack the MinkowskiEngine tensors over the batch dimension for faster indexing
        stacked_feature_maps = self.stack_sparse_tensors(
            sparse_feature_maps, level_spatial_shapes[-1]
        )
        if position_encoding is not None:
            assert not self.use_rope
            stacked_pos_encodings = self.stack_sparse_tensors(
                position_encoding, level_spatial_shapes[-1]
            )
        else:
            stacked_pos_encodings = None

        device = stacked_feature_maps.device
        spatial_dim = level_spatial_shapes.size(-1)
        n_batch_tokens = torch.tensor(
            [scores.size(0) for scores in token_salience_scores],
            device=sparse_feature_maps[0].device,
        )

        for layer_idx, layer in enumerate(self.layers):
            # Get the top tokens for this layer according to ratio

            # Per-batch number of tokens for this layer, capped by the predetermined
            # cap (max_tokens_non_sa) and tokens actually available (n_batch_tokens)
            batch_k = (
                (n_batch_tokens * self.layer_filter_ratio[layer_idx])
                .long()
                .clamp_max_(self.max_tokens_non_sa)
                .clamp_max_(n_batch_tokens)
            )
            token_batch_offsets = seq_lengths_to_batch_offsets(batch_k)
            total_topk_tokens = token_batch_offsets[-1]

            # Pre-allocate the buffers to hold the topk token data
            batch_topk_scores = token_salience_scores[0].new_empty(total_topk_tokens)
            topk_indices = [
                torch.empty(k, device=device, dtype=torch.long) for k in batch_k
            ]
            batch_topk_spatial_indices = token_spatial_indices[0].new_empty(
                spatial_dim + 2, total_topk_tokens
            )
            batch_topk_spatial_indices[0] = torch.repeat_interleave(
                torch.arange(batch_k.size(0), device=device), batch_k
            )

            # Take each batch's topk
            cursor = 0
            for b, batch_scores in enumerate(token_salience_scores):
                k = int(batch_k[b])
                slice_b = slice(cursor, cursor + k)
                scores_b = batch_topk_scores[slice_b]
                indices_b = topk_indices[b]
                torch.topk(batch_scores.detach(), k, out=(scores_b, indices_b))

                # Collate spatial index info
                batch_topk_spatial_indices[1:-1, slice_b] = token_spatial_indices[b][
                    indices_b
                ].T
                batch_topk_spatial_indices[-1, slice_b] = token_level_indices[b][
                    indices_b
                ]
                cursor += k

            # Extract the embeddings at each spatial index
            batch_topk_spatial_indices_t = batch_topk_spatial_indices.T.contiguous()
            query_for_layer = batch_sparse_index(
                stacked_feature_maps, batch_topk_spatial_indices_t, True
            )[0]
            if position_encoding is not None:
                pos_encoding_for_layer = batch_sparse_index(
                    stacked_pos_encodings, batch_topk_spatial_indices_t, True
                )[0]
            else:
                pos_encoding_for_layer = None

            # Compute classification score for each embedding
            electron_score = self.enhance_score_predictor(query_for_layer).squeeze(-1)
            query_for_layer = layer(
                query_for_layer,
                token_batch_offsets,
                batch_topk_scores,
                batch_topk_spatial_indices,
                stacked_feature_maps,
                level_spatial_shapes,
                electron_score,  # score_tgt
                pos_encoding_for_layer,
            )

            stacked_feature_maps = scatter_to_sparse_tensor(
                stacked_feature_maps, batch_topk_spatial_indices_t, query_for_layer
            )

        stacked_feature_maps = self.update_background_embedding(stacked_feature_maps)

        return stacked_feature_maps

    @staticmethod
    def stack_sparse_tensors(
        tensor_list: Union[list[ME.SparseTensor], list[Tensor]],
        full_scale_spatial_shape: Union[Tensor, list[int]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Stacks a list of sparse tensors along a new _level_ dimension.

        Every input tensor is expected to have shape
        ``[batch, *spatial_shape, feature_dim]`` and identical sparse / dense
        dimensionalities.  The function converts ``MinkowskiEngine`` tensors to
        regular PyTorch COO format and concatenates their indices / values while
        inserting a new _level_ coordinate, returning a single sparse COO tensor
        whose shape is  ``[max_batch, *max_spatial, #levels, max_feature]``.

        Args:
            tensor_list (list[ME.SparseTensor | torch.Tensor]): Input sparse
                tensors to be stacked.
            full_scale_spatial_shape (Tensor | list[int]): Spatial extents
                (``len == tensor.ndim - 2``) used when converting Minkowski
                tensors.
            device (torch.device, optional): Device for the resulting tensor.
                Defaults to the device of the first tensor (or of the
                ``full_scale_spatial_shape`` argument when the list is empty).

        Returns:
            torch.Tensor: A sparse COO tensor that contains all input tensors
                stacked along the new level dimension.

        Notes:
            - If the input tensors are already coalesced, this function bypasses an
                explicit coalescing of the output tensor by sorting the concatenated
                sparse indices without the accompanying deduplication step.
        """

        # Determine device
        if device is None:
            if tensor_list:
                device = tensor_list[0].device
            else:  # empty input – use device of spatial shape tensor
                device = (
                    full_scale_spatial_shape.device
                    if isinstance(full_scale_spatial_shape, torch.Tensor)
                    else torch.device("cpu")
                )

        # Convert spatial shape to tensor if needed
        full_scale_spatial_shape = torch.as_tensor(
            full_scale_spatial_shape, device=device
        )

        # Convert possible Minkowski tensors to Pytorch sparse
        converted = [
            (
                minkowski_to_torch_sparse(t, full_scale_spatial_shape)
                if isinstance(t, ME.SparseTensor)
                else t
            )
            for t in tensor_list
        ]

        # Early exit for empty tensor list
        if len(converted) == 0:
            empty_sz = torch.Size([0] + list(full_scale_spatial_shape) + [0, 0])
            return torch.sparse_coo_tensor(
                torch.empty(
                    full_scale_spatial_shape.numel() + 2,
                    0,
                    dtype=torch.long,
                    device=device,
                ),
                torch.empty(0, device=device),
                empty_sz,
                is_coalesced=True,
            )

        # Determine shape information and run shape checks
        sparse_dims = [t.sparse_dim() for t in converted]
        dense_dims = [t.dense_dim() for t in converted]
        if len(set(sparse_dims)) != 1:
            raise ValueError(f"Inconsistent sparse_dim values: {sparse_dims}")
        if len(set(dense_dims)) != 1:
            raise ValueError(f"Inconsistent dense_dim values:  {dense_dims}")

        sparse_dim = sparse_dims[0]  # includes batch + spatial dims
        dense_dim = dense_dims[0]  # usually 1  (feature)
        level_dim = sparse_dim  # we append the level as last sparse dim

        spatial_rank_expected = sparse_dim - 1  # batch excluded, feature excluded
        if full_scale_spatial_shape.numel() != spatial_rank_expected:
            raise ValueError(
                "full_scale_spatial_shape must have length "
                f"{spatial_rank_expected}, got {full_scale_spatial_shape.numel()}"
            )

        # Collect information for constructing output tensor
        total_nnz = 0
        max_sizes = list(converted[0].shape)  # copy to mutate in-place
        all_coalesced = True

        for t in converted:
            total_nnz += t._nnz()
            all_coalesced &= t.is_coalesced()
            for i, dim_size in enumerate(t.shape):
                max_sizes[i] = max(dim_size, max_sizes[i])

        if total_nnz == 0:  # All tensors are empty
            final_shape = torch.Size(max_sizes[:-1] + [len(converted), max_sizes[-1]])
            return torch.sparse_coo_tensor(
                torch.empty(level_dim + 1, 0, dtype=torch.long, device=device),
                torch.empty(0, device=device),
                final_shape,
                is_coalesced=True,
            )

        # Allocate output index and value tensors
        indices_out = torch.empty(
            level_dim + 1, total_nnz, dtype=torch.long, device=device
        )

        sample_values = converted[0].values()
        if dense_dim == 0:
            values_out = torch.empty(
                total_nnz, dtype=sample_values.dtype, device=device
            )
        else:
            values_out = torch.empty(
                (total_nnz,) + sample_values.shape[1:],
                dtype=sample_values.dtype,
                device=device,
            )

        # Fill output tensors
        cursor = 0
        for lvl, t in enumerate(converted):
            nnz = t._nnz()
            if nnz == 0:
                continue
            slicer = slice(cursor, cursor + nnz)
            values_out[slicer] = t.values()
            indices_out[:sparse_dim, slicer] = t.indices()
            indices_out[level_dim, slicer] = lvl
            cursor += nnz

        # If the input tensors are all coalesced, we can "pre-coalesce" the output tensor
        # with a flattening+sort since we know deduplication is not needed
        if all_coalesced:
            sparse_sizes = torch.tensor(
                final_shape[:-1],
                device=device,
                dtype=torch.long,
            )
            flat_keys, _ = flatten_nd_indices(indices_out, sparse_sizes)
            perm = torch.argsort(flat_keys.squeeze(0))
            indices_out = indices_out[:, perm]
            values_out = values_out[perm]

        # Construct final output tensor
        final_shape = torch.Size(max_sizes[:-1] + [len(converted), max_sizes[-1]])
        return torch.sparse_coo_tensor(
            indices_out,
            values_out,
            final_shape,
            is_coalesced=all_coalesced,
        ).coalesce()  # coalesce is no-op if already coalesced

    def update_background_embedding(
        self,
        stacked_feature_maps: Tensor,
        level_spatial_shapes: Tensor,
        sparse_feature_maps: Tensor,
        stacked_ij_indices_for_layer: Tensor,
    ):
        raise NotImplementedError("Not updated or used currently")
        # learned background embedding
        background_indices = self.get_background_indices(
            stacked_feature_maps, stacked_ij_indices_for_layer
        )

        background_ij = background_indices[:, 1:3]
        background_level = background_indices[:, 3]
        background_xy = ij_indices_to_normalized_xy(
            background_ij, level_spatial_shapes[background_level]
        )
        background_xy_level = torch.cat(
            [
                background_xy,
                background_level.unsqueeze(-1).to(background_xy)
                / (len(sparse_feature_maps) - 1),
            ],
            -1,
        )

        background_pos_encoding = self.background_embedding(background_xy_level)
        background_pos_encoding = background_pos_encoding.float()
        stacked_feature_maps = scatter_to_sparse_tensor(
            stacked_feature_maps, background_indices, background_pos_encoding
        )

    @staticmethod
    def get_background_indices(stacked_feature_maps, foreground_indices):
        (
            linear_sparse_indices,
            index_tensor_linearized,
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
