import time
from typing import Optional, Union

import torch
from torch import Tensor, nn
from typing_extensions import Self
from itertools import chain, repeat

from emsim.config.transformer import SegmentationHeadConfig
from emsim.networks.positional_encoding.rope import FreqGroupPattern
from emsim.networks.transformer.blocks import FFNBlock
from emsim.networks.transformer.blocks.neighborhood_attn import (
    SparseNeighborhoodAttentionBlock,
    get_multilevel_neighborhoods,
)
from emsim.networks.transformer.blocks.self_attn import (
    MultilevelSelfAttentionBlockWithRoPE,
)
from emsim.utils.sparse_utils.batching import (
    batch_offsets_from_sparse_tensor_indices,
    batch_offsets_to_indices,
    batch_offsets_to_seq_lengths,
    seq_lengths_to_batch_offsets,
    seq_lengths_to_indices,
    split_batch_concatted_tensor,
)
from emsim.utils.sparse_utils.indexing.indexing import batch_sparse_index, sparse_select
from emsim.utils.sparse_utils.indexing.script_funcs import flatten_nd_indices


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
        embed_dim: int,
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
        self.nhood_attn = SparseNeighborhoodAttentionBlock(
            embed_dim,
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
        self.self_attn = MultilevelSelfAttentionBlockWithRoPE(
            embed_dim,
            n_heads,
            position_dim=2,
            dropout=dropout,
            bias=attn_proj_bias,
            norm_first=norm_first,
            rope_spatial_base_theta=rope_spatial_base_theta,
            rope_level_base_theta=rope_level_base_theta,
            rope_share_heads=rope_share_heads,
            rope_freq_group_pattern=rope_freq_group_pattern,
        )
        self.ffn = FFNBlock(
            embed_dim, dim_feedforward, dropout, activation_fn, norm_first
        )

    def forward(
        self,
        queries: Tensor,
        query_batch_offsets: Tensor,
        query_positions: Tensor,
        stacked_feature_map: Tensor,
        level_spatial_shapes: Tensor,
        background_embedding: Optional[Tensor] = None,
        is_background_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        dn_info_dict: Optional[
            dict[str, Union[Tensor, list[Tensor], int, list[int]]]
        ] = None,
    ) -> Tensor:
        max_level_index = level_spatial_shapes.argmax(dim=0)
        max_level_index = torch.unique(max_level_index)
        assert len(max_level_index) == 1
        query_level_indices = max_level_index.expand(query_positions.shape[0])
        queries = self.nhood_attn(
            queries,
            query_positions,
            query_batch_offsets,
            stacked_feature_map,
            level_spatial_shapes,
            background_embedding=background_embedding,
            query_level_indices=query_level_indices,
            query_mask=is_background_mask,
        )

        queries = self.self_attn(
            queries,
            spatial_positions=query_positions,
            level_indices=query_level_indices,
            level_spatial_shapes=level_spatial_shapes,
            batch_offsets=query_batch_offsets,
            attn_mask=attn_mask,
        )

        queries = self.ffn(queries)

        return queries

    def reset_parameters(self):
        self.nhood_attn.reset_parameters()
        self.self_attn.reset_parameters()
        self.ffn.reset_parameters()


def stack_bg_onto_queries(
    queries: Tensor,
    query_positions: Tensor,
    query_batch_offsets: Tensor,
    background_queries: Tensor,  # [batch_size x (1 or n_dn_groups+1) x embed_dim]
    attn_mask: Optional[Tensor] = None,
    dn_info_dict: Optional[
        dict[str, Union[Tensor, list[Tensor], list[int], int]]
    ] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Augment the object query tensor with background queries for processing by the
    layers in the segmentation head.

    This function takes the object query tensor in batch-concatenated format, with or
    without denoising queries, and properly interleaves the given background queries
    so that each image's background embedding (and potentially each denoising group's
    background embedding) can be made part of the self-attention and FFN operations
    within the segmentation head.

    The query position tensor is also updated, with a dummy all-zero position
    inserted in the proper location for each background query added.

    If provided, this function also expands the attention mask used for masking
    denoising groups from the main queries and each other.

    Args:
        queries (Tensor): Batch-concatenated object query tensor, of shape
            [n_queries, embed_dim].
        query_positions (Tensor): Batch-concatenated tensor of query positions,
            of shape [n_queries, position_dim].
        query_batch_offsets (Tensor): Batch offsets tensor for input queries.
        background_queries (Tensor): Background query tensor to be interleaved into the
            query tensor. Must be of shape [batch_size, n_dn_groups+1, embed_dim],
            where n_dn_groups may be 0 if denoising queries are not present.
        attn_mask (Optional[Tensor]): Attention mask for denoising training. Must be
            a boolean tensor of shape [batch_size, max_queries, max_queries], where
            max_queries is the maximum number of queries among images. If provided,
            this function expands the attention mask to account for the additional
            background queries inserted.
        dn_info_dict (Optional[dict]): Metadata dictionary returned from the
            denoising query generator when stacking denoising queries onto the main
            queries. Must be provided if denoising queries are present. Contains
            information like the number of denoising groups, to ensure proper
            insertion of background queries.

    Returns:
        queries_with_bg (Tensor): Batch-concatenated query tensor with background
            queries inserted. The queries will be in the order
            [main queries, background query, dn group 0 queries, dn group 0 bg query,
             dn group 1 queries, dn group 1 bg query, ...]
        pos_with_bg (Tensor): Batch-concatenated query position tensor, with positions
            in the same order as queries_with_bg and background queries given all-0
            positions.
        new_offsets (Tensor): `query_batch_offsets` for the new enlarged tensors.
        is_background (Tensor): Boolean tensor of size [queries_with_bg.shape[0]] that
            is True at positions where the output tensor are the background queries.
        new_attn_mask (Tensor): Expanded denoising training attention mask, of shape
            [batch_size, new_max_queries, new_max_queries], where new_max_queries is
            larger than the input max_queries size to account for the main queries and
            each denoising query group having the additional background query.
            Returns None if input attn_mask is None.
    """
    batch_size = len(query_batch_offsets) - 1

    # components for stacking
    queries_split: list[Tensor] = split_batch_concatted_tensor(
        queries, query_batch_offsets
    )
    pos_split: list[Tensor] = split_batch_concatted_tensor(
        query_positions, query_batch_offsets
    )
    # bg_query_split = background_queries.split(1, dim=0)
    bg_split = background_queries.unbind(0)  # list[B] (G+1, D)
    zero_pos = query_positions.new_zeros(1, query_positions.size(-1))

    if dn_info_dict is None:  # not denoising
        assert attn_mask is None  # assume attn_mask only for masking dn groups
        # stack background query onto object queries
        queries_with_bg_list: list[Tensor] = []
        pos_with_bg_list: list[Tensor] = []
        is_background_list: list[Tensor] = []
        for q, p, bg in zip(queries_split, pos_split, bg_split):
            queries_with_bg_list.extend([q, bg])
            pos_with_bg_list.extend([p, zero_pos])
            is_background_list.extend(
                [
                    torch.zeros(q.size(0), dtype=torch.bool, device=q.device),
                    torch.ones(bg.size(0), dtype=torch.bool, device=bg.device),
                ]
            )
        queries_with_bg = torch.cat(queries_with_bg_list)
        pos_with_bg = torch.cat(pos_with_bg_list)
        is_background = torch.cat(is_background_list)

        new_offsets = query_batch_offsets + torch.arange(
            len(query_batch_offsets), device=query_batch_offsets.device
        )

        return queries_with_bg, pos_with_bg, new_offsets, is_background, None

    ### with denoising groups ###

    # need to account for denoising by adding bg query to every dn group
    # and updating attn_mask
    n_obj_per_image = dn_info_dict["n_objects_per_image"]
    assert isinstance(n_obj_per_image, list)
    assert torch.jit.isinstance(n_obj_per_image, list[int])
    n_dn_groups = dn_info_dict["n_denoising_groups"]
    assert isinstance(n_dn_groups, int)
    n_main_q_per_image = dn_info_dict["n_main_queries_per_image"]
    assert isinstance(n_main_q_per_image, list)
    assert torch.jit.isinstance(n_main_q_per_image, list[int])
    old_lens = [
        main + obj * n_dn_groups * 2
        for main, obj in zip(n_main_q_per_image, n_obj_per_image)
    ]

    queries_with_bg_list: list[Tensor] = []
    pos_with_bg_list: list[Tensor] = []
    is_background_list: list[Tensor] = []
    new_lens: list[int] = []

    for b, (queries_b, pos_b, bg_b) in enumerate(
        zip(queries_split, pos_split, bg_split)
    ):
        n_q_main_b = n_main_q_per_image[b]
        assert isinstance(n_q_main_b, int)
        dn_group_size_b = n_obj_per_image[b] * 2  # (pos + neg)
        assert isinstance(dn_group_size_b, int)
        q_main, q_dn = queries_b.tensor_split([n_q_main_b])
        p_main, p_dn = pos_b.tensor_split([n_q_main_b])
        bg_main, bg_dn = bg_b.tensor_split([1])

        # Add background query to main queries
        queries_with_bg_list.extend([q_main, bg_main])
        pos_with_bg_list.extend([p_main, zero_pos])

        is_background_list.extend(
            [
                torch.zeros(q_main.size(0), dtype=torch.bool, device=q_main.device),
                torch.ones(bg_main.size(0), dtype=torch.bool, device=bg_main.device),
            ]
        )

        # Add background query to each denoising group
        q_dn_groups = q_dn.split(dn_group_size_b)
        p_dn_groups = p_dn.split(dn_group_size_b)
        bg_dn_groups = bg_dn.split(1)
        assert len(q_dn_groups) == len(p_dn_groups) == len(bg_dn_groups) == n_dn_groups
        for q, p, bg in zip(q_dn_groups, p_dn_groups, bg_dn_groups):
            queries_with_bg_list.extend([q, bg])
            pos_with_bg_list.extend([p, zero_pos])
            is_background_list.extend(
                [
                    torch.zeros(q.size(0), dtype=torch.bool, device=q.device),
                    torch.ones(bg.size(0), dtype=torch.bool, device=bg.device),
                ]
            )

        new_len_b = n_q_main_b + 1 + n_dn_groups * (dn_group_size_b + 1)
        new_lens.append(new_len_b)

    # concatted queries and positions with background queries interleaved
    queries_with_bg = torch.cat(queries_with_bg_list)
    pos_with_bg = torch.cat(pos_with_bg_list)
    is_background = torch.cat(is_background_list)
    # new batch offsets
    new_offsets = seq_lengths_to_batch_offsets(
        torch.tensor(new_lens, device=query_batch_offsets.device)
    )
    assert isinstance(new_offsets, Tensor)

    if attn_mask is None:
        return queries_with_bg, pos_with_bg, new_offsets, is_background, None

    # stacked query and position tensors complete, now need to build new attn mask
    max_new_q = max(new_lens)
    new_attn_mask = attn_mask.new_zeros(batch_size, max_new_q, max_new_q)

    for b, (n_main_b, n_obj, n_total_new_b, n_total_old_b) in enumerate(
        zip(n_main_q_per_image, n_obj_per_image, new_lens, old_lens)
    ):
        new_mask_b = new_attn_mask[b]
        old_mask_b = attn_mask[b]
        dn_group_size_b = n_obj * 2
        dn_group_size_plus_1 = dn_group_size_b + 1

        # copy over the old main-to-main part of the old mask (upper left corner)
        # everything here lines up because we haven't reached a background query
        # all of this chunk should be False already
        assert not old_mask_b[:n_main_b, :n_main_b].any()
        new_mask_b[:n_main_b, :n_main_b] = old_mask_b[:n_main_b, :n_main_b]

        # index tracking
        main_bg_idx = n_main_b  # index of background query for main queries
        dn_start = main_bg_idx + 1

        # mask out denoising queries from main queries
        new_mask_b[:dn_start, dn_start:n_total_new_b] = True

        # mask denoising groups from each other
        for g in range(n_dn_groups):
            g_start = dn_start + g * dn_group_size_plus_1
            g_end = g_start + dn_group_size_plus_1

            # mask out this group from other dn groups
            new_mask_b[g_start:g_end, dn_start:g_start] = True  # earlier groups
            new_mask_b[g_start:g_end, g_end:n_total_new_b] = True  # later groups

        # mask main queries from denoising queries if it was done in original mask
        if old_mask_b[n_main_b:n_total_old_b, :n_main_b].any():
            assert old_mask_b[n_main_b:n_total_old_b, :n_main_b].all()  # sanity check
            # mask out main queries + main bg query from denoising queries
            new_mask_b[:dn_start:n_total_new_b, : n_main_b + 1] = True

        # mask out padding part
        new_mask_b[n_total_new_b:, :] = True
        new_mask_b[:, n_total_new_b:] = True

        assert torch.equal(new_mask_b, new_attn_mask[b])

    return queries_with_bg, pos_with_bg, new_offsets, is_background, new_attn_mask


def unstack_bg_from_queries(
    queries_with_bg: Tensor, batch_offsets: Tensor, dn_info_dict: Optional[dict] = None
) -> tuple[Tensor, Tensor]:
    """Unpack the background-augmented query tensors into the original foreground-query
    and background-query tensors.

    Args:
        queries_with_bg (Tensor): Query tensor in the stacking order from
            stack_bg_onto_queries, with shape [N, embed_dim] where N has all images'
            queries stacked together, with each image's queries in the order
            (main queries, background query for main,
             n_dn_groups * (dn queries, background query for dn)
            ). n_dn_groups may be 0 if denoising queries are not present.
        batch_offsets (Tensor): Batch offsets tensor for bg-augmented queries
        dn_info_dict (Optional[dict]): Must be included when denoising queries are
            present. Contains information like the number of denoising groups, to be
            used in properly separating the query tensors.

    Returns:
        foreground_queries (Tensor): Query tensor with the background queries pulled
            out, in the same stacking order as the input query tensor to
            stack_bg_onto_queries.
        background_queries (Tensor): Background query tensor of shape [B, 1, embed_dim]
            or [B, (1 + n_dn_groups), embed_dim] depending on if denoising queries
            are present. This is the same shape and stacking order as the corresponding
            input to stack_bg_onto_queries.
    """
    batch_size = len(batch_offsets) - 1
    embed_dim = queries_with_bg.shape[-1]
    device = queries_with_bg.device

    if dn_info_dict is not None:
        n_obj_per_image: list[int] = dn_info_dict["n_objects_per_image"]
        n_dn_groups: int = dn_info_dict["n_denoising_groups"]
        n_main_q_per_image: list[int] = dn_info_dict["n_main_queries_per_image"]
    else:
        n_dn_groups = 0
        n_main_q_per_image: list[int] = (batch_offsets.diff() - 1).tolist()

    # get indices of every background query (main plus all dn groups for each image)
    bg_indices_list: list[Tensor] = []
    for b in range(batch_size):
        batch_start = batch_offsets[b]
        n_main_b = n_main_q_per_image[b]

        # background query for main queries
        bg_indices_list.append((batch_start + n_main_b).unsqueeze(0))

        if n_dn_groups > 0:
            # compute background query index for all denoising groups
            dn_group_size = n_obj_per_image[b] * 2
            start_dn = batch_start + n_main_b + 1
            bg_dn = (start_dn + dn_group_size) + torch.arange(
                n_dn_groups, device=device
            ) * (dn_group_size + 1)

            bg_indices_list.append(bg_dn)

    bg_indices = torch.cat(bg_indices_list)

    background_queries = queries_with_bg[bg_indices]
    background_queries = background_queries.view(batch_size, n_dn_groups + 1, embed_dim)

    is_fg = torch.ones(queries_with_bg.size(0), dtype=torch.bool, device=device)
    is_fg[bg_indices] = False

    foreground_queries = queries_with_bg[is_fg]

    return foreground_queries, background_queries


class PatchedSegmentationMapPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
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
        self.embed_dim = embed_dim
        layers = []
        for _ in range(n_transformer_layers):
            layers.append(
                SegmentationMapLayer(
                    embed_dim,
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

    def forward(
        self,
        queries: Tensor,
        query_batch_offsets: Tensor,
        query_positions: Tensor,
        stacked_feature_map: Tensor,
        level_spatial_shapes: Tensor,
        background_embedding: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        dn_info_dict: Optional[dict] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        batch_size = len(query_batch_offsets) - 1
        # Stack the background embeddings onto the queries
        if background_embedding is not None:
            # maxpool background embedding over levels to get bg query
            background_query = background_embedding.max(1, keepdim=True).values
            assert background_query.shape == (batch_size, 1, self.embed_dim)
            if dn_info_dict is not None:
                # make separate bg query for main queries and each dn group
                n_dn_groups: int = dn_info_dict["n_denoising_groups"]
                background_query = background_query.expand(-1, n_dn_groups + 1, -1)
            else:
                n_dn_groups = 0
            (
                queries,
                query_positions_stacked,
                query_batch_offsets_stacked,
                is_background_mask,
                attn_mask,
            ) = stack_bg_onto_queries(
                queries,
                query_positions,
                query_batch_offsets,
                background_query,
                attn_mask,
                dn_info_dict,
            )
        else:
            query_batch_offsets_stacked = query_batch_offsets
            query_positions_stacked = query_positions
            is_background_mask = None

        # Run the queries through the transformer layers
        for layer in self.layers:
            queries = layer(
                queries,
                query_batch_offsets_stacked,
                query_positions_stacked,
                stacked_feature_map,
                level_spatial_shapes,
                background_embedding,
                is_background_mask=is_background_mask,
                attn_mask=attn_mask,
                dn_info_dict=dn_info_dict,
            )

        if is_background_mask is not None:
            background_query = queries[is_background_mask].view(
                batch_size, n_dn_groups + 1, queries.shape[-1]
            )
            queries = queries[~is_background_mask]

        ##### build output segmentation tensors #####
        # 1. find feature level corresponding to full-scale pixels
        max_level_index = level_spatial_shapes.argmax(dim=0)
        max_level_index = torch.unique(max_level_index)
        assert len(max_level_index) == 1  # same level for all batches
        max_level_index = int(max_level_index.item())

        assert torch.equal(
            level_spatial_shapes.new_tensor(stacked_feature_map.shape[1:3]),
            level_spatial_shapes[max_level_index],
        ), print(f"{stacked_feature_map.shape=}")

        # 2. get the full-scale feature map
        fullscale_feature_map = sparse_select(stacked_feature_map, 3, max_level_index)
        assert isinstance(fullscale_feature_map, Tensor)
        assert fullscale_feature_map.ndim == 4  # (batch, height, width, feature)

        # 3. Get indices of n x n patches around each query location
        if level_spatial_shapes.ndim == 3:
            max_level_shape = level_spatial_shapes[0, max_level_index].view(1, 2)
        else:
            assert level_spatial_shapes.ndim == 2
            max_level_shape = level_spatial_shapes[max_level_index].view(1, 2)
        patch_indices, patch_oob, _ = get_multilevel_neighborhoods(
            query_positions, max_level_shape, [self.query_patch_diameter]
        )
        oob_check = torch.any(
            (patch_indices < 0)
            | (patch_indices >= level_spatial_shapes[max_level_index]),
            -1,
        )
        if not torch.equal(oob_check, patch_oob):
            raise ValueError("oob_check and patch_oob not equal")

        # Indices: [n_total_query x n_patch_pixels x (i, j)]
        # -> [n_total_query x n_patch_pixels x (batch, i, j, query)]
        indices = patch_indices.new_empty(
            patch_indices.shape[:-1] + (patch_indices.shape[-1] + 2,)
        )
        query_seq_lengths: Tensor = batch_offsets_to_seq_lengths(query_batch_offsets)
        indices[..., 0] = seq_lengths_to_indices(query_seq_lengths).unsqueeze(-1)
        indices[..., 1:-1] = patch_indices
        for b in range(query_batch_offsets.size(0) - 1):
            batch_start = int(query_batch_offsets[b])
            batch_end = int(query_batch_offsets[b + 1])
            indices[batch_start:batch_end, :, -1] = torch.arange(
                batch_end - batch_start, device=indices.device
            ).unsqueeze(-1)

        # 4. Extract the patches from the full-scale pixel feature map (foreground pixels)
        patch_embeddings, patch_is_specified_mask = batch_sparse_index(
            fullscale_feature_map, indices[..., :-1]  # index (batch, i, j)
        )
        assert isinstance(patch_embeddings, Tensor)
        if not torch.all(
            (patch_oob & (~patch_is_specified_mask)) == patch_oob
        ):  # sanity check
            specified_oob_mask = patch_oob & patch_is_specified_mask
            raise ValueError(
                f"oob indices: {patch_indices[patch_oob]}\n"
                f"Specified oob indices: {patch_indices[specified_oob_mask]}"
            )

        # 5. compute segmentation logits of each query within its patch of pixels
        # dot product each query vector with all of its patch's embeddings
        # [n_total_query x embed_dim] @ [n_total_query x patch_pixels x embed_dim]
        # -> [n_total_query x patch_pixels]
        patch_segmentation_logits = torch.bmm(
            queries.unsqueeze(1), patch_embeddings.transpose(-1, -2)
        ).squeeze(1)

        #### 6. now need to potentially split foreground pixel logits into main queries and
        # dn groups
        batch_indices = batch_offsets_to_indices(query_batch_offsets)
        dn_group_indices = indices.new_zeros(
            queries.size(0),  # will be filled in if dn groups are present
        )

        if dn_info_dict is not None:
            # dn group indices with convention that 0th group is main queries
            # and 1-N are dn groups
            # need to separate denoising queries into their own sparse tensor
            # get tensor of shape [n_total_queries] that has the index
            # of each query's denoising group (with dn group 0 being main (non-dn)
            # queries)
            n_dn_groups = dn_info_dict["n_denoising_groups"]
            for b in range(batch_size):
                batch_start = int(query_batch_offsets[b])
                batch_end = int(query_batch_offsets[b + 1])
                dn_group_sizes = (
                    dn_group_indices.new_tensor(dn_info_dict["n_objects_per_image"]) * 2
                )
                n_main_queries = dn_group_indices.new_tensor(
                    dn_info_dict["n_main_queries_per_image"]
                )

                dn_group_indices[batch_start + int(n_main_queries[b]) : batch_end] = (
                    torch.repeat_interleave(
                        torch.arange(n_dn_groups, device=dn_group_indices.device)
                        + 1,  # +1 to make main queries index 0
                        int(dn_group_sizes[b]),
                    )
                )

            # verify (n_dn_groups + 1) unique indices
            assert dn_group_indices.amax() == n_dn_groups

            # use dn group indices to separate out main and dn

            # boolean masks of indices that are in main queries (non-denoising)
            indices_w_dn_group = torch.cat(  # add dn group index to fg indices
                [
                    indices[:, :, :-1],
                    dn_group_indices[:, None, None].expand(-1, indices.size(1), 1),
                    indices[:, :, -1:],
                ],
                dim=-1,
            )  # (batch, i, j, dn group, query index within image)
            fg_indices = indices_w_dn_group[patch_is_specified_mask]
            fg_is_main = fg_indices[:, -2] == 0

            fg_logits = patch_segmentation_logits[patch_is_specified_mask]

            # main foreground sparse tensor inputs
            main_fg_indices = fg_indices[fg_is_main][
                :, [0, 1, 2, 4]
            ]  # drop dn group index
            main_fg_logits = fg_logits[fg_is_main]

            # denoising foreground sparse tensor inputs
            dn_fg_indices = fg_indices[~fg_is_main]
            dn_fg_indices[:, -2] -= 1  # reindex dn groups from 0 to N-1
            dn_query_reindexing_offset = (  # reindex each dn group's dn queries from 0 to N-1
                n_main_queries[dn_fg_indices[:, 0]]
                + dn_group_sizes[dn_fg_indices[:, 0]] * dn_fg_indices[:, -2]
            )
            dn_fg_indices[:, -1] -= dn_query_reindexing_offset
            dn_fg_logits = fg_logits[~fg_is_main]

        else:  # no denoising, so all foreground logits are main
            main_fg_indices = indices[patch_is_specified_mask]
            main_fg_logits = patch_segmentation_logits[patch_is_specified_mask]
            dn_fg_indices = None
            dn_fg_logits = None
            n_main_queries = batch_offsets_to_seq_lengths(query_batch_offsets)
            n_dn_groups = 0

        ###########
        # 7. handle background logits if background queries present
        ###########
        if background_query is not None:
            ### Need to compute background segmentation logits using bg queries
            # go from [batch_size, (n_dn_groups + 1), embed_dim] to
            # [n_total_queries, embed_dim]
            broadcasted_bg_queries = background_query[batch_indices, dn_group_indices]

            # [n_total_queries, patch_pixels]
            bg_logits = torch.bmm(
                broadcasted_bg_queries.unsqueeze(1),
                patch_embeddings.transpose(-1, -2),
            ).squeeze(1)

            # since bg query is the same within a denoising group, filter down
            # bg queries to unique (batch, i, j, dn group) indices
            # bg_indices: [n_total_queries x patch_pixels x (batch, i, j, dn group)]
            bg_indices = torch.cat(
                [
                    indices[..., :-1],
                    dn_group_indices[:, None, None].expand(-1, indices.size(1), 1),
                ],
                dim=-1,
            )

            # find all unique (batch, i, j, dn group) indices among bg indices
            # flatten to linear (raveled) indices to call torch.unique
            bg_indices_2d = bg_indices[patch_is_specified_mask]
            bg_indices_flat: Tensor = flatten_nd_indices(
                bg_indices_2d.T,
                bg_indices_2d.new_tensor(
                    fullscale_feature_map.shape[:-1] + (n_dn_groups + 1,)
                ),
            )[0].squeeze(0)

            unique_flat_indices, unique_inverse = torch.unique(
                bg_indices_flat, return_inverse=True
            )
            unique_row_indices: Tensor = unique_inverse.new_full(
                (unique_flat_indices.size(0),), bg_indices_flat.size(0)
            )
            unique_row_indices.scatter_reduce_(
                0,
                unique_inverse,
                torch.arange(bg_indices_flat.size(0), device=bg_indices_flat.device),
                "amin",
            )
            # unique_row_indices has all indices of unique bg queries
            unique_bg_indices = bg_indices_2d[unique_row_indices]
            unique_bg_logits = bg_logits[patch_is_specified_mask][unique_row_indices]

            # bg logits for main and potentially dn now computed

            if dn_info_dict is not None:
                # need to split bg logits into main and dn, similar to fg logits
                bg_is_main = unique_bg_indices[:, -1] == 0

                main_bg_indices = unique_bg_indices[bg_is_main]
                # make bg query the last in each image
                main_bg_indices[:, -1] = n_main_queries[main_bg_indices[:, 0]]
                main_bg_logits = unique_bg_logits[bg_is_main]

                main_indices = torch.cat([main_fg_indices, main_bg_indices]).T
                main_logits = torch.cat([main_fg_logits, main_bg_logits])

                dn_bg_indices = unique_bg_indices[~bg_is_main]
                dn_bg_indices[:, -1] -= 1  # reindex dn groups from 0 to N-1
                # 5th index: dn query index within group (make bg index of dn group size)
                dn_bg_indices = torch.cat(
                    [dn_bg_indices, dn_group_sizes[dn_bg_indices[:, 0], None]], dim=1
                )
                dn_bg_logits = unique_bg_logits[~bg_is_main]

                assert dn_fg_logits is not None and dn_fg_indices is not None
                dn_logits = torch.cat([dn_fg_logits, dn_bg_logits])
                dn_indices = torch.cat([dn_fg_indices, dn_bg_indices]).T

            else:  # no denoising
                # make dn query last index
                unique_bg_indices[:, -1] = n_main_queries[unique_bg_indices[:, 0]]

                main_indices = torch.cat([main_fg_indices, unique_bg_indices]).T
                main_logits = torch.cat([main_fg_logits, unique_bg_logits])
                dn_indices = None
                dn_logits = None

        else:  # no background queries
            main_indices = main_fg_indices.T
            main_logits = main_fg_logits
            dn_indices = dn_fg_indices.T if dn_fg_indices is not None else None
            dn_logits = dn_fg_logits

        ### 8. now finally construct the segmentation sparse tensors
        main_segmentation_logits = torch.sparse_coo_tensor(
            main_indices,
            main_logits,
            size=fullscale_feature_map.shape[:-1] + (int(n_main_queries.amax() + 1),),
        ).coalesce()

        if dn_indices is None:
            return main_segmentation_logits, None

        assert dn_logits is not None
        dn_segmentation_logits = torch.sparse_coo_tensor(
            dn_indices,
            dn_logits,
            size=fullscale_feature_map.shape[:-1]
            + (n_dn_groups, int(dn_group_sizes.amax() + 1)),
        ).coalesce()

        return main_segmentation_logits, dn_segmentation_logits

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    @classmethod
    def from_config(cls, config: SegmentationHeadConfig) -> Self:
        return cls(
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            n_transformer_layers=config.n_layers,
            dropout=config.dropout,
            attn_proj_bias=config.attn_proj_bias,
            activation_fn=config.activation_fn,
            norm_first=config.norm_first,
            rope_share_heads=config.rope.share_heads,
            rope_spatial_base_theta=config.rope.spatial_base_theta,
            rope_level_base_theta=config.rope.level_base_theta,
            rope_freq_group_pattern=config.rope.freq_group_pattern,
            query_patch_diameter=config.query_patch_diameter,
        )


def sparse_binary_segmentation_map(segmentation_map: Tensor):
    assert segmentation_map.is_sparse
    return torch.sparse_coo_tensor(
        segmentation_map.indices(),
        segmentation_map.values() > 0.0,
        segmentation_map.shape,
        device=segmentation_map.device,
    )
