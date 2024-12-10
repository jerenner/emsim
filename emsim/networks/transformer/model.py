from typing import Union, Optional

import numpy as np
import torch
import torchvision
from torch import Tensor, nn
import MinkowskiEngine as ME

from emsim.networks.positional_encoding import (
    ij_indices_to_normalized_xy,
    normalized_xy_of_stacked_feature_maps,
)
from emsim.networks.transformer.std_dev_head import StdDevHead
from ...utils.batching_utils import split_batch_concatted_tensor
from ...utils.misc_utils import inverse_sigmoid
from ...utils.sparse_utils import (
    batch_offsets_from_sparse_tensor_indices,
    sparse_index_select,
)
from ..positional_encoding import FourierEncoding
from .decoder import EMTransformerDecoder, TransformerDecoderLayer
from .encoder import EMTransformerEncoder, TransformerEncoderLayer
from ..segmentation_map import SegmentationMapPredictor, PatchedSegmentationMapPredictor
from ..me_salience_mask_predictor import MESparseMaskPredictor
from ..denoising_generator import DenoisingGenerator


class EMTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        n_feature_levels: int,
        n_deformable_points: int,
        backbone_indice_keys: Optional[list[str]] = None,
        dropout: float = 0.1,
        activation_fn="gelu",
        norm_first: bool = True,
        attn_proj_bias: bool = False,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        level_filter_ratio: tuple = (0.25, 0.5, 1.0, 1.0),
        layer_filter_ratio: tuple = (1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
        encoder_max_tokens: int = 10000,
        n_query_embeddings: int = 1000,
        decoder_look_forward_twice: bool = True,
        decoder_detach_updated_positions: bool = True,
        sparse_library: str = "minkowskiengine",
        dimension: int = 2,
        mask_main_queries_from_denoising: bool = False,
    ):
        super().__init__()
        self.two_stage_num_proposals = n_query_embeddings
        if sparse_library == "spconv":
            from ..salience_mask_predictor import SpconvSparseMaskPredictor

            self.salience_mask_predictor = SpconvSparseMaskPredictor(d_model, d_model)
        elif sparse_library == "minkowskiengine":
            self.salience_mask_predictor = MESparseMaskPredictor(d_model, d_model)
        else:
            raise ValueError(f"Unrecognized sparse_library: `{sparse_library=}`")
        self.pos_embedding = FourierEncoding(3, d_model, dtype=torch.double)
        self.n_levels = n_feature_levels
        self.classification_head = nn.Linear(d_model, 1)

        self.alpha = nn.Parameter(torch.Tensor(3), requires_grad=True)

        self.encoder = EMTransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                n_deformable_value_levels=n_feature_levels,
                n_deformable_points=n_deformable_points,
                dropout=dropout,
                activation_fn=activation_fn,
                norm_first=norm_first,
                attn_proj_bias=attn_proj_bias,
            ),
            num_layers=n_encoder_layers,
            score_predictor=self.classification_head,
        )
        self.encoder_output_norm = nn.LayerNorm(d_model)
        self.query_pos_offset_head = nn.Sequential(
            nn.Linear(d_model, d_model, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(d_model, d_model, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(d_model, 2, dtype=torch.double),
        )
        self.segmentation_head = PatchedSegmentationMapPredictor(d_model)
        self.std_head = StdDevHead(d_model)

        if sparse_library == "spconv":
            import spconv.pytorch as spconv

            self.salience_unpoolers = nn.ModuleList(
                [
                    spconv.SparseInverseConv2d(1, 1, 3, indice_key=key)
                    for key in backbone_indice_keys[::-1]
                ]
            )
        elif sparse_library == "minkowskiengine":
            self.salience_unpoolers = nn.ModuleList(
                [
                    torch.compiler.disable(
                        ME.MinkowskiConvolutionTranspose(
                            1, 1, 3, stride=2, dimension=dimension
                        )
                    )
                    for _ in range(len(level_filter_ratio) - 1)
                ]
            )
        else:
            raise ValueError(f"Unrecognized sparse_library: `{sparse_library=}`")
        self.register_buffer("level_filter_ratio", torch.tensor(level_filter_ratio))
        self.register_buffer("layer_filter_ratio", torch.tensor(layer_filter_ratio))
        self.encoder_max_tokens = encoder_max_tokens

        self.object_query_embedding = nn.Embedding(n_query_embeddings, d_model)
        self.decoder = EMTransformerDecoder(
            TransformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                n_deformable_value_levels=n_feature_levels,
                n_deformable_points=n_deformable_points,
                dropout=dropout,
                activation_fn=activation_fn,
                norm_first=norm_first,
                attn_proj_bias=attn_proj_bias,
            ),
            num_layers=n_decoder_layers,
            class_head=self.classification_head,
            position_offset_head=self.query_pos_offset_head,
            std_head=self.std_head,
            segmentation_head=self.segmentation_head,
            look_forward_twice=decoder_look_forward_twice,
            detach_updated_positions=decoder_detach_updated_positions,
        )
        self.mask_main_queries_from_denoising = mask_main_queries_from_denoising

    def reset_parameters(self):
        self.salience_mask_predictor.reset_parameters()
        self.pos_embedding.reset_parameters()
        self.classification_head.reset_parameters()
        nn.init.uniform_(self.alpha, -0.3, 0.3)
        self.encoder.reset_parameters()
        self.encoder_output_norm.reset_parameters()
        for layer in self.query_pos_offset_head:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.salience_unpoolers:
            layer.reset_parameters()
        self.object_query_embedding.reset_parameters()
        self.decoder.reset_parameters()
        self.segmentation_head.reset_parameters()

    def forward(
        self,
        encoded_features: list[ME.SparseTensor],
        image_size: Optional[Union[Tensor, list[int]]],
        denoising_queries: Optional[Tensor] = None,
        denoising_reference_points: Optional[Tensor] = None,
        denoising_batch_offsets: Optional[Tensor] = None,
    ):
        assert len(encoded_features) == self.n_levels

        pos_embed = self.get_position_encoding(encoded_features, image_size)
        pos_embed = [p.to(encoded_features[0].features.dtype) for p in pos_embed]
        pos_embed = [
            ME.SparseTensor(
                pos,
                coordinate_map_key=feat.coordinate_map_key,
                coordinate_manager=feat.coordinate_manager,
            )
            for pos, feat in zip(pos_embed, encoded_features)
        ]

        feat_plus_pos_embed = [
            feat + pos for feat, pos in zip(encoded_features, pos_embed)
        ]
        score_dict = self.level_wise_salience_filtering(feat_plus_pos_embed, image_size)

        encoder_out = self.encoder(
            encoded_features,
            pos_embed,
            score_dict["spatial_shapes"],
            score_dict["selected_token_scores"],
            score_dict["selected_token_ij_level_indices"],
            score_dict["selected_normalized_xy_positions"],
            score_dict["per_layer_subset_indices"],
        )
        encoder_out_batch_offsets = batch_offsets_from_sparse_tensor_indices(
            encoder_out.indices()
        )
        encoder_out_batch_sizes = torch.cat(
            [
                encoder_out_batch_offsets,
                encoder_out_batch_offsets.new_tensor([encoder_out.indices().shape[1]]),
            ]
        ).diff()
        encoder_out_normalized = torch.sparse_coo_tensor(
            encoder_out.indices(),
            self.encoder_output_norm(encoder_out.values()),
            size=encoder_out.shape,
            is_coalesced=encoder_out.is_coalesced(),
        ).coalesce()
        encoder_out_logits: Tensor = self.classification_head(
            encoder_out_normalized.values()
        )

        num_topk = self.two_stage_num_proposals * 4
        topk_by_image = [
            torch.topk(scores, min(num_topk, size), 0)
            for scores, size in zip(
                split_batch_concatted_tensor(
                    encoder_out_logits.squeeze(-1), encoder_out_batch_offsets
                ),
                encoder_out_batch_sizes,
            )
        ]
        topk_scores = torch.cat([topk[0] for topk in topk_by_image])
        topk_indices = torch.cat(
            [
                topk[1] + offset
                for topk, offset in zip(topk_by_image, encoder_out_batch_offsets)
            ]
        )
        topk_bijl_indices = encoder_out.indices()[:, topk_indices].T

        # deduplication via non-maximum suppression
        nms_topk_indices = self.nms_on_topk_index(
            topk_scores, topk_indices, topk_bijl_indices, iou_threshold=0.3
        )
        # sparse tensor with the
        nms_encoder_out = torch.sparse_coo_tensor(
            encoder_out.indices()[:, nms_topk_indices],
            encoder_out.values()[nms_topk_indices],
            encoder_out.shape,
        ).coalesce()
        nms_encoder_out_normalized = encoder_out_normalized.values()[nms_topk_indices]
        nms_topk_logits = encoder_out_logits[nms_topk_indices]
        nms_topk_query_batch_offsets = batch_offsets_from_sparse_tensor_indices(
            nms_encoder_out.indices()
        )
        nms_topk_query_batch_sizes = torch.cat(
            [
                nms_topk_query_batch_offsets,
                nms_topk_query_batch_offsets.new_tensor(
                    [nms_encoder_out.indices().shape[1]]
                ),
            ]
        ).diff()
        nms_topk_position_offsets = self.query_pos_offset_head(
            nms_encoder_out_normalized.double()
        )
        # nms_topk_masks = self.segmentation_head(
        #     encoder_out, nms_encoder_out_normalized, nms_topk_query_batch_offsets
        # )

        nms_proposal_xy = normalized_xy_of_stacked_feature_maps(
            nms_encoder_out, score_dict["spatial_shapes"]
        )
        nms_encoder_out_positions = torch.sigmoid(
            inverse_sigmoid(nms_proposal_xy) + nms_topk_position_offsets
        )
        #####
        reference_points = nms_encoder_out_positions.detach()  # \in [0, 1]
        queries = self.object_query_embedding.weight.unsqueeze(0).expand(
            encoder_out.shape[0], -1, -1
        )
        queries = [q[:size] for q, size in zip(queries, nms_topk_query_batch_sizes)]
        queries = torch.cat(queries)

        if self.training and (
            denoising_queries is not None and denoising_reference_points is not None
        ):
            queries, reference_points, attn_mask, dn_batch_mask_dict = (
                DenoisingGenerator.stack_main_and_denoising_queries(
                    queries,
                    reference_points,
                    nms_topk_query_batch_offsets,
                    denoising_queries,
                    denoising_reference_points,
                    denoising_batch_offsets,
                    self.decoder.layers[0].n_heads,
                    self.mask_main_queries_from_denoising,
                )
            )
            denoising = True
            query_batch_offsets = dn_batch_mask_dict["stacked_batch_offsets"]
        else:
            attn_mask = None
            denoising = False
            query_batch_offsets = nms_topk_query_batch_offsets

        decoder_out = self.decoder(
            queries=queries,
            query_reference_points=reference_points,
            query_batch_offsets=query_batch_offsets,
            stacked_feature_maps=encoder_out_normalized,
            spatial_shapes=score_dict["spatial_shapes"],
            attn_mask=attn_mask,
        )

        if denoising:
            decoder_out, denoising_out = self.unstack_main_denoising_outputs(
                decoder_out, dn_batch_mask_dict
            )
        else:
            denoising_out = None
        # denoising_out = None

        # segmentation_logits = self.segmentation_head(
        #     encoder_out,
        #     decoder_out_queries[-1],
        #     nms_topk_query_batch_offsets,
        #     decoder_out_positions[-1],
        # )

        return (
            decoder_out["logits"],
            decoder_out["positions"],
            decoder_out["std"],
            decoder_out["queries"],
            decoder_out["segmentation_logits"],
            nms_topk_query_batch_offsets,
            denoising_out,
            nms_topk_logits,
            nms_encoder_out_positions,
            encoder_out,
            score_dict,
        )

    def get_position_encoding(
        self,
        # encoded_features: Union[list[spconv.SparseConvTensor], list[ME.SparseTensor]],
        encoded_features: list[ME.SparseTensor],
        full_spatial_shape: Optional[Tensor],
    ):
        if isinstance(encoded_features[0], ME.SparseTensor):
            ij_indices = [encoded.C[:, 1:] for encoded in encoded_features]
            normalized_xy = [
                ij_indices_to_normalized_xy(ij, full_spatial_shape) for ij in ij_indices
            ]
        # elif isinstance(encoded_features[0], spconv.SparseConvTensor):
        #     spatial_sizes = [
        #         encoded.indices.new_tensor(encoded.spatial_shape)
        #         for encoded in encoded_features
        #     ]
        #     ij_indices = [encoded.indices[:, 1:] for encoded in encoded_features]
        #     normalized_xy = [
        #         ij_indices_to_normalized_xy(ij, ss)
        #         for ij, ss in zip(ij_indices, spatial_sizes)
        #     ]
        else:
            raise ValueError(
                "Expected features to be either spconv.SparseConvTensor or "
                f"MinkowskiEngine.SparseTensor, got {type(encoded_features[0])}"
            )

        normalized_x_y_level = [
            torch.cat(
                [xy, xy.new_full([xy.shape[0], 1], i / (len(encoded_features) - 1))], -1
            )
            for i, xy in enumerate(normalized_xy)
        ]

        batch_offsets = torch.cumsum(
            torch.tensor([pos.shape[0] for pos in normalized_x_y_level]), 0
        )[:-1]
        stacked_pos = torch.cat(normalized_x_y_level, 0)
        embedding = self.pos_embedding(stacked_pos)
        embedding = torch.tensor_split(embedding, batch_offsets, 0)
        return embedding

    def level_wise_salience_filtering(
        self, features: list[ME.SparseTensor], full_spatial_shape: Tensor
    ):
        batch_size = len(features[0].decomposition_permutations)

        token_nums = torch.tensor(
            [
                [dp.shape[0] for dp in feat.decomposition_permutations]
                for feat in features
            ],
            device=features[0].device,
        ).T
        assert token_nums.shape == (batch_size, len(features))

        score = None
        scores = []
        spatial_shapes = []
        selected_scores = [[] for _ in range(batch_size)]
        selected_bijl_indices = [[] for _ in range(batch_size)]
        selected_normalized_xy = [[] for _ in range(batch_size)]
        selected_level_indices = [[] for _ in range(batch_size)]
        for level_index, feature_map in enumerate(features):
            if level_index > 0:
                upsampler = self.salience_unpoolers[level_index - 1]
                alpha = self.alpha[level_index - 1]
                upsampled_score = upsampler(score)
                upsampled_score = upsampled_score * alpha
                map_times_upsampled = feature_map * upsampled_score
                feature_map = feature_map + map_times_upsampled

            score: ME.SparseTensor = self.salience_mask_predictor(feature_map)

            # get the indices of each batch element's entries in the sparse values
            batch_element_indices = score.decomposition_permutations
            token_counts = torch.tensor(
                [b.numel() for b in batch_element_indices],
                device=self.level_filter_ratio.device,
            )
            focus_token_counts = (
                token_counts * self.level_filter_ratio[level_index]
            ).int()

            score_values_split = [score.F[i].squeeze(1) for i in batch_element_indices]
            score_bij_indices_split = [score.C[i] for i in batch_element_indices]

            # scale indices by tensor stride
            score_bij_indices_split = [
                indices // indices.new_tensor([1, *score.tensor_stride])
                for indices in score_bij_indices_split
            ]

            # take top k elements for each batch element for this level
            top_elements = [
                values.topk(count)
                for values, count in zip(score_values_split, focus_token_counts)
            ]

            selected_scores_by_batch, selected_inds_by_batch = [
                list(i) for i in zip(*top_elements)
            ]
            selected_bij_indices_by_batch: list[Tensor] = [
                spatial_indices[batch_indices]
                # spatial_indices[batch_indices, 1:]
                for spatial_indices, batch_indices in zip(
                    score_bij_indices_split, selected_inds_by_batch
                )
            ]
            selected_normalized_xy_positions_by_batch = [
                ij_indices_to_normalized_xy(ij_indices[:, 1:], full_spatial_shape)
                for ij_indices in selected_bij_indices_by_batch
            ]
            # append level index as a z dimension
            selected_bij_indices_by_batch = [
                torch.cat(
                    [
                        bij_indices,
                        bij_indices.new_full([bij_indices.shape[0], 1], level_index),
                    ],
                    dim=1,
                )
                for bij_indices in selected_bij_indices_by_batch
            ]

            scores.append(score)
            spatial_shapes.append(
                full_spatial_shape // full_spatial_shape.new_tensor(score.tensor_stride)
            )
            for i in range(batch_size):
                selected_scores[i].append(selected_scores_by_batch[i])
                selected_normalized_xy[i].append(
                    selected_normalized_xy_positions_by_batch[i]
                )
                selected_bijl_indices[i].append(selected_bij_indices_by_batch[i])
                selected_level_indices[i].append(
                    selected_bij_indices_by_batch[i].new_full(
                        [selected_bij_indices_by_batch[i].shape[0]], level_index
                    )
                )

        # now concatenate over the level dimension
        selected_scores = [torch.cat(scores_b, 0) for scores_b in selected_scores]
        selected_normalized_xy = [
            torch.cat(positions_b) for positions_b in selected_normalized_xy
        ]
        selected_bijl_indices = [
            torch.cat(indices_b, 0) for indices_b in selected_bijl_indices
        ]
        selected_level_indices = [
            torch.cat(indices_b, 0) for indices_b in selected_level_indices
        ]

        selected_token_sorted_indices = [
            torch.sort(scores_b, descending=True)[1] for scores_b in selected_scores
        ]
        per_layer_token_counts = [
            (indices.shape[0] * self.layer_filter_ratio).int()
            for indices in selected_token_sorted_indices
        ]

        # cap the number of selected tokens at the given limit
        per_layer_token_counts = [
            counts.clamp_max((self.encoder_max_tokens * self.layer_filter_ratio).int())
            for counts in per_layer_token_counts
        ]

        per_layer_subset_indices = [
            [indices[:count] for count in counts]
            for indices, counts in zip(
                selected_token_sorted_indices, per_layer_token_counts
            )
        ]

        return {
            "score_feature_maps": scores,
            "spatial_shapes": torch.stack(spatial_shapes),
            # "selected_token_scores": torch.nested.as_nested_tensor(selected_scores),
            "selected_token_scores": selected_scores,
            # "selected_normalized_xy_positions": torch.nested.as_nested_tensor(
            #     selected_normalized_xy
            # ),
            "selected_normalized_xy_positions": selected_normalized_xy,
            # "selected_token_ij_level_indices": torch.nested.as_nested_tensor(
            #     selected_ij_indices
            # ),
            "selected_token_ij_level_indices": selected_bijl_indices,
            # "selected_token_level_indices": torch.nested.as_nested_tensor(
            #     selected_level_indices
            # ),
            "selected_token_level_indices": selected_level_indices,
            # "selected_token_sorted_indices": torch.nested.as_nested_tensor(
            #     selected_token_sorted_indices
            # ),
            "selected_token_sorted_indices": selected_token_sorted_indices,
            "per_layer_subset_indices": per_layer_subset_indices,
        }

    def unstack_main_denoising_outputs(
        self, decoder_out: dict[str, Tensor], dn_batch_mask_dict: dict[str, Tensor]
    ):

        main_out = {}
        denoising_out = {
            "electron_batch_offsets": dn_batch_mask_dict["electron_batch_offsets"]
        }

        for key, value in decoder_out.items():
            if isinstance(value, Tensor):
                main_out[key], denoising_out[key] = (
                    DenoisingGenerator.unstack_main_and_denoising_tensor(
                        value, dn_batch_mask_dict
                    )
                )
            else:
                assert key == "segmentation_logits"
                main_out[key] = []
                denoising_out[key] = []
                for layer_seg_logits in value:
                    main = []
                    denoising = []
                    for logits, main_end, n_dn in zip(
                        layer_seg_logits,
                        dn_batch_mask_dict["n_main_queries_per_image"],
                        dn_batch_mask_dict["n_denoising_queries_per_image"],
                    ):
                        logits: Tensor
                        logits = logits.coalesce()
                        dn_end = main_end + n_dn
                        main_i = sparse_index_select(
                            logits, logits.ndim - 1, torch.arange(0, main_end)
                        )
                        main.append(main_i)
                        denoising.append(
                            sparse_index_select(
                                logits, logits.ndim - 1, torch.arange(main_end, dn_end)
                            )
                        )

                    def restack_sparse_segmaps(segmaps: list[Tensor]):
                        max_elecs = max([segmap.shape[-1] for segmap in segmaps])
                        segmaps = [
                            segmap.sparse_resize_(
                                [*segmap.shape[:-1], max_elecs],
                                segmap.sparse_dim(),
                                segmap.dense_dim(),
                            )
                            for segmap in segmaps
                        ]
                        return torch.stack(segmaps, 0).coalesce()

                    main_out[key].append(restack_sparse_segmaps(main))
                    denoising_out[key].append(restack_sparse_segmaps(denoising))

        return main_out, denoising_out

    @torch.no_grad()
    def nms_on_topk_index(
        self,
        topk_scores: Tensor,
        topk_indices: Tensor,
        topk_spatial_indices: Tensor,
        iou_threshold=0.3,
    ):
        image_index, y, x, level = topk_spatial_indices.unbind(-1)
        all_image_indices = image_index.unique_consecutive()

        coords = torch.stack([x - 1.0, y - 1.0, x + 1.0, y + 1.0], -1)
        image_level_index = level + self.n_levels * image_index

        nms_indices = torchvision.ops.batched_nms(
            coords, topk_scores, image_level_index, iou_threshold=iou_threshold
        )
        max_queries_per_image = self.two_stage_num_proposals
        result_indices = []
        for i in all_image_indices:
            topk_index_per_image = topk_indices[
                nms_indices[image_index[nms_indices] == i]
            ]
            result_indices.append(topk_index_per_image[:max_queries_per_image])
        return torch.cat(result_indices)

    def gen_encoder_output_proposals(self, memory):
        pass
