from emsim.utils.batching_utils import (
    split_batch_concatted_tensor,
)
from emsim.utils.sparse_utils import (
    batch_offsets_from_sparse_tensor_indices,
    gather_from_sparse_tensor,
)


import torch
from torch import Tensor, nn


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
    ):
        queries = self.mask_embed(queries)
        # unbind over the level dimension
        fullscale_feature_map = stacked_feature_map.unbind(-2)[-1].coalesce()
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


class PatchedSegmentationMapPredictor(SegmentationMapPredictor):
    def __init__(
        self, d_model: int, mask_head_hidden_layers: int = 3, query_patch_diameter=7
    ):
        super().__init__(d_model, mask_head_hidden_layers)
        self.query_patch_diameter = query_patch_diameter

    def reset_parameters(self):
        for layer in self.mask_embed:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(
        self,
        stacked_feature_map: Tensor,
        queries: Tensor,
        query_batch_offsets: Tensor,
        query_positions: Tensor,
        image_spatial_shapes: Tensor,
    ):
        queries = self.mask_embed(queries)
        # unbind over the level dimension
        fullscale_feature_map = stacked_feature_map.unbind(-2)[-1].coalesce()
        assert fullscale_feature_map.ndim == 4  # (batch, height, width, feature)
        assert image_spatial_shapes.shape == (fullscale_feature_map.shape[0], 2)

        split_queries = split_batch_concatted_tensor(queries, query_batch_offsets)
        split_positions = split_batch_concatted_tensor(
            query_positions, query_batch_offsets
        )

        patch_map_indices = []
        patch_map_values = []
        for i, (im_queries, im_positions, shape) in enumerate(
            zip(split_queries, split_positions, image_spatial_shapes)
        ):
            H = shape[0]
            W = shape[1]
            HW = im_positions.new_tensor([H, W], dtype=torch.int)
            im_query_indices = (im_positions.flip(-1) * HW).int()
            axis = (
                torch.arange(
                    self.query_patch_diameter,
                    device=im_query_indices.device,
                    dtype=im_query_indices.dtype,
                )
                - self.query_patch_diameter // 2
            )
            index_grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1)
            patch_indices = im_query_indices.unsqueeze(1).unsqueeze(1) + index_grid
            patch_indices = patch_indices.clamp_min(0).clamp_max(
                HW.expand_as(patch_indices) - 1
            )
            patch_indices = torch.cat(
                [
                    patch_indices.new_tensor(i).expand(*patch_indices.shape[:-1], 1),
                    patch_indices,
                    torch.arange(
                        patch_indices.shape[0],
                        dtype=patch_indices.dtype,
                        device=patch_indices.device,
                    )
                    .view(-1, 1, 1, 1)
                    .expand(*patch_indices.shape[:-1], 1),
                ],
                -1,
            )
            patch_values = (
                im_queries.unsqueeze(1)
                .unsqueeze(1)
                .expand(*patch_indices.shape[:-1], -1)
            )
            patch_map_indices.append(patch_indices)
            patch_map_values.append(patch_values)

        patch_map_indices = torch.cat(patch_map_indices)
        patch_map_values = torch.cat(patch_map_values)

        feature_map_values = gather_from_sparse_tensor(
            fullscale_feature_map, patch_map_indices[..., :-1]
        )[0]

        patch_map_logits = torch.einsum(
            "qhwf,qhwf->qhw", patch_map_values, feature_map_values
        )

        nonzero_logits = patch_map_logits.nonzero(as_tuple=True)
        patch_segmap = torch.sparse_coo_tensor(
            patch_map_indices[nonzero_logits].T,
            patch_map_logits[nonzero_logits],
            (*fullscale_feature_map.shape[:-1], max(len(q) for q in split_queries)),
        ).coalesce()
        return patch_segmap


def sparse_binary_segmentation_map(segmentation_map: Tensor):
    assert segmentation_map.is_sparse
    return torch.sparse_coo_tensor(
        segmentation_map.indices(),
        segmentation_map.values() > 0.0,
        segmentation_map.shape,
        device=segmentation_map.device,
    )
