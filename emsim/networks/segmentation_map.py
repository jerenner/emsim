from emsim.utils.batching_utils import (
    deconcat_add_batch_dim,
    split_batch_concatted_tensor,
)
from emsim.utils.sparse_utils import batch_offsets_from_sparse_tensor_indices


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
                [indices.view(-1, indices.shape[-1]) for indices in split_segmentation_logit_indices]
            ).T,
            torch.cat([logits.flatten() for logits in split_segmentation_logits]),
            (*fullscale_feature_map.shape[:-1], max(len(q) for q in split_queries)),
        ).coalesce()


def sparse_binary_segmentation_map(segmentation_map: Tensor):
    assert segmentation_map.is_sparse
    return torch.sparse_coo_tensor(
        segmentation_map.indices(),
        segmentation_map.values() > 0.0,
        segmentation_map.shape,
        device=segmentation_map.device,
    )
