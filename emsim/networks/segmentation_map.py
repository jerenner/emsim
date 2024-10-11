from emsim.utils.batching_utils import deconcat_add_batch_dim
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
        batched_queries, query_pad_mask = deconcat_add_batch_dim(
            queries, query_batch_offsets
        )
        # unbind over the level dimension
        fullscale_feature_map = stacked_feature_map.unbind(-2)[-1].coalesce()
        assert fullscale_feature_map.ndim == 4  # (batch, height, width, feature)
        feature_map_batch_offsets = batch_offsets_from_sparse_tensor_indices(
            fullscale_feature_map.indices()
        )
        batched_features, feature_pad_mask = deconcat_add_batch_dim(
            fullscale_feature_map.values(), feature_map_batch_offsets
        )
        batched_feature_indices, index_pad_mask = deconcat_add_batch_dim(
            fullscale_feature_map.indices().T, feature_map_batch_offsets
        )
        assert torch.equal(feature_pad_mask, index_pad_mask)

        segmentation_logits = torch.bmm(batched_features, batched_queries.mT) # (batch, pixel/token, query)

        # now put the segmentation logits in a sparse tensor
        query_index = torch.arange(
            segmentation_logits.shape[-1], device=batched_feature_indices.device
        )
        segmentation_logit_indices = torch.cat(
            [
                batched_feature_indices.unsqueeze(-2).expand(
                    -1, -1, segmentation_logits.shape[-1], -1
                ),
                query_index.expand(*segmentation_logits.shape[:-1], -1).unsqueeze(-1),
            ],
            -1,
        ) # (batch, height, width, query)
        segmentation_logit_pad_mask = torch.logical_or(
            query_pad_mask.unsqueeze(-2), feature_pad_mask.unsqueeze(-1)
        )
        num_nonpad_per_image = segmentation_logit_pad_mask.logical_not().sum([-1, -2])
        out_logits = segmentation_logits.new_zeros(num_nonpad_per_image.sum().item())
        out_indices = segmentation_logit_indices.new_zeros(
            num_nonpad_per_image.sum().item(), 4
        )
        out_offsets = torch.cat(
            [num_nonpad_per_image.new_zeros([1]), num_nonpad_per_image.cumsum(-1)]
        ).to("cpu")

        for batch_pad_mask, batch_logits, batch_indices, start_index, stop_index in zip(
            segmentation_logit_pad_mask.logical_not(),
            segmentation_logits,
            segmentation_logit_indices,
            out_offsets[:-1],
            out_offsets[1:],
        ):
            out_logits[start_index:stop_index] = batch_logits.flatten()[
                batch_pad_mask.flatten()
            ]
            out_indices[start_index:stop_index] = batch_indices.flatten(0, -2)[
                batch_pad_mask.flatten()
            ]

        return torch.sparse_coo_tensor(
            out_indices.T,
            out_logits,
            (*fullscale_feature_map.shape[:-1], segmentation_logits.shape[-1]),
            device=fullscale_feature_map.device,
        ).coalesce()


def sparse_binary_segmentation_map(segmentation_map: Tensor):
    assert segmentation_map.is_sparse
    return torch.sparse_coo_tensor(
        segmentation_map.indices(),
        segmentation_map.values() > 0.0,
        segmentation_map.shape,
        device=segmentation_map.device
    )
