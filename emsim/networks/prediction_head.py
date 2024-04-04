import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils.window_utils import windowed_keys_for_queries
from ..utils.sparse_utils import spconv_to_torch_sparse

import spconv.pytorch as spconv


class PredictionHead(nn.Module):
    def __init__(
        self,
        query_dim: int,
        pixel_feature_dim: int,
        hidden_dim: int,
        activation="relu",
        mask_window_size: int = 5,
    ):
        super().__init__()
        self.mask_window_size = mask_window_size

        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "gelu":
            activation_fn = nn.GELU

        self.class_prediction_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim), activation_fn(), nn.Linear(hidden_dim, 2)
        )

        self.pixel_mask_head = nn.Sequential(
            nn.Linear(pixel_feature_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, query_dim),
        )

    def forward(
        self, decoded_query_dict: dict[str, Tensor], image_feature_tensor: Tensor
    ):
        if isinstance(image_feature_tensor, spconv.SparseConvTensor):
            image_feature_tensor = spconv_to_torch_sparse(image_feature_tensor)

        mask_logits = self.predict_mask_logits(decoded_query_dict, image_feature_tensor)

    def predict_mask_logits(
        self, decoded_query_dict: dict[str, Tensor], image_feature_tensor: Tensor
    ):
        # Compute the mask logits: query feature dotted with MLPed pixel feature
        mask_keys, key_indices, _, key_pad_mask = windowed_keys_for_queries(
            decoded_query_dict["indices"],
            image_feature_tensor,
            self.mask_window_size,
            self.mask_window_size,
        )
        mask_keys = self.pixel_mask_head(mask_keys)
        mask_logits = torch.einsum(
            "qf,qhwf->qhw", decoded_query_dict["queries"], mask_keys
        )
        mask_logits = mask_logits + mask_logits.new_zeros(
            mask_logits.shape
        ).masked_fill(key_pad_mask, -torch.inf)

        # Number the queries in each batch element
        mask_batch_offsets = torch.unique_consecutive(
            key_indices[:, 0, 0, 0], return_counts=True
        )[-1].cumsum(0)
        mask_batch_offsets = torch.cat(
            [mask_batch_offsets.new_zeros([1]), mask_batch_offsets]
        )
        per_batch_mask_index = torch.cat(
            [
                torch.arange(end - start, device=mask_batch_offsets.device)
                for start, end in zip(mask_batch_offsets[:-1], mask_batch_offsets[1:])
            ]
        )

        # Append the query number to the mask index so each mask logit has an index of
        # batch * y * x * mask
        sparse_mask_indices = torch.cat(
            [
                key_indices,
                per_batch_mask_index.reshape(-1, 1, 1, 1).expand(
                    -1, *key_indices.shape[1:-1], -1
                ),
            ],
            -1,
        )

        # Find the individual mask pixels corresponding to a zero pixel
        flat_mask_logits = mask_logits.flatten()
        nonempty_pixel_indices = flat_mask_logits.nonzero().squeeze(-1)

        # Finally create the predicted segmentation mask logits
        predicted_mask_logit_tensor = torch.sparse_coo_tensor(
            sparse_mask_indices.flatten(0, -2)[nonempty_pixel_indices].T,
            flat_mask_logits[nonempty_pixel_indices],
            size=[*image_feature_tensor.shape[:-1], per_batch_mask_index.max()],
        )

        return predicted_mask_logit_tensor
