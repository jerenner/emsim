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
            nn.Linear(query_dim, hidden_dim), activation_fn(), nn.Linear(hidden_dim, 1)
        )

        self.pixel_mask_head = nn.Sequential(
            nn.Linear(pixel_feature_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, query_dim),
        )

        self.query_portion_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, pixel_feature_dim),
        )

    def forward(
        self, decoded_query_dict: dict[str, Tensor], image_feature_tensor: Tensor
    ):
        if isinstance(image_feature_tensor, spconv.SparseConvTensor):
            image_feature_tensor = spconv_to_torch_sparse(image_feature_tensor)

        binary_mask_logits, portion_logits = self.predict_masks(decoded_query_dict, image_feature_tensor)

    def predict_masks(
        self, decoded_query_dict: dict[str, Tensor], image_feature_tensor: Tensor
    ):
        """Predicts the binary segmentation masks for each query and the per-query
        soft assignment logits for each pixel.

        Args:
            decoded_query_dict (dict[str, Tensor]): Output of TransformerDecoder
            image_feature_tensor (Tensor): Output of backbone

        Returns:
            predicted_masks: Boolean sparse Tensor of size batch x height x width x query
            with True if the pixel is part of that query's mask
            pixel_portions: Sparse FloatTensor of shape batch x height x width x query
            where specified
        """
        # Compute the mask logits: query feature dotted with MLPed pixel feature
        mask_keys, key_indices, _, key_pad_mask = windowed_keys_for_queries(
            decoded_query_dict["indices"],
            image_feature_tensor,
            self.mask_window_size,
            self.mask_window_size,
        )
        mlped_mask_keys = self.pixel_mask_head(mask_keys)
        binary_mask_logits = torch.einsum(
            "qf,qhwf->qhw", decoded_query_dict["queries"], mlped_mask_keys
        )

        # Compute the portion logits: MLPed query feature dotted with pixel feature
        mlped_queries = self.query_portion_head(decoded_query_dict["queries"])
        portion_logits = torch.einsum("qf,qhwf->qhw", mlped_queries, mask_keys)

        # Add pad mask bias
        pad_mask_bias = key_pad_mask.new_zeros(
            key_pad_mask.shape, dtype=binary_mask_logits.dtype
        ).masked_fill(key_pad_mask, -torch.inf)
        binary_mask_logits = binary_mask_logits + pad_mask_bias
        portion_logits = portion_logits + pad_mask_bias

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
        # batch * y * x * query
        sparse_mask_indices = torch.cat(
            [
                key_indices,
                per_batch_mask_index.reshape(-1, 1, 1, 1).expand(
                    -1, *key_indices.shape[1:-1], -1
                ),
            ],
            -1,
        )

        # Find the indices corresponding to empty pixels
        nonzero_keys = mask_keys.any(-1)

        # Finally create the sparse logit tensors
        sparse_indices = sparse_mask_indices[nonzero_keys].T
        sparse_tensor_shape = [
            *image_feature_tensor.shape[:-1],
            per_batch_mask_index.max(),
        ]
        predicted_mask_logit_tensor = torch.sparse_coo_tensor(
            sparse_indices,
            binary_mask_logits[nonzero_keys],
            size=sparse_tensor_shape,
        ).coalesce()
        predicted_portion_logit_tensor = torch.sparse_coo_tensor(
            sparse_indices,
            portion_logits[nonzero_keys],
            size=sparse_tensor_shape,
        ).coalesce()

        return predicted_mask_logit_tensor, predicted_portion_logit_tensor

    def compute_electron_energies(
        self,
        decoded_query_dict: dict[str, Tensor],
        image_feature_tensor: Tensor,
        mask_logit_tensor: Tensor,
    ):
        mask_portions = torch.sparse.softmax(mask_logit_tensor, -1)


def _sparse_sigmoid_threshold(sparse_tensor):
    sparse_tensor = sparse_tensor.coalesce()
    return torch.sparse_coo_tensor(
        sparse_tensor.indices(),
        sparse_tensor.values().sigmoid() > 0.5,
        sparse_tensor.shape,
    ).coalesce()
