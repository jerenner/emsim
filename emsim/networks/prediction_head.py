import torch
import torch.nn.functional as F
from torch import Tensor, nn
from math import log

from ..utils.window_utils import windowed_keys_for_queries
from ..utils.sparse_utils import spconv_to_torch_sparse, batch_sparse_index

import spconv.pytorch as spconv


class PredictionHead(nn.Module):
    def __init__(
        self,
        query_dim: int,
        pixel_feature_dim: int,
        hidden_dim: int,
        activation="relu",
        mask_window_size: int = 7,
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

        self.position_std_dev_head = nn.Sequential(
            nn.Linear(query_dim, hidden_dim), activation_fn(), nn.Linear(hidden_dim, 3)
        )

    def forward(
        self,
        decoded_query_dict: dict[str, Tensor],
        image_feature_tensor: Tensor,
        input_frame: Tensor,
    ):
        decoded_query_dict = decoded_query_dict.copy()
        if isinstance(image_feature_tensor, spconv.SparseConvTensor):
            image_feature_tensor = spconv_to_torch_sparse(image_feature_tensor)
        if isinstance(input_frame, spconv.SparseConvTensor):
            input_frame = spconv_to_torch_sparse(input_frame)

        decoded_query_dict["is_electron_logit"], decoded_query_dict["is_electron"] = (
            self.predict_class(decoded_query_dict)
        )
        (
            decoded_query_dict["binary_mask_logits"],
            decoded_query_dict["portion_logits"],
            decoded_query_dict["key_indices"],
            decoded_query_dict["nonzero_key_mask"],
        ) = self.predict_masks(decoded_query_dict, image_feature_tensor)

        decoded_query_dict["position_std_dev_cholesky"] = (
            self.predict_position_std_dev_cholesky(decoded_query_dict)
        )

        (
            decoded_query_dict["binary_mask_logits_sparse"],
            decoded_query_dict["portion_logits_sparse"],
        ), decoded_query_dict[
            "sparse_indices"
        ] = _batched_query_key_map_to_sparse_tensors(
            [
                decoded_query_dict["binary_mask_logits"],
                decoded_query_dict["portion_logits"],
            ],
            decoded_query_dict["key_indices"],
            decoded_query_dict["nonzero_key_mask"],
            image_feature_tensor.shape[:-1],
        )

        # decoded_query_dict["energy_portions"], decoded_query_dict["energies"] = (
        #     self.compute_electron_energies(energy_portion_logits_sparse, input_frame)
        # )

        return decoded_query_dict

    def predict_class(self, decoded_query_dict: dict[str, Tensor]):
        electron_prob_logit = self.class_prediction_head(
            decoded_query_dict["queries"]
        ).squeeze(-1)
        is_electron_prediction = electron_prob_logit > 0.0
        return electron_prob_logit, is_electron_prediction

    def predict_masks(
        self, decoded_query_dict: dict[str, Tensor], image_feature_tensor: Tensor
    ):
        """Predicts the binary segmentation masks for each query and the per-query
        soft assignment logits for each pixel.

        Args:
            decoded_query_dict (dict[str, Tensor]): Output of TransformerDecoder
            image_feature_tensor (Tensor): Output of backbone

        Returns:
            dict[str, Tensor]: decoded_query_dict with new elements:
                binary_mask_logits
                portion_logits
                key_indices
                nonzero_key_mask
        """
        # Compute the mask logits: query feature dotted with MLPed pixel feature
        mask_keys, key_indices, _, key_pad_mask, is_specified_mask = (
            windowed_keys_for_queries(
                decoded_query_dict["indices"],
                image_feature_tensor,
                self.mask_window_size,
                self.mask_window_size,
            )
        )
        mlped_mask_keys = self.pixel_mask_head(mask_keys)
        binary_mask_logits = torch.einsum(
            "qf,qhwf->qhw", decoded_query_dict["queries"], mlped_mask_keys
        )

        # Compute the portion logits: MLPed query feature dotted with pixel feature
        mlped_queries = self.query_portion_head(decoded_query_dict["queries"])
        portion_logits = torch.einsum("qf,qhwf->qhw", mlped_queries, mask_keys)

        # Add pad mask bias
        pad_mask_bias = (
            key_pad_mask.new_zeros(key_pad_mask.shape, dtype=binary_mask_logits.dtype)
            .masked_fill(key_pad_mask, -1e6)
            .masked_fill(is_specified_mask.logical_not(), -1e6)
        )
        binary_mask_logits = binary_mask_logits + pad_mask_bias
        portion_logits = portion_logits + pad_mask_bias

        return binary_mask_logits, portion_logits, key_indices, is_specified_mask

    def predict_position_std_dev_cholesky(
        self, decoded_query_dict: dict[str, Tensor], epsilon: float = 1e-6, max_cov: float = 1e5,
    ):
        logdiag, offdiag = self.position_std_dev_head(
            decoded_query_dict["queries"]
        ).split([2, 1], -1)
        logdiag = logdiag.clamp_max(log(max_cov))
        diag = logdiag.exp() + epsilon

        cholesky = torch.diag_embed(diag)
        tril_indices = torch.tril_indices(2, 2, offset=-1, device=cholesky.device)
        cholesky[..., tril_indices[0], tril_indices[1]] = offdiag

        return cholesky

    def compute_electron_energies(
        self,
        energy_portions_logits_sparse: Tensor,
        sparse_input_frame: Tensor,
    ):
        assert sparse_input_frame.is_sparse
        energy_portions = torch.sparse.softmax(energy_portions_logits_sparse, -1)

        portion_indices, portion_values = (
            energy_portions.indices(),
            energy_portions.values(),
        )
        pixel_energies, _ = batch_sparse_index(
            sparse_input_frame.cuda(),
            torch.cat(
                [
                    portion_indices[:-1],
                    portion_indices.new_zeros([1, 1]).expand(
                        -1, portion_indices.shape[1]
                    ),
                ],
                0,
            ).T,
        )
        portioned_energies = portion_values * pixel_energies


def _batched_query_key_map_to_sparse_tensors(
    batched_tensors: list[Tensor],
    key_indices: Tensor,
    nonzero_key_mask: Tensor,
    sparse_tensor_shape,
):
    if isinstance(batched_tensors, Tensor):
        batched_tensors = [batched_tensors]

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
    # batch, y, x, query
    sparse_mask_indices = torch.cat(
        [
            key_indices,
            per_batch_mask_index.reshape(-1, 1, 1, 1).expand(
                -1, *key_indices.shape[1:-1], -1
            ),
        ],
        -1,
    )

    # Finally create the sparse logit tensors
    sparse_indices = sparse_mask_indices.flatten(0, -2).T
    sparse_tensor_shape = [
        *sparse_tensor_shape,
        per_batch_mask_index.max() + 1,
    ]
    out_tensors = [
        torch.sparse_coo_tensor(
            sparse_indices, tensor.flatten(), size=sparse_tensor_shape
        ).coalesce()
        for tensor in batched_tensors
    ]

    return out_tensors, sparse_mask_indices


def _sparse_sigmoid_threshold(sparse_tensor):
    sparse_tensor = sparse_tensor.coalesce()
    return torch.sparse_coo_tensor(
        sparse_tensor.indices(),
        sparse_tensor.values().sigmoid() > 0.5,
        sparse_tensor.shape,
    ).coalesce()
