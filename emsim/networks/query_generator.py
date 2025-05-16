from typing import Optional

import spconv.pytorch as spconv
import torch
from torch import Tensor, nn

from ..utils.sparse_utils.conversion import (
    spconv_to_torch_sparse,
)
from ..utils.sparse_utils.batching import (
    batch_offsets_from_sparse_tensor_indices,
)
from ..utils.sparse_utils.indexing.indexing import batch_sparse_index

# arbitrarily high value, will likely OOM if this many queries are actually generated
MAX_QUERIES_PER_BATCH = int(1e7)


class QueryGenerator(nn.Module):
    def __init__(
        self,
        in_pixel_features: int,
        query_dim: int = 32,
        count_cdf_threshold: float = 0.95,
        vae_hidden_dim: int = 64,
        vae_encoder_kernel_size: int = 3,
        max_queries_per_batch: Optional[int] = None,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.vae_encoder = spconv.SparseSequential(
            spconv.SubMConv2d(
                in_pixel_features, vae_hidden_dim, vae_encoder_kernel_size
            ),
            nn.ReLU(),
            spconv.SubMConv2d(vae_hidden_dim, 2 * query_dim + 4, 1),
            nn.LayerNorm(2 * query_dim + 4),
        )
        self.count_cdf_threshold = count_cdf_threshold
        self.max_queries_per_batch = max_queries_per_batch or MAX_QUERIES_PER_BATCH

    @property
    def count_cdf_threshold(self):
        return self._count_cdf_threshold

    @count_cdf_threshold.setter
    def count_cdf_threshold(self, threshold):
        assert 0.0 < threshold <= 1.0
        self._count_cdf_threshold = threshold

    def forward(
        self,
        pixel_features: spconv.SparseConvTensor,
        predicted_occupancy_logits: Tensor,
    ):
        query_indices = get_query_indices(
            predicted_occupancy_logits,
            self.count_cdf_threshold,
            self.max_queries_per_batch,
        ).T
        vae_params = spconv_to_torch_sparse(self.vae_encoder(pixel_features))
        query_pixel_vae_params, is_specified_mask = batch_sparse_index(
            vae_params, query_indices
        )
        assert is_specified_mask.all()

        # Split VAE params
        mu, logvar, log_beta_params = torch.split(
            query_pixel_vae_params,
            [self.query_dim, self.query_dim, 4],
            -1,
        )
        y_alpha, y_beta, x_alpha, x_beta = torch.split(
            log_beta_params.clamp_max(10).exp(), [1, 1, 1, 1], -1
        )

        # Generate queries from VAE
        queries = torch.distributions.Normal(mu, logvar.exp()).rsample()

        # Generate fractional offsets
        y = torch.distributions.Beta(y_alpha, y_beta).rsample()
        x = torch.distributions.Beta(x_alpha, x_beta).rsample()

        query_fractional_offsets = torch.cat([y, x], -1)
        return {
            "queries": queries,
            "indices": query_indices,
            "subpixel_coordinates": query_fractional_offsets,
            "positions": query_indices[..., 1:] + query_fractional_offsets,
            "batch_offsets": batch_offsets_from_sparse_tensor_indices(query_indices.T),
            "occupancy_logits": predicted_occupancy_logits,
        }


def cdf_threshold_min_predicted_electrons(
    predicted_occupancy_logits: Tensor,
    threshold: float,
    max_queries: int,
):
    assert predicted_occupancy_logits.is_sparse
    assert 0.0 < threshold <= 1.0
    with torch.no_grad():
        probs = torch.sparse.softmax(predicted_occupancy_logits, -1)
        occupancy_cdf = torch.cumsum(probs.values(), -1)
        electron_counts = queries_for_thresold(occupancy_cdf, threshold)
        while electron_counts.sum() > max_queries:
            threshold /= 2
            electron_counts = queries_for_thresold(occupancy_cdf, threshold)

        return electron_counts


def queries_for_thresold(cdf, threshold):
    cdf_over_threshold = cdf > threshold
    temp = torch.arange(
        cdf_over_threshold.shape[-1], 0, -1, device=cdf_over_threshold.device
    )
    temp = temp * cdf_over_threshold
    min_electron_count_over_cdf_threshold = torch.argmax(temp, 1)
    return min_electron_count_over_cdf_threshold


def indices_and_counts_to_indices_with_duplicates(indices, counts):
    return torch.repeat_interleave(indices, counts, -1)


def get_query_indices(
    count_logits: Tensor, count_cdf_threshold: float, max_queries_per_batch: int
):
    thresholded_electron_counts = cdf_threshold_min_predicted_electrons(
        count_logits, count_cdf_threshold, max_queries_per_batch
    )
    assert thresholded_electron_counts.numel() == count_logits.indices().shape[-1]
    query_pixels = torch.repeat_interleave(
        count_logits.indices(), thresholded_electron_counts, -1
    )
    return query_pixels


def split_queries_by_batch(query_dict: dict[str, Tensor]):
    batch_offsets = query_dict["batch_offsets"]
    batch_offsets = torch.cat(
        [batch_offsets, batch_offsets.new_tensor([query_dict["queries"].shape[0]])]
    )

    return [
        {
            key: value[batch_start:batch_end]
            for key, value in query_dict.items()
            if key != "batch_offsets"
        }
        for batch_start, batch_end in zip(batch_offsets[:-1], batch_offsets[1:])
    ]


def batch_merge_query_dicts(query_dicts: list[dict[str, Tensor]]):
    out_dict = {
        key: torch.cat([in_dict[key] for in_dict in query_dicts])
        for key in query_dicts[0]
    }
    out_dict["batch_offsets"] = batch_offsets_from_sparse_tensor_indices(
        out_dict["indices"].T
    )
    return out_dict
