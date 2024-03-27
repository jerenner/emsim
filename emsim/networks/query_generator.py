import spconv.pytorch as spconv

import torch
from torch import nn, Tensor

from ..utils.sparse_utils import gather_from_sparse_tensor, spconv_to_torch_sparse


class QueryGenerator(nn.Module):
    def __init__(
        self,
        in_pixel_features: int,
        query_dim: int = 32,
        count_cdf_threshold: float = 0.95,
        vae_hidden_dim: int = 64,
        vae_encoder_kernel_size: int = 3,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.vae_encoder = spconv.SparseSequential(
            spconv.SubMConv2d(
                in_pixel_features, vae_hidden_dim, vae_encoder_kernel_size
            ),
            nn.ReLU(),
            spconv.SubMConv2d(vae_hidden_dim, 2 * query_dim + 4, 1),
        )
        self.count_cdf_threshold = count_cdf_threshold

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
            predicted_occupancy_logits, self.count_cdf_threshold
        ).T
        vae_params = spconv_to_torch_sparse(self.vae_encoder(pixel_features))
        query_pixel_vae_params = gather_from_sparse_tensor(vae_params, query_indices)

        # Split VAE params
        mu, logvar, x_alpha, x_beta, y_alpha, y_beta = torch.split(
            query_pixel_vae_params.exp(),
            [self.query_dim, self.query_dim, 1, 1, 1, 1],
            -1,
        )

        # Generate queries from VAE
        queries = torch.distributions.Normal(mu, logvar.exp()).rsample()

        # Generate fractional offsets
        x = torch.distributions.Beta(x_alpha, x_beta).rsample()
        y = torch.distributions.Beta(y_alpha, y_beta).rsample()

        query_fractional_offsets = torch.cat([x, y], -1)
        return {
            "queries": queries,
            "indices": query_indices,
            "subpixel_coordinates": query_fractional_offsets,
        }


def cdf_threshold_min_predicted_electrons(
    predicted_occupancy_logits: Tensor, threshold: float = 0.95
):
    assert predicted_occupancy_logits.is_sparse
    assert 0.0 < threshold <= 1.0
    probs = torch.sparse.softmax(predicted_occupancy_logits, -1)
    occupancy_cdf = torch.cumsum(probs.values(), -1)
    cdf_over_thresholded = occupancy_cdf > threshold
    temp = torch.arange(
        cdf_over_thresholded.shape[-1], 0, -1, device=cdf_over_thresholded.device
    )
    temp = temp * cdf_over_thresholded
    min_electron_count_over_cdf_threshold = torch.argmax(temp, 1)
    return min_electron_count_over_cdf_threshold


def indices_and_counts_to_indices_with_duplicates(indices, counts):
    return torch.repeat_interleave(indices, counts, -1)


def get_query_indices(count_logits: Tensor, count_cdf_threshold: float = 0.95):
    thresholded_electron_counts = cdf_threshold_min_predicted_electrons(
        count_logits, count_cdf_threshold
    )
    assert thresholded_electron_counts.numel() == count_logits.indices().shape[-1]
    query_pixels = torch.repeat_interleave(
        count_logits.indices(), thresholded_electron_counts, -1
    )
    return query_pixels
