import spconv.pytorch as spconv

import torch
from torch import nn, Tensor

from ..utils.sparse_utils import gather_from_sparse_tensor, spconv_to_torch_sparse

class QueryGenerator(nn.Module):
    def __init__(self, in_pixel_features: int, count_cdf_threshold: float = 0.95, beta_distn_hidden_dim=32, beta_distn_kernel_size=3):
        super().__init__()
        self.beta_distn_generator = spconv.SparseSequential(
            spconv.SubMConv2d(in_pixel_features, beta_distn_hidden_dim, beta_distn_kernel_size),
            nn.ReLU(),
            spconv.SubMConv2d(beta_distn_hidden_dim, 4, 1)
        )
        self.count_cdf_threshold = count_cdf_threshold

    def forward(self, pixel_features: spconv.SparseConvTensor, predicted_occupancy_logits: Tensor):
        query_indices = get_query_indices(predicted_occupancy_logits, self.count_cdf_threshold).T
        log_beta_params = spconv_to_torch_sparse(self.beta_distn_generator(pixel_features))
        query_pixel_log_beta_params = gather_from_sparse_tensor(log_beta_params, query_indices)

        x_alpha, x_beta, y_alpha, y_beta = torch.split(query_pixel_log_beta_params.exp(), [1, 1, 1, 1], -1)
        x = torch.distributions.Beta(x_alpha, x_beta).rsample()
        y = torch.distributions.Beta(y_alpha, y_beta).rsample()

        query_fractional_offsets = torch.cat([x, y], -1)
        return query_indices, query_fractional_offsets


def cdf_threshold_min_predicted_electrons(predicted_occupancy_logits: Tensor, threshold: float = 0.99):
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
    thresholded_electron_counts = cdf_threshold_min_predicted_electrons(count_logits, count_cdf_threshold)
    assert thresholded_electron_counts.numel() == count_logits.indices().shape[-1]
    query_pixels = torch.repeat_interleave(count_logits.indices(), thresholded_electron_counts, -1)
    return query_pixels
