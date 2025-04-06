import spconv.pytorch as spconv
import torch
from torch import Tensor, nn

from emsim.utils.sparse_utils.batching.batching import (
    batch_offsets_from_sparse_tensor_indices,
    split_batch_concatted_tensor,
)


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.hidden_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=1)
        out = self.layer2(z)
        return out


def global_mean_second_half_features(z, half_dim) -> Tensor:
    z_local, z_global = torch.split(z, half_dim, -1)
    z_global = z_global.mean(dim=0, keepdim=True).expand(z_local.shape[0], -1)
    return torch.cat([z_local, z_global], dim=-1)


class SparseMaskPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: Tensor, batch_offsets: Tensor):
        z = self.layer1(x)
        z_unbatched = split_batch_concatted_tensor(z, batch_offsets)
        z_unbatched = [
            global_mean_second_half_features(z_b, self.hidden_dim // 2)
            for z_b in z_unbatched
        ]
        z = torch.cat(z_unbatched, 0)
        out = self.layer2(z)
        return out

    def forward_sparse_tensor(self, x: Tensor) -> Tensor:
        assert x.is_sparse
        assert x.is_coalesced()

        batch_offsets = batch_offsets_from_sparse_tensor_indices(x.indices())
        mask_values = self(x.values(), batch_offsets)
        out = torch.sparse_coo_tensor(
            x.indices(), mask_values, x.shape[:-1] + (1,), requires_grad=x.requires_grad
        ).coalesce()
        return out

    def forward_sparse_tensor_list(self, x: list[Tensor]) -> list[Tensor]:
        assert not isinstance(x, Tensor)
        assert all([isinstance(x_, Tensor) for x_ in x])
        return [self.forward_sparse_tensor(x_) for x_ in x]


class SpconvSparseMaskPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer1 = spconv.SparseSequential(
            nn.LayerNorm(in_dim), spconv.SubMConv2d(in_dim, hidden_dim, 1), nn.GELU()
        )
        self.layer2 = spconv.SparseSequential(
            spconv.SubMConv2d(hidden_dim, hidden_dim // 2, 1),
            nn.GELU(),
            spconv.SubMConv2d(hidden_dim // 2, hidden_dim // 4, 1),
            nn.GELU(),
            spconv.SubMConv2d(hidden_dim // 4, 1, 1),
        )

    def forward(self, x: spconv.SparseConvTensor):
        assert isinstance(x, spconv.SparseConvTensor)
        z = self.layer1(x)
        z = self.batchwise_global_mean_second_half_features(z)
        out = self.layer2(z)
        return out

    def reset_parameters(self):
        for layer in self.layer1:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.layer2:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    @staticmethod
    def batchwise_global_mean_second_half_features(z: spconv.SparseConvTensor):
        indices_per_batch = [z.indices[:, 0] == i for i in z.indices[:, 0].unique()]
        features_per_batch = [z.features[indices] for indices in indices_per_batch]
        features_per_batch = [
            global_mean_second_half_features(z_i, z_i.shape[-1] // 2)
            for z_i in features_per_batch
        ]
        new_features = z.features.clone()
        for ind, feat in zip(indices_per_batch, features_per_batch):
            new_features[ind] = feat
        z = z.replace_feature(new_features)
        return z
