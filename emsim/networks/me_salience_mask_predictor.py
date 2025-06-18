from typing import Optional

import MinkowskiEngine as ME
import torch
from torch import Tensor, nn

from emsim.utils.sparse_utils.minkowskiengine import MinkowskiGELU, MinkowskiLayerNorm


assert issubclass(MinkowskiGELU, nn.Module), "MinkowskiEngine not imported successfully"


@torch.compiler.disable
class MESparseMaskPredictor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dimension: int = 2,
        meanpool_half_features: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.meanpool_half_features = meanpool_half_features
        self.layer1 = nn.Sequential(
            MinkowskiLayerNorm(in_dim),
            ME.MinkowskiConvolution(in_dim, hidden_dim, 1, dimension=dimension),
            MinkowskiGELU(),
        )
        self.layer2 = nn.Sequential(
            ME.MinkowskiConvolution(
                hidden_dim, hidden_dim // 2, 1, dimension=dimension
            ),
            MinkowskiGELU(),
            ME.MinkowskiConvolution(
                hidden_dim // 2, hidden_dim // 4, 1, dimension=dimension
            ),
            MinkowskiGELU(),
            ME.MinkowskiConvolution(hidden_dim // 4, 1, 1, dimension=dimension),
        )

    def forward(self, x: ME.SparseTensor):
        assert isinstance(x, ME.SparseTensor)
        z = self.layer1(x)
        if self.meanpool_half_features:
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
    def batchwise_global_mean_second_half_features(z: ME.SparseTensor):
        batch_indices = z.decomposition_permutations
        batch_features = z.decomposed_features
        subpooled_batch_features = [
            _global_mean_second_half_features(z_i, z_i.shape[-1] // 2)
            for z_i in batch_features
        ]
        new_features = z.F.clone()
        for ind, feat in zip(batch_indices, subpooled_batch_features):
            new_features[ind] = feat
        z = ME.SparseTensor(
            new_features,
            coordinate_map_key=z.coordinate_map_key,
            coordinate_manager=z.coordinate_manager,
        )
        return z


def _global_mean_second_half_features(z: Tensor, half_dim: Optional[int] = None):
    if half_dim is None:
        half_dim = z.shape[-1] // 2
    z_local, z_global = torch.split(z, half_dim, -1)
    z_global = z_global.mean(dim=0, keepdim=True).expand(z_local.shape[0], -1)
    return torch.cat([z_local, z_global], dim=-1)
