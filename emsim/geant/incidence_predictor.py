import torch
from torch import nn
from torch.distributions import MultivariateNormal


class GaussianIncidencePointPredictor(nn.Module):
    def __init__(self, backbone, hidden_dim=512, mean_parameterization="sigmoid"):
        super().__init__()
        self.mean_parameterization = mean_parameterization
        self.backbone = backbone

        self.predictor = nn.Sequential(
            nn.Linear(self.backbone.num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        patch_shape = torch.as_tensor(x.shape[-2:], dtype=torch.float, device=x.device)
        patch_center_coords = patch_shape / 2
        x = self.backbone(x)
        x = self.predictor(x)

        mean_vector, cholesky_diagonal, cholesky_offdiag = torch.split(x, [2, 2, 1], -1)

        cholesky_diagonal = cholesky_diagonal.exp()
        cholesky = torch.diag_embed(cholesky_diagonal)
        tril_indices = torch.tril_indices(2, 2, offset=-1, device=x.device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_offdiag

        return MultivariateNormal(mean_vector, scale_tril=cholesky)
