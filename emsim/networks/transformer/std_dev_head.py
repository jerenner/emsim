import torch
from torch import nn


class StdDevHead(nn.Module):
    def __init__(self, d_model: int, scaling_factor: float = .001, eps: float = 1e-6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3)
        )
        self.register_buffer("scaling_factor", torch.as_tensor(scaling_factor))
        self.eps = eps

    def forward(self, x):
        x = self.layers(x)
        chol_diag, chol_offdiag = torch.split(x, [2, 1], -1)
        chol_diag = chol_diag.exp()
        chol_diag = torch.clamp_min(chol_diag, self.eps)

        cholesky = torch.diag_embed(chol_diag)
        cholesky[..., 1, 0] = chol_offdiag.squeeze(-1)
        cholesky = cholesky * self.scaling_factor
        return cholesky

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
