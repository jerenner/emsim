from typing import Union
from typing_extensions import Self

import torch
from torch import nn

from emsim.utils.misc_utils import get_layer
from emsim.config.transformer import StdDevHeadConfig

class StdDevHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        activation_fn: Union[str, type[nn.Module]] = "gelu",
        scaling_factor: float = 0.001,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert n_layers > 1, "Expected at least 1 hidden layer"
        if isinstance(activation_fn, str):
            activation_fn = get_layer(activation_fn)
        layers = [nn.Linear(in_dim, hidden_dim), activation_fn()]
        for _ in range(n_layers - 1):
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), activation_fn()]
            )
        layers.append(nn.Linear(hidden_dim, 3))
        self.layers = nn.Sequential(*layers)

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

    @classmethod
    def from_config(cls, config: StdDevHeadConfig) -> Self:
        return cls(
            in_dim=config.in_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            activation_fn=config.activation_fn,
            scaling_factor=config.scaling_factor,
            eps=config.eps
        )
