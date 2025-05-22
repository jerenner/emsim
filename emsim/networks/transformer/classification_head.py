from typing import Union

import torch
from torch import nn, Tensor

from emsim.utils.misc_utils import get_layer


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        activation_fn: Union[str, type[nn.Module]] = "gelu",
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        assert n_layers > 1, "Expected at least 1 hidden layer"
        self.dtype = dtype
        if isinstance(activation_fn, str):
            activation_fn = get_layer(activation_fn)
        layers = [nn.Linear(in_dim, hidden_dim, dtype=dtype), activation_fn()]
        for _ in range(n_layers - 1):
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim, dtype=dtype), activation_fn()]
            )
        layers.append(nn.Linear(hidden_dim, 1, dtype=dtype))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
