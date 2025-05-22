from typing import Union

import torch
from torch import nn, Tensor

from emsim.utils.misc_utils import get_layer


class PositionOffsetHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        predict_box: bool = False,
        activation_fn: Union[str, type[nn.Module]] = "gelu",
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        assert n_layers > 1, "Expected at least 1 hidden layer"
        self.predict_box = predict_box
        self.dtype = dtype
        if isinstance(activation_fn, str):
            activation_fn = get_layer(activation_fn)
        layers = [nn.Linear(in_dim, hidden_dim, dtype=dtype), activation_fn()]
        for _ in range(n_layers - 1):
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim, dtype=dtype), activation_fn()]
            )
        out_dim = 6 if predict_box else 2
        layers.append(nn.Linear(hidden_dim, out_dim, dtype=dtype))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    @staticmethod
    def split_point_and_box(x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.shape[-1] == 6
        out = torch.tensor_split(x, 2, dim=-1)
        assert len(out) == 2
        return out

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
