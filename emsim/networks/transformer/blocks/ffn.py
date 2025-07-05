from typing import Union

from torch import Tensor, nn

from emsim.utils.misc_utils import get_layer


class FFNBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation_fn: Union[str, type[nn.Module]] = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.norm_first = norm_first
        if isinstance(activation_fn, str):
            activation_fn = get_layer(activation_fn)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor):
        if self.norm_first:
            x = x + self.mlp(self.norm(x))
        else:
            x = self.norm(x + self.mlp(x))
        return x

    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
