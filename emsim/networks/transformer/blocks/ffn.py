from torch import Tensor, nn


class FFNBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation_fn: nn.Module = nn.GELU,
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first

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
