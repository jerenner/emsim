import torch
from torch import nn, Tensor


class FourierEncoding(nn.Module):
    def __init__(self, in_features, out_features, dtype=torch.float):
        super().__init__()
        self.in_dim = in_features
        assert out_features % 2 == 0
        self.out_dim = out_features

        self.weight = nn.Linear(in_features, out_features // 2, bias=False, dtype=dtype)
        self._scaling = 1000

    def reset_parameters(self):
        self.weight.reset_parameters()

    def forward(self, positions: Tensor) -> Tensor:
        assert positions.min() >= 0 and positions.max() <= 1
        positions = positions * 2 * torch.pi
        proj = self.weight(positions)
        out = torch.cat([proj.sin(), proj.cos()], -1)
        out = out * self._scaling
        return out


class PixelPositionalEncoding(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.encoding = FourierEncoding(2, out_features)

    def forward(self, pixel_indices: Tensor, image_size: Tensor):
        positions = pixel_indices / image_size
        out = self.encoding(positions)
        return out


class SubpixelPositionalEncoding(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.encoding = FourierEncoding(2, out_features)

    def forward(self, subpixel_positions: Tensor):
        out = self.encoding(subpixel_positions)
        return out
