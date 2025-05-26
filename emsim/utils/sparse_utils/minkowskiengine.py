from typing import Any, Union

from torch import nn

from . import imports
from .imports import ME, MinkowskiNonlinearityBase


class MinkowskiLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps: float = 0.00001,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        output = self.layer_norm(input.F)
        if isinstance(input, ME.TensorField):
            return ME.TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return ME.SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )


# fmt: off
class _DummyBaseclass:
    pass
if MinkowskiNonlinearityBase is not None:  # successfully imported in utils
    class MinkowskiGELU(MinkowskiNonlinearityBase):  # type: ignore
        MODULE = nn.GELU
else:
    class MinkowskiGELU(_DummyBaseclass):
        pass
# fmt: on


@imports.requires_minkowskiengine
def get_me_layer(layer: Union[str, nn.Module]):
    if isinstance(layer, nn.Module):
        return layer
    if layer.lower() == "relu":
        return ME.MinkowskiReLU
    elif layer.lower() == "gelu":
        return MinkowskiGELU
    elif layer.lower() == "batchnorm1d":
        return ME.MinkowskiBatchNorm
    else:
        raise ValueError(f"Unexpected layer {layer}")
