from typing import Union, Optional, Any

from torch import Tensor, nn
import torch
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiNonlinearityBase


def torch_sparse_to_minkowski(tensor: Tensor):
    assert isinstance(tensor, Tensor)
    assert tensor.is_sparse
    features = tensor.values()
    coordinates = tensor.indices()
    if features.ndim == 1:
        features = features.unsqueeze(-1)
        coordinates = coordinates[:-1]
    coordinates = coordinates.transpose(0, 1).contiguous().int()
    return ME.SparseTensor(
        features, coordinates, requires_grad=tensor.requires_grad, device=tensor.device
    )


def minkowski_to_torch_sparse(
    tensor: ME.SparseTensor, full_scale_spatial_shape: Union[Tensor, list[int]]
):
    if isinstance(tensor, Tensor):
        assert tensor.is_sparse
        return tensor
    assert isinstance(tensor, ME.SparseTensor)
    min_coords = torch.zeros([tensor.dimension], dtype=torch.int, device=tensor.device)
    if full_scale_spatial_shape is not None:
        max_coords = torch.tensor(
            full_scale_spatial_shape, dtype=torch.int, device=tensor.device
        )
    else:
        max_coords = None
    return __me_sparse(tensor, min_coords, max_coords)


def __me_sparse(
    tensor: ME.SparseTensor,
    min_coords: Optional[Tensor] = None,
    max_coords: Optional[Tensor] = None,
    contract_coords=True,
):
    r"""Copied from MinkowskiEngine's SparseTensor.sparse() method to fix
    device placement bugs.
    """
    if min_coords is not None:
        assert min_coords.dtype == torch.int
        assert min_coords.numel() == tensor._D
    if max_coords is not None:
        assert max_coords.dtype == torch.int
        assert min_coords.numel() == tensor._D

    def torch_sparse_Tensor(coords, feats, size=None):
        if size is None:
            if feats.dtype == torch.float64 or feats.dtype == torch.float32:
                return torch.sparse_coo_tensor(coords, feats, dtype=feats.dtype)
            else:
                raise ValueError("Feature type not supported.")
        else:
            if feats.dtype == torch.float64 or feats.dtype == torch.float32:
                return torch.sparse_coo_tensor(coords, feats, size, dtype=feats.dtype)
            else:
                raise ValueError("Feature type not supported.")

    # Use int tensor for all operations
    tensor_stride = torch.tensor(
        tensor.tensor_stride, dtype=torch.int, device=tensor.device
    )

    # New coordinates
    coords = tensor.C
    coords, batch_indices = coords[:, 1:], coords[:, 0]

    if min_coords is None:
        min_coords, _ = coords.min(0, keepdim=True)
    elif min_coords.ndim == 1:
        min_coords = min_coords.unsqueeze(0)

    assert (
        min_coords % tensor_stride
    ).sum() == 0, "The minimum coordinates must be divisible by the tensor stride."

    if max_coords is not None:
        if max_coords.ndim == 1:
            max_coords = max_coords.unsqueeze(0)
        assert (
            max_coords % tensor_stride
        ).sum() == 0, "The maximum coordinates must be divisible by the tensor stride."

    coords -= min_coords

    if coords.ndim == 1:
        coords = coords.unsqueeze(1)
    if batch_indices.ndim == 1:
        batch_indices = batch_indices.unsqueeze(1)

    # return the contracted tensor
    if contract_coords:
        coords = coords // tensor_stride
        if max_coords is not None:
            max_coords = max_coords // tensor_stride
        min_coords = min_coords // tensor_stride

    new_coords = torch.cat((batch_indices, coords), dim=1).long()

    size = None
    if max_coords is not None:
        size = max_coords - min_coords
        # Squeeze to make the size one-dimensional
        size = size.squeeze()

        max_batch = tensor._manager.number_of_unique_batch_indices()
        size = torch.Size([max_batch, *size, tensor.F.size(1)])

    sparse_tensor = torch_sparse_Tensor(
        new_coords.t().to(tensor.F.device), tensor.F, size
    )
    tensor_stride = torch.tensor(
        tensor.tensor_stride, dtype=torch.int, device=tensor.device
    )
    return sparse_tensor, min_coords, tensor_stride




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


class MinkowskiGELU(MinkowskiNonlinearityBase):
    MODULE = nn.GELU
