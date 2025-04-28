import torch
from torch import Tensor


@torch.jit.script
def validate_nd(tensor: Tensor, dims: int, name: str = "tensor") -> None:
    """Validates that a tensor has exactly 'dims' dimensions."""
    if tensor.ndim != dims:
        raise ValueError(f"Expected {name} to be {dims}D, got ndim={tensor.ndim}")


@torch.jit.script
def validate_atleast_nd(tensor: Tensor, min_dims: int, name: str = "tensor") -> None:
    """Validates that a tensor has at least min_dims dimensions."""
    if tensor.ndim < min_dims:
        raise ValueError(
            f"Expected {name} to have at least {min_dims} dimensions, got {tensor.ndim}"
        )


@torch.jit.script
def validate_dim_size(
    tensor: Tensor, dim: int, expected_size: int, name: str = "tensor"
) -> None:
    """Validates that a given dimension has an exact size."""
    if tensor.size(dim) != expected_size:
        raise ValueError(
            "Expected "
            f"{name} to have shape[{dim}]={expected_size}, got shape {tensor.shape}"
        )
