import torch
from torch import Tensor


def ij_indices_to_normalized_xy(ij_indices: Tensor, spatial_shape: Tensor):
    """Rescales pixel coordinates in the format (i, j) to a normalized
    (x, y) format, with x and y being between 0 and 1. The normalized positions
    will be in fp64 for accuracy.

    Args:
        ij_indices (Tensor): N x 2 tensor, where N is the number of
            points and each row is the point's position in (i, j) format
        spatial_shape (Tensor): 1D tensor holding the (height, width)
    """
    ij_indices = ij_indices.double()
    ij_indices = ij_indices + 0.5
    ij_indices = ij_indices / spatial_shape
    xy_positions = torch.flip(ij_indices, [1])
    assert torch.all(xy_positions > 0.0)
    assert torch.all(xy_positions < 1.0)
    return xy_positions


def normalized_xy_of_stacked_feature_maps(
    stacked_feature_maps: Tensor, spatial_shapes: Tensor
):
    assert isinstance(stacked_feature_maps, Tensor)
    assert stacked_feature_maps.is_sparse
    assert stacked_feature_maps.sparse_dim() == 4
    assert stacked_feature_maps.shape[3] == spatial_shapes.shape[0]  # number of levels
    ij_indices = stacked_feature_maps.indices()[1:3].T
    ij_level = stacked_feature_maps.indices()[3]
    ij_spatial_shapes = spatial_shapes[ij_level]
    xy = ij_indices_to_normalized_xy(ij_indices, ij_spatial_shapes)
    assert xy.dtype == torch.double
    return xy
