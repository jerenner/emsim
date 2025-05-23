from torch import Tensor
import torch

from emsim.utils.sparse_utils.indexing.indexing import batch_sparse_index


@torch.jit.script
def sparse_split_heads(sparse_tensor: Tensor, n_heads: int) -> Tensor:
    """
    Splits a sparse tensor into multiple heads.

    Args:
        sparse_tensor (Tensor): The input sparse tensor.
        n_heads (int): The number of heads to split into.

    Returns:
        Tensor: The split sparse tensor with shape (*sparse_tensor.shape[:-1], n_heads, head_dim).
    """
    assert isinstance(sparse_tensor, Tensor)
    assert sparse_tensor.is_sparse
    assert sparse_tensor.ndim >= 4
    n_specified_elements = sparse_tensor.indices().shape[1]
    embed_dim = sparse_tensor.shape[-1]
    assert embed_dim % n_heads == 0
    head_dim = embed_dim // n_heads
    repeated_indices = torch.repeat_interleave(sparse_tensor.indices(), n_heads, 1)
    new_indices = torch.cat(
        [
            repeated_indices,
            torch.arange(n_heads, device=sparse_tensor.device)
            .repeat(n_specified_elements)
            .unsqueeze(0),
        ]
    )
    new_values = sparse_tensor.values().view(n_specified_elements * n_heads, head_dim)

    new_shape = sparse_tensor.shape[:-1]
    new_shape = new_shape + (n_heads, head_dim)
    new_sparse_tensor = torch.sparse_coo_tensor(
        new_indices, new_values, new_shape, is_coalesced=sparse_tensor.is_coalesced()
    ).coalesce()
    return new_sparse_tensor


@torch.jit.script
def multilevel_sparse_bilinear_grid_sample(
    sparse_tensor: Tensor,
    xy_coordinates: Tensor,
    xy_batch_index: Tensor,
    xy_level_index: Tensor,
    level_spatial_shapes: Tensor,
    weight_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Bilinearly samples into a 2D sparse tensor. Similar to F.grid_sample
    except the sampled tensor is expected to be sparse and the interpolation
    points are not in a grid. Assumes align_corners=False.

    Args:
        sparse_tensor (Tensor): torch.sparse.sparse_coo_tensor with shape
        (batch, height, width, level, head, head_dim)
        xy_coordinates (Tensor): Sampling point coordinates, with shape
        (n_queries, n_levels, n_points, n_heads, 2),
        with the last dimension in order (x, y), with all points within
        range (-1, 1)
        xy_batch_index (Tensor): Tensor of shape
        (n_queries, n_levels, n_points, n_heads) with the index of the batch
        element for each point in xy_coordinates
        xy_level_index (Tensor): Tensor of shape
        (n_queries, n_levels, n_points, n_heads) with the index of the level
        for each point in xy_coordinates
        level_spatial_shapes (Tensor): n_levels x 2 tensor with the
        (height, width) of each level's feature map
    """
    assert sparse_tensor.is_sparse
    assert xy_coordinates.dtype == torch.double
    bsz, _, _, n_levels, n_heads, head_dim = sparse_tensor.shape
    n_queries, _, n_points, _, _ = xy_coordinates.shape
    assert n_levels == xy_coordinates.shape[1]
    assert n_heads == xy_coordinates.shape[3]
    assert xy_coordinates.shape[0] == xy_batch_index.shape[0]
    assert xy_coordinates.shape[0] == xy_level_index.shape[0]
    assert xy_batch_index.min() >= 0
    assert xy_batch_index.max() < bsz
    assert xy_level_index.min() >= 0
    assert xy_level_index.max() < xy_level_index.shape[1]

    x = xy_coordinates[..., 0]
    y = xy_coordinates[..., 1]

    head_indices = (
        torch.arange(n_heads, device=xy_batch_index.device)
        .view(1, 1, 1, n_heads)
        .expand_as(xy_batch_index)
    )

    spatial_shapes_expanded = level_spatial_shapes.view(1, n_levels, 1, 1, 2).expand(
        n_queries, -1, n_points, n_heads, -1
    )
    height = spatial_shapes_expanded[..., 0].contiguous()
    width = spatial_shapes_expanded[..., 1].contiguous()

    x = (((x + 1) * width - 1) / 2)
    y = (((y + 1) * height - 1) / 2)

    # x = x.view(-1)
    # y = y.view(-1)

    x0 = x.floor().int()
    y0 = y.floor().int()
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = x0.clamp_min(0).clamp_max(width.view_as(x) - 1)
    x1 = x1.clamp_min(0).clamp_max(width.view_as(x) - 1)
    y0 = y0.clamp_min(0).clamp_max(height.view_as(y) - 1)
    y1 = y1.clamp_min(0).clamp_max(height.view_as(y) - 1)

    wa = ((x1 - x) * (y1 - y)).unsqueeze(-1).to(weight_dtype)
    wb = ((x1 - x) * (y - y0)).unsqueeze(-1).to(weight_dtype)
    wc = ((x - x0) * (y1 - y)).unsqueeze(-1).to(weight_dtype)
    wd = ((x - x0) * (y - y0)).unsqueeze(-1).to(weight_dtype)

    x0_y0 = torch.stack([xy_batch_index, x0, y0, xy_level_index, head_indices], -1)
    x0_y1 = torch.stack([xy_batch_index, x0, y1, xy_level_index, head_indices], -1)
    x1_y0 = torch.stack([xy_batch_index, x1, y0, xy_level_index, head_indices], -1)
    x1_y1 = torch.stack([xy_batch_index, x1, y1, xy_level_index, head_indices], -1)

    xy = torch.stack([x0_y0, x0_y1, x1_y0, x1_y1], -2)
    w = torch.stack([wa, wb, wc, wd], -1)

    val = batch_sparse_index(sparse_tensor, xy)[0]

    out = torch.matmul(w, val.to(w)).squeeze(-2)

    return out
