import math

import spconv.pytorch as spconv
import torch
from torch import Tensor, nn
from torch.nn.init import constant_, xavier_uniform_

from ..utils.sparse_utils import gather_from_sparse_tensor
from ..utils.misc_utils import inverse_sigmoid


# @torch.jit.script
def multi_scale_deformable_attention(
    stacked_value_tensors: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    query_offsets: Tensor,
    attention_weights: Tensor,
):
    assert isinstance(stacked_value_tensors, Tensor)
    assert stacked_value_tensors.is_sparse
    assert stacked_value_tensors.ndim == 5  # (batch, height, width, level, feature)

    n_queries, n_levels, n_points, n_heads, _ = sampling_locations.shape
    embed_dim = stacked_value_tensors.shape[-1]

    stacked_value_tensors = sparse_split_heads(stacked_value_tensors, n_heads)
    # now (batch, height, width, level, heads, head_dim)

    sampling_locations = sampling_locations * 2 - 1  # rescale to (-1, 1)

    batch_sizes = torch.diff(torch.cat([query_offsets, torch.tensor([n_queries])]))
    xy_batch_indices = torch.cat(
        [
            torch.full([size], i, device=sampling_locations.device, dtype=torch.long)
            for i, size in enumerate(batch_sizes)
        ]
    )
    xy_batch_indices = xy_batch_indices.view(-1, 1, 1, 1).expand(
        n_queries, n_levels, n_points, n_heads
    )
    xy_level_indices = (
        torch.arange(n_levels, device=xy_batch_indices.device)
        .view(1, -1, 1, 1)
        .expand(n_queries, n_levels, n_points, n_heads)
    )

    sampled_values = multilevel_sparse_bilinear_grid_sample(
        stacked_value_tensors,
        sampling_locations,
        xy_batch_indices,
        xy_level_indices,
        value_spatial_shapes,
    )
    sampled_values = sampled_values.to(attention_weights)

    assert sampled_values.shape == (
        n_queries,
        n_levels,
        n_points,
        n_heads,
        embed_dim // n_heads,
    )
    assert attention_weights.shape == (n_queries, n_levels, n_points, n_heads)
    output = torch.einsum("qlphd,qlph->qhd", sampled_values, attention_weights)
    output = output.view(n_queries, embed_dim)
    return output


def sparse_split_heads(sparse_tensor: Tensor, n_heads: int):
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
    repeated_indices = torch.repeat_interleave(sparse_tensor.indices(), n_heads, 1)
    new_indices = torch.cat(
        [
            repeated_indices,
            torch.arange(n_heads, device=sparse_tensor.device)
            .repeat(n_specified_elements)
            .unsqueeze(0),
        ]
    )
    new_values = sparse_tensor.values().view(
        n_specified_elements * n_heads, embed_dim // n_heads
    )

    assert embed_dim % n_heads == 0
    head_dim = embed_dim // n_heads
    new_sparse_tensor = torch.sparse_coo_tensor(
        new_indices,
        new_values,
        (*sparse_tensor.shape[:-1], n_heads, head_dim),
    ).coalesce()
    return new_sparse_tensor


## Based on https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
## and https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/multi_scale_deform_attn.py
class SparseMSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_levels: int = 4,
        num_heads: int = 8,
        num_points: int = 4,
        double_presigmoid_sampling_offsets=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.double_presigmoid_sampling_offsets = double_presigmoid_sampling_offsets
        self.sampling_offsets = nn.Linear(
            embed_dim,
            num_levels * num_points * num_heads * 2,
            dtype=(
                torch.double
                if double_presigmoid_sampling_offsets
                else torch.get_default_dtype()
            ),
        )
        self.attention_weights = nn.Linear(
            embed_dim, num_points * num_levels * num_heads
        )
        self.value_proj = spconv.SubMConv2d(embed_dim, embed_dim, 1)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(
            self.num_heads, dtype=self.sampling_offsets.bias.dtype
        ) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        batch_offsets: Tensor,
        xy_reference_points: Tensor,
        stacked_value_tensors: Tensor,
        spatial_shapes: Tensor,
    ):
        """Forward function for SparseMSDeformableAttention

        Args:
            query (Tensor): Batch-flattened query tensor of shape [n_query x embed_dim]
            query_offsets (Tensor): batchsize-long tensor with the ith element
                being the start index in the query tensor for batch element i
            reference_points (Tensor): normalized reference points for queries
                with shape [n_query x 2], where the last two
                dimensions are the (x, y) coordinates in the range (0, 1).
                Should be of dtype double (float64)
            value_tensors (Tensor): Sparse tensor of all feature maps with the
                shape [batch x height x width x level x channels]
            spatial_shapes (Tensor): Tensor of shape [n_levels x 2] with the
                (height, width) of each level's feature map
        """
        assert xy_reference_points.dtype == torch.double
        n_total_queries = query.shape[0]

        sampling_offsets = self.sampling_offsets(
            query.to(self.sampling_offsets.weight)
        ).view(n_total_queries, self.num_levels, self.num_points, self.num_heads, 2)
        attention_weights = self.attention_weights(query).view(
            n_total_queries, self.num_levels * self.num_points, self.num_heads
        )
        attention_weights = attention_weights.softmax(-2)
        attention_weights = attention_weights.view(
            n_total_queries, self.num_levels, self.num_points, self.num_heads
        )

        # spatial_scaler = spatial_shapes.flip([1])  # flip i,j to x,y
        # sampling_locations = xy_reference_points.reshape(
        #     n_total_queries, 1, 1, 1, 2
        # ) + sampling_offsets / spatial_scaler.reshape(1, 1, self.num_levels, 1, 2)

        # n_queries, num_levels, num_points, num_heads, 2
        sampling_locations = torch.sigmoid(
            inverse_sigmoid(xy_reference_points.view(n_total_queries, 1, 1, 1, 2))
            + sampling_offsets
        )

        output = multi_scale_deformable_attention(
            stacked_value_tensors,
            spatial_shapes,
            sampling_locations,
            batch_offsets,
            attention_weights,
        )

        output = self.output_proj(output)
        return output


def sparse_interp(sparse_tensor: Tensor, pixel_coordinates: Tensor):
    """
    Interpolates into a 2d sparse tensor. Similar to F.grid_sample except the
    interpolant coordinates are in un-normalized i,j form
    """
    # translate pixel coordinates so we can floor to find the up-left pixel center
    height, width = sparse_tensor.shape[-3], sparse_tensor.shape[-2]
    shifted_pixel_coordinates = pixel_coordinates - 0.5
    shifted_pixel_coordinates = shifted_pixel_coordinates.clamp_min(0.0)

    pixel_indices_00 = shifted_pixel_coordinates.floor().int()
    pixel_indices_01 = pixel_indices_00 + pixel_indices_00.new_tensor([0, 1])
    pixel_indices_10 = pixel_indices_00 + pixel_indices_00.new_tensor([1, 0])
    pixel_indices_11 = pixel_indices_00 + pixel_indices_00.new_tensor([1, 1])

    max_index = torch.tensor([height - 1, width - 1], device=pixel_indices_00.device)
    pixel_indices_01.clamp_max_(max_index)
    pixel_indices_10.clamp_max_(max_index)
    pixel_indices_11.clamp_max_(max_index)

    values_00 = gather_from_sparse_tensor(sparse_tensor, pixel_indices_00)[0]
    values_01 = gather_from_sparse_tensor(sparse_tensor, pixel_indices_01)[0]
    values_10 = gather_from_sparse_tensor(sparse_tensor, pixel_indices_10)[0]
    values_11 = gather_from_sparse_tensor(sparse_tensor, pixel_indices_11)[0]

    # distance to the n+1,n+1 pixel center
    down_weight, right_weight = shifted_pixel_coordinates.frac().split(1, -1)
    up_weight, left_weight = 1 - down_weight, 1 - right_weight

    weight_00 = up_weight * left_weight
    weight_01 = up_weight * right_weight
    weight_10 = down_weight * left_weight
    weight_11 = down_weight * right_weight

    if values_00.ndim < weight_00.ndim:
        assert values_00.ndim == weight_00.ndim - 1
        weight_00.squeeze_()
        weight_01.squeeze_()
        weight_10.squeeze_()
        weight_11.squeeze_()

    return (
        weight_00 * values_00
        + weight_01 * values_01
        + weight_10 * values_10
        + weight_11 * values_11
    )


def multilevel_sparse_bilinear_grid_sample(
    sparse_tensor: Tensor,
    xy_coordinates: Tensor,
    xy_batch_index: Tensor,
    xy_level_index: Tensor,
    level_spatial_shapes: Tensor,
):
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
    assert xy_level_index.max() < xy_level_index.shape[0]

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

    x = ((x + 1) * width - 1) / 2
    y = ((y + 1) * height - 1) / 2

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

    wa = ((x1 - x) * (y1 - y)).unsqueeze(-1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(-1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(-1)
    wd = ((x - x0) * (y - y0)).unsqueeze(-1)

    x0_y0 = torch.stack([xy_batch_index, x0, y0, xy_level_index, head_indices], -1)
    x0_y1 = torch.stack([xy_batch_index, x0, y1, xy_level_index, head_indices], -1)
    x1_y0 = torch.stack([xy_batch_index, x1, y0, xy_level_index, head_indices], -1)
    x1_y1 = torch.stack([xy_batch_index, x1, y1, xy_level_index, head_indices], -1)

    xy = torch.stack([x0_y0, x0_y1, x1_y0, x1_y1], -2)
    w = torch.stack([wa, wb, wc, wd], -1)

    val = gather_from_sparse_tensor(sparse_tensor, xy)[0]

    out = torch.matmul(w, val.to(w)).squeeze(-2)

    return out
