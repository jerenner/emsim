import math

import spconv.pytorch as spconv
import torch
from torch import Tensor, nn
from torch.nn.init import constant_, xavier_uniform_

from ..utils.batching_utils import split_batch_concatted_tensor
from ..utils.sparse_utils import gather_from_sparse_tensor, spconv_to_torch_sparse


# @torch.jit.script
def multi_scale_deformable_attention(
    sparse_values: list[Tensor],
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    query_offsets: Tensor,
    attention_weights: Tensor,
):
    assert all(t.is_sparse for t in sparse_values)
    sparse_values = [
        sp.unbind(0) for sp in sparse_values
    ]  # split feature maps on batch dimension

    n_queries, n_heads, n_levels, n_points, _ = sampling_locations.shape
    embed_dim = sparse_values[0][0].shape[-1]
    sampling_locations = split_batch_concatted_tensor(sampling_locations, query_offsets)

    sampled_values = []
    for batch_index, locations in enumerate(sampling_locations):
        element_sampled_values = []
        for level, level_feature_maps in enumerate(sparse_values):
            locations_level = locations[:, :, level]  # query x heads x points x 2
            value_level = level_feature_maps[batch_index].coalesce()

            # split heads in value tensor
            n_specified_elements = value_level.indices().shape[-1]
            repeated_indices = torch.repeat_interleave(
                value_level.indices(), n_heads, 1
            )
            new_indices = torch.cat(
                [
                    repeated_indices,
                    torch.arange(n_heads, device=value_level.device)
                    .repeat(n_specified_elements)
                    .unsqueeze(0),
                ]
            )
            new_values = (
                value_level.values()
                .reshape(n_specified_elements, n_heads, embed_dim // n_heads)
                .transpose(1, 2)
                .reshape(n_specified_elements * n_heads, embed_dim // n_heads)
            )
            value_level = torch.sparse_coo_tensor(
                new_indices,
                new_values,
                (
                    value_level.shape[0],
                    value_level.shape[1],
                    n_heads,
                    embed_dim // n_heads,
                ),
            )

            batch_n_queries = locations.shape[0]
            head_features = []
            for h, (locations_level_head, value_level_head) in enumerate(
                zip(locations_level.unbind(1), value_level.unbind(-2))
            ):
                level_head_sampled_values = sparse_interp(
                    value_level_head.coalesce(),
                    locations_level_head.reshape(batch_n_queries * n_points, 2),
                )  # n_queries*points, head_dim
                level_head_sampled_values = level_head_sampled_values.reshape(
                    batch_n_queries, n_points, embed_dim // n_heads
                )
                head_features.append(level_head_sampled_values)
            level_sampled_values = torch.stack(
                head_features, 1
            )  # n_queries, n_heads, n_points, head_dim
            # locations_level_with_head_index = torch.cat([
            #     locations_level,
            #     torch.arange(n_heads, device=locations_level).view(1, n_heads, 1, 1).expand(batch_n_queries,-1, n_points, -1)
            # ], -1)
            # level_sampled_values = sparse_interp(
            #     value_level, locations_level_with_head_index.reshape(batch_n_queries*n_heads*n_points, 3)
            # )
            # level_sampled_values = level_sampled_values.reshape(
            #     batch_n_queries, n_heads, n_points, embed_dim
            # )
            element_sampled_values.append(level_sampled_values)
        sampled_values.append(torch.stack(element_sampled_values, 2))
    sampled_values = torch.cat(sampled_values, 0)  # re-stack the batch

    assert sampled_values.shape == (
        n_queries,
        n_heads,
        n_levels,
        n_points,
        embed_dim // n_heads,
    )
    assert attention_weights.shape == (n_queries, n_heads, n_levels, n_points)
    output = sampled_values * attention_weights.unsqueeze(-1)
    output = output.sum([2, 3]).view(n_queries, embed_dim)
    return output


## Based on https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
## and https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/multi_scale_deform_attn.py
class SparseMSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_levels: int = 4,
        num_heads: int = 8,
        num_points: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )
        self.value_proj = spconv.SubMConv2d(embed_dim, embed_dim, 1)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
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
        query_offsets: Tensor,
        reference_points: Tensor,
        value_tensors: list[spconv.SparseConvTensor],
    ):
        """Forward function for SparseMSDeformableAttention

        Args:
            query (Tensor): Batch-flattened query tensor of shape [n_query x embed_dim]
            query_offsets (Tensor): batchsize-long tensor with the ith element
                being the start index in the query tensor for batch element i
            reference_points (Tensor): un-normalized reference points for queries
                with shape [n_query x 2], where the last two
                dimensions are the i, j pixel coordinates in the scale of the
                input image (i.e., the highest-resolution feature map)
            value_tensors (list[spconv.SparseConvTensor]): List of feature map
                sparse tensors from the backbone
        """
        n_total_queries = query.shape[0]
        value = [spconv_to_torch_sparse(self.value_proj(v)) for v in value_tensors]

        sampling_offsets = self.sampling_offsets(query).view(
            n_total_queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            n_total_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            n_total_queries, self.num_heads, self.num_levels, self.num_points
        )

        spatial_shapes = torch.stack(
            [
                torch.tensor(tensor.spatial_shape, device=sampling_offsets.device)
                for tensor in value_tensors
            ],
            -1,
        )
        base_level_shape = spatial_shapes[:, -1]
        spatial_scaler = spatial_shapes / base_level_shape.unsqueeze(-1)

        # n_queries, num_heads, num_levels, num_points, 2
        sampling_locations = (
            reference_points.reshape(n_total_queries, 1, 1, 1, 2) + sampling_offsets
        ) * spatial_scaler.transpose(0, 1).reshape(1, 1, self.num_levels, 1, 2)

        output = multi_scale_deformable_attention(
            value, spatial_shapes, sampling_locations, query_offsets, attention_weights
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
