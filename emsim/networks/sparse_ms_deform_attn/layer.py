import torch
from torch import nn, Tensor
import math
from torch.nn.init import constant_, xavier_uniform_

from ...utils.misc_utils import inverse_sigmoid
from .utils import sparse_split_heads, multilevel_sparse_bilinear_grid_sample


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
        self.value_proj = nn.Linear(embed_dim, embed_dim)
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

        stacked_value_tensors = torch.sparse_coo_tensor(
            stacked_value_tensors.indices(),
            self.value_proj(stacked_value_tensors.values()),
            size=stacked_value_tensors.shape,
            is_coalesced=stacked_value_tensors.is_coalesced(),
        ).coalesce()

        sampling_offsets = self.sampling_offsets(
            query.to(self.sampling_offsets.weight) # cast to double
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
            torch.get_default_dtype()
        )

        output = self.output_proj(output)
        return output


@torch.jit.script
def multi_scale_deformable_attention(
    stacked_value_tensors: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    query_offsets: Tensor,
    attention_weights: Tensor,
    interp_weight_dtype: torch.dtype = torch.float32,
) -> Tensor:
    assert isinstance(stacked_value_tensors, Tensor)
    assert stacked_value_tensors.is_sparse
    assert stacked_value_tensors.ndim == 5  # (batch, height, width, level, feature)

    n_queries, n_levels, n_points, n_heads, _ = sampling_locations.shape
    embed_dim = stacked_value_tensors.shape[-1]

    stacked_value_tensors = sparse_split_heads(stacked_value_tensors, n_heads)
    # now (batch, height, width, level, heads, head_dim)

    sampling_locations = sampling_locations * 2 - 1  # rescale to (-1, 1)

    batch_sizes: list[int] = torch.diff(
        torch.cat(
            [
                query_offsets,
                torch.tensor(
                    [n_queries], device=query_offsets.device, dtype=query_offsets.dtype
                ),
            ]
        )
    ).tolist()
    xy_batch_indices = torch.cat(
        [
            torch.full([size], i, device=sampling_locations.device, dtype=torch.int)
            for i, size in enumerate(batch_sizes)
        ]
    )
    xy_batch_indices = xy_batch_indices.view(-1, 1, 1, 1).expand(
        n_queries, n_levels, n_points, n_heads
    )
    xy_level_indices = (
        torch.arange(n_levels, device=xy_batch_indices.device, dtype=torch.int)
        .view(1, -1, 1, 1)
        .expand(n_queries, n_levels, n_points, n_heads)
    )

    sampled_values = multilevel_sparse_bilinear_grid_sample(
        stacked_value_tensors,
        sampling_locations,
        xy_batch_indices,
        xy_level_indices,
        value_spatial_shapes,
        interp_weight_dtype,
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
