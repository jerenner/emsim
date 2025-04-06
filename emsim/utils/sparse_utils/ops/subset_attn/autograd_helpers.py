from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from emsim.utils.sparse_utils.indexing.script_funcs import gather_and_mask


@torch.jit.script
def prep_qkv(
    query_tensor: Tensor,
    sparse_tensor_values: Tensor,
    index_search: Tensor,
    is_specified_mask: Tensor,
    Wk: Tensor,
    Wv: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    n_heads: int,
    head_dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the key and value tensors and reshapes them and the query tensor
    for multi-head attention computation.

    Args:
        query_tensor (Tensor): Query features of shape [n_queries, embed_dim]
        sparse_tensor_values (Tensor): Values from sparse tensor of shape
            [num_sparse_values, embed_dim]
        index_search (Tensor): Long tensor of shape [n_queries, n_keys_per_query]
            with elements corresponding to the indices of each key along
            sparse_tensor_values's first dimension. If created by
            get_sparse_index_mapping, indices of unspecified keys will be
            masked to 0 to potentially speed up lookup.
        is_specified_mask (Tensor): Boolean mask of shape
            [n_queries, n_keys_per_query] indicating which indices are
            specified in the sparse tensor
        Wk (Tensor): Key projection matrix of shape [embed_dim, embed_dim]
        Wv (Tensor): Value projection matrix of shape [embed_dim, embed_dim]
        bias_k (Optional[Tensor]): Key projection bias of shape [embed_dim]
        bias_v (Optional[Tensor]): Value projection bias of shape [embed_dim]
        n_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head

    Returns:
        - q (Tensor): Query tensor of shape [n_heads, n_queries, head_dim]
        - k (Tensor): Key tensor of shape
            [n_heads, n_queries, n_keys_per_query, head_dim]
        - v (Tensor): Value tensor of shape
            [n_heads, n_queries, n_keys_per_query, head_dim]
        - selected (Tensor): Selected features from sparse tensor before k and v
            projections, of shape [n_queries, n_keys_per_query, embed_dim]
    """
    assert query_tensor.ndim == 2
    assert index_search.ndim == 2
    assert sparse_tensor_values.ndim == 2

    n_queries = query_tensor.size(0)
    n_keys_per_query = index_search.size(1)

    selected = gather_and_mask(sparse_tensor_values, index_search, is_specified_mask)

    # Stack weight matrices to batch the k and v projections
    W_stacked = torch.cat([Wk, Wv])  # (2*embed_dim, embed_dim)

    # Handle stacking of biases if present
    if bias_k is not None or bias_v is not None:
        bias_k = bias_k if bias_k is not None else Wk.new_zeros(Wk.size(0))
        bias_v = bias_v if bias_v is not None else Wv.new_zeros(Wv.size(0))
        bias_stacked = torch.cat([bias_k, bias_v])  # (2*embed_dim)
    else:
        bias_stacked = None

    # (n_queries, n_keys_per_query, 2*embed_dim)
    kv = F.linear(selected, W_stacked, bias_stacked)
    k, v = kv.chunk(2, -1)  # (n_queries, n_keys_per_query, embed_dim) * 2

    # split heads
    # (n_queries, embed_dim) -> (n_queries, n_heads, head_dim)
    k = k.view(n_queries, n_keys_per_query, n_heads, head_dim)
    v = v.view(n_queries, n_keys_per_query, n_heads, head_dim)
    q = query_tensor.view(n_queries, n_heads, head_dim)

    # Move n_head dim forward
    q = q.transpose(-2, -3).contiguous()  # (n_heads, n_queries, head_dim)

    # (n_heads, n_queries, n_keys_per_query, head_dim)
    # standard batched-heads approach for multiplication with q and attn_weights
    # but with added n_keys_per_query dim that k broadcasts over q and v
    # contracts with attn_weights
    k = k.permute(2, 0, 1, 3).contiguous()
    v = v.permute(2, 0, 1, 3).contiguous()

    return q, k, v, selected


@torch.jit.script
def linear_grads(
    grad_output: Optional[Tensor],
    inputs: Tensor,
    need_weight_grad: bool,
    need_bias_grad: bool,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Efficiently computes gradients for weights and biases of a linear layer.
    Computes only the gradients required. If both the weight and bias gradient
    are required, computes them efficiently with the bias trick by concatenating
    a column of 1s onto the weight matrix before matmuling. The product of matmuling
    this augmented matrix with the gradient is then the weight and bias gradients
    stacked together.

    This function supports both regular and stacked gradients. When grad_output
    is 3D, with the leading dimension representing a stacking of the k and v
    gradients, the returned tensors are 3D and 2D, respectively.

    Args:
        grad_output (Tensor): Gradient of output, of shape [batch_size, out_features]
            or [num_projections, batch_size, out_features] for stacked mode
        inputs (Tensor): Input tensor, of shape [batch_size, in_features]
        need_weight_grad (bool): Whether weight gradients are needed
        need_bias_grad (bool): Whether bias gradients are needed

    Returns:
        - weight_grad (Optional[Tensor]): Gradient for weights, of shape
            [out_features, in_features] for non-stacked mode,
            [num_projections, out_features, in_features] for stacked mode,
            or None if need_weight_grad is False
        - bias_grad (Optional[Tensor]): Gradient for bias, of shape
            [out_features] for non-stacked mode, [num_projections, out_features]
            for stacked mode, or None if need_bias_grad is False
    """
    if grad_output is None:
        return None, None

    if not (grad_output.ndim == 2 or grad_output.ndim == 3):
        raise ValueError(
            f"Expected grad_output.ndim to be 2 or 3, got {grad_output.ndim}"
        )

    is_stacked_mode = grad_output.ndim == 3

    if need_weight_grad and need_bias_grad:
        # Set up bias trick
        ones = inputs.new_ones(inputs.size(0), 1)
        augmented_input = torch.cat([inputs, ones], dim=1)

        if is_stacked_mode:
            # fmt: off
                combined_grad = torch.bmm(
                    grad_output.transpose(-1, -2), # (num_proj, out_features, batch_size)
                    augmented_input.unsqueeze(0).expand(
                        grad_output.size(0), -1, -1
                    ),  # (num_proj, batch_size, in_features+1)
                )  # (num_proj, out_features, in_features+1)
        # fmt: on
        else:
            combined_grad = torch.mm(grad_output.t(), augmented_input)
        return combined_grad[..., :-1], combined_grad[..., -1]
    elif need_weight_grad:
        if is_stacked_mode:
            # fmt: off
                weight_grad = torch.bmm(
                    grad_output.transpose(-1, -2),  # (num_proj, out_features, batch_size)
                    inputs.unsqueeze(0).expand(
                        grad_output.size(0), -1, -1
                    ), # (num_proj, batch_size, in_features)
                )  # (num_proj, out_features, in_features)
        # fmt: on
        else:
            weight_grad = torch.mm(grad_output.t(), inputs)
        return weight_grad, None
    elif need_bias_grad:
        bias_grad = grad_output.sum(-2)
        return None, bias_grad
    return None, None
