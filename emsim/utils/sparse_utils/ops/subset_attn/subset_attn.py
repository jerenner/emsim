from typing import Optional

import torch
from torch import Tensor

from emsim.utils.sparse_utils.indexing.script_funcs import get_sparse_index_mapping

from .autograd import GatherAndSubsetAttentionFunction


@torch.jit.script
def batch_sparse_index_subset_attn(
    sparse_tensor: Tensor,
    key_index_tensor: Tensor,
    query_tensor: Tensor,
    n_heads: int,
    key_weight: Tensor,
    value_weight: Tensor,
    key_bias: Optional[Tensor] = None,
    value_bias: Optional[Tensor] = None,
    key_pos_encoding: Optional[Tensor] = None,
    key_positions: Optional[Tensor] = None,
    rope_freqs: Optional[Tensor] = None,
    scale_factor: Optional[float] = None,
    check_all_specified: bool = False,
):
    """Performs batch selection of elements from a torch sparse tensor followed by
    multi-head attention. Each query attends only to its own specified subset of keys
    (representing that query's local neighborhood, for example).

    The implementation uses a custom autograd function to avoid storing large
    intermediate tensors, recalculating them during the backward pass as needed.

    Notes:
        - Key indices in key_index_tensor pointing to spatial locations in sparse_tensor
            that do not have specified values will be masked out in the attention
            calculation similar to masking out padding in standard attention.
        - Queries whose keys are all unspecified will get an output vector of all 0.
        - For rotary position embeddings, either provide key_pos_encoding OR both
          key_positions and rope_freqs. Providing both options simultaneously is
          not supported.

    Args:
        sparse_tensor (Tensor): Sparse tensor of dimension ..., M; where ... are
            S leading sparse dimensions and M is the dense feature dimension.
        key_index_tensor (Tensor): Long tensor of dimension ..., L, S; where ... are
            leading batch dimensions, L is the number of keys per query, and S is
            the number of sparse dimensions. Negative indices and indices outside
            the spatial dimension of the sparse tensor are not supported and will
            be considered unspecified.
        query_tensor (Tensor): Query features of shape ..., M; where ... matches
            the batch dimensions from key_index_tensor, and M is the feature dimension.
        n_heads (int): Number of attention heads to use.
        key_weight (Tensor): Key projection matrix of shape [M, M].
        value_weight (Tensor): Value projection matrix of shape [M, M].
        key_bias (Optional[Tensor]): Optional bias vector for key projection of shape [M].
        value_bias (Optional[Tensor]): Optional bias vector for value projection of shape [M].
        key_pos_encoding (Optional[Tensor]): Optional positional encoding for keys
            of shape [..., L, M], where ... matches the batch dimensions from key_index_tensor.
            Used for rotary position embedding (RoPE). If specified, M and the
            head dim must both be divisible by 2. Cannot be used together with
            key_positions and rope_freqs.
        key_positions (Optional[Tensor]): Position information for each key of shape
            [..., L, P], where ... matches the batch dimensions from key_index_tensor
            and P is the dimensionality of the position representation. Used together
            with rope_freqs to compute rotary position embedding (RoPE) on-the-fly.
            Cannot be used together with key_pos_encoding.
        rope_freqs (Optional[Tensor]): Frequency values for rotary embeddings of shape
            [P, G, M] or [P, M], where P matches the position dimension from key_positions,
            G is the number of frequency groups, and M is the feature dimension.
            Used together with key_positions to compute rotary position embedding (RoPE)
            on-the-fly. Cannot be used together with key_pos_encoding.
        scale_factor (Optional[float]): Optional scaling factor for attention scores.
            If None, will default is 1/sqrt(M).
        check_all_specified (bool): If True, this function will raise a ValueError
            if any of the indices in `key_index_tensor` are not specified in `sparse_tensor`.
            If False, unspecified indices will be masked out in the attention calculation.
            Defaults to False.

    Returns:
        - Tensor: Output tensor after attention of shape [..., M], where ... are the
            batch dimensions from key_index_tensor and query_tensor.
        - Tensor: Boolean mask of shape [..., L], indicating which keys were actually
            specified in the sparse tensor.
    """
    if key_index_tensor.is_nested:
        raise ValueError("Nested key index tensor not supported")
        # return __gather_nested_index(sparse_tensor, key_index_tensor, check_all_specified)

    if query_tensor.shape[:-1] != key_index_tensor.shape[:-2]:
        error_str = "Expected the first n-1 dims of query_tensor and the first"
        error_str += "n-2 dims of index_tensor to match, got "
        error_str += str(query_tensor.shape)
        error_str += " and "
        error_str += str(key_index_tensor.shape)
        raise ValueError(error_str)

    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor_values = sparse_tensor.values()

    index_search, is_specified_mask = get_sparse_index_mapping(
        sparse_tensor, key_index_tensor
    )
    if check_all_specified and not is_specified_mask.all():
        raise ValueError(
            "`check_all_specified` was set to True but not all gathered values "
            "were specified"
        )

    # Call into custom grad function
    attended = GatherAndSubsetAttentionFunction.apply(
        query_tensor,
        n_heads,
        sparse_tensor_values,
        index_search,
        is_specified_mask,
        key_weight,
        value_weight,
        key_bias,
        value_bias,
        key_pos_encoding,
        key_positions,
        rope_freqs,
        scale_factor,
    )

    out_shape = query_tensor.shape[:-1] + (value_weight.size(0),)
    assert attended.shape == out_shape
    assert is_specified_mask.shape == key_index_tensor.shape[:-1]

    return attended, is_specified_mask
