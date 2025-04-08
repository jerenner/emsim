from typing import Optional, Union

import torch
from torch import Tensor

from .rotary_embedding import (
    calculate_rope,
    rotate_k,
    rotate_k_backward,
    calculate_rope_backward,
)
from .autograd_helpers import (
    select_values_and_project_kv,
    linear_grads,
    split_heads,
    permute_for_attention,
    permute_for_attention_backward,
)


class GatherAndSubsetAttentionFunction(torch.autograd.Function):
    """Custom autograd function that implements memory-efficient attention
    where each query attends to its own local subset of keys. This implementation
    avoids keeping large intermediate tensors in memory by recalculating them
    during the backward pass, saving significant memory for only a minor increase
    in time to run the backward.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query_tensor: Tensor,
        n_heads: int,
        sparse_tensor_values: Tensor,
        index_search: Tensor,
        is_specified_mask: Tensor,
        Wk: Tensor,
        Wv: Tensor,
        bias_k: Optional[Tensor] = None,
        bias_v: Optional[Tensor] = None,
        key_pos_encoding: Optional[Tensor] = None,
        key_positions: Optional[Tensor] = None,
        rope_freqs: Optional[Tensor] = None,
        scale_factor: float = None,  # scaling for attn, default 1/sqrt(d)
    ) -> Tensor:
        """Performs sparse neighborhood attention with minimal memory usage.

        This function computes attention where each query attends only to its
        local neighborhood of keys, without materializing the full attention matrix
        or storing intermediate tensors.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context to save tensors for backward
            query_tensor (Tensor): Query features of shape [n_queries, embed_dim]
            n_heads (int): Number of attention heads
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
            key_pos_encoding (Optional[Tensor]): Positional encoding for keys of shape
                [n_queries, n_keys_per_query, n_heads, head_dim/2]. Used for rotary
                position embedding (RoPE). Cannot be used together with key_positions
                and rope_freqs.
            key_positions (Optional[Tensor]): Position information for each key of
                shape [n_queries, n_keys_per_query, position_dim]. Used together with
                rope_freqs to compute rotary position embedding (RoPE) on-the-fly.
                Cannot be used together with key_pos_encoding.
            rope_freqs (Optional[Tensor]): Frequency values for rotary embeddings of
                shape [position_dim, n_freq_groups, n_heads, head_dim/2] or
                [position_dim, n_freq_groups, 1, head_dim/2]. Used together with
                key_positions to compute rotary position embedding (RoPE) on-the-fly.
                Cannot be used together with key_pos_encoding.
            scale_factor (Optional[float]): Scaling factor for attention scores.
                Default is 1/sqrt(embed_dim).

        Returns:
            Tensor: Output tensor after attention of shape [n_queries, embed_dim]

        Note:
            - The output tensor has NOT gone through the output projection (W_o)
                that is encapsulated within most implementations of standard
                multi-head attention. The decision to exclude the output projection
                from this op was driven by the motivation to remove any extra
                complexity that would have diminishing memory performance benefits.
                You will need to add this as an extra nn.Linear layer that gets applied
                to this op's output before it gets passed to a transformer FFN block.
                The residual connection and normalization are also not included.
        """

        #########
        # The forward pass has several steps:
        # 1. Shape checks
        # 2. Setting up for the backward pass (saving tensors and shape ints)
        # 3. Input projection: Retrieving the input values from sparse_tensor_values
        #   using index_search. Then, as in standard multi-head attention, pass these
        #   values through the key and value projections and unstack the heads of the
        #   resulting tensors
        # 4. If we have the two RoPE inputs (key positions and RoPE freqs), compute
        #   the RoPE encoding rotation vector
        # 5. If we computed the RoPE rotation vector or were given it, apply it to k
        # 6. Permute the dimensions of the q, k, and v tensors
        #   to heads-batched order (similar to standard MHA)
        # 7. Compute the attention scores by matmuling q and k along the head_dim
        #   dimension (with scale factor) (again similar to standard MHA)
        # 8. Mask out the unspecified keys so queries can't attend to them and
        #   apply softmax to compute attention weights (the masking is slightly
        #   different from standard MHA due to the per-query subset structure of
        #   the keys but mostly straightforward)
        # 9. Compute the output values by matmuling the attention weights with
        #   the value tensor, contracting out the n_queries_per_keys dimension.

        # The k and v tensors are the major memory consumers because they are
        # n_keys_per_query times larger than the q and attn_scores tensors. The
        # custom autograd op lets us have each query element attend to its own
        # subset of key elements. In tensor terms, this means that the k and v
        # tensors are 4D while the q (query) tensor is 3D as in standard multi-head
        # attention. The matmuls between q and k/v are performed with unsqueezes and
        # broadcasts.

        ctx.set_materialize_grads(False)

        #### Step 1: shape checks

        assert query_tensor.ndim == 2  # (n_queries, embed_dim)
        assert index_search.ndim == 2  # (n_queries, n_keys_per_query)

        n_queries = query_tensor.size(0)
        embed_dim = query_tensor.size(1)
        n_keys_per_query = index_search.size(1)
        head_dim = embed_dim // n_heads

        assert query_tensor.size(0) == index_search.size(0) == n_queries
        assert index_search.shape == is_specified_mask.shape
        assert Wk.ndim == 2
        assert Wv.ndim == 2

        # embed_dim
        # kv projection
        assert Wk.size(1) == Wv.size(1) == sparse_tensor_values.size(-1) == embed_dim
        # attn calculation
        assert Wk.size(0) == query_tensor.size(1) == embed_dim

        # rope validations
        if key_pos_encoding is not None and (
            key_positions is not None or rope_freqs is not None
        ):
            raise ValueError(
                "Cannot provide both key_pos_encoding and (key_positions, rope_freqs)"
            )
        if (key_positions is not None) ^ (rope_freqs is not None):
            raise ValueError("Cannot provide only one of key_positions and rope_freqs")

        if key_pos_encoding is not None:
            assert head_dim % 2 == 0, "head_dim must be even to use RoPE"
            assert key_pos_encoding.shape == (
                n_queries,
                n_keys_per_query,
                n_heads,
                head_dim / 2,
            )

        if key_positions is not None and rope_freqs is not None:
            # check shapes
            assert head_dim % 2 == 0, "head_dim must be even to use RoPE"

            # (n_queries, n_keys_per_query, position_dim)
            assert key_positions.ndim == 3

            # (position_dim, n_groups, n_heads or 1, head_dim)
            assert rope_freqs.ndim == 4

            assert key_positions.shape[0] == n_queries
            assert key_positions.shape[1] == n_keys_per_query
            assert rope_freqs.shape[-1] == head_dim / 2

            position_dim = key_positions.shape[-1]
            assert rope_freqs.shape[0] == position_dim

        #### Step 2: backward pass preparation

        # save shape info
        ctx.n_queries = n_queries
        ctx.embed_dim = embed_dim
        ctx.n_heads = n_heads
        ctx.head_dim = head_dim
        ctx.n_keys_per_query = n_keys_per_query

        # default scale factor
        if scale_factor is None:
            scale_factor = embed_dim ** (-1 / 2)
        ctx.scale_factor = scale_factor

        # save tensors
        ctx.save_for_backward(
            query_tensor,
            sparse_tensor_values,
            Wk,
            Wv,
            bias_k,
            bias_v,
            (
                key_pos_encoding
                if not (key_positions is not None and rope_freqs is not None)
                else None
            ),
            key_positions,
            rope_freqs,
        )
        ctx.index_search = index_search
        ctx.is_specified_mask = is_specified_mask

        #### Step 3: Sparse values selection and input projection

        # fmt: off
        k, v, _ = select_values_and_project_kv(
            sparse_tensor_values, index_search, is_specified_mask,
            Wk, Wv, bias_k, bias_v,
        )
        # fmt: on

        q, k, v = [split_heads(x, n_heads) for x in [query_tensor, k, v]]

        #### Step 4: RoPE encoding calculation

        if key_positions is not None and rope_freqs is not None:
            key_pos_encoding = calculate_rope(key_positions, rope_freqs)

        #### Step 5: Rotate the keys by applying RoPE

        if key_pos_encoding is not None:
            k = rotate_k(k, key_pos_encoding, needs_autograd=False)

        #### Step 6: Permutation

        q, k, v = [permute_for_attention(x) for x in [q, k, v]]

        #### Step 7: Attention scores calculation

        q = q.unsqueeze(-2)  # (n_heads, n_queries, 1, head_dim)
        # fmt: off
        attn_scores = torch.matmul(
            q * scale_factor,   # (n_heads, n_queries, 1, head_dim)
            k.transpose(-1, -2) # (n_heads, n_queries, head_dim, n_keys_per_query)
        ).squeeze(-2)           # (n_heads, n_queries, n_keys_per_query)
        # fmt: on

        #### Step 8: Masking and softmax

        attn_scores.masked_fill_(~is_specified_mask, -torch.inf)
        attn_weights = attn_scores.softmax(-1)
        # nans expected if all of the keys that a query tried to attend to were unspecified
        attn_weights.nan_to_num_(0.0)

        #### Step 9: Compute the output values

        # fmt: off
        output = torch.matmul(
            attn_weights.unsqueeze(-2), # (n_heads, n_queries, 1, n_keys_per_query)
            v,                          # (n_heads, n_queries, n_keys_per_query, head_dim)
            out=q,  # memory optimization
        ).squeeze(-2)                   # (n_heads, n_queries, head_dim)
        # fmt: on

        output = output.transpose(-2, -3)  # (n_queries, n_heads, head_dim)
        output = output.reshape(n_queries, embed_dim)

        ctx.attn_weights = attn_weights

        return output

    @staticmethod
    def _determine_needed_intermediate_grads(
        needed_grads: dict[str, bool],
    ) -> dict[str, bool]:
        """Determine which intermediate tensors need to be computed to compute
        all needed gradients
        """
        # All upstream gradients require grad of attention scores
        compute_grad_attn_scores = (
            needed_grads["query"]
            or needed_grads["sparse_values"]
            or needed_grads["key_weight"]
            or needed_grads["value_weight"]
            or needed_grads["key_bias"]
            or needed_grads["value_bias"]
            or needed_grads["key_rope_encoding"]
            or needed_grads["key_positions"]
            or needed_grads["rope_freqs"]
        )

        # Query tensor is its own branch, so no other grads depend on it
        compute_grad_query = needed_grads["query"]

        # Everything else is upstream of k
        compute_grad_k = (
            needed_grads["sparse_values"]
            or needed_grads["key_weight"]
            or needed_grads["value_weight"]
            or needed_grads["key_bias"]
            or needed_grads["value_bias"]
            or needed_grads["key_rope_encoding"]
            or needed_grads["key_positions"]
            or needed_grads["rope_freqs"]
        )

        # Decide whether we need to go into the key branch
        # (only gradient that doesn't depend on it is the value projection)
        compute_grads_key_branch = (
            needed_grads["sparse_values"]
            or needed_grads["key_weight"]
            or needed_grads["key_bias"]
            or needed_grads["key_rope_encoding"]
            or needed_grads["key_positions"]
            or needed_grads["rope_freqs"]
        )

        # Do we need to compute the grads of the on-the-fly RoPE encoding
        compute_grads_rope_inputs = (
            needed_grads["key_positions"] or needed_grads["rope_freqs"]
        )

        # Do we need to traverse the value branch
        compute_grads_value_branch = (
            needed_grads["sparse_values"]
            or needed_grads["value_weight"]
            or needed_grads["value_bias"]
        )

        # Input projections
        compute_grads_input_projections = (
            needed_grads["key_weight"]
            or needed_grads["value_weight"]
            or needed_grads["key_bias"]
            or needed_grads["value_bias"]
        )

        # sparse_tensor_values is upstream of everything so nothing depends on it
        compute_grads_sparse_values = needed_grads["sparse_values"]

        return {
            "attn_scores": compute_grad_attn_scores,
            "query": compute_grad_query,
            "k": compute_grad_k,
            "key_branch": compute_grads_key_branch,
            "rope_inputs": compute_grads_rope_inputs,
            "value_branch": compute_grads_value_branch,
            "input_projections": compute_grads_input_projections,
            "sparse_values": compute_grads_sparse_values,
        }

    @staticmethod
    def _compute_grad_attn_scores(
        grad_output: Tensor, v: Tensor, attn_weights: Tensor, is_specified_mask: Tensor
    ) -> Tensor:
        # fmt: off
        grad_attn_weights = torch.matmul(
            grad_output.unsqueeze(-2), # (n_heads, n_queries, 1, head_dim)
            v.transpose(-1, -2),       # (n_heads, n_queries, head_dim, n_keys_per_query)
        ).squeeze(-2)                  # (n_heads, n_queries, n_keys_per_query)
        # fmt: on

        # softmax gradient: dL/dz = S * (dL/dS - sum_j(S_j * dL/dS_j))
        # where z = attn_scores, S = softmax(z), dL/dS = grad_attn_weights
        # and j indexes keys
        grad_attn_scores = attn_weights * (
            grad_attn_weights - (attn_weights * grad_attn_weights).sum(-1, keepdim=True)
        )

        grad_attn_scores.masked_fill_(~is_specified_mask, 0)

        return grad_attn_scores

    @staticmethod
    def _compute_grad_query(grad_attn_scores: Tensor, k: Tensor, scale_factor: float):
        # fmt: off
        grad_q = torch.matmul(
            grad_attn_scores.unsqueeze(-2),  # (n_heads, n_queries, 1, n_keys_per_query)
            k,                               # (n_heads, n_queries, n_keys_per_query, head_dim)
        ).squeeze(-2)                        # (n_heads, n_queries, head_dim

        grad_q *= scale_factor

        # Flip dims back and stack heads
        grad_q = permute_for_attention_backward(grad_q)  # (n_queries, n_heads, head_dim)
        grad_query = grad_q.flatten(-2, -1)  # (n_queries, embed_dim)
        # fmt: on
        return grad_query

    @staticmethod
    def _compute_grads_k_and_pos_encoding(
        grad_attn_scores: Tensor,
        q: Tensor,
        k: Tensor,
        scale_factor: float,
        key_rope_encoding: Union[Tensor, None],
        needs_grad_k: bool,
        needs_grad_key_pos: bool,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if not needs_grad_k and not needs_grad_key_pos:
            return None, None
        # fmt: off
        grad_k_maybe_rotated = torch.matmul(
            grad_attn_scores.unsqueeze(-1), # (n_heads, n_queries, n_keys_per_query, 1)
            q.unsqueeze(-2) * scale_factor, # (n_heads, n_queries, 1, head_dim)
        )                                   # (n_heads, n_queries, n_keys_per_query, head_dim)
        # fmt: on

        # (n_queries, n_keys_per_query, n_heads, head_dim)
        grad_k_maybe_rotated = permute_for_attention_backward(grad_k_maybe_rotated)

        if key_rope_encoding is not None:
            # Handle backpropagation through RoPE
            grad_k, grad_key_pos_encoding = rotate_k_backward(
                grad_k_maybe_rotated,
                k,
                key_rope_encoding,
                needs_grad_k,
                needs_grad_key_pos,
                needs_autograd=False,
            )
        else:
            grad_k = grad_k_maybe_rotated
            grad_key_pos_encoding = None

        if grad_k is None:
            assert not needs_grad_k
            return None, grad_key_pos_encoding

        # (n_heads, n_queries, embed_dim)
        grad_k = grad_k.flatten(-2, -1)

        return grad_k, grad_key_pos_encoding

    @staticmethod
    def _compute_grad_v(attn_weights: Tensor, grad_output: Tensor) -> Tensor:
        # fmt: off
        grad_v = torch.matmul(
            attn_weights.unsqueeze(-1), # (n_heads, n_queries, n_keys_per_query, 1)
            grad_output.unsqueeze(-2)   # (n_heads, n_queries, 1, head_dim)
        )                               # (n_heads, n_queries, n_keys_per_query, head_dim)
        # fmt: on

        # (n_queries, n_keys_per_query, n_heads, head_dim)
        grad_v = permute_for_attention_backward(grad_v)
        grad_v = grad_v.flatten(-2, -1)  # (n_queries, n_keys_per_query, embed_dim)
        return grad_v

    @staticmethod
    def _compute_grads_k_v_projections(
        grad_k_flat: Tensor,
        grad_v_flat: Tensor,
        selected: Tensor,
        needs_grad_Wk: bool,
        needs_grad_Wv: bool,
        needs_grad_bias_k: bool,
        needs_grad_bias_v: bool,
    ) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        selected_flat = selected.view(-1, selected.size(-1))

        if (needs_grad_Wk or needs_grad_bias_k) and (
            needs_grad_Wv and needs_grad_bias_v
        ):
            # need grads from both projections - batch the two gradient
            # calculations to save a matmul call (bmm vs 2x mm)

            # stack gradients for batched k and v backward (adding leading dim)
            grad_kv_flat = torch.stack([grad_k_flat, grad_v_flat])

            grad_W_stacked, grad_bias_stacked = linear_grads(
                grad_kv_flat,
                selected_flat,
                needs_grad_Wk or needs_grad_Wv,
                needs_grad_bias_k or needs_grad_bias_v,
            )

            if grad_W_stacked is not None:
                grad_Wk, grad_Wv = grad_W_stacked
                grad_Wk = grad_Wk if needs_grad_Wk else None
                grad_Wv = grad_Wv if needs_grad_Wv else None

            if grad_bias_stacked is not None:
                grad_bias_k, grad_bias_v = grad_bias_stacked
                grad_bias_k = grad_bias_k if needs_grad_bias_k else None
                grad_bias_v = grad_bias_v if needs_grad_bias_v else None

        else:
            # only need one projection's grad. call linear_grads twice
            # since it will safely return None, None for the one where
            # needs_grads bools are False
            grad_Wk, grad_bias_k = linear_grads(
                grad_k_flat, selected_flat, needs_grad_Wk, needs_grad_bias_k
            )

            grad_Wv, grad_bias_v = linear_grads(
                grad_v_flat, selected_flat, needs_grad_Wv, needs_grad_bias_v
            )
        return grad_Wk, grad_Wv, grad_bias_k, grad_bias_v

    @staticmethod
    def _compute_grad_sparse_values(
        grad_k_flat: Tensor,
        grad_v_flat: Tensor,
        Wk: Tensor,
        Wv: Tensor,
        is_specified_mask: Tensor,
        sparse_tensor_values: Tensor,
        index_search: Tensor,
    ) -> Tensor:
        n_queries = is_specified_mask.size(0)
        n_keys_per_query = is_specified_mask.size(1)
        embed_dim = grad_k_flat.size(-1)

        # two matrix multiplies - faster if we batch them
        grad_k_v_stacked = torch.stack([grad_k_flat, grad_v_flat])
        W_stacked = torch.stack([Wk, Wv])
        # fmt: off
        grad_selected = torch.bmm(
            grad_k_v_stacked,  # (2, n_queries * n_keys_per_query, embed_dim)
            W_stacked,         # (2, embed_dim, embed_dim)
        )                      # (2, n_queries * n_keys_per_query, embed_dim)
        # fmt: on
        grad_selected = grad_selected.sum(0)  # = elementwise add of k, v contributions
        grad_selected = grad_selected.view(n_queries, n_keys_per_query, embed_dim)

        # Zero out grads for masked selecteds
        grad_selected.masked_fill_(~is_specified_mask.unsqueeze(-1), 0)

        # Scatter grads back into the sparse values
        grad_sparse_values = torch.zeros_like(sparse_tensor_values)
        grad_sparse_values.index_add_(
            0, index_search.view(-1), grad_selected.view(-1, embed_dim)
        )
        return grad_sparse_values

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Optional[Tensor], ...]:
        """Implements the backward pass for sparse neighborhood attention.

        This custom backward operation recalculates intermediate values that were
        not stored during the forward pass to save memory, then calculates gradients
        for only the input tensors that require gradients.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context containing saved tensors
            grad_output (Tensor): Gradient of the loss with respect to the output,
                shape [n_queries, embed_dim]

        Returns:
            tuple[Optional[Tensor], ...]: Gradients for all inputs in the same order as
            the forward method:
                - grad_query: [n_queries, embed_dim] or None
                - None (for n_heads)
                - grad_sparse_values: [num_sparse_values, embed_dim] or None
                - None (for index_search)
                - None (for is_specified_mask)
                - grad_Wk: [embed_dim, embed_dim] or None
                - grad_Wv: [embed_dim, embed_dim] or None
                - grad_bias_k: [embed_dim] or None
                - grad_bias_v: [embed_dim] or None
                - grad_key_pos_encoding: [n_queries, n_keys_per_query, embed_dim] or None
                - None (for scale_factor)
        """

        ######
        # The backward pass is broken into steps:
        # 1. Retrieve all inputs and set up variables
        # 2. Deduce which intermediate values need to be recalculated
        # 3. Repeat the first few steps of the forward pass
        #   - Input and key and value projections, and unstacking the heads of q, k, and v
        #   - Calculation of the RoPE encoding and applying it to the key tensor
        #   - Permutation of q, k, and v into batched-heads structure (as standard)
        # 4. If any gradients are required, compute the gradient of the attention scores
        #   as it is downstream of all of the input parameters which may need gradients
        # 5. If query_vector needs a gradient, compute its gradient
        # 6. If any gradients along the key branch (key projection, RoPE parameters,
        #   or the input sparse tensor values) are needed, compute the grad of k
        # 7. If RoPE was applied, apply its backward to the grad of k, and get the
        #   gradients of the RoPE encoding or its inputs (key positions and rope
        #   frequencies) if we need those
        # 8. If the value projection or the sparse tensor values need gradients,
        #   compute the grad of v
        # 9. If the input key and/or value projections need gradients, compute those
        # 10. Finally, if the sparse tensor values need gradients, compute those
        #
        # All intermediate tensors are deleted as soon as they aren't needed anymore
        #   (i.e., when there aren't any more tensors upstream of them to compute)
        ######

        ##### Step 1: retrieve values and set up variables

        # retrieve tensors
        (
            query_tensor,
            sparse_tensor_values,
            Wk,
            Wv,
            bias_k,
            bias_v,
            key_pos_encoding,
            key_positions,
            rope_freqs,
        ) = ctx.saved_tensors
        index_search: Tensor = ctx.index_search
        is_specified_mask: Tensor = ctx.is_specified_mask

        attn_weights: Tensor = ctx.attn_weights

        # retrieve shape info
        embed_dim: int = ctx.embed_dim
        n_heads: int = ctx.n_heads

        # retrieve scale factor
        scale_factor: float = ctx.scale_factor

        # account for which inputs need gradients
        needed_grads = {
            "query": ctx.needs_input_grad[0],
            "sparse_values": ctx.needs_input_grad[2],
            "key_weight": ctx.needs_input_grad[5],
            "value_weight": ctx.needs_input_grad[6],
            "key_bias": bias_k is not None and ctx.needs_input_grad[7],
            "value_bias": bias_v is not None and ctx.needs_input_grad[8],
            "key_rope_encoding": (
                key_pos_encoding is not None and ctx.needs_input_grad[9]
            ),
            "key_positions": (key_positions is not None and ctx.needs_input_grad[10]),
            "rope_freqs": rope_freqs is not None and ctx.needs_input_grad[11],
        }

        # initialize grad vars
        grad_query = None
        grad_sparse_values = None
        grad_Wk = None
        grad_Wv = None
        grad_bias_k = None
        grad_bias_v = None
        grad_key_pos_encoding = None
        grad_key_positions = None
        grad_rope_freqs = None

        # initialize flattened grad vars
        grad_k_flat = None
        grad_v_flat = None

        # initialize pre-rotation k copy for RoPE grads
        k_unrotated_copy = None

        if grad_output is None:
            return (
                grad_query,  # query_tensor
                None,  # n_heads
                grad_sparse_values,  # sparse_tensor_values
                None,  # index_search
                None,  # is_specified_mask
                grad_Wk,  # Wk
                grad_Wv,  # Wv
                grad_bias_k,  # bias_k
                grad_bias_v,  # bias_v
                grad_key_pos_encoding,  # key_pos_encoding
                grad_key_positions,  # key_positions
                grad_rope_freqs,  # rope_freqs
                None,  # scale_factor
            )

        #### Step 2: Decide which which gradients to compute

        needed_intermediates = (
            GatherAndSubsetAttentionFunction._determine_needed_intermediate_grads(
                needed_grads
            )
        )

        #### Step 3: repeat the first few operations of the forward pass

        # recompute k, v
        # fmt: off
        k, v, selected = select_values_and_project_kv(
            sparse_tensor_values, index_search, is_specified_mask,
            Wk, Wv, bias_k, bias_v,
        )
        # fmt: on

        if key_positions is not None and rope_freqs is not None:
            key_pos_encoding = calculate_rope(key_positions, rope_freqs)

        [grad_output, q, k, v] = [
            split_heads(x, n_heads) for x in [grad_output, query_tensor, k, v]
        ]

        if key_pos_encoding is not None:
            if needed_intermediates["key_branch"]:
                # used later in backward pass to compute RoPE grads
                k_unrotated_copy = k.clone()
            k = rotate_k(k, key_pos_encoding, needs_autograd=False)

        # split heads and permute the grad tensor and input tensors
        # (n_heads, n_queries, head_dim)
        grad_output = permute_for_attention(grad_output)
        q = permute_for_attention(q)

        # (n_heads, n_queries, n_keys_per_query, head_dim)
        k = permute_for_attention(k)
        v = permute_for_attention(v)

        #### Step 4: Compute gradient of attention scores

        if needed_intermediates["attn_scores"]:
            grad_attn_scores = (
                GatherAndSubsetAttentionFunction._compute_grad_attn_scores(
                    grad_output, v, attn_weights, is_specified_mask
                )
            )
        del v  # big tensor we no longer need

        #### Step 5: Compute query gradient

        if needed_grads["query"]:
            grad_query = GatherAndSubsetAttentionFunction._compute_grad_query(
                grad_attn_scores, k, scale_factor
            )
        del k

        #### Step 6: Compute gradient of k, representing the entry to the key branch

        if needed_intermediates["k"]:
            if needed_intermediates["key_branch"]:
                #### Step 7: Compute the backward pass of RoPE and compute the RoPE
                #       encoding's gradient, if needed.

                # combining the computation of grad_k and un-rotating it into one
                # function makes the main backward's logic less complex
                grad_k, grad_key_pos_encoding = (
                    GatherAndSubsetAttentionFunction._compute_grads_k_and_pos_encoding(
                        grad_attn_scores,
                        q,
                        k_unrotated_copy,
                        scale_factor,
                        key_pos_encoding,
                        needs_grad_k=(
                            needed_grads["key_weight"]
                            or needed_grads["key_bias"]
                            or needed_grads["sparse_values"]
                        ),
                        needs_grad_key_pos=(
                            needed_grads["key_rope_encoding"]
                            or needed_grads["key_positions"]
                            or needed_grads["rope_freqs"]
                        ),
                    )
                )
                del k_unrotated_copy

                #### Step 7.5: Compute the gradients of the RoPE encoding's inputs
                #       if RoPE was computed on the fly

                if needed_intermediates["rope_inputs"]:
                    assert not needed_grads["key_rope_encoding"]  # mutually exclusive
                    grad_key_positions, grad_rope_freqs = calculate_rope_backward(
                        grad_key_pos_encoding,
                        key_positions,
                        rope_freqs,
                        needed_grads["key_positions"],
                        needed_grads["rope_freqs"],
                    )
                    grad_key_pos_encoding = None

                # Flatten for grad calcs
                if grad_k is not None:
                    # [n_queries * n_keys_per_query * n_heads, head_dim]
                    grad_k_flat = grad_k.view(-1, embed_dim)
            del grad_attn_scores
            del q

            ##### Step 8: Enter value branch

            if needed_intermediates["value_branch"]:
                grad_v = GatherAndSubsetAttentionFunction._compute_grad_v(
                    attn_weights, grad_output
                )

                # Flatten for grad calcs
                # [n_queries * n_keys_per_query * n_heads, head_dim]
                grad_v_flat = grad_v.view(-1, embed_dim)

            ##### Step 9: Input projection gradients

            if needed_intermediates["input_projections"]:
                # need to get at least one of the projection gradients
                grad_Wk, grad_Wv, grad_bias_k, grad_bias_v = (
                    GatherAndSubsetAttentionFunction._compute_grads_k_v_projections(
                        grad_k_flat,
                        grad_v_flat,
                        selected,
                        needed_grads["key_weight"],
                        needed_grads["value_weight"],
                        needed_grads["key_bias"],
                        needed_grads["value_bias"],
                    )
                )
            del selected

            ##### Step 10: Gradients of the original sparse tensor values

            if needed_grads["sparse_values"]:
                grad_sparse_values = (
                    GatherAndSubsetAttentionFunction._compute_grad_sparse_values(
                        grad_k_flat,
                        grad_v_flat,
                        Wk,
                        Wv,
                        is_specified_mask,
                        sparse_tensor_values,
                        index_search,
                    )
                )

        return (
            grad_query,  # query_tensor
            None,  # n_heads
            grad_sparse_values,  # sparse_tensor_values
            None,  # index_search
            None,  # is_specified_mask
            grad_Wk,  # Wk
            grad_Wv,  # Wv
            grad_bias_k,  # bias_k
            grad_bias_v,  # bias_v
            grad_key_pos_encoding,  # key_pos_encoding
            grad_key_positions,  # key_positions
            grad_rope_freqs,  # rope_freqs
            None,  # scale_factor
        )
