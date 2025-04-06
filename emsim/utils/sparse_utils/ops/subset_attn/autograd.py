from typing import Optional, Union

import torch
from torch import Tensor

from .rotary_embedding import (
    calculate_rope,
    rotate_k,
    rotate_k_backward,
    calculate_rope_backward,
)
from .autograd_helpers import prep_qkv, linear_grads


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
                [n_queries, n_keys_per_query, embed_dim]. Used for rotary position
                embedding (RoPE). If specified, both embed_dim and the head dim must be
                divisible by 2. Cannot be used together with key_positions
                and rope_freqs.
            key_positions (Optional[Tensor]): Position information for each key of
                shape [n_queries, n_keys_per_query, position_dim]. Used together with
                rope_freqs to compute rotary position embedding (RoPE) on-the-fly.
                Cannot be used together with key_pos_encoding.
            rope_freqs (Optional[Tensor]): Frequency values for rotary embeddings of
                shape [position_dim, n_freq_groups, embed_dim] or
                [position_dim, embed_dim]. Used together with key_positions to
                compute rotary position embedding (RoPE) on-the-fly. Cannot be used
                together with key_pos_encoding.
            scale_factor (Optional[float]): Scaling factor for attention scores.
                Default is 1/sqrt(embed_dim).

        Returns:
            Tensor: Output tensor after attention of shape [n_queries, embed_dim]
        """
        ctx.set_materialize_grads(False)

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
            assert key_pos_encoding.shape == (n_queries, n_keys_per_query, embed_dim)
            assert embed_dim % 2 == 0, "embed_dim must be even to use RoPE"
            assert head_dim % 2 == 0, "head_dim must be even to use RoPE"

        if key_positions is not None and rope_freqs is not None:
            # check shapes
            assert (
                key_positions.ndim == 3
            )  # (n_queries, n_keys_per_query, position_dim)

            # (position_dim, embed_dim) or (position_dim, n_groups, embed_dim)
            assert rope_freqs.ndim in (2, 3)
            if rope_freqs.ndim == 2:
                rope_freqs = rope_freqs.unsqueeze(1)

            assert key_positions.shape[0] == n_queries
            assert key_positions.shape[1] == n_keys_per_query
            assert rope_freqs.shape[-1] == embed_dim

            position_dim = key_positions.shape[-1]
            assert key_positions.shape[-1] == position_dim

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

        # fmt: off
        q, k, v, _ = prep_qkv(
            query_tensor, sparse_tensor_values, index_search, is_specified_mask,
            Wk, Wv, bias_k, bias_v, n_heads, head_dim,
        )
        # fmt: on

        if key_positions is not None and rope_freqs is not None:
            key_pos_encoding = calculate_rope(key_positions, rope_freqs)

        if key_pos_encoding is not None:
            k, _, _ = rotate_k(k, key_pos_encoding)

        # fmt: off
        attn_scores = torch.matmul(
            q.unsqueeze(-2) * scale_factor, # (n_heads, n_queries, 1, head_dim)
            k.transpose(-1, -2)             # (n_heads, n_queries, head_dim, n_keys_per_query)
        ).squeeze(-2)                       # (n_heads, n_queries, n_keys_per_query)
        # fmt: on

        attn_scores.masked_fill_(~is_specified_mask, -torch.inf)
        attn_weights = attn_scores.softmax(-1)
        # nans expected if all of the keys a query tried to attend to were unspecified
        attn_weights.nan_to_num_(0.0)

        # fmt: off
        output = torch.matmul(
            attn_weights.unsqueeze(-2), # (n_heads, n_queries, 1, n_keys_per_query)
            v,                          # (n_heads, n_queries, n_keys_per_query, head_dim)
        ).squeeze(-2)                   # (n_heads, n_queries, head_dim)
        # fmt: on

        output = output.transpose(-2, -3)  # (n_queries, n_heads, head_dim)
        output = output.reshape(n_queries, embed_dim)

        ctx.attn_weights = attn_weights

        return output


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
        grad_q = grad_q.transpose(-2, -3)  # (n_queries, n_heads, head_dim)
        grad_query = grad_q.flatten(-2, -1)  # (n_queries, embed_dim)
        # fmt: on
        return grad_query

    @staticmethod
    def _compute_grad_k(
        grad_attn_scores: Tensor,
        q: Tensor,
        scale_factor: float,
        key_pos_encoding: Union[Tensor, None],
        needs_grad_key_pos: bool,
        k_complex: Union[Tensor, None],
        key_pos_complex: Union[Tensor, None],
    ) -> tuple[Tensor, Optional[Tensor]]:
        # fmt: off
        grad_k = torch.matmul(
            grad_attn_scores.unsqueeze(-1), # (n_heads, n_queries, n_keys_per_query, 1)
            q.unsqueeze(-2) * scale_factor, # (n_heads, n_queries, 1, head_dim)
        )                                   # (n_heads, n_queries, n_keys_per_query, head_dim)
        # fmt: on

        if key_pos_encoding is not None:
            # Handle backpropagation through RoPE
            grad_k, grad_key_pos_encoding = rotate_k_backward(
                grad_k, k_complex, key_pos_complex, needs_grad_key_pos
            )
        else:
            grad_key_pos_encoding = None

        # (n_queries, n_keys_per_query, n_heads, head_dim)
        grad_k = grad_k.permute(1, 2, 0, 3)
        grad_k = grad_k.flatten(-2, -1)  # (n_queries, n_keys_per_query, embed_dim)
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
        grad_v = grad_v.permute(1, 2, 0, 3)
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

            grad_W_stacked, grad_bias_stacked = (
                linear_grads(
                    grad_kv_flat,
                    selected_flat,
                    needs_grad_Wk or needs_grad_Wv,
                    needs_grad_bias_k or needs_grad_bias_v,
                )
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
        n_queries: int = ctx.n_queries
        embed_dim: int = ctx.embed_dim
        n_heads: int = ctx.n_heads
        head_dim: int = ctx.head_dim

        # retrieve scale factor
        scale_factor: float = ctx.scale_factor

        # account for which inputs need gradients
        needs_grad_query = ctx.needs_input_grad[0]
        needs_grad_sparse_values = ctx.needs_input_grad[2]
        needs_grad_Wk = ctx.needs_input_grad[5]
        needs_grad_Wv = ctx.needs_input_grad[6]
        needs_grad_bias_k = bias_k is not None and ctx.needs_input_grad[7]
        needs_grad_bias_v = bias_v is not None and ctx.needs_input_grad[8]
        needs_grad_key_pos_encoding = (
            key_pos_encoding is not None and ctx.needs_input_grad[9]
        )
        needs_grad_key_positions = (
            key_positions is not None and ctx.needs_input_grad[10]
        )
        needs_grad_rope_freqs = rope_freqs is not None and ctx.needs_input_grad[11]

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

        # initialize rope vars
        k_complex = None
        key_pos_complex = None

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

        # recompute q, k, v
        # fmt: off
        q, k, v, selected = prep_qkv(
            query_tensor, sparse_tensor_values, index_search, is_specified_mask,
            Wk, Wv, bias_k, bias_v, n_heads, head_dim,
        )
        # fmt: on

        if key_positions is not None and rope_freqs is not None:
            key_pos_encoding = calculate_rope(key_positions, rope_freqs)

        if key_pos_encoding is not None:
            k, k_complex, key_pos_complex = rotate_k(k, key_pos_encoding)

        # split heads on the grad tensor
        grad_output = grad_output.reshape(n_queries, n_heads, head_dim)

        # (n_heads, n_queries, head_dim)
        grad_output = grad_output.transpose(-2, -3).contiguous()

        if (
            needs_grad_query
            or needs_grad_sparse_values
            or needs_grad_Wk
            or needs_grad_Wv
            or needs_grad_bias_k
            or needs_grad_bias_v
            or needs_grad_key_pos_encoding
            or needs_grad_key_positions
            or needs_grad_rope_freqs
        ):
            grad_attn_scores = (
                GatherAndSubsetAttentionFunction._compute_grad_attn_scores(
                    grad_output, v, attn_weights, is_specified_mask
                )
            )
        del v  # big tensor we no longer need

        if needs_grad_query:
            grad_query = GatherAndSubsetAttentionFunction._compute_grad_query(
                grad_attn_scores, k, scale_factor
            )
        del k

        if (
            needs_grad_sparse_values
            or needs_grad_Wk
            or needs_grad_Wv
            or needs_grad_bias_k
            or needs_grad_bias_v
            or needs_grad_key_pos_encoding
            or needs_grad_key_positions
            or needs_grad_rope_freqs
        ):
            if (
                needs_grad_sparse_values
                or needs_grad_Wk
                or needs_grad_bias_k
                or needs_grad_key_pos_encoding
                or needs_grad_key_positions
                or needs_grad_rope_freqs
            ):
                grad_k, grad_key_pos_encoding = (
                    GatherAndSubsetAttentionFunction._compute_grad_k(
                        grad_attn_scores,
                        q,
                        scale_factor,
                        key_pos_encoding,
                        (
                            needs_grad_key_pos_encoding
                            or needs_grad_key_positions
                            or needs_grad_rope_freqs
                        ),
                        k_complex,
                        key_pos_complex,
                    )
                )
                del k_complex, key_pos_complex

                if needs_grad_key_positions or needs_grad_rope_freqs:
                    assert not needs_grad_key_pos_encoding  # mutually exclusive
                    grad_key_positions, grad_rope_freqs = calculate_rope_backward(
                        grad_key_pos_encoding,
                        key_positions,
                        rope_freqs,
                        needs_grad_key_positions,
                        needs_grad_rope_freqs,
                    )
                    grad_key_pos_encoding = None

                # Flatten for grad calcs
                grad_k_flat = grad_k.view(-1, embed_dim)
            del grad_attn_scores
            del q

            if needs_grad_sparse_values or needs_grad_Wv or needs_grad_bias_v:
                grad_v = GatherAndSubsetAttentionFunction._compute_grad_v(
                    attn_weights, grad_output
                )

                # Flatten for grad calcs
                grad_v_flat = grad_v.view(-1, embed_dim)

            if needs_grad_Wk or needs_grad_Wv or needs_grad_bias_k or needs_grad_bias_v:
                # need to get at least one of the projection gradients
                grad_Wk, grad_Wv, grad_bias_k, grad_bias_v = (
                    GatherAndSubsetAttentionFunction._compute_grads_k_v_projections(
                        grad_k_flat,
                        grad_v_flat,
                        selected,
                        needs_grad_Wk,
                        needs_grad_Wv,
                        needs_grad_bias_k,
                        needs_grad_bias_v,
                    )
                )
            del selected

            if needs_grad_sparse_values:
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
