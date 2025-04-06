from typing import Optional

import torch
from torch import Tensor


@torch.jit.script
def calculate_rope(key_positions: Tensor, rope_freqs: Tensor) -> Tensor:
    """Computes the positional encoding for keys using the provided positions and frequency values.

    This function calculates the position encoding by matrix-multiplying key
    positions with rotary frequency embeddings, then summing over frequency
    groups.
    This function may be used in combination with the others in its module for a
    memory-efficient RoPE application over many positions.

    Args:
        key_positions (Tensor): Position information for each key of shape
            [n_queries, n_keys_per_query, position_dim], where position_dim is the
            dimensionality of the position representation.
        rope_freqs (Tensor): Frequency values for rotary embeddings of shape
            [position_dim, n_freq_groups, embed_dim] or [position_dim, embed_dim],
            which will be reshaped to [position_dim, 1, embed_dim] if needed.

    Returns:
        Tensor: Computed positional encoding of shape
            [n_queries, n_keys_per_query, embed_dim]
    """
    if key_positions.ndim != 3:
        raise ValueError(
            f"Expected 3 dimensions for `key_positions`, got {key_positions.ndim}"
        )
    if rope_freqs.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2 or 3 dimnensions for `rope_freqs, got {rope_freqs.ndim}"
        )
    n_queries, n_keys_per_query, position_dim = key_positions.shape
    if rope_freqs.size(0) != position_dim:
        error_msg = "Expected first dimension of `rope_freqs` and last dimension of "
        error_msg += "key_positions to match, got "
        error_msg += str(rope_freqs.size(0)) + " and " + str(key_positions.size(-1))
        raise ValueError(error_msg)
    if rope_freqs.ndim == 2:  # only one freq_group
        rope_freqs = rope_freqs.unsqueeze(1)
    _, n_freq_groups, embed_dim = rope_freqs.shape
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even to use RoPE, got {embed_dim}")

    # fmt: off
    key_pos_encoding = torch.mm(
        key_positions.reshape(-1, position_dim),  # (n_queries*n_keys_per_query, position_dim)
        rope_freqs.reshape(position_dim, -1),  # (position_dim, n_freq_groups*embed_dim)
    ).view(n_queries, n_keys_per_query, n_freq_groups, embed_dim)
    # fmt: on

    key_pos_encoding = key_pos_encoding.sum(-2)  # sum over freq_groups
    return key_pos_encoding


@torch.jit.script
def rotate_k(keys: Tensor, rope_encoding: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Applies rotary position encoding (RoPE) to the key tensor via
    complex multiplication.

    Args:
        keys (Tensor): Post-input projection key tensor of shape
            [n_heads, n_queries, n_keys_per_query, head_dim], i.e. as returned
            from GatherAndSubsetAttentionFunction._prep_qkv
        rope_encoding (Tensor): Position encoding of shape
            [n_queries, n_keys_per_query, embed_dim]

    Returns:
        - k_rotated (Tensor): Key tensor after rotation, of shape
            [n_heads, n_queries, n_keys_per_query, head_dim]
        - keys_complex (Tensor): Complex representation of key tensor of shape
            [n_heads, n_queries, n_keys_per_query, head_dim/2].
            Used later in backward pass.
        - rope_encoding_complex: Complex representation of position encoding of shape
            [n_heads, n_queries, n_keys_per_query, head_dim/2].
            Used later in backward pass.
    """
    assert keys.ndim == 4
    if keys.size(-1) % 2 != 0:
        raise ValueError(f"head_dim ({keys.size(-1)}) must be even to use RoPE")
    n_heads, n_queries, n_keys_per_query, head_dim = keys.shape
    rope_encoding = rope_encoding.reshape(
        n_queries, n_keys_per_query, n_heads, head_dim
    )

    # (n_heads, n_queries, n_keys_per_query, head_dim)
    rope_encoding = rope_encoding.permute(2, 0, 1, 3).contiguous()

    # Convert to complex and apply rotation
    to_complex_shape = keys.shape[:-1] + (
        head_dim // 2,
        2,
    )
    keys_complex = torch.view_as_complex(keys.view(to_complex_shape))
    rope_encoding_complex = torch.view_as_complex(rope_encoding.view(to_complex_shape))

    # multiply and convert back to real
    k_rotated = keys_complex * rope_encoding_complex
    k_rotated = torch.view_as_real(k_rotated).reshape_as(keys)

    # complex tensors are used later in the backward pass
    return k_rotated, keys_complex, rope_encoding_complex


@torch.jit.script
def rotate_k_backward(
    grad_k_rotated: Tensor,
    k_complex: Tensor,
    key_pos_complex: Tensor,
    needs_grad_key_pos: bool,
) -> tuple[Tensor, Optional[Tensor]]:
    """Perform the backward pass of applying rotary positional encoding (RoPE)

    Computes gradients through complex number operations used in the RoPE
    forward pass. For complex multiplication z = x * y, the gradients are:
    dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x).

    Args:
        grad_k_rotated (Tensor): Gradient of loss with respect to rotated keys,
            of shape [n_heads, n_queries, n_keys_per_query, head_dim]
        k_complex (Tensor): Complex representation of the keys from forward pass
            of shape [n_heads, n_queries, n_keys_per_query, head_dim/2],
            as returned from _rotate_k.
        key_pos_complex (Tensor): Complex representation of positional encodings
            of shape [n_heads, n_queries, n_keys_per_query, head_dim/2],
            as returned from _rotate_k.
        needs_grad_key_pos (bool): Whether gradients for positional encodings
            are needed

    Returns:
        grad_keys (Tensor): Gradient tensor for the unrotated keys,
            of shape [n_heads, n_queries, n_keys_per_query, head_dim]
        grad_rope_encoding (Tensor): Gradient tensor for the positional encodings
            of shape [n_queries, n_keys_per_query, embed_dim],
            or None if not needed
    """
    n_heads, n_queries, n_keys_per_query, head_dim = grad_k_rotated.shape
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even to use RoPE")

    grad_k_rotated = grad_k_rotated.reshape(
        n_heads,
        n_queries,
        n_keys_per_query,
        head_dim // 2,
        2,
    )
    grad_k_rotated_complex = torch.view_as_complex(grad_k_rotated)

    # Complex multiplication gradient
    # For z = x * y, we have dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x)
    grad_k_complex = grad_k_rotated_complex * key_pos_complex.conj()

    grad_keys = torch.view_as_real(grad_k_complex).reshape(
        n_heads, n_queries, n_keys_per_query, head_dim
    )

    if needs_grad_key_pos:
        grad_key_pos_complex = grad_k_rotated_complex * k_complex.conj()

        # Convert back to real and reshape
        grad_rope_encoding = torch.view_as_real(grad_key_pos_complex)
        grad_rope_encoding = grad_rope_encoding.reshape(
            n_heads, n_queries, n_keys_per_query, head_dim
        )

        # (n_queries, n_keys_per_query, n_heads, head_dim)
        grad_rope_encoding = grad_rope_encoding.permute(1, 2, 0, 3).contiguous()
        grad_rope_encoding = grad_rope_encoding.view(
            n_queries, n_keys_per_query, head_dim * n_heads
        )

        return grad_keys, grad_rope_encoding
    return grad_keys, None


@torch.jit.script
def calculate_rope_backward(
    grad_key_pos_encoding: Tensor,
    key_positions: Tensor,
    rope_freqs: Tensor,
    needs_grad_key_positions: bool,
    needs_grad_rope_freqs: bool,
    squeeze_rope_freqs: bool = False,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Calculates gradients for the calculate_rope function.

    This function implements the backward pass for the calculation of the rotary
    positional encoding tensor that gets multiplied with the query/key tensor. It
    propagates the gradients from key_pos_encoding to key_positions and rope_freqs.

    Args:
        grad_key_pos_encoding (Tensor): Gradient of loss with respect to the positional
            encoding, of shape [n_queries, n_keys_per_query, embed_dim]
        key_positions (Tensor): Position tensor from the forward pass, of shape
            [n_queries, n_keys_per_query, position_dim]
        rope_freqs (Tensor): Frequency values tensor from the forward pass, of shape
            [position_dim, n_freq_groups, embed_dim] or [position_dim, embed_dim].
            If 2D, grad_rope_freqs will be 2D as well
        needs_grad_key_positions (bool): Whether grad for key_positions is required
        needs_grad_rope_freqs (bool): Whether grad for rope_freqs is required
        squeeze_rope_freqs (bool): If True and n_freq_groups is 1, will squeeze
            grad_rope_freqs to [position_dim, embed_dim] instead of
            [position_dim, 1, embed_dim]. Does nothing otherwise.

    Returns:
        tuple[Optional[Tensor], Optional[Tensor]]:
            - grad_key_positions: Gradient tensor for key positions of shape
              [n_queries, n_keys_per_query, position_dim], or None if not needed
            - grad_rope_freqs: Gradient tensor for rope frequencies of shape
              [position_dim, n_freq_groups, embed_dim] or [position_dim, embed_dim]
              if rope_freqs is 2D, or None if not needed
    """
    if key_positions.ndim != 3:
        raise ValueError(
            f"Expected 3 dimensions for `key_positions`, got {key_positions.ndim}"
        )
    if rope_freqs.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2 or 3 dimnensions for `rope_freqs, got {rope_freqs.ndim}"
        )
    n_queries, n_keys_per_query, position_dim = key_positions.shape
    if rope_freqs.size(0) != position_dim:
        error_msg = "Expected first dimension of `rope_freqs` and last dimension of "
        error_msg += "key_positions to match, got "
        error_msg += str(rope_freqs.size(0)) + " and " + str(key_positions.size(-1))
        raise ValueError(error_msg)
    embed_dim = rope_freqs.size(-1)
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be even to use RoPE")

    if rope_freqs.ndim == 2:
        rope_freqs_2d = True
        rope_freqs = rope_freqs.unsqueeze(1)
    else:
        rope_freqs_2d = False

    n_queries, n_keys_per_query, position_dim = key_positions.shape
    _, n_freq_groups, embed_dim = rope_freqs.shape

    # Backward of sum: distribute gradient across n_freq_groups
    grad_mm_result = grad_key_pos_encoding.unsqueeze(-2).expand(
        -1, -1, n_freq_groups, -1
    )
    # Reshape to match the mm result
    grad_mm_result = grad_mm_result.reshape(
        n_queries * n_keys_per_query, n_freq_groups * embed_dim
    )

    # Flatten inputs as in forward pass
    key_positions_flat = key_positions.reshape(-1, position_dim)
    rope_freqs_flat = rope_freqs.reshape(position_dim, -1)

    # Gradient for matrix multiplication: If C = A @ B
    # Then grad_A = grad_C @ B^T and grad_B = A^T @ grad_C
    if needs_grad_key_positions:
        grad_key_positions_flat = torch.mm(grad_mm_result, rope_freqs_flat.t())
        grad_key_positions = grad_key_positions_flat.view(
            n_queries, n_keys_per_query, position_dim
        )
    else:
        grad_key_positions = None

    if needs_grad_rope_freqs:
        grad_rope_freqs_flat = torch.mm(key_positions_flat.t(), grad_mm_result)
        grad_rope_freqs = grad_rope_freqs_flat.view(
            position_dim, n_freq_groups, embed_dim
        )
        if rope_freqs_2d and squeeze_rope_freqs:
            grad_rope_freqs = grad_rope_freqs.squeeze(1)
    else:
        grad_rope_freqs = None

    return grad_key_positions, grad_rope_freqs
