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
            [position_dim, n_freq_groups, n_heads, head_dim], where n_freq_groups
            and n_heads can be 1 for broadcasting.

    Returns:
        Tensor: Computed positional encoding of shape
            [n_queries, n_keys_per_query, n_heads, embed_dim]
    """
    if key_positions.ndim != 3:
        raise ValueError(
            f"Expected 3 dimensions for `key_positions`, got {key_positions.ndim}"
        )
    if rope_freqs.ndim != 4:
        raise ValueError(
            f"Expected 4 dimnensions for `rope_freqs, got {rope_freqs.ndim}"
        )

    n_queries, n_keys_per_query, position_dim = key_positions.shape
    position_dim_freqs, n_freq_groups, n_heads, head_dim = rope_freqs.shape

    if rope_freqs.size(0) != position_dim:
        error_msg = "Expected first dimension of `rope_freqs` and last dimension of "
        error_msg += "key_positions to match, got "
        error_msg += str(rope_freqs.size(0)) + " and " + str(key_positions.size(-1))
        raise ValueError(error_msg)
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even to use RoPE, got {head_dim}")

    # [n_queries*n_keys_per_query, position_dim]
    key_positions_flat = key_positions.reshape(-1, position_dim)

    # [position_dim, n_freq_groups*n_heads*head_dim]
    rope_freqs_flat = rope_freqs.reshape(position_dim, -1)

    # Compute position encoding
    key_pos_encoding = torch.mm(
        key_positions_flat,
        rope_freqs_flat,
    ).view(n_queries, n_keys_per_query, n_freq_groups, n_heads, head_dim)

    # Sum over frequency groups
    key_pos_encoding = key_pos_encoding.sum(
        dim=2
    )  # [n_queries, n_keys_per_query, n_heads, head_dim]

    return key_pos_encoding


@torch.jit.script
def rotate_k(k: Tensor, rope_encoding: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Applies rotary position encoding (RoPE) to the key tensor via
    complex multiplication.

    Args:
        k (Tensor): Key tensor of real dtype and shape
            [n_queries, n_keys_per_query, n_heads, head_dim]
        rope_encoding (Tensor): Position encoding of real dtype and shape
            [n_queries, n_keys_per_query, n_heads, head_dim] or
            [n_queries, n_keys_per_query, 1,       head_dim] (broadcasted over heads)

    Returns:
        - k_rotated (Tensor): Key tensor after rotation, of shape
            [n_queries, n_keys_per_query, n_heads, head_dim] and real dtype
        - keys_complex (Tensor): Complex representation of (unrotated) key tensor,
            of shape [n_queries, n_keys_per_query, n_heads, head_dim/2] and complex dtype.
            Used later in backward pass.
        - rope_encoding_complex: Complex representation of position encoding of shape
            [n_queries, n_keys_per_query, n_heads, head_dim/2] or
            [n_queries, n_keys_per_query, 1,       head_dim/2]
            and complex dtype.
            Used later in backward pass.
    """
    if k.ndim != 4 or rope_encoding.ndim != 4:
        raise ValueError(
            "Expected k and rope_encoding to be 4D, got shapes "
            f"{k.shape} and {rope_encoding.shape}"
        )
    if k.size(-1) % 2 != 0:
        raise ValueError(f"head_dim ({k.size(-1)}) must be even to use RoPE")
    if k.is_complex() or rope_encoding.is_complex():
        raise ValueError(
            "Expected keys and rope_encoding to be real, got dtypes "
            f"{k.dtype}, {rope_encoding.dtype}"
        )

    # Convert to complex and apply rotation
    keys_complex_shape = k.shape[:-1] + (k.size(-1) // 2, 2)
    keys_complex = torch.view_as_complex(k.view(keys_complex_shape))
    rope_encoding_complex_shape = rope_encoding.shape[:-1] + (
        rope_encoding.size(-1) // 2,
        2,
    )
    rope_encoding_complex = torch.view_as_complex(
        rope_encoding.view(rope_encoding_complex_shape)
    )

    # multiply and convert back to real
    keys_rotated = keys_complex * rope_encoding_complex
    keys_rotated = torch.view_as_real(keys_rotated).reshape_as(k)

    # complex tensors are used later in the backward pass
    return keys_rotated, keys_complex, rope_encoding_complex


@torch.jit.script
def rotate_k_backward(
    grad_k_rotated: Tensor,
    k_complex: Tensor,
    rope_encoding_complex: Tensor,
    needs_grad_keys: bool = True,
    needs_grad_rope_encoding: bool = True,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Perform the backward pass of applying rotary positional encoding (RoPE)

    Computes gradients through complex number operations used in the RoPE
    forward pass. For complex multiplication z = x * y, the gradients are:
    dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x).

    Args:
        grad_k_rotated (Tensor): Gradient of loss with respect to rotated keys,
            of shape [n_queries, n_keys_per_query, n_heads, head_dim]
        k_complex (Tensor): Complex representation of the keys from forward pass
            of complex dtype and shape
            [n_queries, n_keys_per_query, n_heads, head_dim/2], as returned from
            rotate_k.
        rope_encoding_complex (Tensor): Complex representation of positional encodings
            of complex dtype and shape
            [n_queries, n_keys_per_query, n_heads, head_dim/2] or
            [n_queries, n_keys_per_query, 1,       head_dim/2], as returned from
            rotate_k.
        needs_grad_keys (bool): Whether gradients for keys are needed. Default: True
        needs_grad_rope_encoding (bool): Whether gradients for positional encodings
            are needed. Default: True

    Returns:
        grad_keys (Tensor): Gradient tensor for the unrotated keys,
            of shape [n_queries, n_keys_per_query, n_heads, head_dim] and real dtype,
            or None if not needed
        grad_rope_encoding (Tensor): Gradient tensor for the positional encodings
            of real dtype and shape
            [n_queries, n_keys_per_query, n_heads, head_dim] or
            [n_queries, n_keys_per_query, 1,       head_dim], or None if not needed
    """
    if grad_k_rotated.ndim != 4 or grad_k_rotated.is_complex():
        raise ValueError(
            "Expected grad_k_rotated to be a 4D real tensor, got "
            f"shape {grad_k_rotated.shape} and dtype {grad_k_rotated.dtype}"
        )
    if k_complex.ndim != 4 or not k_complex.is_complex():
        raise ValueError(
            "Expected k_complex to be a 4D complex tensor, got "
            f"shape {k_complex.shape} and dtype {k_complex.dtype}"
        )
    if rope_encoding_complex.ndim != 4 or not rope_encoding_complex.is_complex():
        raise ValueError(
            "Expected rope_encoding_complex to be a 4D complex tensor, got "
            f"shape {rope_encoding_complex.shape} and dtype {rope_encoding_complex.dtype}"
        )
    if grad_k_rotated.size(-1) % 2 != 0:
        raise ValueError(
            f"head_dim ({grad_k_rotated.size(-1)}) must be even to use RoPE"
        )

    # Check for no grads needed
    if not needs_grad_keys and not needs_grad_rope_encoding:
        # Early return
        return None, None

    # Convert grad_k_rotated to complex
    to_complex_shape = grad_k_rotated.shape[:-1] + (grad_k_rotated.size(-1) // 2, 2)
    grad_k_rotated_complex = torch.view_as_complex(
        grad_k_rotated.view(to_complex_shape)
    )

    # Complex multiplication gradient
    # For z = x * y, we have dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x)
    if needs_grad_keys:
        grad_k_complex = grad_k_rotated_complex * rope_encoding_complex.conj()
        grad_keys = torch.view_as_real(grad_k_complex).reshape_as(grad_k_rotated)
    else:
        grad_keys = None

    if needs_grad_rope_encoding:
        grad_key_pos_complex = grad_k_rotated_complex * k_complex.conj()
        grad_rope_encoding = torch.view_as_real(grad_key_pos_complex).reshape_as(
            grad_k_rotated
        )
    else:
        grad_rope_encoding = None

    return grad_keys, grad_rope_encoding


@torch.jit.script
def calculate_rope_backward(
    grad_key_pos_encoding: Tensor,
    key_positions: Tensor,
    rope_freqs: Tensor,
    needs_grad_key_positions: bool,
    needs_grad_rope_freqs: bool,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Calculates gradients for the calculate_rope function.

    This function implements the backward pass for the calculation of the rotary
    positional encoding tensor that gets multiplied with the query/key tensor. It
    propagates the gradients from key_pos_encoding to key_positions and rope_freqs.

    Args:
        grad_key_pos_encoding (Tensor): Real-valued gradient of loss with respect to
            the positional encoding, of shape
            [n_queries, n_keys_per_query, n_heads, head_dim]
        key_positions (Tensor): Position tensor from the forward pass, of shape
            [n_queries, n_keys_per_query, position_dim]
        rope_freqs (Tensor): Frequency values tensor from the forward pass, of shape
            [position_dim, n_freq_groups, n_heads, head_dim], with n_freq_groups and/or
            n_heads also allowed to be 1.
        needs_grad_key_positions (bool): Whether grad for key_positions is required
        needs_grad_rope_freqs (bool): Whether grad for rope_freqs is required

    Returns:
        tuple[Optional[Tensor], Optional[Tensor]]:
            - grad_key_positions: Gradient tensor for key positions of shape
              [n_queries, n_keys_per_query, position_dim], or None if not needed
            - grad_rope_freqs: Gradient tensor for rope frequencies of same
              shape as input tensor rope_freqs, or None if not needed
    """
    if key_positions.ndim != 3:
        raise ValueError(
            f"Expected 3 dimensions for `key_positions`, got {key_positions.ndim}"
        )
    if rope_freqs.ndim != 4:
        raise ValueError(
            f"Expected 4 dimensions for `rope_freqs`, got {rope_freqs.ndim}"
        )
    if grad_key_pos_encoding.ndim != 4:
        raise ValueError(
            f"Expected 4 dimensions for `grad_key_pos_encoding`, got {grad_key_pos_encoding.ndim}"
        )

    n_queries, n_keys_per_query, position_dim = key_positions.shape
    position_dim_freqs, n_freq_groups, n_heads, head_dim = rope_freqs.shape

    # potentially different than n_heads if rope_freqs was broadcasted over heads
    expanded_n_heads = grad_key_pos_encoding.size(2)

    if position_dim_freqs != position_dim:
        error_msg = "Expected first dimension of `rope_freqs` and last dimension of "
        error_msg += "key_positions to match, got "
        error_msg += f"{position_dim_freqs} and {position_dim}"
        raise ValueError(error_msg)

    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even to use RoPE, got {head_dim}")

    # Backward of sum: distribute gradient across n_freq_groups
    grad_mm_result = grad_key_pos_encoding.unsqueeze(2).expand(
        -1, -1, n_freq_groups, -1, -1
    )

    # Reshape to match the mm result
    grad_mm_result = grad_mm_result.reshape(
        n_queries * n_keys_per_query, n_freq_groups * expanded_n_heads * head_dim
    )

    # expand rope_freqs to account for broadcasting
    expanded_rope_freqs = rope_freqs.expand(-1, -1, expanded_n_heads, -1)

    # Flatten inputs as in forward pass
    key_positions_flat = key_positions.reshape(-1, position_dim)
    expanded_rope_freqs_flat = expanded_rope_freqs.reshape(position_dim, -1)

    # Gradient for matrix multiplication: If C = A @ B
    # Then grad_A = grad_C @ B^T and grad_B = A^T @ grad_C
    if needs_grad_key_positions:
        grad_key_positions_flat = torch.mm(grad_mm_result, expanded_rope_freqs_flat.t())
        grad_key_positions = grad_key_positions_flat.view(
            n_queries, n_keys_per_query, position_dim
        )
    else:
        grad_key_positions = None

    if needs_grad_rope_freqs:
        grad_rope_freqs_flat = torch.mm(key_positions_flat.t(), grad_mm_result)
        grad_rope_freqs = grad_rope_freqs_flat.view(
            position_dim, n_freq_groups, expanded_n_heads, head_dim
        )

        # handle broadcasting case
        if n_heads == 1 and expanded_n_heads > 1:
            grad_rope_freqs = grad_rope_freqs.sum(2, keepdim=True)
    else:
        grad_rope_freqs = None

    return grad_key_positions, grad_rope_freqs
