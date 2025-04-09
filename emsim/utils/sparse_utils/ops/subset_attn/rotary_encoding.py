from typing import Optional

import torch
from torch import Tensor


@torch.jit.script
def calculate_rope(key_positions: Tensor, rope_freqs: Tensor) -> Tensor:
    """Computes the positional encoding for keys using the provided positions and frequency values.

    This function calculates the position encoding by matrix-multiplying key
    positions with rotary frequency encodings, then summing over frequency
    groups.
    The returned positional encoding will be in real space, and must be converted
    to complex coordinates with e.g. torch.polar before multiplying with the
    complex representation of the key embedding.
    This function may be used in combination with the others in its module for a
    memory-efficient RoPE application over many positions.
    This implementation allows for grouping of position dimensions into specific
    frequency groups. The intention is to allow dimensions with potentially different
    spatial characteristics (e.g., x and y vs time for videos) to be grouped
    separately. This generalization is experimental and under active research.
    If dimension i is not in frequency group j, then rope_freqs[i, j] should be 0.
    For traditional RoPE, keep n_freq_groups as 1.

    Args:
        key_positions (Tensor): Position information for each key of shape
            [n_queries, n_keys_per_query, position_dim], where position_dim is the
            dimensionality of the position representation.
        rope_freqs (Tensor): Frequency values for rotary encodings of shape
            [position_dim, n_freq_groups, n_heads, head_dim/2], where n_freq_groups
            and n_heads can be 1 for broadcasting.

    Returns:
        Tensor: Computed positional encoding of shape
            [n_queries, n_keys_per_query, n_heads, head_dim/2]
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
    position_dim_freqs, n_freq_groups, n_heads, half_head_dim = rope_freqs.shape

    if position_dim_freqs != position_dim:
        raise ValueError(
            "Expected first dimension of `rope_freqs` and last dimension of "
            "key_positions to match, got "
            f"{rope_freqs.size(0)} and {key_positions.size(-1)}"
        )

    # [n_queries*n_keys_per_query, position_dim]
    key_positions_flat = key_positions.reshape(-1, position_dim)

    # [position_dim, n_freq_groups*n_heads*head_dim/2]
    rope_freqs_flat = rope_freqs.reshape(position_dim, -1)

    # Compute position encoding
    key_rope_encoding = torch.mm(
        key_positions_flat,
        rope_freqs_flat,
    ).view(n_queries, n_keys_per_query, n_freq_groups, n_heads, half_head_dim)

    # Sum over frequency groups
    # [n_queries, n_keys_per_query, n_heads, head_dim/2]
    key_rope_encoding = key_rope_encoding.sum(dim=2)
    return key_rope_encoding


@torch.jit.script
def rotate_keys(
    keys: Tensor, key_rope_encoding: Tensor, needs_autograd: bool = True
) -> Tensor:
    """Applies rotary position encoding (RoPE) to the key tensor via
    complex multiplication.

    Args:
        keys (Tensor): Key tensor of real dtype and shape
            [n_queries, n_keys_per_query, n_heads, head_dim]
        key_rope_encoding (Tensor): Position encoding of real dtype and shape
            [n_queries, n_keys_per_query, n_heads, head_dim/2] or
            [n_queries, n_keys_per_query, 1,       head_dim/2] (broadcasted over heads)
        needs_autograd (bool): If you need this function to be tracked by autograd,
            keep this at True. If False, additional autograd-incompatible
            memory optimizations are applied. The function will fail in the backward
            pass if this option is False, so the optimizations are not applied by
            default for safety.

    Returns:
        - keys_rotated (Tensor): Key tensor after rotation, of shape
            [n_queries, n_keys_per_query, n_heads, head_dim] and real dtype
    """
    if keys.ndim != 4 or key_rope_encoding.ndim != 4:
        raise ValueError(
            "Expected keys and key_rope_encoding to be 4D, got shapes "
            f"{keys.shape} and {key_rope_encoding.shape}"
        )
    if keys.size(-1) != key_rope_encoding.size(-1) * 2:
        raise ValueError(
            "Expected key_rope_encoding to have last dimension equal to half of keys's"
            f"head dim, got {key_rope_encoding.size(-1)} and {keys.size(-1)}"
        )
    if keys.is_complex() or key_rope_encoding.is_complex():
        raise ValueError(
            "Expected keys and key_rope_encoding to be real, got dtypes "
            f"{keys.dtype}, {key_rope_encoding.dtype}"
        )

    # Convert to complex and apply rotation
    keys_complex_shape = keys.shape[:-1] + (keys.size(-1) // 2, 2)
    keys_complex = torch.view_as_complex(keys.view(keys_complex_shape))
    key_rope_encoding_complex = torch.polar(
        torch.ones_like(key_rope_encoding),
        key_rope_encoding,
    )

    # multiply and convert back to real
    if needs_autograd:
        keys_rotated = keys_complex * key_rope_encoding_complex
    else:
        # can use an in-place op rather than creating a new tensor
        keys_rotated = keys_complex
        keys_rotated *= key_rope_encoding_complex
    keys_rotated = torch.view_as_real(keys_rotated).reshape_as(keys)

    return keys_rotated


@torch.jit.script
def rotate_keys_backward(
    grad_keys_rotated: Tensor,
    keys: Tensor,
    key_rope_encoding: Tensor,
    needs_grad_keys: bool = True,
    needs_grad_rope_encoding: bool = True,
    needs_autograd: bool = True,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Perform the backward pass of applying rotary positional encoding (RoPE)

    Computes gradients through complex number operations used in the RoPE
    forward pass. For complex multiplication z = x * y, the gradients are:
    dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x).

    Args:
        grad_keys_rotated (Tensor): Gradient of loss with respect to rotated keys,
            of shape [n_queries, n_keys_per_query, n_heads, head_dim]
        keys (Tensor): Original, un-rotated key tensor of real dtype and shape
            [n_queries, n_keys_per_query, n_heads, head_dim].
        key_rope_encoding (Tensor): Real representation of positional encodings
            of real dtype and shape
            [n_queries, n_keys_per_query, n_heads, head_dim/2] or
            [n_queries, n_keys_per_query, 1,       head_dim/2]
        needs_grad_keys (bool): Whether gradients for keys are needed. Default: True
        needs_grad_rope_encoding (bool): Whether gradients for positional encodings
            are needed. Default: True
        needs_autograd (bool): If you need this function to be tracked by autograd,
            keep this at True. If False, additional autograd-incompatible
            memory optimizations are applied. The function will fail in the backward
            pass if this option is False, so the optimizations are not applied by
            default for safety.

    Returns:
        grad_keys (Tensor): Gradient tensor for the unrotated keys,
            of shape [n_queries, n_keys_per_query, n_heads, head_dim] and real dtype,
            or None if not needed
        grad_key_rope_encoding (Tensor): Gradient tensor for the positional encodings
            of real dtype and shape
            [n_queries, n_keys_per_query, n_heads, head_dim/2] or
            [n_queries, n_keys_per_query, 1,       head_dim/2], or None if not needed
    """
    if grad_keys_rotated.ndim != 4 or grad_keys_rotated.is_complex():
        raise ValueError(
            "Expected grad_k_rotated to be a 4D real tensor, got "
            f"shape {grad_keys_rotated.shape} and dtype {grad_keys_rotated.dtype}"
        )
    if keys.ndim != 4 or keys.is_complex():
        raise ValueError(
            "Expected keys to be a 4D real tensor, got "
            f"shape {keys.shape} and dtype {keys.dtype}"
        )
    if key_rope_encoding.ndim != 4 or key_rope_encoding.is_complex():
        raise ValueError(
            "Expected key_rope_encoding to be a 4D real tensor, got "
            f"shape {key_rope_encoding.shape} and dtype {key_rope_encoding.dtype}"
        )

    # Check for no grads needed
    if not needs_grad_keys and not needs_grad_rope_encoding:
        # Early return
        return None, None

    # Convert grad_k_rotated to complex
    to_complex_shape = grad_keys_rotated.shape[:-1] + (
        grad_keys_rotated.size(-1) // 2,
        2,
    )
    grad_keys_rotated_complex = torch.view_as_complex(
        grad_keys_rotated.view(to_complex_shape)
    )

    # Complex multiplication gradient
    # For z = x * y, we have dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x)

    # Unconditionally recompute complex version of key_rope_encoding tensor since it's
    # required by both branches
    key_rope_encoding_complex = torch.polar(
        torch.ones_like(key_rope_encoding),
        key_rope_encoding,
    )

    # Gradient for key tensor
    if needs_grad_keys:
        if needs_autograd or needs_grad_rope_encoding:
            grad_keys_complex = (
                grad_keys_rotated_complex * key_rope_encoding_complex.conj()
            )
        else:
            # Can modify tensor in-place rather than creating a new one
            # Need to check needs_grad_rope_encoding because we'll need
            # grad_keys_rotated_complex in that branch
            grad_keys_complex = grad_keys_rotated_complex
            grad_keys_complex *= key_rope_encoding_complex.conj()
        grad_keys = torch.view_as_real(grad_keys_complex).reshape_as(grad_keys_rotated)
    else:
        grad_keys = None

    # Gradient for position encoding
    if needs_grad_rope_encoding:
        # Recompute complex version of key tensor
        keys_complex_shape = keys.shape[:-1] + (keys.size(-1) // 2, 2)
        keys_complex = torch.view_as_complex(keys.view(keys_complex_shape))

        # Compute gradient with respect to key_rope_encoding_complex
        if needs_autograd:
            grad_key_rope_encoding_complex = (
                grad_keys_rotated_complex * keys_complex.conj()
            )
        else:
            # Can modify tensor in-place rather than creating a new one
            grad_key_rope_encoding_complex = grad_keys_rotated_complex
            grad_key_rope_encoding_complex *= keys_complex.conj()

        # Check if broadcasting happened
        is_broadcasted = (
            key_rope_encoding_complex.size(2) == 1 and keys_complex.size(2) > 1
        )

        if is_broadcasted:
            # Sum gradients across broadcasted dimension (heads)
            grad_key_rope_encoding_complex = grad_key_rope_encoding_complex.sum(
                dim=2, keepdim=True
            )

        # Then compute gradient with respect to key_rope_encoding (the phase angle)
        # Since key_rope_encoding_complex = exp(i*key_rope_encoding), the gradient is:
        # dL/d(key_rope_encoding)
        #   = Im(dL/d(key_rope_encoding_complex) / key_rope_encoding_complex)
        if needs_autograd:
            grad_key_rope_encoding = (
                grad_key_rope_encoding_complex / key_rope_encoding_complex
            ).imag
        else:
            # Can modify tensor in-place rather than creating a new one
            grad_key_rope_encoding = grad_key_rope_encoding_complex
            grad_key_rope_encoding /= key_rope_encoding_complex
            grad_key_rope_encoding = grad_key_rope_encoding.imag
    else:
        grad_key_rope_encoding = None

    return grad_keys, grad_key_rope_encoding


@torch.jit.script
def calculate_rope_backward(
    grad_key_rope_encoding: Tensor,
    key_positions: Tensor,
    rope_freqs: Tensor,
    needs_grad_key_positions: bool,
    needs_grad_rope_freqs: bool,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Calculates gradients for the calculate_rope function.

    This function implements the backward pass for the calculation of the rotary
    positional encoding tensor that gets multiplied with the query/key tensor. It
    propagates the gradients from key_rope_encoding to key_positions and rope_freqs.

    Args:
        grad_key_rope_encoding (Tensor): Real-valued gradient of loss with respect to
            the positional encoding, of shape
            [n_queries, n_keys_per_query, n_heads, head_dim/2]
        key_positions (Tensor): Position tensor from the forward pass, of shape
            [n_queries, n_keys_per_query, position_dim]
        rope_freqs (Tensor): Frequency values tensor from the forward pass, of shape
            [position_dim, n_freq_groups, n_heads, head_dim/2], with n_freq_groups
            and/or n_heads also allowed to be 1.
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
    if grad_key_rope_encoding.ndim != 4:
        raise ValueError(
            f"Expected 4 dimensions for `grad_key_rope_encoding`, got {grad_key_rope_encoding.ndim}"
        )

    # Check for no grads needed
    if not needs_grad_key_positions and not needs_grad_rope_freqs:
        # Early return
        return None, None

    n_queries, n_keys_per_query, position_dim = key_positions.shape
    position_dim_freqs, n_freq_groups, n_heads, half_head_dim = rope_freqs.shape

    # potentially different than n_heads if rope_freqs was broadcasted over heads
    expanded_n_heads = grad_key_rope_encoding.size(2)

    if position_dim_freqs != position_dim:
        raise ValueError(
            "Expected first dimension of `rope_freqs` and last dimension of "
            "key_positions to match, got "
            f"{position_dim_freqs} and {position_dim}"
        )

    # Backward of sum: distribute gradient across n_freq_groups
    grad_mm_result = grad_key_rope_encoding.unsqueeze(2).expand(
        -1, -1, n_freq_groups, -1, -1
    )

    # Reshape to match the mm result
    grad_mm_result = grad_mm_result.reshape(
        n_queries * n_keys_per_query, n_freq_groups * expanded_n_heads * half_head_dim
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
            position_dim, n_freq_groups, expanded_n_heads, half_head_dim
        )

        # handle broadcasting case
        if n_heads == 1 and expanded_n_heads > 1:
            grad_rope_freqs = grad_rope_freqs.sum(2, keepdim=True)
    else:
        grad_rope_freqs = None

    return grad_key_positions, grad_rope_freqs
