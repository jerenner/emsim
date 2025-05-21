from typing import Optional

import torch
from torch import Tensor


@torch.jit.script
def _validate_real(tensor: Tensor, name: str) -> None:
    if tensor.is_complex():
        raise ValueError(f"Expected {name} to be real, got dtype {tensor.dtype}")


@torch.jit.script
def _validate_at_least_nd(tensor: Tensor, name: str, min_dims: int) -> None:
    if tensor.ndim < min_dims:
        raise ValueError(
            "Expected at least "
            f"{min_dims} dimensions for {name}, got shape {tensor.shape}"
        )


@torch.jit.script
def _validate_4d(tensor: Tensor, name: str) -> None:  # for rope frequences
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4 dimensions for {name}, got shape {tensor.shape}")


@torch.jit.script
def _validate_same_ndims(
    tensor_1: Tensor, name_1: str, tensor_2: Tensor, name_2: str
) -> None:
    if tensor_1.ndim != tensor_2.ndim:
        raise ValueError(
            "Expected " + name_1 + " and " + name_2 + " to have the same number of "
            f"dims, got shapes {tensor_1.shape} and {tensor_2.shape}",
        )


@torch.jit.script
def _validate_head_dim(embeddings: Tensor, rope_encoding: Tensor) -> None:
    if embeddings.size(-1) != rope_encoding.size(-1) * 2:
        raise ValueError(
            "Expected rope_encoding to have last dimension equal to 1/2 embedding's "
            f"head dim, got {rope_encoding.size(-1)} and {embeddings.size(-1)}"
        )


@torch.jit.script
def _validate_position_dim(positions: Tensor, rope_freqs: Tensor) -> None:
    if positions.size(-1) != rope_freqs.size(0):
        raise ValueError(
            "Expected first dimension of `rope_freqs` and last dimension of "
            "positions to match, got "
            f"{rope_freqs.size(0)} and {positions.size(-1)}"
        )


@torch.jit.script
def calculate_rope(positions: Tensor, rope_freqs: Tensor) -> Tensor:
    """Computes the positional encoding for embeddings tensors using the provided
    positions and frequency values.

    This function calculates the rotary position encoding by matrix-multiplying
    embedding positions with rotary frequency encodings, then summing over frequency
    groups.
    The returned positional encoding will be in real space, and must be converted
    to complex coordinates with e.g. torch.polar before multiplying with the
    complex representation of the embedding tensor (this conversion is handled by
    rotate_embeddings).
    This function may be used in combination with the others in its module for a
    memory-efficient RoPE application over many positions.
    This implementation allows for grouping of position dimensions into specific
    frequency groups. The intention is to allow dimensions with potentially different
    spatial characteristics (e.g., x and y vs time for videos) to be grouped
    separately. This generalization is experimental and under active research.
    If dimension i is not in frequency group j, then rope_freqs[i, j] should be 0.
    For traditional RoPE, keep n_freq_groups as 1.

    Args:
        positions (Tensor): Position information for each embedding element of shape
            [..., position_dim], where ... are arbitrary batch dimensions and
            position_dim is the dimensionality of the position representation.
        rope_freqs (Tensor): Frequency values for rotary encodings of shape
            [position_dim, n_freq_groups, n_heads, head_dim/2], where n_freq_groups
            and n_heads can be 1 for broadcasting.

    Returns:
        Tensor: Computed positional encoding of shape
            [..., n_heads, head_dim/2]
    """
    _validate_at_least_nd(positions, "positions", 2)
    _validate_4d(rope_freqs, "rope_freqs")
    _validate_position_dim(positions, rope_freqs)

    batch_dims = positions.shape[:-1]
    position_dim = positions.size(-1)
    _, n_freq_groups, n_heads, half_head_dim = rope_freqs.shape

    # flatten batch dimensions
    positions_flat = positions.reshape(-1, position_dim)

    # [position_dim, n_freq_groups*n_heads*head_dim/2]
    rope_freqs_flat = rope_freqs.reshape(position_dim, -1)

    # Compute position encoding
    rope_encoding = torch.mm(positions_flat, rope_freqs_flat)
    # shape: [prod(batch_dims), n_freq_groups*n_heads*head_dim/2]

    # reshape back to input batch dims
    output_shape = batch_dims + (n_freq_groups, n_heads, half_head_dim)
    rope_encoding = rope_encoding.view(output_shape)

    # Sum over frequency groups
    # [*batch_dims, n_heads, head_dim/2]
    rope_encoding = rope_encoding.sum(dim=-3)
    return rope_encoding


@torch.jit.script
def _upcast_if_needed(tensor: Tensor) -> Tensor:
    """Helper function to upcast a tensor to float32 for complex number operations
    if in lower precision. Pytorch complex number ops currently only support
    32-bit or higher."""
    if tensor.dtype not in (torch.float32, torch.float64):
        return tensor.to(torch.float32)
    return tensor


@torch.jit.script
def rotate_embeddings(
    embeddings: Tensor, rope_encoding: Tensor, needs_autograd: bool = True
) -> Tensor:
    """Applies rotary position encoding (RoPE) to the embeddings tensor via
    complex multiplication.

    Args:
        embeddings (Tensor): Embeddings tensor to be rotated (usually a query or
            key tensor) of real dtype and shape [..., n_heads, head_dim]
        rope_encoding (Tensor): Position encoding of real dtype and shape
            [..., n_heads, head_dim/2] or
            [..., 1,       head_dim/2] (broadcasted over heads)
        needs_autograd (bool): If you need this function to be tracked by autograd,
            keep this at True. If False, additional autograd-incompatible
            memory optimizations are applied. The function will fail in the backward
            pass if this option is False, so the optimizations are not applied by
            default for safety.

    Returns:
        - embeddings_rotated (Tensor): Embedding tensor after rotation, of shape
            [..., n_heads, head_dim] and real dtype

    Note:
        - This function uses Pytorch's complex number operations, which only support
            single and double precision. If `embeddings` and `rope_encoding` are
            half precision or lower, they are temporarily upcasted to float32 for
            this function, and the output is downcasted back to `embeddings`'s original
            dtype before returning it.
    """
    _validate_same_ndims(embeddings, "embeddings", rope_encoding, "rope_encoding")
    _validate_at_least_nd(embeddings, "embeddings", 3)
    _validate_real(embeddings, "embeddings")
    _validate_real(rope_encoding, "rope_encoding")
    _validate_head_dim(embeddings, rope_encoding)

    # Save original dtype
    embeddings_dtype = embeddings.dtype

    # Upcast if needed
    embeddings = _upcast_if_needed(embeddings)
    rope_encoding = _upcast_if_needed(rope_encoding)

    # Convert to complex and apply rotation
    emb_complex_shape = embeddings.shape[:-1] + (embeddings.size(-1) // 2, 2)
    embeddings_complex = torch.view_as_complex(embeddings.reshape(emb_complex_shape))
    rope_encoding_complex = torch.polar(torch.ones_like(rope_encoding), rope_encoding)

    # multiply and convert back to real
    if needs_autograd:
        embeddings_rotated = embeddings_complex * rope_encoding_complex
    else:
        # can use an in-place op rather than creating a new tensor
        embeddings_rotated = embeddings_complex
        embeddings_rotated *= rope_encoding_complex
    embeddings_rotated = torch.view_as_real(embeddings_rotated).reshape_as(embeddings)

    # Cast back
    embeddings_rotated = embeddings_rotated.to(embeddings_dtype)

    return embeddings_rotated


@torch.jit.script
def rotate_embeddings_backward(
    grad_embeddings_rotated: Tensor,
    embeddings: Tensor,
    rope_encoding: Tensor,
    needs_grad_embeddings: bool = True,
    needs_grad_rope_encoding: bool = True,
    needs_autograd: bool = True,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Perform the backward pass of applying rotary positional encoding (RoPE)

    Computes gradients through complex number operations used in the RoPE
    forward pass. For complex multiplication z = x * y, the gradients are:
    dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x).

    Args:
        grad_embeddings_rotated (Tensor): Gradient of loss with respect to rotated
            embeddings, of shape [..., n_heads, head_dim]
        embeddings (Tensor): Original, un-rotated embeddings tensor of real dtype and
            shape [..., n_heads, head_dim].
        rope_encoding (Tensor): Real representation of positional encodings
            of real dtype and shape
            [..., n_heads, head_dim/2] or
            [..., 1,       head_dim/2]
        needs_grad_embeddings (bool): Whether gradients for embeddings are needed.
            Default: True
        needs_grad_rope_encoding (bool): Whether gradients for positional encodings
            are needed. Default: True
        needs_autograd (bool): If you need this function to be tracked by autograd,
            keep this at True. If False, additional autograd-incompatible
            memory optimizations are applied. The function will fail in the backward
            pass if this option is False, so the optimizations are not applied by
            default for safety.

    Returns:
        grad_embeddings (Tensor): Gradient tensor for the unrotated embeddings,
            of shape [..., n_heads, head_dim] and real dtype,
            or None if not needed
        grad_rope_encoding (Tensor): Gradient tensor for the positional encodings
            of real dtype and shape
            [..., n_heads, head_dim/2] or
            [..., 1,       head_dim/2], or None if not needed

    Note:
        - This function uses Pytorch's complex number operations, which only support
            single and double precision. If any of the input tensors are
            half precision or lower, they are temporarily upcasted to float32 for
            this function, and the output gradients are downcasted back to the original
            dtype of `embeddings` and `rope_encoding`, respectively, before returning
            them.
    """
    _validate_same_ndims(embeddings, "embeddings", rope_encoding, "rope_encoding")
    _validate_real(grad_embeddings_rotated, "grad_embeddings_rotated")
    _validate_real(embeddings, "embeddings")
    _validate_real(rope_encoding, "rope_encoding")

    if grad_embeddings_rotated.shape != embeddings.shape:
        raise ValueError(
            "Expected grad_embeddings_rotated and embeddings to have the same shape, "
            f"got {grad_embeddings_rotated.shape} and {embeddings.shape}"
        )
    _validate_head_dim(embeddings, rope_encoding)

    # Check for no grads needed
    if not needs_grad_embeddings and not needs_grad_rope_encoding:
        # Early return
        return None, None

    # Save input dtypes
    embeddings_dtype = embeddings.dtype
    rope_encoding_dtype = rope_encoding.dtype

    # Upcast if needed
    grad_embeddings_rotated = _upcast_if_needed(grad_embeddings_rotated)
    embeddings = _upcast_if_needed(embeddings)
    rope_encoding = _upcast_if_needed(rope_encoding)

    # Convert grad_tensor_rotated to complex
    to_complex_shape = grad_embeddings_rotated.shape[:-1] + (
        grad_embeddings_rotated.size(-1) // 2,
        2,
    )
    grad_embeddings_rotated_complex = torch.view_as_complex(
        grad_embeddings_rotated.reshape(to_complex_shape)
    )

    # Complex multiplication gradient
    # For z = x * y, we have dL/dx = dL/dz * conj(y) and dL/dy = dL/dz * conj(x)

    # Unconditionally recompute complex version of rope_encoding tensor since it's
    # required by both output grads
    rope_encoding_complex = torch.polar(torch.ones_like(rope_encoding), rope_encoding)

    # Gradient for embeddings tensor
    if needs_grad_embeddings:
        if needs_autograd or needs_grad_rope_encoding:
            grad_emb_complex = (
                grad_embeddings_rotated_complex * rope_encoding_complex.conj()
            )
        else:
            # Can modify tensor in-place rather than creating a new one
            # Need to check needs_grad_rope_encoding because we'll need
            # grad_embeddings_rotated_complex in that branch
            grad_emb_complex = grad_embeddings_rotated_complex
            grad_emb_complex *= rope_encoding_complex.conj()
        grad_embeddings = torch.view_as_real(grad_emb_complex).reshape_as(
            grad_embeddings_rotated
        )
        grad_embeddings = grad_embeddings.to(embeddings_dtype)  # downcast
    else:
        grad_embeddings = None

    # Gradient for position encoding
    if needs_grad_rope_encoding:
        # Recompute complex version of embeddings tensor
        emb_complex_shape = embeddings.shape[:-1] + (embeddings.size(-1) // 2, 2)
        embeddings_complex = torch.view_as_complex(
            embeddings.reshape(emb_complex_shape)
        )

        # Compute gradient with respect to rope_encoding_complex
        if needs_autograd:
            grad_rope_encoding_complex = (
                grad_embeddings_rotated_complex * embeddings_complex.conj()
            )
        else:
            # Can modify tensor in-place rather than creating a new one
            grad_rope_encoding_complex = grad_embeddings_rotated_complex
            grad_rope_encoding_complex *= embeddings_complex.conj()

        # Check if broadcasting happened
        is_broadcasted = (
            rope_encoding_complex.size(-2) == 1 and embeddings_complex.size(-2) > 1
        )

        if is_broadcasted:
            # Sum gradients across broadcasted dimension (heads)
            grad_rope_encoding_complex = grad_rope_encoding_complex.sum(
                dim=-2, keepdim=True
            )

        # Then compute gradient with respect to rope_encoding (the phase angle)
        # Since rope_encoding_complex = exp(i*rope_encoding), the gradient is:
        # dL/d(rope_encoding)
        #   = Im(dL/d(rope_encoding_complex) / rope_encoding_complex)
        if needs_autograd:
            grad_rope_encoding = (
                grad_rope_encoding_complex / rope_encoding_complex
            ).imag
        else:
            # Can modify tensor in-place rather than creating a new one
            grad_rope_encoding = grad_rope_encoding_complex
            grad_rope_encoding /= rope_encoding_complex
            grad_rope_encoding = grad_rope_encoding.imag

        grad_rope_encoding = grad_rope_encoding.to(rope_encoding_dtype)  # downcast
    else:
        grad_rope_encoding = None

    return grad_embeddings, grad_rope_encoding


@torch.jit.script
def calculate_rope_backward(
    grad_rope_encoding: Tensor,
    positions: Tensor,
    rope_freqs: Tensor,
    needs_grad_positions: bool,
    needs_grad_rope_freqs: bool,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Calculates gradients for the calculate_rope function.

    This function implements the backward pass for the calculation of the rotary
    positional encoding tensor that gets multiplied with the query/key tensor. It
    propagates the gradients from rope_encoding to positions and rope_freqs.

    Args:
        grad_rope_encoding (Tensor): Real-valued gradient of loss with respect to
            the positional encoding, of shape
            [..., n_heads, head_dim/2]
        positions (Tensor): Position tensor from the forward pass, of shape
            [..., position_dim]
        rope_freqs (Tensor): Frequency values tensor from the forward pass, of shape
            [position_dim, n_freq_groups, n_heads, head_dim/2], with n_freq_groups
            and/or n_heads also allowed to be 1.
        needs_grad_positions (bool): Whether grad for positions is required
        needs_grad_rope_freqs (bool): Whether grad for rope_freqs is required

    Returns:
        tuple[Optional[Tensor], Optional[Tensor]]:
            - grad_positions: Gradient tensor for positions of shape
              [..., position_dim], or None if not needed
            - grad_rope_freqs: Gradient tensor for rope frequencies of same
              shape as input tensor rope_freqs, or None if not needed
    """
    _validate_at_least_nd(positions, "positions", 2)
    _validate_at_least_nd(grad_rope_encoding, "grad_rope_encoding", 3)
    _validate_4d(rope_freqs, "rope_freqs")
    _validate_position_dim(positions, rope_freqs)
    if grad_rope_encoding.shape[:-2] != positions.shape[:-1]:
        raise ValueError(
            "Expected matching batch dims for grad_rope_encoding (first n-2 dims) "
            "and positions (first n-1 dims), got "
            f"shapes {grad_rope_encoding.shape} and {positions.shape}"
        )

    batch_dims = positions.shape[:-1]
    position_dim = positions.size(-1)
    _, n_freq_groups, n_heads, half_head_dim = rope_freqs.shape

    # Check for no grads needed
    if not needs_grad_positions and not needs_grad_rope_freqs:
        # Early return
        return None, None

    # potentially different than n_heads if rope_freqs was broadcasted over heads
    expanded_n_heads = grad_rope_encoding.size(-2)

    # Backward of sum: distribute gradient across n_freq_groups
    grad_mm_result = grad_rope_encoding.unsqueeze(-3).expand(
        (-1,) * len(batch_dims) + (n_freq_groups, -1, -1)
    )

    # Reshape to match the mm result
    grad_mm_result = grad_mm_result.reshape(
        -1, n_freq_groups * expanded_n_heads * half_head_dim
    )

    # expand rope_freqs to account for broadcasting
    expanded_rope_freqs = rope_freqs.expand(-1, -1, expanded_n_heads, -1)

    # Flatten inputs as in forward pass
    positions_flat = positions.reshape(-1, position_dim)
    expanded_rope_freqs_flat = expanded_rope_freqs.reshape(position_dim, -1)

    # Gradient for matrix multiplication: If C = A @ B
    # Then grad_A = grad_C @ B^T and grad_B = A^T @ grad_C
    if needs_grad_positions:
        grad_positions_flat = torch.mm(grad_mm_result, expanded_rope_freqs_flat.t())
        grad_positions = grad_positions_flat.view(batch_dims + (position_dim,))
    else:
        grad_positions = None

    if needs_grad_rope_freqs:
        grad_rope_freqs_flat = torch.mm(positions_flat.t(), grad_mm_result)
        grad_rope_freqs = grad_rope_freqs_flat.view(
            position_dim, n_freq_groups, expanded_n_heads, half_head_dim
        )

        # handle broadcasting case
        if n_heads == 1 and expanded_n_heads > 1:
            grad_rope_freqs = grad_rope_freqs.sum(2, keepdim=True)
    else:
        grad_rope_freqs = None

    return grad_positions, grad_rope_freqs
