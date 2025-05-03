import warnings
from typing import Optional, Union

from emsim.utils.sparse_utils.ops.subset_attn.rotary_encoding import (
    calculate_rope,
    rotate_embeddings,
)
from emsim.utils.sparse_utils.validation import validate_nd
from emsim.utils.misc_utils import can_broadcast_shapes

import torch
from torch import Tensor, nn

# Based on code from
# https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py


def _validate_head_dim_even(head_dim: int):
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")


def init_2d_freqs_rope_mixed_orig(
    head_dim: int,
    num_heads: int,
    theta: float = 10.0,
    rotate: bool = True,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Slightly modified version of the original RoPE-Mixed initialization function."""
    freqs_x = []
    freqs_y = []
    mag = 1 / (
        theta ** (torch.arange(0, head_dim, 4, dtype=dtype, device=device) / head_dim)
    )
    for _ in range(num_heads):
        angles = (
            torch.rand(1, device=device, dtype=dtype) * 2 * torch.pi
            if rotate
            else torch.zeros(1, device=device, dtype=dtype)
        )
        fx = torch.cat(
            [mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1
        )
        fy = torch.cat(
            [mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=-1)
    return freqs  # n_head, head_dim/2, 2


def init_2d_freqs_rope_mixed(
    head_dim: int,
    n_heads: int,
    theta: float = 10.0,
    rotate: bool = True,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Initializes frequency parameters for 2D rotary position embeddings.

    Generates the frequencies used for RoPE (Rotary Position Embeddings) in two dimensions.
    For each head, creates frequency vectors that incorporate both magnitude decay based
    on theta and optional random per-head rotation angles.

    Args:
        head_dim (int): Dimension size of each attention head. Must be divisible by 2.
        n_heads (int): Number of attention heads.
        theta (float): Base value for frequency scaling. Larger values result in longer
            period sinusoids. Default: 10.0
        rotate (bool): Whether to apply random rotation to the frequency vectors.
            When True, each head gets different random rotations. Default: True
        dtype (Optional[torch.dtype]): Data type for the output tensor. Default: None
        device (Optional[torch.device]): Device for the output tensor. Default: None

    Returns:
        Tensor: Frequency parameter tensor of shape [2, n_heads, head_dim/2], containing
            the frequency parameters for x and y dimensions for each attention head.

    Raises:
        ValueError: If head_dim is not divisible by 2.
    """
    _validate_head_dim_even(head_dim)

    # Create frequency magnitudes that decay with head_dim index
    dim_t = torch.arange(0, head_dim, 2, dtype=dtype, device=device)
    dim_t = theta ** (dim_t / head_dim)

    freqs = torch.zeros(2, n_heads, head_dim // 2, device=device, dtype=dtype)
    for dim_index in range(freqs.size(0)):
        for head_index in range(n_heads):
            angle = (
                torch.rand(1, device=device, dtype=dtype) * 2 * torch.pi
                if rotate
                else torch.zeros(1, device=device, dtype=dtype)
            )
            head_freqs = torch.cos(angle) * dim_t  # shape: [head_dim / 2]
            freqs[dim_index, head_index, :] = head_freqs

    return freqs


def init_nd_freqs(
    position_dim: int,
    head_dim: int,
    num_heads: int,
    freq_group_pattern: Tensor,
    enforce_freq_groups_equal: bool = True,
    thetas: Union[Tensor, float] = 10.0,
    rotate: bool = True,
    max_rotation_angle: float = 2 * torch.pi,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[list[Tensor], Tensor]:
    """Initializes frequency parameters for N-dimensional rotary position embeddings.

    Generates the frequencies used for RoPE (Rotary Position Embeddings) in N dimensions.
    For each head and position dimension, creates frequency vectors with magnitude decay
    based on theta values and optional random rotation angles.

    Args:
        position_dim (int): Number of position dimensions (e.g., 2 for 2D, 3 for 3D).
        head_dim (int): Dimension size of each attention head. Must be divisible by 2.
        num_heads (int): Number of attention heads. Can be 1 even if the embeddings to
            be encoded will be split into multiple heads to share RoPE encodings among
            heads.
        freq_group_pattern (Tensor): Boolean tensor of shape
            [n_freq_groups, position_dim] defining frequency group inclusion. The
            (head_dim/2) elements of the RoPE encoding vector will be split among the
            frequency groups. If position i, j is True, then frequency group i
            includes position dimension j.
        enforce_freq_groups_equal (boolean): If True, then this function will raise
            a ValueError if the (head_dim/2) available elements of the RoPE vector
            cannot be evenly split between the frequency groups. If False, then
            trailing frequency groups may have fewer RoPE encodings assigned to them.
        thetas (Union[Tensor], float]): Base value(s) for frequency scaling.
            Can be a single float (applied to all dimensions and frequency groups)
            or a 2D tensor of shape [n_freq_groups, position_dim], with either
            component dimension allowed to be 1 for broadcasting. Entries corresponding
            to non-included position dimensions in a frequency group will be ignored.
            Larger values of theta result in lower-frequency rotations, and may be more
            suitable for dimensions of greater spatial scale. Default: 10.0
        rotate (bool): Whether to apply random rotation to the frequency vectors. When True,
            each head gets different random rotations. Default: True
        max_rotation_angle (bool): If rotate is True, each head's random rotation is
            uniformly distributed between 0 and max_rotation_angle. Default: 2 * pi
        dtype (Optional[torch.dtype]): Data type for the output tensor. Default: None
        device (Optional[torch.device]): Device for the output tensor. If None, the
            tensor will be created on freq_group_pattern's device. Default: None

    Returns:
        list[Tensor]: n_freq_groups-long list of frequency tensors, each of shape
            [position_dim_g, n_heads, head_dim//(2 * n_freq_groups)] (or
            [position_dim_g, n_heads, head_dim//(2 * n_freq_groups) + 1], if
            enforce_greq_groups_equal is False and some frequency groups have fewer
            encoding dimensions), where position_dim_g is the number of position
            dimensions included in that frequency group (i.e.,
            freq_group_pattern[g].sum()), containing the frequency parameters for each
            position dimension and attention head.
        Tensor: Long tensor of shape [n_freq_groups, 2] of the start and end indices of
            the RoPE encoding dimensions that are assigned to each frequency group

    Raises:
        ValueError: If head_dim is not divisible by 2, if freq_group pattern is not
        2D or has second dimension size not equal to position_dim, if
        enforce_freq_groups_equal is True and (head_dim/2) is not evenly divisible
        by the number of frequency groups, or if thetas is the wrong size.

    Notes:
        Differences from rope-for-vit:
            - Decreasing frequencies over encoding dim instead of increasing
                (theta raised to negative power instead of positive) - similar to
                standard 1D RoPE
            - Configurable max rotation angle
    """
    _validate_head_dim_even(head_dim)

    if device is None:
        device = freq_group_pattern.device

    n_freq_groups = freq_group_pattern.size(0)

    # Validate thetas (base frequencies)
    thetas: Tensor = torch.as_tensor(thetas, dtype=dtype, device=device)
    if thetas.ndim != 2 and thetas.numel() != 1:
        raise ValueError(
            "Expected thetas to either be a scalar or a 2D tensor, got shape "
            f"{thetas.shape}."
        )

    # broadcast thetas
    if thetas.numel() == 1 or any(torch._shape_as_tensor(thetas) == 1):
        thetas = thetas.expand(n_freq_groups, position_dim)

    if thetas.shape != (n_freq_groups, position_dim):
        raise ValueError(
            "Expected thetas to be broadcastable to [n_freq_groups, position_dim] "
            f"([{n_freq_groups}, {position_dim}]), got shape {thetas.shape}"
        )

    # Assign RoPE encoding dims to frequency groups
    half_head_dim = head_dim // 2
    base_dim = half_head_dim // n_freq_groups
    remainder = half_head_dim % n_freq_groups

    if remainder > 0 and enforce_freq_groups_equal:
        raise ValueError(
            f"RoPE encodings ({half_head_dim}) not evenly divisible by frequency "
            f"groups ({n_freq_groups})"
        )

    # Create tensor with base dimensions and add remainder to first elements
    encodings_per_freq_group = torch.full((n_freq_groups,), base_dim, device=device)
    encodings_per_freq_group[:remainder] += 1

    # Initialize grouped RoPE frequencies
    freqs = []
    encoding_ranges: list[tuple[int, int]] = []
    encoding_start = 0
    for g in range(n_freq_groups):
        freq_group_size = encodings_per_freq_group[g]
        n_pos_dims_this_freq_group = freq_group_pattern[g].sum()
        freqs_g = torch.zeros(
            (n_pos_dims_this_freq_group, num_heads, freq_group_size),
            dtype=dtype,
            device=device,
        )

        encoding_ranges.append((encoding_start, encoding_start + freq_group_size))
        encoding_start = encoding_start + freq_group_size

        group_dim_counter = 0
        # loop over all position dims, skipping excluded ones for this freq group
        for dim_index in range(position_dim):
            if not freq_group_pattern[g, dim_index]:
                continue
            theta_g_dim = thetas[g, dim_index]

            # Create frequency magnitudes that decay with RoPE encoding index
            dim_t = torch.arange(0, freq_group_size, 1, dtype=dtype, device=device)
            dim_t = theta_g_dim ** (-dim_t / freq_group_size)

            for head_index in range(num_heads):
                angle = (
                    torch.rand(1, device=device, dtype=dtype) * max_rotation_angle
                    if rotate
                    else torch.zeros(1, device=device, dtype=dtype)
                )
                head_freqs = torch.cos(angle) * dim_t
                freqs_g[group_dim_counter, head_index, :] = head_freqs
            group_dim_counter += 1

        freqs.append(freqs_g)

    encoding_ranges = torch.tensor(encoding_ranges, dtype=torch.long, device=device)
    return freqs, encoding_ranges


class RoPEEncodingND(nn.Module):
    """N-dimensional Rotary Position Embedding (RoPE) module.

    Implements rotary position embeddings for arbitrary dimensional positional inputs.
    This module applies RoPE to queries and keys in attention mechanisms, enabling
    position-aware attention across N spatial dimensions.

    Args:
        position_dim (int): Number of position dimensions (e.g., 2 for 2D, 3 for 3D).
        embed_dim (int): Total embedding dimension, must be divisible by n_heads.
        n_heads (int): Number of attention heads.
        share_heads (bool): If True, then only one set of frequencies per frequency
            group is created, that is shared among all attention heads, similar to
            traditional 1D RoPE. Defaults to False.
        freq_group_pattern (Optional[Tensor]): Boolean tensor of shape
            [n_freq_groups, position_dim] defining frequency group inclusion. The
            (head_dim/2) elements of the RoPE encoding vector will be split among the
            frequency groups. If position i, j is True, then frequency group i
            includes position dimension j. If None, freq_group_pattern will default
            to an all-True tensor of shape [1, position_dim]; i.e., one frequency group
            with all position dimensions.
        enforce_freq_groups_equal (boolean): If True, then this function will raise
            a ValueError if the (head_dim/2) available elements of the RoPE vector
            cannot be evenly split between the frequency groups. If False, then
            trailing frequency groups may have fewer RoPE encodings assigned to them.
        rope_base_theta (Union[Tensor], float]): Base value(s) for frequency scaling.
            Can be a single float (applied to all dimensions and frequency groups)
            or a 2D tensor of shape [n_freq_groups, position_dim], with either
            component dimension allowed to be 1 for broadcasting. Entries corresponding
            to non-included position dimensions in a frequency group will be ignored.
            Larger values of theta result in lower-frequency rotations, and may be more
            suitable for dimensions of greater spatial scale. Default: 10.0
        dtype (torch.dtype): Data type for the internal parameters. Default: torch.float
    """

    def __init__(
        self,
        position_dim: int,
        embed_dim: int,
        n_heads: int,
        share_heads: bool = False,
        freq_group_pattern: Optional[Tensor] = None,
        enforce_freq_groups_equal: bool = True,
        rope_base_theta: Union[Tensor, float] = 10.0,
        dtype=torch.float,
    ):
        """Initialize the module"""
        super().__init__()
        self.embed_dim = embed_dim
        if embed_dim % n_heads != 0:
            raise ValueError(
                "Expected embed_dim to be divisible by n_heads, got "
                f"{embed_dim} and {n_heads}"
            )
        self.head_dim = embed_dim // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"Expected head_dim to be divisible by 2, got {self.head_dim}"
            )
        self.position_dim = position_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.share_heads = share_heads

        if freq_group_pattern is None:
            # default frequency group pattern: one group with all position dimensions
            freq_group_pattern = torch.ones(1, position_dim, dtype=torch.bool)
        freq_group_pattern = torch.as_tensor(freq_group_pattern, dtype=torch.bool)

        self.enforce_freq_groups_equal = enforce_freq_groups_equal

        self.validate_freq_group_pattern(freq_group_pattern)
        self.register_buffer("freq_group_pattern", freq_group_pattern)
        self.n_freq_groups = freq_group_pattern.size(0)

        self._base_theta = torch.as_tensor(rope_base_theta, dtype=dtype)
        self.dtype = dtype
        self._init_freq_param()

    def _init_freq_param(self):
        """Initialize the frequency parameters for the RoPE module.

        Creates and stores the frequency parameters as trainable parameters and
        precomputes the indices used to construct the full sparse RoPE frequency
        tensor.
        """

        effective_n_heads = self.n_heads if not self.share_heads else 1

        freqs, encoding_ranges = init_nd_freqs(
            self.position_dim,
            self.head_dim,
            effective_n_heads,
            self.freq_group_pattern,
            self.enforce_freq_groups_equal,
            self._base_theta,
            dtype=self.dtype,
        )
        self.validate_grouped_freqs(freqs, encoding_ranges)

        self.freqs = nn.ParameterList(freqs)
        self.register_buffer("encoding_ranges", encoding_ranges)

        # Precompute indices for grouped_rope_freqs_tensor
        indices_list = []

        for g, ranges in enumerate(encoding_ranges):
            range_start, range_end = ranges
            range_size = range_end - range_start
            pos_dims = torch.nonzero(self.freq_group_pattern[g], as_tuple=True)[0]

            # Create indexing tensors for this frequency group
            # Order matches output tensor shape: [position_dim, n_freq_groups, n_heads, head_dim//2]
            pos_idx = pos_dims.view(-1, 1, 1).expand(-1, effective_n_heads, range_size)
            g_idx = torch.full(
                (pos_dims.size(0), effective_n_heads, range_size),
                g,
                dtype=torch.long,
                device=pos_dims.device,
            )
            head_idx = (
                torch.arange(effective_n_heads, device=pos_dims.device)
                .view(1, -1, 1)
                .expand(pos_dims.size(0), -1, range_size)
            )
            dim_idx = (
                torch.arange(range_start, range_end, device=pos_dims.device)
                .view(1, 1, -1)
                .expand(pos_dims.size(0), effective_n_heads, -1)
            )

            # Stack with dimension order matching output tensor
            indices = torch.stack(
                [
                    pos_idx.flatten(),
                    g_idx.flatten(),
                    head_idx.flatten(),
                    dim_idx.flatten(),
                ],
                dim=0,
            )
            indices_list.append(indices)

        # Concatenate all indices
        indices = torch.cat(indices_list, dim=1)

        # store indices for construction ofm freq tensor in forward pass
        pos_indices, group_indices, head_indices, enc_indices = indices.unbind(0)
        self.register_buffer("freq_pos_indices", pos_indices)
        self.register_buffer("freq_group_indices", group_indices)
        self.register_buffer("freq_head_indices", head_indices)
        self.register_buffer("freq_enc_indices", enc_indices)

    def validate_freq_group_pattern(self, freq_group_pattern: Tensor):
        if freq_group_pattern.ndim != 2:
            raise ValueError(
                "Expected 2D tensor for freq_group_pattern, got shape "
                f"{freq_group_pattern.size()}"
            )
        if freq_group_pattern.size(1) != self.position_dim:
            raise ValueError(
                "Expected second dimension of freq_group_pattern to have size equal to "
                f"position_dim, got freq_group_pattern shape {freq_group_pattern.size()} "
                f"and position_dim={self.position_dim}"
            )
        n_freq_groups = freq_group_pattern.size(0)
        half_head_dim = self.head_dim // 2
        remainder = half_head_dim % n_freq_groups

        if remainder > 0 and self.enforce_freq_groups_equal:
            raise ValueError(
                f"RoPE encodings ({half_head_dim}) not evenly divisible by frequency "
                f"groups ({n_freq_groups})"
            )

    def validate_grouped_freqs(self, freqs: list[Tensor], encoding_ranges: Tensor):
        # Validate number of frequency groups
        n_freq_groups = len(freqs)
        if self.freq_group_pattern.size(0) != n_freq_groups:
            raise ValueError(
                "Expected the first dimension of freq_group_pattern (shape: "
                f"{self.freq_group_pattern.shape}) to have size equal to the length of the"
                f"freqs list ({len(freqs)})"
            )

        # Validate head_dim is consistent
        half_head_dim_list = [freqs.size(2) for freqs in freqs]
        if len(set(half_head_dim_list)) != 1 and self.enforce_freq_groups_equal:
            raise ValueError(
                "Expected tensors in freqs to all have the same number of "
                f"RoPE encodings; got {half_head_dim_list}"
            )

        # Validate n_heads is consistent
        n_heads_list = [freqs.size(1) for freqs in freqs]
        n_heads_set = set(n_heads_list)
        if not (
            len(n_heads_set) == 1
            or (len(n_heads_set) == 2 and len(n_heads_set - set((1,)) == 1))
        ):
            raise ValueError(
                "Expected tensors in freqs to have number of attention heads "
                f"all equal and/or 1, got {n_heads_list}"
            )

        # Validate encoding ranges
        if encoding_ranges.size(0) != n_freq_groups:
            raise ValueError(
                "Expected first dim of encoding_ranges to be equal to n_freq_groups "
                f"({n_freq_groups}), got shape {encoding_ranges}"
            )

        if not (
            torch.all(encoding_ranges[:, 0] <= encoding_ranges[:, 1])
            and torch.all(encoding_ranges[:-1, 1] == encoding_ranges[1:, 0])
        ):
            raise ValueError(
                "Expected encoding_ranges to be a 2D tensor of contiguous, "
                f"non-overlapping slices, got {encoding_ranges}"
            )

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Apply rotary position embeddings to query and optionally key tensors.

        Applies position-dependent rotations to query and key tensors based on
        their associated position information.

        Args:
            query (Tensor): Query tensor of shape [..., embed_dim].
            query_pos (Tensor): Position tensor for query of shape
                [..., position_dim]. The leading dimensions must match those of query.
                It is assumed that the positions are NOT normalized to the standard
                [0, 1] range and are instead the true positions.
            key (Optional[Tensor]): Key tensor of shape [..., embed_dim]. Default: None
            key_pos (Optional[Tensor]): Position tensor for key of shape
                [..., position_dim]. If None and key is provided, query_pos will be
                used. It is assumed that the positions are NOT normalized to the
                standard [0, 1] range and are instead the true positions. Default: None

        Returns:
            Union[Tensor, tuple[Tensor, Tensor]]:
                - If key is None: Rotated query tensor of same shape as input query.
                - If key is provided: Tuple of (rotated query, rotated key) tensors.

        Note:
            - For query/key embeddings with a regular grid structure, a default
                position grid may be obtained from the static method `position_grid`.

        Raises:
            ValueError: If the tensor shapes are incompatible.

        Warns:
            UserWarning: If position coordinates appear to be normalized
                (in [0,1] range).
        """

        self.shape_check(query, query_pos)
        if query_pos.numel() > 0 and query_pos.min() > 0.0 and query_pos.max() <= 1.0:
            warnings.warn(
                "Expected un-normalized (i.e., not inside [0,1]) coordinates "
                "for position but found potentially normalized coordinates. "
                "Did you accidentally pass in normalized coordinates?\n(Your coord "
                f"range: [{query_pos.min().item(), query_pos.max().item()}])",
                UserWarning,
            )
        if key_pos is not None:
            self.shape_check(key, key_pos)
        freq_tensor = self.grouped_rope_freqs_tensor(self.freqs)

        query_rot_vec = self.calculate_rope(query_pos, freq_tensor)

        query_batch_dims = query.shape[:-1]

        # unstack query heads
        query = query.reshape(query_batch_dims + (self.n_heads, self.head_dim))

        query_rotated = self.rotate_embeddings(query, query_rot_vec)
        # stack heads back
        query_rotated = query_rotated.view(query_batch_dims + (self.embed_dim,))

        if key is None:
            return query_rotated

        if key_pos is not None:
            key_rot_vec = self.calculate_rope(key_pos, freq_tensor)
        else:
            key_rot_vec = query_rot_vec

        key_batch_dims = key.shape[:-1]
        # unstack key heads
        key = key.reshape(key_batch_dims + (self.n_heads, self.head_dim))

        key_rotated = self.rotate_embeddings(key, key_rot_vec)
        # stack heads back
        key_rotated = key_rotated.view(key_batch_dims + (self.embed_dim,))

        return query_rotated, key_rotated

    @staticmethod
    def position_grid(
        embeddings_shape: Union[tuple[int, ...], Tensor],
        start_dim: int = 1,
        end_dim: int = -1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Generates a regularly-spaced grid of positions based on the input shape.

        This function may be used to generate a tensor of positions corresponding to
        the tensor indices of each element in the embeddings tensor. This is
        potentially useful for regularly-spaced queries and keys, such as embeddings
        corresponding to text tokens or image pixels. The exact form of the position
        grid tensor is torch.stack(torch.meshgrid(
            *[torch.arange(size) for size in embeddings_shape[start_dim:end_dim]],
            indexing="ij"
        ))

        Args:
            embeddings_shape (Tensor): The full shape of the embeddings tensor.
            start_dim (int, optional): Start index of the position dimensions in
                embeddings_shape, inclusive. Defaults to 1 (i.e., one batch dim).
            end_dim (int, optional): End index of the position dimensions in
                embeddings_shape, exclusive. Defaults to -1 (i.e., one feature dim).
            device(torch.device, optional): The device on which to create the tensor.
                Defaults to None.
            dtype(torch.device, optional): The dtype for the created tensor. Defaults
                to None.

        Returns:
            Tensor: Created position grid tensor, of shape
                [*embeddings_shape[start_dim:end_dim],
                 len(embeddings_shape[start_dim:end_dim])]
        """
        grid = torch.stack(
            torch.meshgrid(
                *[
                    torch.arange(size, device=device, dtype=dtype)
                    for size in embeddings_shape[start_dim:end_dim]
                ],
                indexing="ij",
            ),
            dim=-1,
        )
        return grid

    def grouped_rope_freqs_tensor(
        self,
        grouped_rope_freqs: list[Tensor],
    ) -> Tensor:
        """Use frequency group information to build the full RoPE frequency tensor that
        is multiplied by the positions to produce RoPE encodings.

        This function takes the per-group RoPE frequencies to construct the RoPE
        frequency tensor. The RoPE frequency tensor has shape
        [position_dim, n_freq_groups, n_heads, head_dim/2], and is zero at positions
        where a position dimension is not included in a frequency group. The
        frequencies are stored separately per frequency group in tensors of shape
        [position_dim_g, n_heads, group_encoding_dim] because each frequency group may
        have a different number of active position dimensions and/or assigned encoding
        dimensions.

        Args:
            grouped_rope_freqs (list[Tensor]): List of per-group frequency tensors, as
                generated by init_nd_freqs, each of shape
                [
                    position_dim_g,
                    n_heads,
                    {head_dim//(2 * n_freq_groups), head_dim//(2 * n_freq_groups) + 1}
                ],
                where position_dim_g is the number of position dimensions included in
                frequency group g.

        Returns:
            Tensor: RoPE frequency tensor of shape
                [position_dim, n_freq_groups, n_heads, head_dim/2] or
                [position_dim, n_freq_groups,       1, head_dim/2], with nonzero
                elements corresponding to position dimensions included in each
                frequency group. It may be passed to `calculate_rope` with the
                positions tensor to compute RoPE encodings.
        """
        if isinstance(grouped_rope_freqs, Tensor):
            grouped_rope_freqs = [grouped_rope_freqs]

        # Create output tensor
        rope_freqs = grouped_rope_freqs[0].new_zeros(
            self.position_dim,
            self.n_freq_groups,
            self.n_heads if not self.share_heads else 1,
            self.head_dim // 2,
        )

        values = torch.cat([fg.flatten() for fg in grouped_rope_freqs])

        rope_freqs.index_put_(
            (
                self.freq_pos_indices,
                self.freq_group_indices,
                self.freq_head_indices,
                self.freq_enc_indices,
            ),
            values,
        )

        return rope_freqs

    @staticmethod
    def calculate_rope(positions: Tensor, rope_freqs: Tensor) -> Tensor:
        """Creates rotation vectors from position coordinates and RoPE frequencies.

        Transforms positional information into rotation vectors for RoPE.

        Args:
            positions (Tensor): Position tensor of shape [..., position_dim].
            rope_freqs (Tensor): Frequency tensor for rotary encodings of shape
                [position_dim, n_freq_groups, n_heads, head_dim/2].

        Returns:
            Tensor: Real-valued positional encodings of shape
                [..., n_heads, head_dim/2].
        """
        return calculate_rope(positions.to(rope_freqs), rope_freqs)

    @staticmethod
    def rotate_embeddings(query_or_key: Tensor, rope_encoding: Tensor) -> Tensor:
        """Applies rotary embeddings to query or key tensor using complex
        multiplication.

        Rotates the query or key tensor using the rotation vectors via complex
        multiplication.

        Args:
            query_or_key (Tensor): Query or key tensor of shape
                [..., n_heads, head_dim].
            rope_encoding (Tensor): Real-valued RoPE encoding tensor of shape
                [..., n_heads, head_dim/2].

        Returns:
            Tensor: Rotated query or key tensor of same shape as input query_or_key.
        """
        # Unsqueeze rope_encoding if needed
        dim_diff = query_or_key.ndim - rope_encoding.ndim
        if dim_diff > 0:
            rope_encoding = rope_encoding.view((1,) * dim_diff + rope_encoding.shape)
        return rotate_embeddings(query_or_key, rope_encoding)

    def shape_check(self, query_or_key: Tensor, query_or_key_pos: Tensor):
        """Validates the shapes of query/key and their position tensors.

        Args:
            query_or_key (Tensor): Query or key tensor of shape [..., embed_dim].
            query_or_key_pos (Tensor): Position tensor of shape [..., position_dim].
                Must be broadcastable to the shape of query_or_key.

        Raises:
            ValueError: If tensor shapes are incompatible.
        """
        if not can_broadcast_shapes(
            query_or_key.shape[:-1], query_or_key_pos.shape[:-1]
        ):
            raise ValueError(
                "Expected leading dims of query_or_key_pos to be broadcastable to "
                "leading dims of query_or_key, but got shapes "
                f"{query_or_key_pos.shape} and {query_or_key.shape}, respectively."
            )
        if query_or_key.shape[-1] != self.embed_dim:
            raise ValueError(
                "Expected query_or_key to have last dim equal to embed_dim "
                f"(={self.embed_dim}), got {query_or_key.shape[-1]}"
            )
        if query_or_key_pos.shape[-1] != self.position_dim:
            raise ValueError(
                "Expected query_or_key_pos to have last dim equal to pos_dim "
                f"(={self.position_dim}), got {query_or_key_pos.shape[-1]}"
            )

    def reset_parameters(self):
        """Resets frequency parameters"""
        freqs, _ = init_nd_freqs(
            self.position_dim,
            self.head_dim,
            self.n_heads if not self.share_heads else 1,
            self.freq_group_pattern,
            self.enforce_freq_groups_equal,
            self._base_theta,
            dtype=self.dtype,
            device=self.freqs[0].device,
        )
        with torch.no_grad():
            for param, init in zip(self.freqs, freqs):
                param.copy_(init)


def prep_multilevel_positions(
    spatial_positions: Tensor,
    batch_indices: Tensor,
    level_indices: Tensor,
    level_spatial_shapes: Tensor,
):
    """Standardizes positional coordinates across multiple resolution levels.

    Converts indices or positions from multiple resolution levels to a standardized
    coordinate system by rescaling each level to match the finest level's resolution.
    This enables consistent position encoding across hierarchical feature maps.

    Args:
        spatial_positions (Tensor): Indices or positions of shape [num_points, position_dim],
            where each row contains the N-D position of each point. If floating point,
            they're treated as coordinates; if integer, they're treated as indices.
        batch_indices (Tensor): Integer tensor of shape [num_points], containing the
            batch index for each position in spatial_positions.
        level_indices (Tensor): Integer tensor of shape [num_points], containing the
            level index for each position in spatial_positions.
        level_spatial_shapes (Tensor): Tensor of shape [num_levels, 2] or
            [batch_size, num_levels, 2] specifying the spatial dimensions
            (height, width) of each level.

    Returns:
        Tensor: Rescaled positions of shape [num_points, position_dim + 1] with floating
            point dtype, where the second dimension has the level index concatenated onto
            the end of the spatial coordinates, and the spatial coordinates are
            standardized to the finest resolution level.

    Raises:
        ValueError: If tensors don't have the expected shape, dimensions, or dtypes.
    """
    validate_nd(spatial_positions, 2, "spatial_positions")
    validate_nd(batch_indices, 1, "batch_indices")
    validate_nd(level_indices, 1, "level_indices")
    num_points = spatial_positions.size(0)

    if not torch.is_floating_point(spatial_positions):
        # convert from indices to coordinates of pixel centers
        spatial_positions = spatial_positions + 0.5

    # batch, level, pos_dim or level, pos_dim
    assert level_spatial_shapes.ndim in (2, 3)

    # Initialize output tensor
    multilevel_positions = spatial_positions.new_zeros(
        num_points, spatial_positions.size(1) + 1
    )

    # Early exit
    if num_points == 0:
        return multilevel_positions

    if level_spatial_shapes.ndim == 2:
        level_spatial_shapes = level_spatial_shapes.unsqueeze(0).expand(
            torch.max(batch_indices) + 1, -1, -1
        )

    batch_max_spatial_shape = level_spatial_shapes.max(-2)[0]
    max_spatial_shapes = batch_max_spatial_shape[batch_indices]
    indexed_spatial_shapes = level_spatial_shapes[batch_indices, level_indices]

    # Fill in rescaled positions
    multilevel_positions[:, :-1] = spatial_positions / (
        indexed_spatial_shapes / max_spatial_shapes
    )

    # Fill in level indices
    multilevel_positions[:, -1] = level_indices.to(multilevel_positions)

    return multilevel_positions


def get_multilevel_freq_group_pattern(
    position_dim: int, pattern_name: str, device=None
) -> Tensor:
    """Get a predefined frequency group pattern for RoPE encodings of multilevel features.

    Creates a frequency group pattern tensor for use with RoPEEncodingND based on
    predefined patterns that determine how spatial and level dimensions are encoded.

    Args:
        position_dim (int): Spatial dimension of the features to be encoded (2 for 2D
            images, etc.). The output tensor will have this many spatial dimensions
            plus 1 dimension for the feature level
        pattern_name (str): Name of the pattern to use. Options:
            - "single": All dimensions (*spatial, level) in a single frequency group
            - "partition": Spatial dimensions and level in separate groups
            - "closure": Three groups - Spatial, level, and (*spatial, level)
        device (torch.device, optional): Device for the created tensor. Defaults to None.

    Returns:
        Tensor: Boolean tensor encoding the frequency group pattern, of shape
            [n_freq_groups, position_dim + 1]

    Raises:
        ValueError: If an unrecognized pattern name is provided.
    """
    if pattern_name == "single":
        out = torch.ones(1, position_dim + 1, device=device)
    elif pattern_name == "partition":
        out = torch.zeros(2, position_dim + 1, device=device)
        out[0, :-1] = True  # Spatial dimensions in one group
        out[1, -1] = True  # Level dimension in second group
    elif pattern_name == "closure":
        out = torch.zeros(3, position_dim + 1, device=device)
        out[0, :-1] = True  # Spatial dimensions in one group
        out[1, -1] = True  # Level dimension in second group
        out[2, :] = True  # Third group has all dimensions
    else:
        raise ValueError(f"Unrecognized pattern_name {pattern_name}")

    return out
