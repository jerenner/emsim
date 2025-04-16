import warnings
from typing import Optional, Union

import torch
from torch import Tensor, nn

# Based on code from
# https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py


def init_2d_freqs(
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
    on theta and optional random rotation angles.

    Args:
        head_dim (int): Dimension size of each attention head. Must be divisible by 4.
        n_heads (int): Number of attention heads.
        theta (float): Base value for frequency scaling. Larger values result in longer
            period sinusoids. Default: 10.0
        rotate (bool): Whether to apply random rotation to the frequency vectors.
            When True, each head gets different random rotations. Default: True
        dtype (Optional[torch.dtype]): Data type for the output tensor. Default: None
        device (Optional[torch.device]): Device for the output tensor. Default: None

    Returns:
        Tensor: Frequency parameter tensor of shape [n_head, head_dim/2, 2], containing
            the frequency parameters for x and y dimensions for each attention head.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    # Create frequency magnitudes that decay with head_dim index
    dim_t = torch.arange(0, head_dim, 2, dtype=dtype, device=device)
    dim_t = theta ** (dim_t / head_dim)

    freqs_list = []
    for dim_index in range(2):
        dim_freqs = []
        for head_index in range(n_heads):
            angle = torch.rand(1, device=device) * 2 * torch.pi if rotate else 0


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


def init_nd_freqs(
    position_dim: int,
    head_dim: int,
    num_heads: int,
    thetas: Union[Tensor, list[float], float] = 10.0,
    rotate: bool = True,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Initializes frequency parameters for N-dimensional rotary position embeddings.

    Generates the frequencies used for RoPE (Rotary Position Embeddings) in N dimensions.
    For each head and position dimension, creates frequency vectors with magnitude decay
    based on theta values and optional random rotation angles.

    Args:
        position_dim (int): Number of position dimensions (e.g., 2 for 2D, 3 for 3D).
        head_dim (int): Dimension size of each attention head. Must be divisible by 4.
        num_heads (int): Number of attention heads.
        thetas (Union[Tensor, list[float], float]): Base value(s) for frequency scaling.
            Can be a single float (applied to all dimensions), a list of floats, or a tensor
            of shape [position_dim]. Larger values result in longer period sinusoids.
            Default: 10.0
        rotate (bool): Whether to apply random rotation to the frequency vectors. When True,
            each head gets different random rotations. Default: True
        dtype (Optional[torch.dtype]): Data type for the output tensor. Default: None
        device (Optional[torch.device]): Device for the output tensor. Default: None

    Returns:
        Tensor: Frequency tensor of shape [n_head, head_dim/(2 * n_groups), pos_dim],
            containing the frequency parameters for each dimension and attention head.
    """
    thetas: Tensor = torch.as_tensor(thetas, dtype=dtype, device=device)
    if thetas.numel() == 1:
        thetas = thetas.expand(position_dim)
    mag = 1 / (
        thetas.view(-1, 1)
        ** (
            torch.arange(0, head_dim, 4, dtype=dtype, device=device).view(1, -1)
            / head_dim
        )
    )
    freqs = [[] for _ in range(position_dim)]
    for _ in range(num_heads):
        angles = (
            torch.rand(1, device=device) * 2 * torch.pi
            if rotate
            else torch.zeros(1, device=device)
        )
        for i, dim_freqs in enumerate(freqs):
            f = torch.cat(
                [
                    mag[i] * torch.cos(angles + torch.pi * 2 * i / (2 * position_dim)),
                    mag[i]
                    * torch.cos(angles + torch.pi * ((2 * i) + 1) / (2 * position_dim)),
                ],
                dim=-1,
            )
            dim_freqs.append(f)
    freqs = [torch.stack(dim_freqs, dim=0) for dim_freqs in freqs]
    freqs = torch.stack(freqs, dim=-1)
    return freqs  # n_head, head_dim/(2 * n_groups), pos_dim


class RoPEEncodingND(nn.Module):
    """N-dimensional Rotary Position Embedding (RoPE) module.

    Implements rotary position embeddings for arbitrary dimensional positional inputs.
    This module applies RoPE to queries and keys in attention mechanisms, enabling
    position-aware attention across N spatial dimensions.

    Args:
        position_dim (int): Number of position dimensions (e.g., 2 for 2D, 3 for 3D).
        embed_dim (int): Total embedding dimension, must be divisible by n_heads.
        n_heads (int): Number of attention heads.
        rope_base_theta (float): Base value for frequency scaling in RoPE. Default: 10.0
        dtype (torch.dtype): Data type for the internal parameters. Default: torch.float
    """

    def __init__(
        self,
        position_dim: int,
        embed_dim: int,
        n_heads: int,
        rope_base_theta: float = 10.0,
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
        self.pos_dim = position_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self._base_theta = torch.as_tensor(rope_base_theta)
        self.dtype = dtype
        self._init_freq_param()

    def _init_freq_param(self):
        """Initialize the frequency parameters for the RoPE module.

        Creates and stores the frequency parameters as a trainable parameter.
        """

        freqs = init_nd_freqs(
            self.pos_dim,
            self.head_dim,
            self.n_heads,
            self._base_theta,
            dtype=self.dtype,
        )
        assert freqs.shape == (self.n_heads, self.head_dim // 2, self.pos_dim)
        self.freqs = nn.Parameter(freqs)

    @staticmethod
    def real_to_complex(tensor: Tensor) -> Tensor:
        """Converts a real tensor to complex representation.

        Reshapes a real tensor of shape [..., 2*N] or shape [..., N, 2] to a
        complex tensor of shape [..., N]. If the last dimension is 2, it is taken
        as representing the real and imaginary parts. If the last dimension is not
        2, it is interpreted as interleaved real and imaginary parts and implicitly
        reshaped to [..., N, 2]

        Args:
            tensor (Tensor): Real-valued tensor to convert, with last dimension
                having an even number of elements or 2.

        Returns:
            Tensor: Complex-valued tensor.
        """
        assert not tensor.is_complex()
        if not tensor.size(-1) == 2:
            assert tensor.size(-1) % 2 == 0, "Last dim must be divisible by 2"
            new_shape = tensor.shape[:-1] + (tensor.size(-1) // 2, 2)
            tensor = tensor.reshape(new_shape)
        return torch.view_as_complex(tensor)

    @staticmethod
    def complex_to_real(tensor: Tensor) -> Tensor:
        """Converts a complex tensor to real representation.

        Reshapes a complex tensor of shape [..., N] to a real tensor of shape [..., 2*N].
        The complex values are flattened into interleaved real and imaginary parts.

        Args:
            tensor (Tensor): Complex-valued tensor to convert.

        Returns:
            Tensor: Real-valued tensor with the last dimension size doubled.
        """
        assert tensor.is_complex()
        tensor_real = torch.view_as_real(tensor)
        assert tensor_real.size(-1) == 2
        tensor_real = tensor_real.flatten(-2, -1)  # flatten out new trailing dim of 2
        assert tensor_real.ndim == tensor.ndim
        return tensor_real

    @torch.amp.autocast("cuda", enabled=False)
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

        Raises:
            ValueError: If the tensor shapes are incompatible or

        Warns:
            UserWarning: If position coordinates appear to be normalized
                (in [0,1] range).
        """

        self.shape_check(query, query_pos)
        if query_pos.numel() > 0 and query_pos.max() <= 1.0:
            warnings.warn(
                "Expected un-normalized (i.e., not inside [0,1]) coordinates "
                "for position but found potentially normalized coordinates. "
                "Did you accidentally pass in normalized coordinates?\n"
                f"(Your coord range: [{query_pos.min(), query_pos.max()}])",
                UserWarning,
            )
        if key_pos is not None:
            self.shape_check(key, key_pos)
        query_rot_vec = self.calculate_rope(query_pos)
        query_rotated = self.apply_rotation(query, query_rot_vec)

        if key is None:
            return query_rotated

        if key_pos is not None:
            key_rot_vec = self.calculate_rope(key_pos)
        else:
            key_rot_vec = query_rot_vec
        key_rotated = self.apply_rotation(key, key_rot_vec)

        return query_rotated, key_rotated

    def calculate_rope(self, positions: Tensor) -> Tensor:
        """Creates complex rotation vectors from position coordinates.

        Transforms positional information into complex rotation vectors for RoPE.

        Args:
            positions (Tensor): Position tensor of shape [..., position_dim].

        Returns:
            Tensor: Complex-valued rotation vectors of shape [..., embed_dim/2].
        """
        leading_dims = positions.shape[:-1]
        rot_vec = torch.mm(
            positions.view(-1, self.pos_dim).to(self.freqs),
            self.freqs.view(-1, self.pos_dim).t(),
        )
        rot_vec = rot_vec.view(leading_dims + (self.embed_dim // 2,))
        out = torch.polar(torch.ones_like(rot_vec), rot_vec)
        assert out.is_complex()
        return out

    @staticmethod
    def apply_rotation(query_or_key: Tensor, rot_vec: Tensor) -> Tensor:
        """Applies rotary embeddings to query or key tensor using complex
        multiplication.

        Rotates the query or key tensor using the rotation vectors via complex
            multiplication.

        Args:
            query_or_key (Tensor): Query or key tensor of shape [..., embed_dim].
            rot_vec (Tensor): Complex rotation vector of shape [..., embed_dim/2].

        Returns:
            Tensor: Rotated query or key tensor of same shape as input query_or_key.
        """
        if not query_or_key.is_complex():
            query_or_key = RoPEEncodingND.real_to_complex(query_or_key)
        if not rot_vec.is_complex():
            rot_vec = RoPEEncodingND.real_to_complex(rot_vec)

        query_or_key_rotated = query_or_key * rot_vec

        return RoPEEncodingND.complex_to_real(query_or_key_rotated)

    def shape_check(self, query_or_key: Tensor, query_or_key_pos: Tensor):
        """Validates the shapes of query/key and their position tensors.

        Args:
            query_or_key (Tensor): Query or key tensor of shape [..., embed_dim].
            query_or_key_pos (Tensor): Position tensor of shape [..., position_dim].

        Raises:
            ValueError: If tensor shapes are incompatible.
        """
        if query_or_key.ndim != query_or_key_pos.ndim:  # ..., seq_len, embed_dim
            raise ValueError(
                "Expected query_or_key and query_or_key_pos to have same number "
                f"of dimensions, got {query_or_key.ndim} and {query_or_key_pos.ndim}"
            )
        if query_or_key.shape[-1] != self.embed_dim:
            raise ValueError(
                "Expected query_or_key to have last dim equal to embed_dim "
                f"(={self.embed_dim}), got {query_or_key.shape[-1]}"
            )
        if query_or_key_pos.shape[-1] != self.pos_dim:
            raise ValueError(
                "Expected query_or_key_pos to have last dim equal to pos_dim "
                f"(={self.pos_dim}), got {query_or_key_pos.shape[-1]}"
            )
        if query_or_key.shape[:-1] != query_or_key_pos.shape[:-1]:
            raise ValueError(
                "Expected query_or_key and query_or_key_pos to have matching leading dims,"
                f" got {query_or_key.shape[:-1]} and {query_or_key_pos.shape[:-1]}"
            )

    def reset_parameters(self):
        """Resets frequency parameters"""
        freqs = init_nd_freqs(
            self.pos_dim,
            self.head_dim,
            self.n_heads,
            self._base_theta,
            dtype=self.freqs.dtype,
            device=self.freqs.device,
        )
        with torch.no_grad():
            self.freqs.copy_(freqs)


class RoPEEncodingNDGroupedFreqs(RoPEEncodingND):
    """N-dimensional Rotary Position Embedding with grouped frequencies.

    Extends RoPEEncodingND by allowing position dimensions to share frequency groups.
    This can be useful for handling heterogeneous position representations where
    some dimensions should use the same frequency bands.

    Args:
        position_dim (int): Number of position dimensions.
        embed_dim (int): Total embedding dimension, must be divisible by n_heads.
        n_heads (int): Number of attention heads.
        pos_dim_to_rope_group (Union[Tensor, list[int]]): Mapping from position
            dimensions to frequency groups. Each element indicates which group a
            position dimension belongs to. Length must equal position_dim.
        rope_base_theta (float): Base value for frequency scaling in RoPE. Default: 10.0
        dtype (torch.dtype): Data type for the internal parameters. Default: torch.float
    """
    def __init__(
        self,
        position_dim: int,
        embed_dim: int,
        n_heads: int,
        pos_dim_to_rope_group: Union[Tensor, list[int]],
        rope_base_theta: float = 10.0,
        dtype=torch.float,
    ):
        self.pos_dim_to_rope_group = torch.as_tensor(pos_dim_to_rope_group)
        if self.pos_dim_to_rope_group.ndim != 1:
            raise ValueError(
                f"Expected 1D pos_dim_to_rope_group, got {pos_dim_to_rope_group.ndim}"
            )
        if len(self.pos_dim_to_rope_group) != position_dim:
            raise ValueError(
                "Expected pos_dim_to_rope_group to have length equal to position_dim,"
                f" got {len(self.pos_dim_to_rope_group)} and {position_dim}"
            )
        self.n_freq_groups = len(self.pos_dim_to_rope_group.unique())
        super().__init__(position_dim, embed_dim, n_heads, rope_base_theta, dtype)
        if self.head_dim % self.n_freq_groups != 0:
            raise ValueError(
                "head_dim must be divisible by number of freq groups, got "
                f"{self.head_dim} and {self.n_freq_groups}"
            )

    def _init_freq_param(self):
        """Intializes frequency parameters, accounting for frequency groups.

        Creates frequency parameters shared within each frequency group.
        """
        freqs = init_nd_freqs(
            self.pos_dim,
            self.head_dim // self.n_freq_groups,
            self.n_heads,
            self._base_theta,
            dtype=self.dtype,
        )
        assert freqs.shape == (
            self.n_heads,
            self.head_dim // 2 // self.n_freq_groups,
            self.pos_dim,
        )
        self.freqs = nn.Parameter(freqs)

    def calculate_rope(self, positions: Tensor):
        """Creates complex rotation vectors using the grouped frequency parameters.

        Applies appropriate frequency parameters to each position dimension based on
        their assigned frequency group.

        Args:
            positions (Tensor): Position tensor of shape [..., position_dim].

        Returns:
            Tensor: Complex-valued rotation vectors of shape [..., embed_dim/2].
        """
        leading_dims = positions.shape[:-1]
        assert positions.shape[-1] == self.pos_dim
        unique_indices, index_counts = torch.unique(
            self.pos_dim_to_rope_group, return_counts=True
        )
        split_positions = [
            positions[..., self.pos_dim_to_rope_group == i] for i in unique_indices
        ]  # (batch_dims, pos_dim_of_group) x n_groups
        split_freqs = [
            self.freqs[..., self.pos_dim_to_rope_group == i] for i in unique_indices
        ]  # (n_heads, head_dim/(2 * n_groups), pos_dim_of_group) x n_groups
        rot_subvecs = [
            # (batch_dims, pos_dim_of_group)
            # x (n_heads, head_dim/(2 * n_groups), pos_dim_of_group)
            # = (batch_dims, n_heads, head_dim/(2 * n_groups))
            torch.einsum("...x,hdx->...hd", pos, freq)
            for pos, freq in zip(split_positions, split_freqs)
        ]
        rot_subvecs = [
            torch.polar(torch.ones_like(subvec), subvec) for subvec in rot_subvecs
        ]
        out = torch.cat(rot_subvecs, -1)
        out_shape = leading_dims + (self.embed_dim // 2,)
        out = out.view(out_shape)
        assert out.is_complex()
        return out

    def reset_parameters(self):
        freqs = init_nd_freqs(
            self.pos_dim,
            self.head_dim // self.n_freq_groups,
            self.n_heads,
            self._base_theta,
            dtype=self.freqs.dtype,
            device=self.freqs.device,
        )
        with torch.no_grad():
            self.freqs.copy_(freqs)


def prep_multilevel_positions(bijl_indices: Tensor, spatial_shapes: Tensor):
    """Standardizes positional coordinates across multiple resolution levels.

    Converts indices or positions from multiple resolution levels to a standardized
    coordinate system by rescaling each level to match the finest level's resolution.
    This enables consistent position encoding across hierarchical feature maps.

    Args:
        bijl_indices (Tensor): Indices or positions of shape [num_points, 4], where
            each row contains (batch_idx, i, j, level_idx). If i,j are floating point,
            they're treated as coordinates; if integer, they're treated as indices.
        spatial_shapes (Tensor): Tensor of shape [num_levels, 2] or
            [batch_size, num_levels, 2] specifying the spatial dimensions
            (height, width) of each level.

    Returns:
        Tensor: Rescaled positions of shape [num_points, 4] with the same dtype as
            bijl_indices, where the i,j coordinates are standardized to the finest
            resolution level.

    Raises:
        ValueError: If bijl_indices doesn't have the expected shape or dimensions.
    """
    if bijl_indices.ndim != 2:
        raise ValueError(
            "Expected bijl_indices to have 2 dimensions, got " f"{bijl_indices.ndim}"
        )
    if bijl_indices.shape[-1] != 4:
        raise ValueError(
            "Expected bijl_indices to have last dimension of 4 (batch, i, j, level),"
            f" got {bijl_indices.shape[-1]}"
        )
    ij = bijl_indices[:, 1:-1]
    if not torch.is_floating_point(ij):
        # convert from indices to coordinates of pixel centers
        ij = ij + 0.5
    batch_level = torch.stack([bijl_indices[:, 0], bijl_indices[:, -1]], -1)
    assert ij.shape[-1] == spatial_shapes.shape[-1]
    assert spatial_shapes.ndim in (2, 3)  # batch, level, 2 or level, 2

    if spatial_shapes.ndim == 2:
        spatial_shapes = spatial_shapes.unsqueeze(0).expand(
            torch.unique(batch_level[:, 0]).shape[0], -1, -1
        )

    max_spatial_shape = spatial_shapes.max(-2)[0][batch_level[:, 0]]
    spatial_shapes = spatial_shapes[batch_level.unbind(-1)]

    rescaled_positions = ij / (spatial_shapes / max_spatial_shape)

    positions = bijl_indices.clone().to(rescaled_positions)
    positions[:, 1:3] = rescaled_positions
    return positions
