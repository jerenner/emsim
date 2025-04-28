import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from emsim.networks.positional_encoding.rope import (
    RoPEEncodingND,
    get_multilevel_freq_group_pattern,
    prep_multilevel_positions,
)
from emsim.networks.transformer.blocks.self_attn import (
    MultilevelSelfAttentionBlockWithRoPE,
)


@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Base configuration for MultilevelSelfAttentionBlockWithRoPE tests."""
    return {
        "embed_dim": 64,
        "n_heads": 4,
        "position_dim": 2,
        "dropout": 0.0,
        "bias": False,
        "norm_first": True,
        "rope_spatial_base_theta": 100.0,
        "rope_level_base_theta": 10.0,
        "rope_share_heads": False,
        "rope_freq_group_pattern": "single",
    }


@pytest.fixture
def module_instance(
    base_config: Dict[str, Any], device: str
) -> MultilevelSelfAttentionBlockWithRoPE:
    """Create a module instance for testing."""
    module = MultilevelSelfAttentionBlockWithRoPE(**base_config).to(device)
    return module


@pytest.fixture
def simple_input_data(device: str) -> Dict[str, torch.Tensor]:
    """Generate simple input tensors for testing."""
    # Parameters
    stacked_sequence_length = 6
    embed_dim = 64
    position_dim = 2
    batch_size = 2

    # Create input tensors
    x = torch.randn(stacked_sequence_length, embed_dim, device=device)
    spatial_positions = (
        torch.rand(stacked_sequence_length, position_dim, device=device) * 10
    )
    level_indices = torch.tensor([0, 0, 1, 1, 1, 0], device=device)
    level_spatial_shapes = torch.tensor([[8, 8], [4, 4]], device=device)
    batch_offsets = torch.tensor([0, 3, 6], device=device)

    return {
        "x": x,
        "spatial_positions": spatial_positions,
        "level_indices": level_indices,
        "level_spatial_shapes": level_spatial_shapes,
        "batch_offsets": batch_offsets,
        "attn_mask": None,
    }


@pytest.fixture
def complex_input_data(device: str) -> Dict[str, torch.Tensor]:
    """Generate more complex input tensors for testing."""
    # Parameters
    stacked_sequence_length = 12
    embed_dim = 64
    position_dim = 2
    num_levels = 3
    batch_size = 3

    # Create input tensors
    x = torch.randn(stacked_sequence_length, embed_dim, device=device)
    spatial_positions = (
        torch.rand(stacked_sequence_length, position_dim, device=device) * 10
    )
    # Simple level indices pattern: [0,1,2,0, 1,2,0,1, 2,0,1,2]
    level_indices = (
        torch.arange(num_levels, device=device)
        .unsqueeze(0)
        .expand(stacked_sequence_length // num_levels, -1)
        .flatten()
    )
    level_spatial_shapes = torch.tensor([[16, 16], [8, 8], [4, 4]], device=device)
    # 3 batches with 4 tokens each
    batch_offsets = torch.tensor([0, 4, 8, 12], device=device)

    # Create an attention mask (True indicates masked positions)
    # We'll create a simple mask where the first token in each batch can't attend to the last token
    attn_mask = torch.zeros(batch_size, 4, 4, dtype=torch.bool, device=device)
    attn_mask[:, 0, 3] = True

    return {
        "x": x,
        "spatial_positions": spatial_positions,
        "level_indices": level_indices,
        "level_spatial_shapes": level_spatial_shapes,
        "batch_offsets": batch_offsets,
        "attn_mask": attn_mask,
    }


@pytest.fixture
def variable_length_input_data(device: str) -> Dict[str, torch.Tensor]:
    """Generate input tensors with variable sequence lengths."""
    # Parameters
    embed_dim = 64
    position_dim = 2
    num_levels = 3

    # Tokens per level per batch (variable)
    tokens_per_level = [
        [16, 8, 4],  # Batch 1
        [12, 6, 3],  # Batch 2 (smaller)
        [20, 10, 5],  # Batch 3 (larger)
    ]

    # Calculate total sequence length and batch offsets
    batch_size = len(tokens_per_level)
    total_seq_length = sum(sum(batch) for batch in tokens_per_level)
    batch_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)

    batch_offsets[1:] = torch.cumsum(
        torch.tensor(tokens_per_level, device=device).sum(1), 0
    )

    # current_offset = 0
    # for i, batch_tokens in enumerate(tokens_per_level):
    #     batch_offsets[i] = current_offset
    #     current_offset += sum(batch_tokens)
    # batch_offsets[-1] = current_offset

    # Create embeddings
    x = torch.randn(total_seq_length, embed_dim, device=device)

    # Create spatial positions and level indices
    spatial_positions = torch.zeros(total_seq_length, position_dim, device=device)
    level_indices = torch.zeros(total_seq_length, dtype=torch.long, device=device)

    # Level spatial shapes
    level_spatial_shapes = torch.tensor([[16, 16], [8, 8], [4, 4]], device=device)

    # Fill spatial positions and level indices
    idx = 0
    for b in range(batch_size):
        for level in range(num_levels):
            num_tokens = tokens_per_level[b][level]
            h, w = level_spatial_shapes[level]

            # Create grid positions
            h_use = num_tokens // w + (1 if num_tokens % w > 0 else 0)
            positions = []
            for i in range(h_use):
                for j in range(min(w, num_tokens - i * w)):
                    positions.append([i, j])

            positions = torch.tensor(positions, device=device)
            num_positions = positions.shape[0]

            # Add to spatial positions
            spatial_positions[idx : idx + num_positions] = positions

            # Set level indices
            level_indices[idx : idx + num_positions] = level

            idx += num_positions

    return {
        "x": x,
        "spatial_positions": spatial_positions,
        "level_indices": level_indices,
        "level_spatial_shapes": level_spatial_shapes,
        "batch_offsets": batch_offsets,
        "attn_mask": None,
    }


class HookedQKV(nn.Module):
    def __init__(self, main_module: nn.Module):
        super().__init__()
        self.module = main_module
        self.captured_values = {"pre_qkv_input": [], "qkv": []}

    def forward(self, x):
        self.captured_values["pre_qkv_input"] = x.detach().clone()
        out = self.module(x)
        self.captured_values["qkv"] = out.detach().clone()
        return out


class HookedCalcAttn(nn.Module):
    def __init__(self, main_module: nn.Module):
        super().__init__()
        self.module = main_module
        self.captured_values = {
            "q": [],
            "k": [],
            "v": [],
            "pad_mask": [],
            "attn_mask": [],
            "attn_output": [],
        }

    def forward(self, q, k, v, pad_mask, attn_mask=None):
        self.captured_values["q"].append(q.detach().clone())
        self.captured_values["k"].append(k.detach().clone())
        self.captured_values["v"].append(v.detach().clone())
        self.captured_values["pad_mask"].append(pad_mask.detach().clone())
        if attn_mask is not None:
            self.captured_values["attn_mask"].append(attn_mask.detach().clone())
        result = self.module(q, k, v, pad_mask, attn_mask)
        self.captured_values["attn_output"].append(result.detach().clone())
        return result


class ModuleWithHooks(nn.Module):
    """Wrapper that adds hooks to capture intermediate values."""

    def __init__(self, module: MultilevelSelfAttentionBlockWithRoPE):
        super().__init__()
        self.module = module
        self.captured_values = {}
        self._install_hooks()

    def _install_hooks(self):
        # Store original methods
        self._orig_qkv = self.module.qkv
        self._orig_calc_attn = self.module._calc_attn

        # Replace with hooked versions
        def hooked_calc_attn(q, k, v, pad_mask, attn_mask=None):
            self.captured_values["q"] = q.detach().clone()
            self.captured_values["k"] = k.detach().clone()
            self.captured_values["v"] = v.detach().clone()
            self.captured_values["pad_mask"] = pad_mask.detach().clone()
            if attn_mask is not None:
                self.captured_values["attn_mask"] = attn_mask.detach().clone()
            result = self._orig_calc_attn(q, k, v, pad_mask, attn_mask)
            self.captured_values["attn_output"] = result.detach().clone()
            return result

        self.module.qkv = HookedQKV(self._orig_qkv)
        self.module._calc_attn = hooked_calc_attn

    def forward(self, *args, **kwargs):
        result = self.module(*args, **kwargs)
        self.captured_values["pre_qkv_input"] = self.module.qkv.captured_values[
            "pre_qkv_input"
        ]
        self.captured_values["qkv"] = self.module.qkv.captured_values["qkv"]
        return result

    def restore_original_methods(self):
        """Restore original methods to the module."""
        self.module.qkv = self._orig_qkv
        self.module._calc_attn = self._orig_calc_attn


@pytest.mark.cuda_if_available
class TestInitialization:
    """Tests for initialization of MultilevelSelfAttentionBlockWithRoPE."""

    def test_basic_initialization(self, base_config: Dict[str, Any], device: str):
        """Test basic initialization with default parameters."""
        module = MultilevelSelfAttentionBlockWithRoPE(**base_config).to(device)

        # Check essential attributes
        assert module.embed_dim == base_config["embed_dim"]
        assert module.n_heads == base_config["n_heads"]
        assert module.position_dim == base_config["position_dim"]
        assert module.norm_first == base_config["norm_first"]

        # Check submodules
        assert isinstance(module.norm, nn.LayerNorm)
        assert isinstance(module.qkv, nn.Linear)
        assert isinstance(module.pos_encoding, RoPEEncodingND)
        assert isinstance(module.out_proj, nn.Linear)
        assert isinstance(module.out_proj_drop, nn.Dropout)

        # Check dimensions
        assert module.qkv.in_features == base_config["embed_dim"]
        assert module.qkv.out_features == 3 * base_config["embed_dim"]
        assert module.out_proj.in_features == base_config["embed_dim"]
        assert module.out_proj.out_features == base_config["embed_dim"]

    @pytest.mark.parametrize(
        "param_name,param_value",
        [
            ("dropout", 0.2),
            ("bias", True),
            ("norm_first", False),
            ("position_dim", 3),
            ("rope_spatial_base_theta", 1000.0),
            ("rope_level_base_theta", 50.0),
            ("rope_share_heads", True),
            ("rope_freq_group_pattern", "partition"),
        ],
    )
    def test_custom_initialization(
        self,
        base_config: Dict[str, Any],
        param_name: str,
        param_value: Any,
        device: str,
    ):
        """Test initialization with custom parameters."""
        # Update config with custom parameter
        config = base_config.copy()
        config[param_name] = param_value

        # Initialize module
        module = MultilevelSelfAttentionBlockWithRoPE(**config).to(device)

        # Check parameter was set correctly
        if param_name == "dropout":
            assert module.attn_drop_rate == param_value
        elif param_name == "bias":
            assert (module.qkv.bias is not None) == param_value
            assert (module.out_proj.bias is not None) == param_value
        elif param_name == "rope_freq_group_pattern":
            # This gets converted to a tensor pattern
            expected_pattern = get_multilevel_freq_group_pattern(
                config["position_dim"], param_value
            )
            pos_encoding_pattern = module.pos_encoding.freq_group_pattern
            assert pos_encoding_pattern.shape == expected_pattern.shape
        else:
            assert getattr(module, param_name) == param_value

    def test_embed_dim_head_compatibility(self, device: str):
        """Test that initialization handles embed_dim not divisible by n_heads."""
        with pytest.raises(ValueError, match="divisible by"):
            MultilevelSelfAttentionBlockWithRoPE(
                embed_dim=65,  # Not divisible by 4
                n_heads=4,
            ).to(device)

    @pytest.mark.parametrize("embed_dim,n_heads", [(64, 4), (128, 8), (256, 16)])
    def test_different_dimensions(
        self, base_config: Dict[str, Any], embed_dim: int, n_heads: int, device: str
    ):
        """Test that module works with different embedding dimensions and head counts."""
        # Update config
        config = base_config.copy()
        config.update({"embed_dim": embed_dim, "n_heads": n_heads})

        # Create module
        module = MultilevelSelfAttentionBlockWithRoPE(**config).to(device)

        # Check key dimensions
        assert module.embed_dim == embed_dim
        assert module.n_heads == n_heads
        assert module.qkv.out_features == 3 * embed_dim

        # Create simple input
        x = torch.randn(6, embed_dim, device=device)
        spatial_positions = torch.rand(6, 2, device=device) * 10
        level_indices = torch.zeros(6, dtype=torch.long, device=device)
        level_spatial_shapes = torch.tensor([[8, 8]], device=device)
        batch_offsets = torch.tensor([0, 3, 6], device=device)

        # Run forward pass
        output = module(
            x, spatial_positions, level_indices, level_spatial_shapes, batch_offsets
        )

        # Check output shape
        assert output.shape == x.shape


@pytest.mark.cuda_if_available
class TestForward:
    """Tests for forward pass of MultilevelSelfAttentionBlockWithRoPE."""

    def test_forward_shape_preservation(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        simple_input_data: Dict[str, torch.Tensor],
    ):
        """Test that the output shape matches the input shape."""
        x = simple_input_data["x"]

        # Run forward pass
        output = module_instance(**simple_input_data)

        # Check output shape
        assert output.shape == x.shape

        # Check output is not NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_batch_processing(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        simple_input_data: Dict[str, torch.Tensor],
    ):
        """Test that batched inputs are processed correctly."""
        # Run forward pass
        output = module_instance(**simple_input_data)

        # Split the output by batch
        outputs_by_batch = torch.split(output, [3, 3])  # 3 tokens per batch

        # Each batch's output should be different due to separate processing
        assert not torch.allclose(outputs_by_batch[0], outputs_by_batch[1])

    def test_forward_with_small_inputs(
        self, module_instance: MultilevelSelfAttentionBlockWithRoPE, device: str
    ):
        """Test forward pass with minimal-sized inputs."""
        # Create minimal inputs - just one token per batch
        batch_size = 2
        x = torch.randn(batch_size, module_instance.embed_dim, device=device)
        spatial_positions = (
            torch.randn(batch_size, module_instance.position_dim, device=device) * 10
        )
        level_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        level_spatial_shapes = torch.tensor([[1, 1]], device=device).float()
        batch_offsets = torch.tensor([0, 1, 2], device=device)

        # Run forward pass
        output = module_instance(
            x, spatial_positions, level_indices, level_spatial_shapes, batch_offsets
        )

        # Check output
        assert output.shape == x.shape

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_variable_length_sequences(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        variable_length_input_data: Dict[str, torch.Tensor],
    ):
        """Test forward pass with variable length sequences."""
        # Run forward pass
        output = module_instance(**variable_length_input_data)

        # Check output shape
        assert output.shape == variable_length_input_data["x"].shape

    @pytest.mark.parametrize("norm_first", [True, False])
    def test_norm_placement(
        self,
        base_config: Dict[str, Any],
        norm_first: bool,
        simple_input_data: Dict[str, torch.Tensor],
        device: str,
    ):
        """Test that norm_first parameter correctly affects normalization placement."""
        # Create module with specified norm_first
        config = base_config.copy()
        config["norm_first"] = norm_first
        module = MultilevelSelfAttentionBlockWithRoPE(**config).to(device)

        # Create hooked module
        hooked_module = ModuleWithHooks(module)

        # Run forward pass
        with torch.no_grad():
            hooked_module(**simple_input_data)

        # Verify behavior based on norm_first
        pre_qkv_input = hooked_module.captured_values["pre_qkv_input"]

        if norm_first:
            # Pre-norm architecture: input to QKV should be normalized
            # Normalized data typically has near-zero mean
            assert torch.abs(pre_qkv_input.mean()) < 0.1
            # And standard deviation close to 1
            assert torch.abs(pre_qkv_input.std() - 1.0) < 0.5
        else:
            # Post-norm architecture: input to QKV would be the original input
            # Should approximately match the statistics of the input data
            assert torch.abs(pre_qkv_input.mean() - simple_input_data["x"].mean()) < 0.1

        # Restore original methods
        hooked_module.restore_original_methods()

    @pytest.mark.parametrize("dropout_value", [0.0, 0.1, 0.5])
    def test_dropout_effect(
        self,
        base_config: Dict[str, Any],
        dropout_value: float,
        simple_input_data: Dict[str, torch.Tensor],
        device: str,
    ):
        """Test that dropout parameter affects the behavior appropriately."""
        # Configure the module with specified dropout
        config = base_config.copy()
        config["dropout"] = dropout_value
        module = MultilevelSelfAttentionBlockWithRoPE(**config).to(device)

        # Output in eval mode (no dropout)
        module.eval()
        eval_output = module(**simple_input_data)

        # Set module to training mode to enable dropout
        module.train()

        # Set seed for testing reproducibility
        torch.manual_seed(1)

        output_1 = module(**simple_input_data)

        # Should be different with dropout active
        if dropout_value == 0.0:
            assert torch.allclose(eval_output, output_1)
        else:
            assert not torch.allclose(eval_output, output_1)

        # Test reproducibility with same seed
        torch.manual_seed(1)
        output_2 = module(**simple_input_data)

        assert torch.allclose(output_1, output_2)

        # Test different seed
        torch.manual_seed(5)
        output_3 = module(**simple_input_data)

        if dropout_value == 0.0:
            assert torch.allclose(output_1, output_3)
        else:
            assert not torch.allclose(output_1, output_3)

    def test_grad_flow(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        simple_input_data: Dict[str, torch.Tensor],
    ):
        """Test that gradients flow through the module properly."""
        # Make inputs require grad
        x = simple_input_data["x"].clone().requires_grad_(True)
        input_data = simple_input_data.copy()
        input_data["x"] = x

        # Run forward pass
        output = module_instance(**input_data)

        # Create a dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that gradients have been computed
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Also check that module parameters received gradients
        for name, param in module_instance.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


@pytest.mark.cuda_if_available
class TestValidation:
    """Tests for input validation in MultilevelSelfAttentionBlockWithRoPE."""

    def test_dimension_validation(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        simple_input_data: Dict[str, torch.Tensor],
    ):
        """Test validation of input tensor dimensions."""
        # Create invalid x (1D instead of 2D)
        invalid_data = simple_input_data.copy()
        invalid_data["x"] = torch.randn(10, device=simple_input_data["x"].device)

        with pytest.raises((torch.jit.Error, ValueError), match="Expected x to be 2D"):
            module_instance(**invalid_data)

    def test_sequence_length_validation(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        simple_input_data: Dict[str, torch.Tensor],
    ):
        """Test validation of sequence length consistency."""
        # Create positions with wrong sequence length
        invalid_data = simple_input_data.copy()
        invalid_data["spatial_positions"] = torch.randn(
            simple_input_data["x"].shape[0] + 1,
            simple_input_data["spatial_positions"].shape[1],
            device=simple_input_data["x"].device,
        )

        with pytest.raises(ValueError, match="Mismatched sequence lengths"):
            module_instance(**invalid_data)

    def test_position_dimension_validation(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        simple_input_data: Dict[str, torch.Tensor],
    ):
        """Test validation of position dimensions."""
        # Create positions with wrong number of dimensions
        invalid_data = simple_input_data.copy()
        invalid_data["spatial_positions"] = torch.randn(
            simple_input_data["x"].shape[0],
            module_instance.position_dim + 1,  # Wrong number of position dimensions
            device=simple_input_data["x"].device,
        )

        with pytest.raises(
            (torch.jit.Error, ValueError), match="Mismatched position dimensions"
        ):
            module_instance(**invalid_data)

    def test_spatial_shapes_validation(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        simple_input_data: Dict[str, torch.Tensor],
    ):
        """Test validation of level spatial shapes dimensions."""
        # Create spatial shapes with wrong number of dimensions
        invalid_data = simple_input_data.copy()
        invalid_data["level_spatial_shapes"] = torch.randn(
            simple_input_data["level_spatial_shapes"].shape[0],
            simple_input_data["level_spatial_shapes"].shape[1]
            + 1,  # Wrong number of dimensions
            device=simple_input_data["x"].device,
        )

        with pytest.raises(ValueError, match="Mismatched position dimensions"):
            module_instance(**invalid_data)

    def test_input_validation(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        simple_input_data: Dict[str, torch.Tensor],
    ):
        """Test that validation properly catches shape mismatches."""
        # Test mismatched sequence lengths
        invalid_data = simple_input_data.copy()
        invalid_data["level_indices"] = invalid_data["level_indices"][
            :-1
        ]  # Remove last element

        with pytest.raises(ValueError, match="Mismatched sequence lengths"):
            module_instance(**invalid_data)


@pytest.mark.cuda_if_available
class TestAttention:
    """Tests for attention calculation in MultilevelSelfAttentionBlockWithRoPE."""

    def test_attention_mask_effect(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        complex_input_data: Dict[str, torch.Tensor],
    ):
        """Test that attention mask correctly affects the output."""
        # Get data with attention mask
        data_with_mask = complex_input_data.copy()

        # Run with mask
        output_with_mask = module_instance(**data_with_mask)

        # Run without mask
        data_without_mask = data_with_mask.copy()
        data_without_mask["attn_mask"] = None
        output_without_mask = module_instance(**data_without_mask)

        # Outputs should differ when mask is applied
        assert not torch.allclose(output_with_mask, output_without_mask)

    def test_calc_attn_method(
        self, module_instance: MultilevelSelfAttentionBlockWithRoPE, device: str
    ):
        """Test the internal _calc_attn method directly."""
        # Create test inputs
        batch_size = 2
        n_heads = module_instance.n_heads
        seq_len = 4
        head_dim = module_instance.embed_dim // n_heads

        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        # Test without attention mask
        output_no_mask = module_instance._calc_attn(q, k, v, pad_mask)
        assert output_no_mask.shape == (batch_size, n_heads, seq_len, head_dim)

        # Test with attention mask (different shapes)
        # Shape: (batch, seq, seq)
        attn_mask_3d = torch.zeros(
            batch_size, seq_len, seq_len, dtype=torch.bool, device=device
        )
        attn_mask_3d[:, 0, 1] = True  # Mask one position

        output_3d_mask = module_instance._calc_attn(q, k, v, pad_mask, attn_mask_3d)
        assert output_3d_mask.shape == (batch_size, n_heads, seq_len, head_dim)
        assert not torch.allclose(output_no_mask, output_3d_mask)

        # Shape: (batch, n_heads, seq, seq)
        attn_mask_4d = torch.zeros(
            batch_size, n_heads, seq_len, seq_len, dtype=torch.bool, device=device
        )
        attn_mask_4d[:, :, 0, 2] = True  # Mask different position

        output_4d_mask = module_instance._calc_attn(q, k, v, pad_mask, attn_mask_4d)
        assert output_4d_mask.shape == (batch_size, n_heads, seq_len, head_dim)
        assert not torch.allclose(output_3d_mask, output_4d_mask)

        # Test with padding
        pad_mask_with_padding = pad_mask.clone()
        pad_mask_with_padding[0, -1] = True  # Last token in first batch is padding

        output_with_padding = module_instance._calc_attn(q, k, v, pad_mask_with_padding)
        assert output_with_padding.shape == (batch_size, n_heads, seq_len, head_dim)
        # Padding should affect the output
        assert not torch.allclose(output_no_mask, output_with_padding)

        # Test with combined padding and attention mask
        output_combined = module_instance._calc_attn(
            q, k, v, pad_mask_with_padding, attn_mask_3d
        )
        assert output_combined.shape == (batch_size, n_heads, seq_len, head_dim)
        assert not torch.allclose(output_with_padding, output_combined)


@pytest.mark.cuda_if_available
class TestPositionEncoding:
    """Tests for position encoding in MultilevelSelfAttentionBlockWithRoPE."""

    def test_multilevel_position_handling(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        complex_input_data: Dict[str, torch.Tensor],
    ):
        """Test that positions from different levels are properly standardized."""
        # We create two sets of position inputs - one with regular positions and
        # another with altered positions for one level
        original_data = complex_input_data.copy()

        # Create modified data where Level 1 positions are scaled differently
        modified_data = complex_input_data.copy()
        level_1_mask = modified_data["level_indices"] == 1
        modified_positions = modified_data["spatial_positions"].clone()
        modified_positions[level_1_mask] *= 2  # Scale level 1 positions
        modified_data["spatial_positions"] = modified_positions

        # Run forward with both inputs
        output_original = module_instance(**original_data)
        output_modified = module_instance(**modified_data)

        # Since positions affect attention through RoPE, outputs should differ
        assert not torch.allclose(output_original, output_modified)

    @pytest.mark.parametrize("rope_freq_pattern", ["single", "partition", "closure"])
    def test_rope_frequency_patterns(
        self,
        base_config: Dict[str, Any],
        rope_freq_pattern: str,
        simple_input_data: Dict[str, torch.Tensor],
        device: str,
    ):
        """Test different RoPE frequency group patterns."""
        # Update config
        config = base_config.copy()
        config["rope_freq_group_pattern"] = rope_freq_pattern
        config["rope_enforce_freq_groups_equal"] = False

        # Create module
        module = MultilevelSelfAttentionBlockWithRoPE(**config).to(device)

        # Run forward pass
        output = module(**simple_input_data)

        # Check output shape
        assert output.shape == simple_input_data["x"].shape

        # Outputs with different patterns should be different
        if rope_freq_pattern != "single":
            default_module = MultilevelSelfAttentionBlockWithRoPE(**base_config).to(
                device
            )
            default_output = default_module(**simple_input_data)
            assert not torch.allclose(output, default_output)


@pytest.mark.cuda_if_available
class TestBatching:
    """Tests for batching operations in MultilevelSelfAttentionBlockWithRoPE."""

    def test_empty_batch_handling(
        self, module_instance: MultilevelSelfAttentionBlockWithRoPE, device: str
    ):
        """Test handling of empty batches."""
        # Create inputs with an empty batch
        embed_dim = module_instance.embed_dim
        stacked_sequence_length = 6

        x = torch.randn(stacked_sequence_length, embed_dim, device=device)
        spatial_positions = torch.rand(stacked_sequence_length, 2, device=device) * 10
        level_indices = torch.zeros(
            stacked_sequence_length, dtype=torch.long, device=device
        )
        level_spatial_shapes = torch.tensor([[8, 8]], device=device)

        # Create batch offsets with an empty batch in the middle: [0, 3, 3, 6]
        # This means batch 0 has 3 tokens, batch 1 has 0 tokens, batch 2 has 3 tokens
        batch_offsets = torch.tensor([0, 3, 3, 6], device=device)

        # Run forward pass
        output = module_instance(
            x, spatial_positions, level_indices, level_spatial_shapes, batch_offsets
        )

        # Check output shape
        assert output.shape == x.shape


@pytest.mark.cuda_if_available
class TestResetParameters:
    """Tests for parameter resetting in MultilevelSelfAttentionBlockWithRoPE."""

    def test_reset_parameters(
        self,
        module_instance: MultilevelSelfAttentionBlockWithRoPE,
        simple_input_data: Dict[str, torch.Tensor],
    ):
        """Test reset_parameters method."""
        # Store original parameter values
        original_qkv_weight = module_instance.qkv.weight.clone()
        original_out_proj_weight = module_instance.out_proj.weight.clone()

        # Reset parameters
        module_instance.reset_parameters()

        # Parameters should be different after reset
        assert not torch.allclose(original_qkv_weight, module_instance.qkv.weight)
        assert not torch.allclose(
            original_out_proj_weight, module_instance.out_proj.weight
        )

        # Run forward pass to ensure functionality
        output = module_instance(**simple_input_data)
        assert output.shape == simple_input_data["x"].shape


@pytest.mark.cuda_if_available
class TestGradients:
    """Integration tests for MultilevelSelfAttentionBlockWithRoPE."""

    def test_varied_sequences(self, base_config: Dict[str, Any], device: str):
        """Integration test with varied sequence lengths and multiple levels."""
        # Create module
        module = MultilevelSelfAttentionBlockWithRoPE(**base_config).to(device)

        # Create complex input with multiple levels and variable sequence lengths
        batch_size = 3
        num_levels = 3

        # Level spatial shapes (different for each level)
        level_spatial_shapes = torch.tensor(
            [[8, 8], [4, 4], [2, 2]], device=device
        ).float()

        # Tokens per level per batch (variable)
        tokens_per_level = [
            [64, 16, 4],  # Batch 1
            [50, 15, 3],  # Batch 2 (partial grid)
            [60, 12, 4],  # Batch 3 (partial grid)
        ]

        # Calculate total sequence length and batch offsets
        seq_lengths = torch.tensor(
            [sum(batch) for batch in tokens_per_level], device=device
        )
        batch_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        batch_offsets[1:] = torch.cumsum(seq_lengths, 0)
        total_seq_length = batch_offsets[-1]

        # Create embeddings
        x = torch.randn(total_seq_length, base_config["embed_dim"], device=device)
        x.requires_grad_(True)

        # Create spatial positions and level indices
        spatial_positions = torch.randn(total_seq_length, 2, device=device)
        level_indices = torch.randint(
            0, num_levels, (total_seq_length,), dtype=torch.long, device=device
        )

        # Run forward pass
        output = module(
            x, spatial_positions, level_indices, level_spatial_shapes, batch_offsets
        )

        # Check output shape
        assert output.shape == x.shape

        # Check output has no NaNs or Infs
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Try backward pass
        output.sum().backward()

        # Check gradients
        assert x.grad is not None
        for name, param in module.named_parameters():
            assert param.grad is not None, f"{name} has no grad"
