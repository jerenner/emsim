import pytest
import torch
from torch import Tensor

from emsim.utils.sparse_utils.batching.batch_utils import (
    split_batch_concatted_tensor,
    normalize_batch_offsets,
    batch_offsets_to_seq_lengths,
    batch_offsets_to_indices,
    deconcat_add_batch_dim,
    remove_batch_dim_and_concat,
    batch_dim_to_leading_index,
    batch_offsets_from_sparse_tensor_indices,
)
from emsim.utils.sparse_utils.validation import validate_atleast_nd


@pytest.fixture
def simple_tensor(device):
    """Create a simple tensor for testing."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device=device)


@pytest.fixture
def simple_batch_offsets(device):
    """Create simple batch offsets for testing."""
    return torch.tensor([0, 3, 7, 9], device=device)


@pytest.mark.cpu_and_cuda
class TestNormalizeBatchOffsets:
    def test_already_normalized(self, device):
        """Test when offsets already end with total length."""
        batch_offsets = torch.tensor([0, 5, 10], device=device)
        total_length = 10
        result = normalize_batch_offsets(batch_offsets, total_length)
        assert torch.equal(result, batch_offsets)

    def test_needs_normalizing(self, device):
        """Test when normalization is needed."""
        batch_offsets = torch.tensor([0, 5, 8], device=device)
        total_length = 10
        result = normalize_batch_offsets(batch_offsets, total_length)
        expected = torch.tensor([0, 5, 8, 10], device=device)
        assert torch.equal(result, expected)

    def test_different_dtype(self, device):
        """Test with different dtypes."""
        batch_offsets = torch.tensor([0, 5, 8], dtype=torch.int32, device=device)
        total_length = 10
        result = normalize_batch_offsets(batch_offsets, total_length)
        assert result.dtype == batch_offsets.dtype


@pytest.mark.cpu_and_cuda
class TestComputeSeqLengths:
    def test_basic_functionality(self, device):
        """Test basic functionality of compute_seq_lengths."""
        batch_offsets = torch.tensor([0, 3, 7, 10], device=device)
        result = batch_offsets_to_seq_lengths(batch_offsets)
        expected = torch.tensor([3, 4, 3], device=device)
        assert torch.equal(result, expected)

    def test_single_batch(self, device):
        """Test with a single batch."""
        batch_offsets = torch.tensor([0, 5], device=device)
        result = batch_offsets_to_seq_lengths(batch_offsets)
        expected = torch.tensor([5], device=device)
        assert torch.equal(result, expected)


@pytest.mark.cpu_and_cuda
class TestValidateTensorDims:
    def test_valid_dims(self, device):
        """Test with valid dimensions."""
        tensor = torch.rand(3, 4, 5, device=device)
        # Should not raise any exception
        validate_atleast_nd(tensor, 3)

    def test_too_few_dims(self, device):
        """Test with too few dimensions."""
        tensor = torch.rand(3, 4, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # type: ignore
            match="Expected tensor to have at least 3 dimensions",
        ):
            validate_atleast_nd(tensor, 3)

    def test_custom_name(self, device):
        """Test with custom tensor name."""
        tensor = torch.rand(3, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # type: ignore
            match="Expected custom_tensor to have at least 2 dimensions",
        ):
            validate_atleast_nd(tensor, 2, name="custom_tensor")


@pytest.mark.cpu_and_cuda
class TestBatchOffsetsToIndices:
    def test_basic_functionality(self, device):
        """Test basic functionality."""
        batch_offsets = torch.tensor([0, 3, 7, 10], device=device)
        result = batch_offsets_to_indices(batch_offsets)
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], device=device)
        assert torch.equal(result, expected)

    def test_with_total_seq_length(self, device):
        """Test with total_seq_length parameter."""
        batch_offsets = torch.tensor([0, 3, 7], device=device)
        total_seq_length = 10
        result = batch_offsets_to_indices(batch_offsets, total_seq_length)
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], device=device)
        assert torch.equal(result, expected)


@pytest.mark.cpu_and_cuda
class TestSplitBatchConcattedTensor:
    def test_basic_functionality(
        self, simple_tensor: Tensor, simple_batch_offsets: Tensor, device
    ):
        """Test basic splitting functionality."""
        result = split_batch_concatted_tensor(simple_tensor, simple_batch_offsets)

        # Expected splits: [1,2,3], [4,5,6,7], [8,9]
        assert len(result) == 3
        assert torch.equal(result[0], torch.tensor([1.0, 2.0, 3.0], device=device))
        assert torch.equal(result[1], torch.tensor([4.0, 5.0, 6.0, 7.0], device=device))
        assert torch.equal(result[2], torch.tensor([8.0, 9.0], device=device))

    def test_multidimensional_tensor(self, device):
        """Test with multidimensional tensors."""
        tensor = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
                [13.0, 14.0],
            ],
            device=device,
        )

        batch_offsets = torch.tensor([0, 3, 5, 7], device=device)
        result = split_batch_concatted_tensor(tensor, batch_offsets)

        assert len(result) == 3
        assert torch.equal(result[0], tensor[0:3])
        assert torch.equal(result[1], tensor[3:5])
        assert torch.equal(result[2], tensor[5:7])


@pytest.mark.cpu_and_cuda
class TestDeconcatAddBatchDim:
    def test_basic_functionality(self, device):
        """Test basic functionality."""
        tensor = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            device=device,
        )

        batch_offsets = torch.tensor([0, 3, 6], device=device)

        result, padding_mask = deconcat_add_batch_dim(tensor, batch_offsets)

        # Expected shape: [2, 3, 2] (2 batches, max_len=3, feature_dim=2)
        assert result.shape == (2, 3, 2)

        # Check values
        assert torch.equal(result[0], tensor[0:3])
        assert torch.equal(result[1], tensor[3:6])

        # No padding needed in this case
        expected_mask = torch.zeros(2, 3, dtype=torch.bool, device=device)
        assert torch.equal(padding_mask, expected_mask)

    def test_with_padding(self, device):
        """Test with sequences of different lengths requiring padding."""
        tensor = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device=device
        )

        batch_offsets = torch.tensor([0, 2, 5], device=device)

        result, padding_mask = deconcat_add_batch_dim(tensor, batch_offsets)

        # Expected shape: [2, 3, 2] (2 batches, max_len=3, feature_dim=2)
        assert result.shape == (2, 3, 2)

        # Check padding mask
        expected_mask = torch.tensor(
            [[False, False, True], [False, False, False]],
            dtype=torch.bool,
            device=device,
        )
        assert torch.equal(padding_mask, expected_mask)

        # Check padded values
        assert torch.equal(result[0, 0:2], tensor[0:2])
        assert torch.all(result[0, 2] == 0.0)  # Padded with zeros

    def test_custom_pad_value(self, device):
        """Test with custom pad value."""
        tensor = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], device=device
        )

        batch_offsets = torch.tensor([0, 1, 4], device=device)

        result, padding_mask = deconcat_add_batch_dim(
            tensor, batch_offsets, pad_value=-1.0
        )

        # Check padded values have custom pad value
        assert torch.all(result[0, 1:3] == -1.0)

    def test_error_handling(self, device):
        """Test error handling."""
        # Test with 1D tensor (less than 2 dims)
        tensor_1d = torch.tensor([1.0, 2.0, 3.0], device=device)
        batch_offsets = torch.tensor([0, 3], device=device)

        with pytest.raises(
            (ValueError, torch.jit.Error),  # type: ignore
            match="Expected tensor to have at least 2 dimensions",
        ):
            deconcat_add_batch_dim(tensor_1d, batch_offsets)


@pytest.mark.cpu_and_cuda
class TestRemoveBatchDimAndConcat:
    def test_basic_functionality(self, device):
        """Test basic functionality."""
        # Create a batched tensor with padding
        tensor = torch.tensor(
            [
                [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
                [[4.0, 4.1], [5.0, 5.1], [0.0, 0.0]],  # Last position is padding
            ],
            device=device,
        )

        padding_mask = torch.tensor(
            [[False, False, False], [False, False, True]], device=device
        )

        result, batch_offsets = remove_batch_dim_and_concat(tensor, padding_mask)

        # Expected shape: [5, 2] (total non-padded elements, feature dim)
        assert result.shape == (5, 2)

        # Check values
        expected_values = torch.tensor(
            [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1], [5.0, 5.1]], device=device
        )
        assert torch.allclose(result, expected_values)

        # Check batch offsets
        expected_offsets = torch.tensor([0, 3, 5], device=device)
        assert torch.equal(batch_offsets, expected_offsets)

    def test_without_padding_mask(self, device):
        """Test without a padding mask."""
        tensor = torch.tensor(
            [[[1.0, 1.1], [2.0, 2.1]], [[3.0, 3.1], [4.0, 4.1]]], device=device
        )

        result, batch_offsets = remove_batch_dim_and_concat(tensor)

        # All elements kept - shape [4, 2]
        assert result.shape == (4, 2)

        # Should be just a view when all sequences are equal length
        assert result[0, 0] == 1.0
        tensor[0, 0, 0] = 9.9
        assert result[0, 0] == 9.9

        # Check batch offsets
        expected_offsets = torch.tensor([0, 2, 4], device=device)
        assert torch.equal(batch_offsets, expected_offsets)

    def test_empty_batch(self, device):
        """Test with an empty batch."""
        # Create a batched tensor with padding
        tensor = torch.tensor(
            [
                [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[4.0, 4.1], [5.0, 5.1], [0.0, 0.0]],  # Last position is padding
            ],
            device=device,
        )

        padding_mask = torch.tensor(
            [[False, False, False], [True, True, True], [False, False, True]],
            device=device,
        )

        result, batch_offsets = remove_batch_dim_and_concat(tensor, padding_mask)

        # Expected shape: [5, 2] (total non-padded elements, feature dim)
        assert result.shape == (5, 2)

        # Check values
        expected_values = torch.tensor(
            [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1], [5.0, 5.1]], device=device
        )
        assert torch.allclose(result, expected_values)

        # Check batch offsets
        expected_offsets = torch.tensor([0, 3, 3, 5], device=device)
        assert torch.equal(batch_offsets, expected_offsets)

    def test_error_handling(self, device):
        """Test error handling."""
        # Test with tensor with less than 3 dimensions
        tensor_2d = torch.rand(3, 4, device=device)
        with pytest.raises(
            (ValueError, torch.jit.Error),  # type: ignore
            match="Expected tensor to have at least 3 dimensions",
        ):
            remove_batch_dim_and_concat(tensor_2d)

        # Test with mismatched padding mask dimensions
        tensor = torch.rand(3, 4, 2, device=device)
        padding_mask_wrong_batch = torch.zeros(2, 4, device=device, dtype=torch.bool)
        padding_mask_wrong_batch[0, -1] = True
        with pytest.raises(
            (ValueError, torch.jit.Error), match="Batch size mismatch"  # type: ignore
        ):
            remove_batch_dim_and_concat(tensor, padding_mask_wrong_batch)


@pytest.mark.cpu_and_cuda
class TestBatchDimToLeadingIndex:
    def test_basic_functionality(self, device):
        """Test basic functionality."""
        tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=device)

        result = batch_dim_to_leading_index(tensor)

        # Expected shape: [2, 2, 3] (original + 1 for batch index)
        assert result.shape == (2, 2, 3)

        # Check batch indices in the first position of the last dimension
        assert torch.equal(result[0, 0, 0], torch.tensor(0, device=device))
        assert torch.equal(result[0, 1, 0], torch.tensor(0, device=device))
        assert torch.equal(result[1, 0, 0], torch.tensor(1, device=device))
        assert torch.equal(result[1, 1, 0], torch.tensor(1, device=device))

        # Check original values in the remaining positions
        assert torch.equal(result[0, 0, 1:], torch.tensor([1, 2], device=device))
        assert torch.equal(result[0, 1, 1:], torch.tensor([3, 4], device=device))
        assert torch.equal(result[1, 0, 1:], torch.tensor([5, 6], device=device))
        assert torch.equal(result[1, 1, 1:], torch.tensor([7, 8], device=device))


@pytest.mark.cpu_and_cuda
class TestBatchOffsetsFromSparseTensorIndices:
    def test_basic_functionality(self, device):
        """Test basic functionality."""
        # Create indices for a sparse tensor with batch dimension first
        indices = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 1, 3]],  # Batch indices  # Other indices
            device=device,
        )

        result = batch_offsets_from_sparse_tensor_indices(indices)

        # Expected: [0, 2, 4, 6] - 3 batches of 2
        expected = torch.tensor([0, 2, 4, 6], device=device)
        assert torch.equal(result, expected)

    def test_non_contiguous_batches(self, device):
        """Test with non-contiguous batch indices."""
        # Sparse indices with gaps in batch indices (batch 1 missing)
        indices = torch.tensor(
            [
                [0, 0, 2, 2, 3],  # Batch indices: 0, 0, 2, 2, 3
                [0, 1, 0, 1, 0],  # Other indices
            ],
            device=device,
        )

        result = batch_offsets_from_sparse_tensor_indices(indices)

        # Each element is the first occurrence of that batch index
        expected = torch.tensor([0, 2, 2, 4, 5], device=device)
        assert torch.equal(result, expected)
