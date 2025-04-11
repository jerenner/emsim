import torch
from hypothesis import strategies as st


# Utility function for checking tensor closeness with detailed error messages
def assert_close(actual, expected, rtol=1e-5, atol=1e-8, msg=""):
    """Assert that tensors are close within tolerance with helpful error output."""
    is_close = torch.allclose(actual, expected, rtol=rtol, atol=atol)
    if not is_close:
        max_diff = (actual - expected).abs().max().item()
        shapes = f"Shapes: actual {actual.shape}, expected {expected.shape}"
        error_msg = f"{msg} Max difference: {max_diff}. {shapes}"
        assert False, error_msg


# Strategies for generating valid tensor dimensions
def valid_dims():
    return st.integers(min_value=1, max_value=8)


def even_dims():
    return st.integers(min_value=2, max_value=8).filter(lambda x: x % 2 == 0)


def batch_dims_strategy():
    return st.lists(
        st.integers(min_value=1, max_value=8),
        min_size=1,
        max_size=4,
    )
