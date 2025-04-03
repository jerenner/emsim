# conftest.py
import os
import pytest
import torch


CUDA_AVAILABLE = torch.cuda.is_available()


def pytest_generate_tests(metafunc: pytest.Metafunc):
    generate_torchscript_parameterization(metafunc)
    generate_cuda_parameterization(metafunc)


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disable CUDA tests even if GPU is available"
    )


def generate_torchscript_parameterization(metafunc: pytest.Metafunc):
    """
    Dynamically parametrizes tests to run with/without TorchScript based on markers.
    - Unmarked tests run in both modes (TorchScript enabled/disabled).
    - Tests with @torchscript or @no_torchscript markers run in one mode only.
    - Generates human-readable test names showing TorchScript state.
    """
    if "enable_ts" not in metafunc.fixturenames:
        return

    # Check for markers
    has_ts_marker = any(
        m.name == "torchscript" for m in metafunc.definition.own_markers
    )
    has_no_ts_marker = any(
        m.name == "no_torchscript" for m in metafunc.definition.own_markers
    )

    if has_ts_marker and has_no_ts_marker:
        raise ValueError(
            "Test cannot have both @torchscript and @no_torchscript markers"
        )

    # Parameterize with descriptive IDs
    if has_ts_marker:
        params = [True]
        ids = ["torchscript=True"]
    elif has_no_ts_marker:
        params = [False]
        ids = ["torchscript=False"]
    else:
        params = [True, False]
        ids = ["torchscript=True", "torchscript=False"]

    metafunc.parametrize("enable_ts", params, ids=ids, indirect=True)


def generate_cuda_parameterization(metafunc: pytest.Metafunc):
    """Handles CPU/CUDA device parameterization"""
    if "device" not in metafunc.fixturenames:
        return

    cuda_marker = metafunc.definition.get_closest_marker("cuda")
    no_cuda = metafunc.config.getoption("--no-cuda")

    devices = ["cpu"]
    if cuda_marker and not no_cuda and CUDA_AVAILABLE:
        devices.append("cuda")

    metafunc.parametrize("device", devices, ids=lambda d: f"device={d}")


@pytest.fixture
def enable_ts(request):
    """Indirect fixture to pass TorchScript enable/disable state to tests"""
    return request.param


@pytest.fixture(autouse=True)
def manage_torchscript(enable_ts):
    """Autouse fixture to set PYTORCH_JIT based on parameter/markers."""
    original = os.environ.get("PYTORCH_JIT")
    os.environ["PYTORCH_JIT"] = "1" if enable_ts else "0"

    yield

    # Restore original value
    if original is not None:
        os.environ["PYTORCH_JIT"] = original
    else:
        os.environ.pop("PYTORCH_JIT", None)
