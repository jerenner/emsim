import pytest
import torch


CUDA_AVAILABLE = torch.cuda.is_available()


def pytest_generate_tests(metafunc: pytest.Metafunc):
    generate_cuda_parameterization(metafunc)


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disable CUDA tests even if GPU is available",
    )


def generate_cuda_parameterization(metafunc: pytest.Metafunc):
    """Handles CPU/CUDA device parameterization"""
    if "device" not in metafunc.fixturenames:
        return

    cuda_marker = metafunc.definition.get_closest_marker("cuda_if_available")
    both_marker = metafunc.definition.get_closest_marker("cpu_and_cuda")
    no_cuda = metafunc.config.getoption("--no-cuda")

    # For tests with cpu_and_cuda marker, run on both CPU and CUDA if available
    if both_marker:
        if CUDA_AVAILABLE and not no_cuda:
            devices = ["cpu", "cuda"]
        else:
            devices = ["cpu"]
        metafunc.parametrize("device", devices, ids=lambda d: f"device={d}")
        return

    # For tests with cuda_if_available marker, use CUDA if available, otherwise CPU
    if cuda_marker:
        if CUDA_AVAILABLE and not no_cuda:
            devices = ["cuda"]
        else:
            devices = ["cpu"]
        metafunc.parametrize("device", devices, ids=lambda d: f"device={d}")
        return

    # Default to CPU for tests without markers
    metafunc.parametrize("device", ["cpu"], ids=lambda d: f"device={d}")
