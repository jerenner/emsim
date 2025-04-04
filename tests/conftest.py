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

    cuda_marker = metafunc.definition.get_closest_marker("cuda")
    no_cuda = metafunc.config.getoption("--no-cuda")

    devices = ["cpu"]
    if cuda_marker and not no_cuda and CUDA_AVAILABLE:
        devices.append("cuda")

    metafunc.parametrize("device", devices, ids=lambda d: f"device={d}")
