import os
import pytest

def test_default_behavior():
    """Runs twice - with both TorchScript states"""
    assert "PYTORCH_JIT" in os.environ

@pytest.mark.torchscript
def test_requires_torchscript():
    """Runs once with TorchScript enabled"""
    assert os.environ["PYTORCH_JIT"] == "1"

@pytest.mark.no_torchscript
def test_requires_no_torchscript():
    """Runs once with TorchScript disabled"""
    assert os.environ["PYTORCH_JIT"] == "0"
