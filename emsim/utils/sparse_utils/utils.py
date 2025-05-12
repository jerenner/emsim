import functools
import torch

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiNonlinearityBase

    has_minkowskiengine = True
except ImportError:
    MinkowskiEngine = None
    MinkowskiNonlinearityBase = None
    has_minkowskiengine = False
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor

    has_spconv = True
except ImportError:
    spconv = None
    SparseConvTensor = None
    has_spconv = False


def requires_minkowskiengine(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not has_minkowskiengine:
            raise ModuleNotFoundError("Could not import MinkowskiEngine")
        return func(*args, **kwargs)

    return wrapper


def requires_spconv(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not has_spconv:
            raise ModuleNotFoundError("Could not import spconv")
        return func(*args, **kwargs)

    return wrapper


def check_pytorch_version(required_version: str) -> bool:
    try:
        current_version = tuple(map(int, torch.__version__.split(".")))
        required_version_tuple = tuple(map(int, required_version.split(".")))

        return current_version >= required_version_tuple
    except ValueError:
        raise ValueError(f"Invalid version string: {required_version}")
