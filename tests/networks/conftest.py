from functools import wraps
from typing import Any, Callable, Dict

import torch
from hypothesis import strategies as st
from torch import Tensor, nn


class ModuleHook:
    """Universal hook for capturing intermediate tensors in a module."""

    def __init__(self, module: nn.Module, points_to_hook: Dict[str, Callable]):
        """
        Initialize hook.

        Args:
            module: Module to hook
            points_to_hook: Dict mapping hook names to functions that return the component to hook
        """
        self.module = module
        self.captured_values: dict[str, dict[str, Any]] = {
            hook_name: {"inputs": {}, "outputs": []} for hook_name in points_to_hook
        }
        self.original_methods = {}

        for hook_name, component_getter in points_to_hook.items():
            component = component_getter(module)
            if isinstance(component, nn.Module):
                # For submodules
                self._hook_module(hook_name, component_getter)
            else:
                # For methods
                self._hook_method(hook_name, component_getter)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()

    def _process_value(self, value: Any) -> Any:
        """Recursively process values, detaching and cloning tensors."""
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        elif isinstance(value, (list, tuple)):
            return type(value)(self._process_value(item) for item in value)
        elif isinstance(value, dict):
            return {k: self._process_value(v) for k, v in value.items()}
        return value

    def _normalize_to_list(self, value: Any) -> list[Any]:
        """Normalize input or output to a list, handling single values and tuples."""
        if isinstance(value, tuple):
            return list(self._process_value(value))
        return [self._process_value(value)]

    def _hook_method(self, hook_name: str, component_getter: Callable):
        """Hook a method to capture inputs and outputs."""
        original_method = component_getter(self.module)
        self.original_methods[hook_name] = original_method

        @wraps(original_method)
        def hooked_method(*args, **kwargs):
            # Capture inputs
            captured_args = self._process_value(args)
            captured_kwargs = self._process_value(kwargs)
            self.captured_values[hook_name]["inputs"] = {
                "args": captured_args,
                "kwargs": captured_kwargs,
            }

            # Call original method
            result = original_method(*args, **kwargs)

            # Capture outputs (normalized to list)
            self.captured_values[hook_name]["outputs"] = self._normalize_to_list(result)

            return result

        # Replace the method
        self._set_attr_nested(hook_name, hooked_method)

    def _hook_module(self, hook_name: str, component_getter: Callable):
        """Hook a module to capture inputs and outputs."""
        component: nn.Module = component_getter(self.module)

        def forward_hook(module, args, kwargs, outputs):
            captured_args = self._process_value(args)
            captured_kwargs = self._process_value(kwargs)
            self.captured_values[hook_name]["inputs"] = {
                "args": captured_args,
                "kwargs": captured_kwargs,
            }

            # Capture outputs (normalized to list)
            self.captured_values[hook_name]["outputs"] = self._normalize_to_list(
                outputs
            )

        # Register hooks
        handle = component.register_forward_hook(forward_hook, with_kwargs=True)
        self.original_methods[hook_name] = handle

    def _set_attr_nested(self, path: str, value: Any):
        """Set attribute following nested path."""
        parts = path.split(".")
        obj = self.module
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def restore(self):
        """Restore original methods and remove hooks."""
        for hook_name, original in self.original_methods.items():
            if callable(original):
                # Restore method
                self._set_attr_nested(hook_name, original)
            else:
                # Remove hook
                original.remove()


@st.composite
def positions_strategy(draw):
    positions_dtype = draw(st.sampled_from([torch.float32, torch.long]))
    if positions_dtype == torch.float32:
        min_position = draw(
            st.floats(min_value=-1e30, max_value=1e30, exclude_max=True)
        )
        max_position = draw(st.floats(min_value=min_position, max_value=1e30))
    else:
        min_position = draw(st.integers(min_value=int(-1e10), max_value=int(1e10) - 1))
        max_position = draw(st.integers(min_value=min_position, max_value=int(1e10)))
    return positions_dtype, min_position, max_position


def tensor_unit_normal(
    tensor: Tensor, pass_all_zeros: bool = True, tolerance: float = 0.1
):
    """Tests whether a Pytorch tensor has near 0 mean and near unit standard deviation.

    Args:
        tensor (Tensor): Tensor to check
        pass_zero_var (bool): If True, the test is also considered successful if the
            tensor is all zeros (even though the standard deviation is not 1)
        tolerance (float): Tolerance for mean and standard deviation.

    Returns:
        boolean: True if the tensor is empty, all zeros and pass_all_zeros is True,
            or if the sample mean and standard deviation are within `tol` of 0
            and 1, respectively. False otherwise.
    """
    n = tensor.numel()
    if n == 0:
        return True
    if pass_all_zeros and torch.all(tensor == 0.0):
        return True

    return tensor.mean().abs().item() < tolerance and tensor.std().item() < tolerance

    # assert tensor.mean().abs().item() < 3 / math.sqrt(n)

    # # 4 standard errors of unit Gaussian standard deviation
    # # Using 4 standard errors for extra tolerance
    # assert tensor.std().item() < 4 / math.sqrt(2 * n)

    # # 3 standard errors of variance
    # # assert tensor.var().item() < 3 * math.sqrt(2.0 / max(n - 1, 1))
