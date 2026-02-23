"""Generic adapters for the calibrax benchmarking framework.

Provides BenchmarkAdapter ABC for non-NNX targets,
NNXBenchmarkAdapter (nnx.Module) for JIT-compatible NNX wrapping,
and AdapterRegistry for managing adapter resolution.

These base classes are intentionally thin — they provide only identity,
target access, and discoverability. Domain-specific methods (predict,
sample, iterate, solve, etc.) are added by sister repos that extend
these bases.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

import flax.nnx as nnx


class BenchmarkAdapter(ABC):
    """Base class for non-NNX benchmark adapters.

    Wraps an arbitrary target (model, data pipeline, solver, etc.)
    with identity and discoverability. Subclasses add domain-specific
    methods — this base imposes no interface beyond ``name``,
    ``target``, and ``can_adapt``.

    For NNX models, use NNXBenchmarkAdapter instead — it inherits from
    nnx.Module for JIT/vmap/grad compatibility.
    """

    def __init__(self, target: Any) -> None:
        """Initialize the adapter with a target.

        Resolves the name from ``target.name``, then ``target.model_name``,
        falling back to ``"unknown"``.

        Args:
            target: The object to adapt for benchmarking.
        """
        self._target = target
        self._name: str = (
            getattr(target, "name", None) or getattr(target, "model_name", None) or "unknown"
        )

    @property
    def target(self) -> Any:
        """Get the wrapped target object.

        Returns:
            The original target passed to the constructor.
        """
        return self._target

    @property
    def name(self) -> str:
        """Get the name of the adapted target.

        Returns:
            The target name string.
        """
        return self._name

    @classmethod
    def can_adapt(cls, target: object) -> bool:
        """Check if this adapter can handle the given target.

        Returns False by default — subclasses override with specific checks.

        Args:
            target: The object to check.

        Returns:
            True if this adapter can wrap the target.
        """
        return False


class NNXBenchmarkAdapter(nnx.Module):
    """JIT-compatible adapter for Flax NNX modules.

    Inherits from nnx.Module so it participates in NNX's graph system.
    This enables:

    - ``nnx.jit`` for JIT-compiled execution
    - ``nnx.vmap`` for batched execution
    - Proper state mutation tracking (RNG, batch norm, etc.)
    - ``nnx.split`` / ``nnx.merge`` for state serialization

    Like BenchmarkAdapter, this base is intentionally thin — subclasses
    add domain-specific methods (predict, sample, solve, etc.).
    """

    def __init__(self, model: nnx.Module) -> None:
        """Initialize the adapter with an NNX module.

        The model becomes a tracked sub-module in the NNX graph system,
        ensuring its parameters and state are handled correctly during
        JIT compilation and other transforms.

        Resolves the name from ``model.name``, then ``model.model_name``,
        falling back to ``"unknown"``.

        Args:
            model: The NNX module to adapt.
        """
        self.model = model
        self._name_value: str = (
            getattr(model, "name", None) or getattr(model, "model_name", None) or "unknown"
        )

    @property
    def name(self) -> str:
        """Get the name of the adapted model.

        Returns:
            The model name string.
        """
        return self._name_value

    @classmethod
    def can_adapt(cls, target: object) -> bool:
        """Check if the target is a Flax NNX Module.

        Args:
            target: The object to check.

        Returns:
            True if the target is an nnx.Module instance.
        """
        return isinstance(target, nnx.Module)


class AdapterRegistry:
    """Registry of benchmark adapters.

    Maintains an ordered list of adapter classes and resolves
    the appropriate adapter for a given target via can_adapt checks.

    Accepts both BenchmarkAdapter subclasses (for non-NNX targets)
    and NNXBenchmarkAdapter subclasses (nnx.Module-based, for JIT-compatible
    NNX adapters).
    """

    def __init__(self) -> None:
        """Initialize an empty adapter registry."""
        self._adapters: list[type] = []

    def register(self, adapter_cls: type) -> None:
        """Register an adapter class (highest priority first).

        The adapter class must have a ``can_adapt(target)`` classmethod
        and accept a target as its first constructor argument.

        Args:
            adapter_cls: The adapter class to register.
        """
        self._adapters.insert(0, adapter_cls)

    def adapt(self, target: Any) -> Any:
        """Find and apply a suitable adapter for the target.

        Args:
            target: The object to adapt.

        Returns:
            An adapter wrapping the target.

        Raises:
            ValueError: If no registered adapter can handle the target.
        """
        for adapter_cls in self._adapters:
            if adapter_cls.can_adapt(target):
                return adapter_cls(target)

        name = type(target).__name__
        msg = f"No adapter found for target of type {name}"
        raise ValueError(msg)

    def reset(self) -> None:
        """Remove all registered adapters."""
        self._adapters.clear()


# Default registry with NNX adapter pre-registered
_default_registry = AdapterRegistry()
_default_registry.register(NNXBenchmarkAdapter)


def adapt(target: Any) -> Any:
    """Adapt a target using the default registry.

    Args:
        target: The object to adapt.

    Returns:
        An adapter wrapping the target.

    Raises:
        ValueError: If no adapter can handle the target.
    """
    return _default_registry.adapt(target)


def register_adapter(adapter_cls: type) -> None:
    """Register an adapter class into the default registry.

    Args:
        adapter_cls: The adapter class to register.
    """
    _default_registry.register(adapter_cls)
