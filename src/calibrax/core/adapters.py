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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol
from typing_extensions import TypeIs

from flax import nnx


def _resolved_name(target: object) -> str:
    """What an adapted object calls itself.

    Args:
        target: The adapted object.

    Returns:
        Its ``name``, else its ``model_name``, else ``"unknown"``.
    """
    return getattr(target, "name", None) or getattr(target, "model_name", None) or "unknown"


class BenchmarkAdapter[TargetT](ABC):
    """Base class for non-NNX benchmark adapters.

    Wraps an arbitrary target (model, data pipeline, solver, etc.)
    with identity and discoverability. Subclasses add domain-specific
    methods — this base imposes no interface beyond ``name``,
    ``target``, and ``can_adapt``.

    For NNX models, use NNXBenchmarkAdapter instead — it inherits from
    nnx.Module for JIT/vmap/grad compatibility.
    """

    def __init__(self, target: TargetT) -> None:
        """Initialize the adapter with a target.

        Resolves the name from ``target.name``, then ``target.model_name``,
        falling back to ``"unknown"``.

        Args:
            target: The object to adapt for benchmarking.
        """
        self._target = target
        self._name: str = _resolved_name(target)

    @property
    def target(self) -> TargetT:
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
    @abstractmethod
    def can_adapt(cls, target: object) -> bool:
        """Check if this adapter can handle the given target.

        ``AdapterRegistry`` picks the first registered adapter whose ``can_adapt`` accepts the
        target, so every adapter states which targets it wraps. An adapter a registry holds
        returns ``TypeIs[TargetT]``, which lets the registry hand the target to its constructor.

        Args:
            target: The object to check.

        Returns:
            True if this adapter can wrap the target.
        """


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
        self._name_value: str = _resolved_name(model)

    @property
    def name(self) -> str:
        """Get the name of the adapted model.

        Returns:
            The model name string.
        """
        return self._name_value

    @classmethod
    def can_adapt(cls, target: object) -> TypeIs[nnx.Module]:
        """Check if the target is a Flax NNX Module.

        Args:
            target: The object to check.

        Returns:
            True if the target is an nnx.Module instance.
        """
        return isinstance(target, nnx.Module)


type Adapter = BenchmarkAdapter[object] | NNXBenchmarkAdapter
"""An adapter of either kind: a plain wrapper or an NNX module."""


class AdapterClass[TargetT](Protocol):
    """An adapter class a registry holds: a type predicate and a constructor for its targets."""

    def can_adapt(self, target: object, /) -> TypeIs[TargetT]:
        """Whether ``target`` is one of the adapter's targets."""
        ...

    def __call__(self, target: TargetT, /) -> BenchmarkAdapter[TargetT] | NNXBenchmarkAdapter:
        """Wrap ``target``."""
        ...


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
        self._factories: list[Callable[[object], Adapter | None]] = []

    def register[TargetT](self, adapter_cls: AdapterClass[TargetT]) -> None:
        """Register an adapter class (highest priority first).

        Args:
            adapter_cls: The adapter class; its ``can_adapt`` is a type predicate for the
                targets its constructor takes.
        """

        def wrap(target: object) -> Adapter | None:
            return adapter_cls(target) if adapter_cls.can_adapt(target) else None

        self._factories.insert(0, wrap)

    def adapt(self, target: object) -> Adapter:
        """Find and apply a suitable adapter for the target.

        Args:
            target: The object to adapt.

        Returns:
            An adapter wrapping the target.

        Raises:
            ValueError: If no registered adapter can handle the target.
        """
        for wrap in self._factories:
            adapter = wrap(target)
            if adapter is not None:
                return adapter

        name = type(target).__name__
        msg = f"No adapter found for target of type {name}"
        raise ValueError(msg)

    def reset(self) -> None:
        """Remove all registered adapters."""
        self._factories.clear()


# Default registry with NNX adapter pre-registered
_default_registry = AdapterRegistry()
_default_registry.register(NNXBenchmarkAdapter)


def adapt(target: object) -> Adapter:  # noqa: DOC502  # raised by AdapterRegistry.adapt
    """Adapt a target using the default registry.

    Args:
        target: The object to adapt.

    Returns:
        An adapter wrapping the target.

    Raises:
        ValueError: If no adapter can handle the target.
    """
    return _default_registry.adapt(target)


def register_adapter[TargetT](adapter_cls: AdapterClass[TargetT]) -> None:
    """Register an adapter class into the default registry.

    Args:
        adapter_cls: The adapter class to register.
    """
    _default_registry.register(adapter_cls)
