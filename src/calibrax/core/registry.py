"""Generic type-safe registry and benchmark registry singleton.

Provides a reusable Registry[T] for named item storage, a
SingletonRegistry[T] for shared-instance registries, a
BenchmarkRegistry singleton specialization, and convenience
functions for benchmark registration.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Self, TypeVar


_C = TypeVar("_C", bound=type)


class Registry[T]:
    """Generic type-safe registry for named items.

    Supports register, get, remove, clear, and iteration.
    Not a singleton — allows multiple independent instances.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        """Register an item under the given name.

        Args:
            name: Unique name for the item.
            item: The item to register.
        """
        self._items[name] = item

    def get(self, name: str) -> T:
        """Retrieve an item by name.

        Args:
            name: Name of the item to retrieve.

        Returns:
            The registered item.

        Raises:
            KeyError: If no item is registered under the given name.
        """
        if name not in self._items:
            raise KeyError(f"'{name}' not found in registry")
        return self._items[name]

    def list_names(self) -> list[str]:
        """Return all registered names.

        Returns:
            List of registered item names.
        """
        return list(self._items.keys())

    def has(self, name: str) -> bool:
        """Check if a name is registered.

        Args:
            name: Name to check.

        Returns:
            True if the name is registered.
        """
        return name in self._items

    def remove(self, name: str) -> None:
        """Remove an item by name.

        Args:
            name: Name of the item to remove.

        Raises:
            KeyError: If no item is registered under the given name.
        """
        if name not in self._items:
            raise KeyError(f"'{name}' not found in registry")
        del self._items[name]

    def clear(self) -> None:
        """Remove all registered items."""
        self._items.clear()

    def __len__(self) -> int:
        """Return the number of registered items."""
        return len(self._items)

    def __contains__(self, name: object) -> bool:
        """Check if a name is registered.

        Args:
            name: Name to check.

        Returns:
            True if the name is registered.
        """
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered names."""
        return iter(self._items)


class SingletonRegistry[T](Registry[T]):
    """Registry that enforces a single shared instance per subclass.

    Subclasses inherit register/get/remove/clear/iteration from Registry[T]
    and gain automatic singleton semantics with a reset() classmethod for
    test isolation.

    Usage:
        class MyRegistry(SingletonRegistry[MyItem]):
            '''Project-specific singleton registry.'''
    """

    _instance: Self | None = None  # type: ignore[misc]

    def __new__(cls) -> Self:
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._items = {}  # type: ignore[attr-defined]
        return cls._instance

    def __init__(self) -> None:
        """No-op init to prevent re-initialization of singleton state."""

    @classmethod
    def reset(cls) -> None:
        """Clear all registered items.

        Intended for test isolation to prevent cross-test leakage.
        """
        if cls._instance is not None:
            cls._instance._items.clear()


class BenchmarkRegistry(SingletonRegistry[object]):
    """Singleton registry for benchmark implementations.

    Ensures a single shared registry instance across the application.
    Provides reset() for test isolation.
    """


def register_benchmark(name: str) -> Callable[[_C], _C]:
    """Class decorator that registers a class into the BenchmarkRegistry.

    Args:
        name: Name to register the class under.

    Returns:
        Decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: _C) -> _C:
        """Register the class and return it unchanged."""
        registry = BenchmarkRegistry()
        registry.register(name, cls)
        return cls

    return decorator


def get_benchmark(name: str) -> object:
    """Retrieve a benchmark by name from the singleton registry.

    Args:
        name: Name of the benchmark.

    Returns:
        The registered benchmark.

    Raises:
        KeyError: If no benchmark is registered under the given name.
    """
    return BenchmarkRegistry().get(name)


def list_benchmarks() -> list[str]:
    """List all registered benchmark names.

    Returns:
        List of registered benchmark names.
    """
    return BenchmarkRegistry().list_names()
