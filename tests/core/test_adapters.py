"""Tests for generic adapters: BenchmarkAdapter, NNXBenchmarkAdapter, AdapterRegistry.

Verifies ABC enforcement, NNX JIT compatibility, adapter resolution,
and generic target wrapping without domain-specific methods.
"""

from unittest.mock import MagicMock

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from calibrax.core.adapters import (
    adapt,
    AdapterRegistry,
    BenchmarkAdapter,
    NNXBenchmarkAdapter,
    register_adapter,
)


class TestBenchmarkAdapter:
    """Tests for BenchmarkAdapter ABC."""

    def test_can_adapt_default_false(self) -> None:
        """Default can_adapt returns False for any target."""
        assert BenchmarkAdapter.can_adapt(object()) is False

    def test_name_from_target_name(self) -> None:
        """Name resolved from target.name attribute."""
        target = MagicMock(spec=["name"])
        target.name = "my_pipeline"

        class ConcreteAdapter(BenchmarkAdapter):
            """Concrete adapter for testing."""

        adapter = ConcreteAdapter(target)
        assert adapter.name == "my_pipeline"

    def test_name_from_target_model_name(self) -> None:
        """Falls back to target.model_name when target.name is absent."""
        target = MagicMock(spec=["model_name"])
        target.model_name = "my_model"

        class ConcreteAdapter(BenchmarkAdapter):
            """Concrete adapter for testing."""

        adapter = ConcreteAdapter(target)
        assert adapter.name == "my_model"

    def test_name_default_unknown(self) -> None:
        """Falls back to 'unknown' when target has no name attributes."""

        class ConcreteAdapter(BenchmarkAdapter):
            """Concrete adapter for testing."""

        adapter = ConcreteAdapter(object())
        assert adapter.name == "unknown"

    def test_target_property_returns_wrapped_object(self) -> None:
        """The target property returns the original wrapped object."""
        original = {"key": "value"}

        class ConcreteAdapter(BenchmarkAdapter):
            """Concrete adapter for testing."""

        adapter = ConcreteAdapter(original)
        assert adapter.target is original

    def test_subclass_with_custom_can_adapt(self) -> None:
        """Subclass overrides can_adapt for specific target types."""

        class DictAdapter(BenchmarkAdapter):
            """Adapter that handles dicts."""

            @classmethod
            def can_adapt(cls, target: object) -> bool:
                return isinstance(target, dict)

        assert DictAdapter.can_adapt({"a": 1}) is True
        assert DictAdapter.can_adapt("not a dict") is False


class TestNNXBenchmarkAdapter:
    """Tests for NNXBenchmarkAdapter (nnx.Module-based)."""

    def test_can_adapt_nnx_module(self) -> None:
        """Returns True for nnx.Module instances."""
        model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
        assert NNXBenchmarkAdapter.can_adapt(model) is True

    def test_can_adapt_non_nnx(self) -> None:
        """Returns False for non-NNX objects."""
        assert NNXBenchmarkAdapter.can_adapt("not a module") is False

    def test_is_nnx_module(self) -> None:
        """NNXBenchmarkAdapter must be an nnx.Module for JIT compatibility."""
        model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
        adapter = NNXBenchmarkAdapter(model)
        assert isinstance(adapter, nnx.Module)

    def test_name_from_model_name_attribute(self) -> None:
        """Resolves name from model.name attribute."""

        class NamedModel(nnx.Module):
            """Model with a name attribute."""

            name = "my_model"

            def __call__(self, x: jax.Array) -> jax.Array:
                return x

        adapter = NNXBenchmarkAdapter(NamedModel())
        assert adapter.name == "my_model"

    def test_name_from_model_model_name_attribute(self) -> None:
        """Falls back to model.model_name when model.name is absent."""

        class LegacyModel(nnx.Module):
            """Model with model_name attribute."""

            model_name = "legacy_model"

            def __call__(self, x: jax.Array) -> jax.Array:
                return x

        adapter = NNXBenchmarkAdapter(LegacyModel())
        assert adapter.name == "legacy_model"

    def test_name_default_unknown(self) -> None:
        """Falls back to 'unknown' for models without name attributes."""
        model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
        adapter = NNXBenchmarkAdapter(model)
        assert adapter.name == "unknown"

    def test_model_accessible(self) -> None:
        """The wrapped model should be accessible as .model attribute."""
        model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
        adapter = NNXBenchmarkAdapter(model)
        assert adapter.model is model

    def test_nnx_split_merge_roundtrip(self) -> None:
        """Adapter supports nnx.split/merge for state serialization."""

        class NamedModel(nnx.Module):
            """Model with name for round-trip test."""

            name = "roundtrip_model"

            def __call__(self, x: jax.Array) -> jax.Array:
                return x

        adapter = NNXBenchmarkAdapter(NamedModel())

        graphdef, state = nnx.split(adapter)
        restored = nnx.merge(graphdef, state)

        assert isinstance(restored, NNXBenchmarkAdapter)
        assert restored.name == "roundtrip_model"
        assert isinstance(restored.model, NamedModel)

    def test_nnx_jit_on_subclass_method(self) -> None:
        """JIT compilation works on subclass methods via unbound form."""

        class LinearAdapter(NNXBenchmarkAdapter):
            """Adapter subclass with a forward method."""

            def forward(self, x: jax.Array) -> jax.Array:
                """Forward pass through the model."""
                return self.model(x)  # type: ignore[operator]

        model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
        adapter = LinearAdapter(model)
        x = jnp.ones((1, 2))

        # Eager execution
        eager_result = adapter.forward(x)

        # JIT execution via unbound method
        jit_forward = nnx.jit(LinearAdapter.forward)
        jit_result = jit_forward(adapter, x)

        assert isinstance(jit_result, jax.Array)
        assert jit_result.shape == (1, 3)
        assert jnp.allclose(eager_result, jit_result)


class TestAdapterRegistry:
    """Tests for AdapterRegistry."""

    def test_register_and_adapt(self) -> None:
        """Registered adapter is used to wrap matching targets."""
        registry = AdapterRegistry()
        registry.register(NNXBenchmarkAdapter)
        model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
        adapter = registry.adapt(model)
        assert isinstance(adapter, NNXBenchmarkAdapter)

    def test_adapt_unknown_raises(self) -> None:
        """Raises ValueError when no adapter can handle the target."""
        registry = AdapterRegistry()
        with pytest.raises(ValueError, match="No adapter found"):
            registry.adapt("not a model")

    def test_reset_clears_adapters(self) -> None:
        """Reset removes all registered adapters."""
        registry = AdapterRegistry()
        registry.register(NNXBenchmarkAdapter)
        registry.reset()
        with pytest.raises(ValueError):
            registry.adapt(nnx.Linear(2, 3, rngs=nnx.Rngs(0)))

    def test_priority_order(self) -> None:
        """Most recently registered adapter takes priority."""

        class SpecialAdapter(BenchmarkAdapter):
            """Higher-priority adapter that handles all objects."""

            @classmethod
            def can_adapt(cls, target: object) -> bool:
                return isinstance(target, nnx.Module)

        registry = AdapterRegistry()
        registry.register(NNXBenchmarkAdapter)
        registry.register(SpecialAdapter)

        model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
        adapter = registry.adapt(model)
        assert isinstance(adapter, SpecialAdapter)


class TestModuleConvenience:
    """Tests for module-level adapt and register_adapter."""

    def test_adapt_with_default_registry(self) -> None:
        """Default registry resolves NNX modules to NNXBenchmarkAdapter."""
        model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
        adapter = adapt(model)
        assert isinstance(adapter, NNXBenchmarkAdapter)

    def test_register_custom_adapter(self) -> None:
        """Custom adapter registered and resolved via default registry."""

        class DictAdapter(BenchmarkAdapter):
            """Custom adapter for dict targets."""

            @classmethod
            def can_adapt(cls, target: object) -> bool:
                return isinstance(target, dict)

        register_adapter(DictAdapter)
        adapter = adapt({"key": "value"})
        assert isinstance(adapter, DictAdapter)
        assert adapter.name == "unknown"
        assert adapter.target == {"key": "value"}
