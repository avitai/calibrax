"""Tests for MetricRegistry, MetricEntry, MetricTier, and calculate_all."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.core.models import MetricDirection
from calibrax.metrics import (
    calculate_all,
    MetricEntry,
    MetricRegistry,
    MetricSignature,
    MetricTier,
)
from calibrax.metrics._registry import register_metric


class TestMetricTier:
    """Tests for MetricTier enum."""

    def test_values(self) -> None:
        """All four tiers should have string values."""
        assert MetricTier.PURE_FUNCTION == "pure_function"
        assert MetricTier.FROZEN_BACKBONE == "frozen_backbone"
        assert MetricTier.LEARNED == "learned"
        assert MetricTier.METRIC_LEARNING == "metric_learning"

    def test_is_str_enum(self) -> None:
        """MetricTier should be usable as a string."""
        assert isinstance(MetricTier.PURE_FUNCTION, str)


class TestMetricEntry:
    """Tests for MetricEntry dataclass."""

    def test_creation_minimal(self) -> None:
        """Should create with required fields only."""
        entry = MetricEntry(name="test", fn=None, tier=MetricTier.FROZEN_BACKBONE)
        assert entry.name == "test"
        assert entry.fn is None
        assert entry.tier == MetricTier.FROZEN_BACKBONE
        assert entry.domain == "general"
        assert entry.direction == MetricDirection.LOWER

    def test_creation_full(self) -> None:
        """Should create with all fields specified."""

        def dummy(p: object, t: object) -> float:
            return 0.0

        entry = MetricEntry(
            name="dummy",
            fn=dummy,
            tier=MetricTier.PURE_FUNCTION,
            domain="image",
            direction=MetricDirection.HIGHER,
            description="A test metric",
            required_extra="image",
        )
        assert entry.fn is dummy
        assert entry.domain == "image"
        assert entry.required_extra == "image"

    def test_frozen(self) -> None:
        """MetricEntry should be immutable."""
        entry = MetricEntry(name="test", fn=None, tier=MetricTier.PURE_FUNCTION)
        with pytest.raises(AttributeError):
            entry.name = "changed"  # type: ignore[misc]

    def test_axiom_defaults(self) -> None:
        """Axiom fields should have correct defaults."""
        entry = MetricEntry(name="test", fn=None, tier=MetricTier.PURE_FUNCTION)
        assert entry.properties.is_true_metric is False
        assert entry.properties.is_symmetric is False
        assert entry.properties.is_proper is False
        assert entry.properties.is_differentiable is True
        assert entry.properties.is_jit_compatible is True
        assert entry.properties.invariances == ()

    def test_invariances(self) -> None:
        """Should accept and store invariance tuples."""
        from calibrax.metrics._types import MetricProperties

        entry = MetricEntry(
            name="test",
            fn=None,
            tier=MetricTier.PURE_FUNCTION,
            properties=MetricProperties(invariances=("translation", "rotation")),
        )
        assert entry.properties.invariances == ("translation", "rotation")


class TestMetricRegistry:
    """Tests for MetricRegistry singleton and operations."""

    def test_singleton(self) -> None:
        """Two calls should return the same instance."""
        r1 = MetricRegistry()
        r2 = MetricRegistry()
        assert r1 is r2

    def test_builtin_metrics_registered(self) -> None:
        """All 13 regression metrics should be registered at import time."""
        registry = MetricRegistry()
        expected = {
            "crps",
            "mse",
            "mae",
            "rmse",
            "r_squared",
            "mape",
            "relative_error",
            "explained_variance",
            "max_error",
            "huber_loss",
            "quantile_loss",
            "log_cosh_loss",
            "smape",
        }
        registered = set(registry.list_names())
        assert expected.issubset(registered)

    def test_crps_registered_as_proper_ensemble_metric(self) -> None:
        """CRPS should expose proper-scoring metadata without joining default batches."""
        registry = MetricRegistry()
        entry = registry.get("crps")
        assert entry.properties.is_proper is True
        assert entry.properties.is_jit_compatible is True
        assert entry.signature == MetricSignature.ENSEMBLE_PREDICTIONS_TARGETS

    def test_get_returns_metric_entry(self) -> None:
        """get() should return a MetricEntry."""
        entry = MetricRegistry().get("mse")
        assert isinstance(entry, MetricEntry)
        assert entry.name == "mse"
        assert entry.tier == MetricTier.PURE_FUNCTION

    def test_get_function(self) -> None:
        """get_function() should return a callable for Tier 0."""
        fn = MetricRegistry().get_function("mse")
        result = fn(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
        assert result == pytest.approx(0.0, abs=1e-7)

    def test_list_by_domain(self) -> None:
        """list_by_domain('general') should include regression metrics."""
        entries = MetricRegistry().list_by_domain("general")
        names = {e.name for e in entries}
        assert "mse" in names

    def test_list_by_tier(self) -> None:
        """list_by_tier(PURE_FUNCTION) should include regression metrics."""
        entries = MetricRegistry().list_by_tier(MetricTier.PURE_FUNCTION)
        names = {e.name for e in entries}
        assert "mse" in names

    def test_unknown_metric_raises(self) -> None:
        """get() for unknown name should raise KeyError."""
        with pytest.raises(KeyError):
            MetricRegistry().get("nonexistent_metric_xyz")

    def test_list_jit_compatible(self) -> None:
        """All built-in regression metrics should be JIT-compatible."""
        entries = MetricRegistry().list_jit_compatible()
        names = {e.name for e in entries}
        assert "mse" in names

    def test_list_true_metrics(self) -> None:
        """MAE and RMSE satisfy metric space axioms; MSE does not."""
        entries = MetricRegistry().list_true_metrics()
        names = {e.name for e in entries}
        assert "mae" in names
        assert "rmse" in names
        assert "mse" not in names

    def test_list_proper_scoring_rules(self) -> None:
        """CRPS should be discoverable as a proper scoring rule."""
        entries = MetricRegistry().list_proper_scoring_rules()
        names = {e.name for e in entries}
        assert "crps" in names
        assert "mse" not in names

    def test_list_by_invariance(self) -> None:
        """Metrics with invariances should be findable."""
        from calibrax.metrics._types import MetricProperties

        registry = MetricRegistry()
        entry = MetricEntry(
            name="test_invariant",
            fn=lambda p, t: 0.0,
            tier=MetricTier.PURE_FUNCTION,
            properties=MetricProperties(invariances=("translation", "rotation")),
        )
        registry.register("test_invariant", entry)
        translation_metrics = registry.list_by_invariance("translation")
        assert any(e.name == "test_invariant" for e in translation_metrics)
        rotation_metrics = registry.list_by_invariance("rotation")
        assert any(e.name == "test_invariant" for e in rotation_metrics)
        scale_metrics = registry.list_by_invariance("scale")
        assert not any(e.name == "test_invariant" for e in scale_metrics)
        # Cleanup
        registry.remove("test_invariant")

    def test_direction_correctness(self) -> None:
        """Metrics should have correct direction metadata."""
        registry = MetricRegistry()
        assert registry.get("mse").direction == MetricDirection.LOWER
        assert registry.get("r_squared").direction == MetricDirection.HIGHER
        assert registry.get("explained_variance").direction == MetricDirection.HIGHER
        assert registry.get("huber_loss").direction == MetricDirection.LOWER

    def test_symmetry_correctness(self) -> None:
        """Symmetric metrics should be marked correctly."""
        registry = MetricRegistry()
        assert registry.get("mse").properties.is_symmetric is True
        assert registry.get("smape").properties.is_symmetric is True
        assert registry.get("mape").properties.is_symmetric is False
        assert registry.get("quantile_loss").properties.is_symmetric is False


class TestRegisterMetricDecorator:
    """Tests for the register_metric decorator."""

    def test_decorator_registers_function(self) -> None:
        """Decorated function should appear in the registry."""

        @register_metric("test_custom_metric", description="Test")
        def custom_metric(predictions: object, targets: object) -> float:
            return 0.0

        registry = MetricRegistry()
        assert registry.has("test_custom_metric")
        entry = registry.get("test_custom_metric")
        assert entry.fn is custom_metric
        assert entry.tier == MetricTier.PURE_FUNCTION
        # Cleanup
        registry.remove("test_custom_metric")


class TestCalculateAllFused:
    """Verify fused calculate_all matches individual metric calls."""

    _FUSED_NAMES = [
        "mse",
        "mae",
        "rmse",
        "r_squared",
        "mape",
        "relative_error",
        "explained_variance",
        "max_error",
        "huber_loss",
        "quantile_loss",
        "log_cosh_loss",
        "smape",
    ]

    def test_fused_returns_all_regression_metrics(self) -> None:
        """Fused path must return the exact 12 same-shape regression metric names."""
        predictions = jnp.array([1.0, 2.5, 3.2, 4.1, 5.8])
        targets = jnp.array([1.1, 2.0, 3.0, 4.0, 5.0])
        fused = calculate_all(predictions, targets, metrics=self._FUSED_NAMES)
        assert set(fused.keys()) == set(self._FUSED_NAMES)

    def test_fused_matches_individual(self) -> None:
        """Fused path must produce values identical to individual metric calls."""
        predictions = jnp.array([1.0, 2.5, 3.2, 4.1, 5.8])
        targets = jnp.array([1.1, 2.0, 3.0, 4.0, 5.0])
        fused = calculate_all(predictions, targets, metrics=self._FUSED_NAMES)
        for name, value in fused.items():
            fn = MetricRegistry().get_function(name)
            individual = fn(predictions, targets)
            assert float(value) == pytest.approx(float(individual), abs=1e-6), (
                f"Mismatch for {name}: fused={value}, individual={individual}"
            )


class TestCalculateAll:
    """Tests for calculate_all (registry-backed)."""

    def test_all_metrics_returned(self) -> None:
        """Default should return same-shape general Tier 0 metrics."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 2.1, 3.1])
        result = calculate_all(predictions, targets)
        expected = {
            "mse",
            "mae",
            "rmse",
            "r_squared",
            "mape",
            "relative_error",
            "explained_variance",
            "max_error",
            "huber_loss",
            "quantile_loss",
            "log_cosh_loss",
            "smape",
        }
        assert expected.issubset(set(result))
        assert "crps" not in result

    def test_subset_metrics(self) -> None:
        """Should return only requested metrics."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 2.1, 3.1])
        result = calculate_all(predictions, targets, metrics=["mse", "mae"])
        assert set(result) == {"mse", "mae"}

    def test_unknown_metric_raises(self) -> None:
        """Unknown metric name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            calculate_all(jnp.ones(3), jnp.zeros(3), metrics=["nonexistent"])

    def test_values_are_numeric_scalars(self) -> None:
        """All returned values should be numeric scalars (JAX arrays or floats)."""
        result = calculate_all(jnp.ones(3), jnp.zeros(3))
        for value in result.values():
            assert isinstance(value, (float, jax.Array))
