"""Tests for metric composition classes."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from calibrax.metrics.composition import (
    MetricCollection,
    MetricSuite,
    ThresholdMetric,
    WeightedMetric,
)
from calibrax.metrics.functional.regression import mae, mse, rmse


class TestMetricCollection:
    """Tests for MetricCollection."""

    def test_compute_functional(self) -> None:
        collection = MetricCollection({"mse": mse, "mae": mae})
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        results = collection.compute_functional(predictions, targets)
        assert results["mse"] == pytest.approx(0.0, abs=1e-5)
        assert results["mae"] == pytest.approx(0.0, abs=1e-5)

    def test_add_remove(self) -> None:
        collection = MetricCollection({"mse": mse})
        assert "mse" in collection.names
        collection.add("mae", mae)
        assert "mae" in collection.names
        collection.remove("mse")
        assert "mse" not in collection.names

    def test_remove_missing_raises(self) -> None:
        collection = MetricCollection({"mse": mse})
        with pytest.raises(KeyError, match="not found"):
            collection.remove("nonexistent")

    def test_names_property(self) -> None:
        collection = MetricCollection({"mse": mse, "mae": mae, "rmse": rmse})
        assert sorted(collection.names) == ["mae", "mse", "rmse"]

    def test_from_registry(self) -> None:
        collection = MetricCollection.from_registry(domain="general")
        assert len(collection.names) >= 6  # At least 6 regression metrics

    def test_empty_collection(self) -> None:
        collection = MetricCollection({})
        predictions = jnp.array([1.0, 2.0])
        targets = jnp.array([1.0, 2.0])
        results = collection.compute_functional(predictions, targets)
        assert results == {}


class TestWeightedMetric:
    """Tests for WeightedMetric."""

    def test_known_weighted_sum(self) -> None:
        weighted = WeightedMetric({"mse": 0.7, "mae": 0.3})
        score = weighted.compute({"mse": 0.01, "mae": 0.05})
        expected = 0.7 * 0.01 + 0.3 * 0.05
        assert score == pytest.approx(expected, abs=1e-10)

    def test_equal_weights(self) -> None:
        weighted = WeightedMetric({"a": 1.0, "b": 1.0})
        score = weighted.compute({"a": 0.2, "b": 0.4})
        assert score == pytest.approx(0.6, abs=1e-10)

    def test_missing_metric_raises(self) -> None:
        weighted = WeightedMetric({"mse": 0.5, "mae": 0.5})
        with pytest.raises(KeyError, match="Required metric"):
            weighted.compute({"mse": 0.01})

    def test_empty_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            WeightedMetric({})

    def test_normalized_weights(self) -> None:
        weighted = WeightedMetric({"a": 2.0, "b": 3.0})
        nw = weighted.normalized_weights
        assert nw["a"] == pytest.approx(0.4, abs=1e-10)
        assert nw["b"] == pytest.approx(0.6, abs=1e-10)
        assert sum(nw.values()) == pytest.approx(1.0, abs=1e-10)

    def test_weights_property(self) -> None:
        weighted = WeightedMetric({"a": 1.0, "b": 2.0})
        assert weighted.weights == {"a": 1.0, "b": 2.0}


class TestMetricSuite:
    """Tests for MetricSuite."""

    def test_add_group_and_compute(self) -> None:
        suite = MetricSuite()
        suite.add_group("regression", ["mse", "mae"])
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        results = suite.compute_all(predictions, targets)
        assert "regression" in results
        assert results["regression"]["mse"] == pytest.approx(0.0, abs=1e-5)

    def test_multiple_groups(self) -> None:
        suite = MetricSuite()
        suite.add_group("error", ["mse"])
        suite.add_group("absolute", ["mae"])
        predictions = jnp.array([1.5, 2.5, 3.5])
        targets = jnp.array([1.0, 2.0, 3.0])
        results = suite.compute_all(predictions, targets)
        assert len(results) == 2
        assert "error" in results
        assert "absolute" in results

    def test_list_groups(self) -> None:
        suite = MetricSuite()
        suite.add_group("a", ["mse"])
        suite.add_group("b", ["mae"])
        assert sorted(suite.list_groups()) == ["a", "b"]

    def test_from_registry_domains(self) -> None:
        suite = MetricSuite.from_registry_domains()
        groups = suite.list_groups()
        assert "general" in groups
        assert len(groups) >= 2  # At least general + one domain

    def test_unknown_metric_raises(self) -> None:
        suite = MetricSuite()
        with pytest.raises(KeyError, match="not found in registry"):
            suite.add_group("bad", ["nonexistent_metric_xyz"])


class TestThresholdMetric:
    """Tests for ThresholdMetric."""

    def test_passes_when_below_max(self) -> None:
        threshold = ThresholdMetric("mse", max_value=1.0)
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        result = threshold.evaluate(predictions, targets)
        assert result["passed"] is True
        assert result["value"] == pytest.approx(0.0, abs=1e-5)

    def test_fails_when_above_max(self) -> None:
        threshold = ThresholdMetric("mse", max_value=0.001)
        predictions = jnp.array([2.0, 3.0, 4.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        result = threshold.evaluate(predictions, targets)
        assert result["passed"] is False

    def test_passes_when_above_min(self) -> None:
        threshold = ThresholdMetric("r_squared", min_value=0.9)
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        result = threshold.evaluate(predictions, targets)
        assert result["passed"] is True

    def test_fails_when_below_min(self) -> None:
        threshold = ThresholdMetric("r_squared", min_value=0.99)
        predictions = jnp.array([1.5, 2.5, 3.5])
        targets = jnp.array([1.0, 2.0, 3.0])
        result = threshold.evaluate(predictions, targets)
        assert result["passed"] is False

    def test_no_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ThresholdMetric("mse")

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            ThresholdMetric("nonexistent_xyz", max_value=1.0)

    def test_evaluate_returns_dict(self) -> None:
        threshold = ThresholdMetric("mse", max_value=1.0)
        predictions = jnp.array([1.0, 2.0])
        targets = jnp.array([1.0, 2.0])
        result = threshold.evaluate(predictions, targets)
        assert "value" in result
        assert "passed" in result
        assert "threshold" in result
        assert "metric_name" in result
        assert result["metric_name"] == "mse"

    def test_properties(self) -> None:
        threshold = ThresholdMetric("mse", min_value=0.0, max_value=1.0)
        assert threshold.metric_name == "mse"
        assert threshold.min_value == 0.0
        assert threshold.max_value == 1.0
