"""Tests for metric wrapper classes."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.regression import mae, mse
from calibrax.metrics.wrappers import (
    BootstrapMetric,
    ClasswiseWrapper,
    MetricTracker,
    MinMaxTracker,
)


class TestBootstrapMetric:
    """Tests for BootstrapMetric."""

    def test_confidence_interval(self) -> None:
        bootstrap = BootstrapMetric(mse, num_bootstraps=200, confidence=0.95, seed=42)
        predictions = jnp.array([1.1, 2.2, 3.3, 4.4, 5.5])
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bootstrap.compute(predictions, targets)
        assert result["lower"] <= result["value"]
        assert result["value"] <= result["upper"]

    def test_point_estimate(self) -> None:
        bootstrap = BootstrapMetric(mse, num_bootstraps=50, seed=0)
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        result = bootstrap.compute(predictions, targets)
        assert result["value"] == pytest.approx(0.0, abs=1e-5)

    def test_samples_length(self) -> None:
        bootstrap = BootstrapMetric(mse, num_bootstraps=100, seed=0)
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.5, 2.5, 3.5])
        result = bootstrap.compute(predictions, targets)
        assert len(result["samples"]) == 100

    def test_reproducible_with_seed(self) -> None:
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = jnp.array([1.5, 2.5, 3.5, 4.5, 5.5])
        b1 = BootstrapMetric(mse, num_bootstraps=50, seed=42)
        b2 = BootstrapMetric(mse, num_bootstraps=50, seed=42)
        r1 = b1.compute(predictions, targets)
        r2 = b2.compute(predictions, targets)
        assert r1["lower"] == pytest.approx(r2["lower"], abs=1e-10)
        assert r1["upper"] == pytest.approx(r2["upper"], abs=1e-10)

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be"):
            BootstrapMetric(mse, confidence=1.5)
        with pytest.raises(ValueError, match="confidence must be"):
            BootstrapMetric(mse, confidence=0.0)

    def test_properties(self) -> None:
        bootstrap = BootstrapMetric(mse, num_bootstraps=100, confidence=0.9)
        assert bootstrap.metric_fn is mse
        assert bootstrap.num_bootstraps == 100
        assert bootstrap.confidence == 0.9


class TestClasswiseWrapper:
    """Tests for ClasswiseWrapper."""

    def test_per_class_values(self) -> None:
        classwise = ClasswiseWrapper(mse)
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.0, 2.0, 3.5, 4.5])
        labels = jnp.array([0, 0, 1, 1])
        result = classwise.compute(predictions, targets, labels)
        assert result["0"] == pytest.approx(0.0, abs=1e-5)
        assert result["1"] > 0.0

    def test_mean_key(self) -> None:
        classwise = ClasswiseWrapper(mse)
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.5, 2.5, 3.0, 4.0])
        labels = jnp.array([0, 0, 1, 1])
        result = classwise.compute(predictions, targets, labels)
        expected_mean = (result["0"] + result["1"]) / 2
        assert result["mean"] == pytest.approx(expected_mean, abs=1e-5)

    def test_custom_class_names(self) -> None:
        classwise = ClasswiseWrapper(mae, class_names=["cat", "dog"])
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0])
        targets = jnp.array([1.0, 2.0, 3.0, 4.0])
        labels = jnp.array([0, 0, 1, 1])
        result = classwise.compute(predictions, targets, labels)
        assert "cat" in result
        assert "dog" in result

    def test_integer_labels(self) -> None:
        classwise = ClasswiseWrapper(mse)
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        labels = jnp.array([0, 1, 1])
        result = classwise.compute(predictions, targets, labels)
        assert "0" in result
        assert "1" in result


class TestMetricTracker:
    """Tests for MetricTracker."""

    def test_best_lower(self) -> None:
        tracker = MetricTracker(mse, direction="lower")
        tracker.increment(jnp.array([1.5, 2.5]), jnp.array([1.0, 2.0]))  # MSE=0.25
        tracker.increment(jnp.array([1.1, 2.1]), jnp.array([1.0, 2.0]))  # MSE=0.01
        tracker.increment(jnp.array([1.3, 2.3]), jnp.array([1.0, 2.0]))  # MSE=0.09
        assert tracker.best() == pytest.approx(0.01, abs=1e-5)

    def test_best_higher(self) -> None:
        tracker = MetricTracker(mse, direction="higher")
        tracker.increment(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))  # 0.0
        tracker.increment(jnp.array([1.5, 2.5]), jnp.array([1.0, 2.0]))  # 0.25
        assert tracker.best() == pytest.approx(0.25, abs=1e-5)

    def test_best_epoch(self) -> None:
        tracker = MetricTracker(mse, direction="lower")
        tracker.increment(jnp.array([2.0, 3.0]), jnp.array([1.0, 2.0]))  # 1.0
        tracker.increment(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))  # 0.0
        tracker.increment(jnp.array([1.5, 2.5]), jnp.array([1.0, 2.0]))  # 0.25
        assert tracker.best_epoch == 1

    def test_history_immutable(self) -> None:
        tracker = MetricTracker(mse, direction="lower")
        tracker.increment(jnp.array([1.0]), jnp.array([1.0]))
        assert isinstance(tracker.history, tuple)

    def test_reset(self) -> None:
        tracker = MetricTracker(mse, direction="lower")
        tracker.increment(jnp.array([1.0]), jnp.array([1.0]))
        tracker.reset()
        assert tracker.history == ()

    def test_empty_best_raises(self) -> None:
        tracker = MetricTracker(mse, direction="lower")
        with pytest.raises(ValueError, match="No values"):
            tracker.best()

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction must be"):
            MetricTracker(mse, direction="invalid")


class TestMinMaxTracker:
    """Tests for MinMaxTracker."""

    def test_tracks_min(self) -> None:
        tracker = MinMaxTracker(mse)
        tracker.update(jnp.array([1.5, 2.5]), jnp.array([1.0, 2.0]))  # 0.25
        tracker.update(jnp.array([1.1, 2.1]), jnp.array([1.0, 2.0]))  # 0.01
        tracker.update(jnp.array([2.0, 3.0]), jnp.array([1.0, 2.0]))  # 1.0
        assert tracker.min == pytest.approx(0.01, abs=1e-5)

    def test_tracks_max(self) -> None:
        tracker = MinMaxTracker(mse)
        tracker.update(jnp.array([1.5, 2.5]), jnp.array([1.0, 2.0]))  # 0.25
        tracker.update(jnp.array([2.0, 3.0]), jnp.array([1.0, 2.0]))  # 1.0
        assert tracker.max == pytest.approx(1.0, abs=1e-5)

    def test_current(self) -> None:
        tracker = MinMaxTracker(mse)
        tracker.update(jnp.array([1.5, 2.5]), jnp.array([1.0, 2.0]))
        tracker.update(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
        assert tracker.current == pytest.approx(0.0, abs=1e-5)

    def test_none_before_update(self) -> None:
        tracker = MinMaxTracker(mse)
        assert tracker.current is None
        assert tracker.min is None
        assert tracker.max is None

    def test_reset(self) -> None:
        tracker = MinMaxTracker(mse)
        tracker.update(jnp.array([1.0]), jnp.array([2.0]))
        tracker.reset()
        assert tracker.current is None
        assert tracker.min is None
        assert tracker.max is None
