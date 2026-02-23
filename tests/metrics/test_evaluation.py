"""Tests for calibrax.metrics.evaluation module."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from calibrax.metrics.evaluation import (
    calculate_all,
    mae,
    mape,
    METRIC_FUNCTIONS,
    mse,
    r_squared,
    relative_error,
    rmse,
)


class TestMSE:
    """Tests for mean squared error."""

    def test_perfect_predictions(self) -> None:
        """MSE should be 0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert mse(targets, targets) == pytest.approx(0.0, abs=1e-7)

    def test_known_value(self) -> None:
        """MSE should match hand-calculated value."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 3.0, 5.0])
        # (0 + 1 + 4) / 3 = 5/3
        assert mse(predictions, targets) == pytest.approx(5.0 / 3.0, rel=1e-5)

    def test_returns_python_float(self) -> None:
        """Result should be a Python float, not JAX scalar."""
        result = mse(jnp.ones(3), jnp.zeros(3))
        assert type(result) is float


class TestMAE:
    """Tests for mean absolute error."""

    def test_perfect_predictions(self) -> None:
        """MAE should be 0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert mae(targets, targets) == pytest.approx(0.0, abs=1e-7)

    def test_known_value(self) -> None:
        """MAE should match hand-calculated value."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([2.0, 4.0, 6.0])
        # (1 + 2 + 3) / 3 = 2.0
        assert mae(predictions, targets) == pytest.approx(2.0, rel=1e-5)


class TestRMSE:
    """Tests for root mean squared error."""

    def test_perfect_predictions(self) -> None:
        """RMSE should be 0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert rmse(targets, targets) == pytest.approx(0.0, abs=1e-7)

    def test_known_value(self) -> None:
        """RMSE should be sqrt of MSE."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 3.0, 5.0])
        expected = (5.0 / 3.0) ** 0.5
        assert rmse(predictions, targets) == pytest.approx(expected, rel=1e-5)

    def test_returns_python_float(self) -> None:
        """Result should be a Python float."""
        result = rmse(jnp.ones(3), jnp.zeros(3))
        assert type(result) is float


class TestRSquared:
    """Tests for coefficient of determination."""

    def test_perfect_fit(self) -> None:
        """R-squared should be ~1.0 for perfect predictions."""
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r_squared(targets, targets) == pytest.approx(1.0, abs=1e-5)

    def test_poor_fit(self) -> None:
        """R-squared should be low for poor predictions."""
        predictions = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = r_squared(predictions, targets)
        assert result < 0.0  # Negative R^2 means worse than mean

    def test_mean_predictor(self) -> None:
        """Predicting the mean should give R-squared ~0."""
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = jnp.full_like(targets, jnp.mean(targets))
        assert r_squared(predictions, targets) == pytest.approx(0.0, abs=1e-5)


class TestMAPE:
    """Tests for mean absolute percentage error."""

    def test_perfect_predictions(self) -> None:
        """MAPE should be ~0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert mape(targets, targets) == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self) -> None:
        """MAPE should match hand-calculated value."""
        predictions = jnp.array([1.1, 2.2, 3.3])
        targets = jnp.array([1.0, 2.0, 3.0])
        # |0.1/1| + |0.2/2| + |0.3/3| = 0.1 + 0.1 + 0.1 => mean = 0.1
        assert mape(predictions, targets) == pytest.approx(0.1, rel=1e-3)

    def test_handles_near_zero_targets(self) -> None:
        """MAPE should not produce inf for near-zero targets."""
        predictions = jnp.array([0.1])
        targets = jnp.array([0.0])
        result = mape(predictions, targets)
        assert jnp.isfinite(result)


class TestRelativeError:
    """Tests for mean relative error."""

    def test_perfect_predictions(self) -> None:
        """Relative error should be ~0 for identical arrays."""
        targets = jnp.array([1.0, 2.0, 3.0])
        assert relative_error(targets, targets) == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self) -> None:
        """Relative error should be L2 norm ratio."""
        predictions = jnp.array([2.0, 4.0])
        targets = jnp.array([1.0, 2.0])
        # diff = [1, 2], norm = sqrt(5)
        # target norm = sqrt(5)
        # relative = sqrt(5) / sqrt(5) = 1.0
        assert relative_error(predictions, targets) == pytest.approx(1.0, rel=1e-5)

    def test_returns_python_float(self) -> None:
        """Result should be a Python float."""
        result = relative_error(jnp.ones(3), jnp.ones(3) * 2)
        assert type(result) is float


class TestShapeValidation:
    """Tests for input shape validation."""

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched shapes should raise ValueError."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            mse(jnp.ones(3), jnp.ones(4))

    def test_2d_arrays_work(self) -> None:
        """Functions should work with multi-dimensional arrays."""
        predictions = jnp.ones((2, 3))
        targets = jnp.zeros((2, 3))
        result = mse(predictions, targets)
        assert result == pytest.approx(1.0)

    def test_scalar_inputs_work(self) -> None:
        """Functions should work with scalar inputs."""
        result = mse(jnp.array(1.0), jnp.array(2.0))
        assert result == pytest.approx(1.0)


class TestMetricFunctions:
    """Tests for the METRIC_FUNCTIONS registry."""

    def test_all_functions_registered(self) -> None:
        """All six metric functions should be in the registry."""
        expected = {"mse", "mae", "rmse", "r_squared", "mape", "relative_error"}
        assert set(METRIC_FUNCTIONS) == expected

    def test_registry_functions_callable(self) -> None:
        """All registered functions should be callable and return float."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 2.1, 3.1])
        for name, fn in METRIC_FUNCTIONS.items():
            result = fn(predictions, targets)
            assert isinstance(result, float), f"{name} did not return float"


class TestCalculateAll:
    """Tests for calculate_all."""

    def test_all_metrics_returned(self) -> None:
        """Default should return all metrics."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 2.1, 3.1])
        result = calculate_all(predictions, targets)
        assert set(result) == set(METRIC_FUNCTIONS)

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

    def test_values_are_python_floats(self) -> None:
        """All returned values should be Python floats."""
        result = calculate_all(jnp.ones(3), jnp.zeros(3))
        for value in result.values():
            assert type(value) is float
