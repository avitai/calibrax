"""Tests for calibrax.validation.convergence module."""

from __future__ import annotations

import pytest

from calibrax.validation.convergence import check_convergence, ConvergenceResult


class TestConvergenceResult:
    """Tests for ConvergenceResult dataclass."""

    def test_frozen(self) -> None:
        """ConvergenceResult should be immutable."""
        result = ConvergenceResult(rates={}, achieved={})
        with pytest.raises(AttributeError):
            result.optimal_tolerance = 0.1  # type: ignore[misc]

    def test_to_dict_from_dict_round_trip(self) -> None:
        """to_dict/from_dict should preserve all fields."""
        original = ConvergenceResult(
            rates={"mse": 0.5},
            achieved={"mse_0.01": True},
            iterations={"mse_0.01": 3},
            optimal_tolerance=0.01,
        )
        reconstructed = ConvergenceResult.from_dict(original.to_dict())
        assert reconstructed.rates == original.rates
        assert reconstructed.achieved == original.achieved
        assert reconstructed.iterations == original.iterations
        assert reconstructed.optimal_tolerance == original.optimal_tolerance


class TestCheckConvergence:
    """Tests for check_convergence."""

    def test_exponential_decay_converges(self) -> None:
        """Exponentially decaying values should show convergence."""
        series = {"mse": [1.0, 0.1, 0.01, 0.001]}
        tolerances = [0.1, 0.01, 0.001]
        result = check_convergence(series, tolerances)
        assert result.rates["mse"] > 0
        assert result.achieved["mse_0.1"] is True
        assert result.achieved["mse_0.01"] is True
        assert result.achieved["mse_0.001"] is True

    def test_flat_values_no_convergence(self) -> None:
        """Flat values should show zero convergence rate."""
        series = {"mse": [1.0, 1.0, 1.0, 1.0]}
        tolerances = [0.5, 0.1]
        result = check_convergence(series, tolerances)
        assert result.rates["mse"] == pytest.approx(0.0, abs=1e-10)

    def test_convergence_rate_geometric_series(self) -> None:
        """Known geometric series should produce correct rate."""
        # Each step halves the value: log(0.5) = -0.693...
        series = {"error": [8.0, 4.0, 2.0, 1.0]}
        result = check_convergence(series, [2.0, 1.0])
        assert result.rates["error"] == pytest.approx(0.6931, abs=0.01)

    def test_multiple_metrics(self) -> None:
        """Multiple metrics should be tracked independently."""
        series = {
            "mse": [1.0, 0.1, 0.01],
            "mae": [2.0, 2.0, 2.0],
        }
        tolerances = [0.5]
        result = check_convergence(series, tolerances)
        assert result.rates["mse"] > 0
        assert result.rates["mae"] == pytest.approx(0.0, abs=1e-10)

    def test_single_value_series(self) -> None:
        """Single-value series should have rate 0."""
        series = {"mse": [0.5]}
        tolerances = [1.0, 0.1]
        result = check_convergence(series, tolerances)
        assert result.rates["mse"] == 0.0
        assert result.achieved["mse_1.0"] is True
        assert result.achieved["mse_0.1"] is False

    def test_empty_series(self) -> None:
        """Empty metric_series should return empty result."""
        result = check_convergence({}, [0.1])
        assert result.rates == {}
        assert result.achieved == {}

    def test_iterations_tracked(self) -> None:
        """Iterations should record when tolerance was first achieved."""
        series = {"mse": [10.0, 1.0, 0.1, 0.01]}
        tolerances = [1.0, 0.01]
        result = check_convergence(series, tolerances)
        assert result.iterations["mse_1.0"] == 2
        assert result.iterations["mse_0.01"] == 4

    def test_optimal_tolerance(self) -> None:
        """Optimal tolerance should be the tightest achieved by all metrics."""
        series = {"mse": [1.0, 0.1, 0.01]}
        tolerances = [1.0, 0.1, 0.001]
        result = check_convergence(series, tolerances)
        assert result.optimal_tolerance == 0.1

    def test_no_optimal_when_none_achieved(self) -> None:
        """optimal_tolerance should be None when nothing achieved."""
        series = {"mse": [100.0, 50.0]}
        tolerances = [0.001]
        result = check_convergence(series, tolerances)
        assert result.optimal_tolerance is None

    def test_near_zero_values(self) -> None:
        """Near-zero values should be clamped (no log of zero)."""
        series = {"mse": [0.001, 0.0001, 0.0]}
        tolerances = [0.001]
        result = check_convergence(series, tolerances)
        assert result.rates["mse"] > 0
