"""Tests for calibration metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.calibration import (
    adaptive_calibration_error,
    brier_decomposition,
    brier_score,
    classwise_ece,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram_bins,
)


class TestBrierScore:
    """Tests for brier_score."""

    def test_perfect_predictions(self) -> None:
        predictions = jnp.array([1.0, 0.0, 1.0, 0.0])
        targets = jnp.array([1, 0, 1, 0])
        assert brier_score(predictions, targets) == pytest.approx(0.0, abs=1e-6)

    def test_worst_predictions(self) -> None:
        predictions = jnp.array([0.0, 1.0, 0.0, 1.0])
        targets = jnp.array([1, 0, 1, 0])
        assert brier_score(predictions, targets) == pytest.approx(1.0, abs=1e-6)

    def test_uniform_predictions(self) -> None:
        predictions = jnp.array([0.5, 0.5, 0.5, 0.5])
        targets = jnp.array([1, 0, 1, 0])
        assert brier_score(predictions, targets) == pytest.approx(0.25, abs=1e-6)

    def test_asymmetric_predictions(self) -> None:
        predictions = jnp.array([0.9, 0.1, 0.8, 0.2])
        targets = jnp.array([1, 0, 1, 0])
        result = brier_score(predictions, targets)
        assert 0.0 < result < 0.25

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([0.5, 0.5])
        targets = jnp.array([1, 0])
        result = brier_score(predictions, targets)
        assert isinstance(result, jax.Array)


class TestExpectedCalibrationError:
    """Tests for expected_calibration_error."""

    def test_perfectly_calibrated(self) -> None:
        # Predictions matching actual frequencies within bins
        predictions = jnp.array([0.1, 0.1, 0.9, 0.9])
        targets = jnp.array([0, 0, 1, 1])
        result = expected_calibration_error(predictions, targets, num_bins=5)
        assert result == pytest.approx(0.0, abs=0.15)

    def test_completely_miscalibrated(self) -> None:
        # All predict 0.9 but none are positive
        predictions = jnp.array([0.9, 0.9, 0.9, 0.9])
        targets = jnp.array([0, 0, 0, 0])
        result = expected_calibration_error(predictions, targets, num_bins=10)
        assert result > 0.5

    def test_range_zero_to_one(self) -> None:
        predictions = jnp.array([0.3, 0.7, 0.2, 0.8, 0.5])
        targets = jnp.array([0, 1, 0, 1, 1])
        result = expected_calibration_error(predictions, targets)
        assert 0.0 <= result <= 1.0

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([0.5, 0.5])
        targets = jnp.array([1, 0])
        result = expected_calibration_error(predictions, targets)
        assert isinstance(result, jax.Array)

    def test_custom_num_bins(self) -> None:
        predictions = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = jnp.array([0, 0, 1, 1, 1])
        result_5 = expected_calibration_error(predictions, targets, num_bins=5)
        result_20 = expected_calibration_error(predictions, targets, num_bins=20)
        # Both should be valid floats in [0, 1]
        assert 0.0 <= result_5 <= 1.0
        assert 0.0 <= result_20 <= 1.0


class TestMaximumCalibrationError:
    """Tests for maximum_calibration_error."""

    def test_perfectly_calibrated(self) -> None:
        predictions = jnp.array([0.1, 0.1, 0.9, 0.9])
        targets = jnp.array([0, 0, 1, 1])
        result = maximum_calibration_error(predictions, targets, num_bins=5)
        assert result == pytest.approx(0.0, abs=0.15)

    def test_worst_case_bin(self) -> None:
        # All predictions in one bin, all wrong
        predictions = jnp.array([0.9, 0.9, 0.9, 0.9])
        targets = jnp.array([0, 0, 0, 0])
        result = maximum_calibration_error(predictions, targets, num_bins=10)
        assert result > 0.5

    def test_mce_geq_ece(self) -> None:
        predictions = jnp.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7])
        targets = jnp.array([0, 1, 0, 1, 1, 0])
        ece = expected_calibration_error(predictions, targets, num_bins=5)
        mce = maximum_calibration_error(predictions, targets, num_bins=5)
        assert mce >= ece - 1e-6

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([0.5, 0.5])
        targets = jnp.array([1, 0])
        result = maximum_calibration_error(predictions, targets)
        assert isinstance(result, jax.Array)


class TestReliabilityDiagramBins:
    """Tests for reliability_diagram_bins."""

    def test_returns_dict_with_expected_keys(self) -> None:
        predictions = jnp.array([0.1, 0.5, 0.9])
        targets = jnp.array([0, 1, 1])
        result = reliability_diagram_bins(predictions, targets, num_bins=5)
        assert isinstance(result, dict)
        assert "bin_edges" in result
        assert "bin_accuracies" in result
        assert "bin_confidences" in result
        assert "bin_counts" in result

    def test_bin_edges_shape(self) -> None:
        predictions = jnp.array([0.1, 0.5, 0.9])
        targets = jnp.array([0, 1, 1])
        result = reliability_diagram_bins(predictions, targets, num_bins=5)
        assert result["bin_edges"].shape == (6,)  # num_bins + 1

    def test_bin_counts_sum_to_n(self) -> None:
        predictions = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = jnp.array([0, 0, 1, 1, 1])
        result = reliability_diagram_bins(predictions, targets, num_bins=5)
        assert int(jnp.sum(result["bin_counts"])) == 5

    def test_arrays_have_correct_shape(self) -> None:
        num_bins = 10
        predictions = jnp.array([0.2, 0.4, 0.6, 0.8])
        targets = jnp.array([0, 0, 1, 1])
        result = reliability_diagram_bins(predictions, targets, num_bins=num_bins)
        assert result["bin_accuracies"].shape == (num_bins,)
        assert result["bin_confidences"].shape == (num_bins,)
        assert result["bin_counts"].shape == (num_bins,)


class TestBrierDecomposition:
    """Tests for brier_decomposition."""

    def test_returns_dict_with_expected_keys(self) -> None:
        predictions = jnp.array([0.1, 0.5, 0.9])
        targets = jnp.array([0, 1, 1])
        result = brier_decomposition(predictions, targets)
        assert isinstance(result, dict)
        assert "calibration" in result
        assert "resolution" in result
        assert "uncertainty" in result
        assert "brier" in result

    def test_decomposition_relationship(self) -> None:
        # brier ≈ calibration - resolution + uncertainty
        predictions = jnp.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7])
        targets = jnp.array([0, 0, 1, 1, 0, 1])
        result = brier_decomposition(predictions, targets, num_bins=5)
        expected_brier = result["calibration"] - result["resolution"] + result["uncertainty"]
        assert result["brier"] == pytest.approx(expected_brier, abs=0.05)

    def test_uncertainty_is_base_rate_variance(self) -> None:
        targets = jnp.array([0, 0, 1, 1, 1])
        predictions = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = brier_decomposition(predictions, targets)
        base_rate = 3.0 / 5.0
        expected_uncertainty = base_rate * (1.0 - base_rate)
        assert result["uncertainty"] == pytest.approx(expected_uncertainty, abs=1e-6)

    def test_calibration_non_negative(self) -> None:
        predictions = jnp.array([0.2, 0.8, 0.3, 0.7])
        targets = jnp.array([0, 1, 1, 0])
        result = brier_decomposition(predictions, targets)
        assert result["calibration"] >= -1e-6

    def test_resolution_non_negative(self) -> None:
        predictions = jnp.array([0.2, 0.8, 0.3, 0.7])
        targets = jnp.array([0, 1, 1, 0])
        result = brier_decomposition(predictions, targets)
        assert result["resolution"] >= -1e-6


class TestAdaptiveCalibrationError:
    """Tests for adaptive_calibration_error."""

    def test_perfectly_calibrated(self) -> None:
        # Simple case: predictions match targets
        predictions = jnp.array([0.0, 0.0, 1.0, 1.0])
        targets = jnp.array([0, 0, 1, 1])
        result = adaptive_calibration_error(predictions, targets, num_bins=2)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_completely_miscalibrated(self) -> None:
        predictions = jnp.array([0.9, 0.9, 0.9, 0.9])
        targets = jnp.array([0, 0, 0, 0])
        result = adaptive_calibration_error(predictions, targets, num_bins=2)
        assert result > 0.5

    def test_range_zero_to_one(self) -> None:
        predictions = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = jnp.array([0, 0, 1, 1, 1])
        result = adaptive_calibration_error(predictions, targets)
        assert 0.0 <= result <= 1.0

    def test_returns_scalar(self) -> None:
        """Result should be a numeric scalar (Python float due to equal-mass binning loop)."""
        predictions = jnp.array([0.5, 0.5])
        targets = jnp.array([1, 0])
        result = adaptive_calibration_error(predictions, targets)
        assert isinstance(result, (float, jax.Array))

    def test_equal_mass_binning(self) -> None:
        # With 4 samples and 2 bins, each bin should have 2 samples
        predictions = jnp.array([0.1, 0.3, 0.7, 0.9])
        targets = jnp.array([0, 0, 1, 1])
        result = adaptive_calibration_error(predictions, targets, num_bins=2)
        # Well-calibrated case — low error
        assert result < 0.3


class TestClasswiseECE:
    """Tests for classwise_ece."""

    def test_perfectly_calibrated_multiclass(self) -> None:
        # 3 classes, predictions match targets
        predictions = jnp.array(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
            ]
        )
        targets = jnp.array([0, 1, 2])
        result = classwise_ece(predictions, targets, num_bins=5)
        assert result == pytest.approx(0.0, abs=0.15)

    def test_requires_2d_predictions(self) -> None:
        predictions = jnp.array([0.5, 0.5, 0.5])
        targets = jnp.array([0, 1, 0])
        with pytest.raises(ValueError, match="2D predictions"):
            classwise_ece(predictions, targets)

    def test_range_zero_to_one(self) -> None:
        predictions = jnp.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.5],
                [0.6, 0.3, 0.1],
            ]
        )
        targets = jnp.array([0, 1, 2, 0])
        result = classwise_ece(predictions, targets, num_bins=5)
        assert 0.0 <= result <= 1.0

    def test_returns_jax_scalar(self) -> None:
        predictions = jnp.array([[0.8, 0.2], [0.3, 0.7]])
        targets = jnp.array([0, 1])
        result = classwise_ece(predictions, targets)
        assert isinstance(result, jax.Array)

    def test_infers_num_classes(self) -> None:
        predictions = jnp.array(
            [
                [0.5, 0.3, 0.2],
                [0.1, 0.6, 0.3],
            ]
        )
        targets = jnp.array([0, 1])
        # Should infer 3 classes from predictions shape
        result = classwise_ece(predictions, targets, num_bins=3)
        assert isinstance(result, jax.Array)


class TestCalibrationMetricRegistration:
    """Tests for calibration metric registration in MetricRegistry."""

    def test_calibration_metrics_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        expected = [
            "brier_score",
            "expected_calibration_error",
            "maximum_calibration_error",
            "adaptive_calibration_error",
            "classwise_ece",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_calibration_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        calibration_metrics = registry.list_by_domain("calibration")
        assert len(calibration_metrics) == 5
        names = {m.name for m in calibration_metrics}
        assert "brier_score" in names
        assert "expected_calibration_error" in names

    def test_brier_score_is_proper(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        entry = registry.get("brier_score")
        assert entry.properties.is_proper is True

    def test_calibration_metrics_direction_lower(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        calibration_metrics = registry.list_by_domain("calibration")
        for m in calibration_metrics:
            assert m.direction == MetricDirection.LOWER
