"""Tests for calibrax.validation.accuracy module."""

from __future__ import annotations

import pytest

from calibrax.validation.accuracy import AccuracyResult, check_accuracy


class TestAccuracyResult:
    """Tests for AccuracyResult dataclass."""

    def test_frozen(self) -> None:
        """AccuracyResult should be immutable."""
        result = AccuracyResult(
            target=0.01,
            achieved=0.005,
            metric_type="mse",
            units="relative",
            passed=True,
            margin=0.005,
        )
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]

    def test_to_dict_from_dict_round_trip(self) -> None:
        """to_dict/from_dict should preserve all fields."""
        original = AccuracyResult(
            target=0.01,
            achieved=0.005,
            metric_type="mse",
            units="eV",
            passed=True,
            margin=0.005,
        )
        reconstructed = AccuracyResult.from_dict(original.to_dict())
        assert reconstructed.target == original.target
        assert reconstructed.achieved == original.achieved
        assert reconstructed.metric_type == original.metric_type
        assert reconstructed.units == original.units
        assert reconstructed.passed == original.passed
        assert reconstructed.margin == original.margin


class TestCheckAccuracy:
    """Tests for check_accuracy."""

    def test_passes_when_under_target(self) -> None:
        """achieved <= target should pass."""
        result = check_accuracy(0.005, 0.01)
        assert result.passed is True

    def test_fails_when_over_target(self) -> None:
        """achieved > target should fail."""
        result = check_accuracy(0.02, 0.01)
        assert result.passed is False

    def test_passes_when_equal_to_target(self) -> None:
        """achieved == target should pass."""
        result = check_accuracy(0.01, 0.01)
        assert result.passed is True

    def test_margin_positive_headroom(self) -> None:
        """Margin should be positive when under target."""
        result = check_accuracy(0.005, 0.01)
        assert result.margin == pytest.approx(0.005)

    def test_margin_negative_overshoot(self) -> None:
        """Margin should be negative when over target."""
        result = check_accuracy(0.02, 0.01)
        assert result.margin == pytest.approx(-0.01)

    def test_custom_metric_type_and_units(self) -> None:
        """Custom metric_type and units should be preserved."""
        result = check_accuracy(0.001, 0.01, metric_type="force_accuracy", units="eV/A")
        assert result.metric_type == "force_accuracy"
        assert result.units == "eV/A"

    def test_default_metric_type_and_units(self) -> None:
        """Default metric_type and units should be set."""
        result = check_accuracy(0.005, 0.01)
        assert result.metric_type == "accuracy"
        assert result.units == "relative"

    def test_large_achieved_value(self) -> None:
        """Very large achieved value should fail with large negative margin."""
        result = check_accuracy(1000.0, 0.01)
        assert result.passed is False
        assert result.margin < 0
