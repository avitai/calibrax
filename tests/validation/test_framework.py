"""Tests for calibrax.validation.framework module."""

from __future__ import annotations

import pytest

from calibrax.validation.framework import ValidationReport


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_frozen(self) -> None:
        """ValidationReport should be immutable."""
        report = ValidationReport(
            name="test",
            reference="ref",
            accuracy_metrics={"mse": 0.01},
        )
        with pytest.raises(AttributeError):
            report.name = "changed"  # type: ignore[misc]

    def test_to_dict_from_dict_round_trip(self) -> None:
        """to_dict/from_dict should preserve all fields."""
        original = ValidationReport(
            name="bench1",
            reference="FEM",
            accuracy_metrics={"mse": 0.001, "mae": 0.01},
            convergence_metrics={"rate": 2.0},
            violations=("MSE too high",),
            passed=False,
            notes="Needs improvement",
        )
        reconstructed = ValidationReport.from_dict(original.to_dict())
        assert reconstructed.name == original.name
        assert reconstructed.reference == original.reference
        assert reconstructed.accuracy_metrics == original.accuracy_metrics
        assert reconstructed.convergence_metrics == original.convergence_metrics
        assert reconstructed.violations == original.violations
        assert reconstructed.passed == original.passed
        assert reconstructed.notes == original.notes

    def test_passed_true_no_violations(self) -> None:
        """Report with no violations can have passed=True."""
        report = ValidationReport(
            name="test",
            reference="ref",
            accuracy_metrics={"mse": 0.001},
            passed=True,
        )
        assert report.passed is True
        assert report.violations == ()

    def test_passed_false_with_violations(self) -> None:
        """Report with violations should have passed=False."""
        report = ValidationReport(
            name="test",
            reference="ref",
            accuracy_metrics={"mse": 1.0},
            violations=("MSE exceeds tolerance",),
            passed=False,
        )
        assert report.passed is False
        assert len(report.violations) == 1

    def test_defaults(self) -> None:
        """Default values should be sensible."""
        report = ValidationReport(
            name="test",
            reference="ref",
            accuracy_metrics={},
        )
        assert report.convergence_metrics == {}
        assert report.violations == ()
        assert report.passed is True
        assert report.notes == ""

    def test_violations_is_tuple(self) -> None:
        """violations should be a tuple (immutable)."""
        report = ValidationReport(
            name="test",
            reference="ref",
            accuracy_metrics={},
            violations=("v1", "v2"),
        )
        assert isinstance(report.violations, tuple)
        assert len(report.violations) == 2

    def test_from_dict_with_missing_optional_fields(self) -> None:
        """from_dict should handle missing optional fields."""
        data = {
            "name": "test",
            "reference": "ref",
            "accuracy_metrics": {"mse": 0.01},
        }
        report = ValidationReport.from_dict(data)
        assert report.convergence_metrics == {}
        assert report.violations == ()
        assert report.passed is True
        assert report.notes == ""
