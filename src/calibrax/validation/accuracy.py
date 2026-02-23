"""Generic accuracy assessment for benchmark validation.

Compares an achieved value against a target, computing pass/fail and margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class AccuracyResult:
    """Assessment of accuracy against a target.

    Attributes:
        target: Target accuracy threshold.
        achieved: Achieved accuracy value.
        metric_type: Type of accuracy (e.g. "accuracy", "mse").
        units: Units of measurement (e.g. "relative", "eV").
        passed: Whether achieved meets the target (achieved <= target).
        margin: Difference between target and achieved (positive = headroom).
    """

    target: float
    achieved: float
    metric_type: str
    units: str
    passed: bool
    margin: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "target": self.target,
            "achieved": self.achieved,
            "metric_type": self.metric_type,
            "units": self.units,
            "passed": self.passed,
            "margin": self.margin,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccuracyResult:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with accuracy result fields.

        Returns:
            Reconstructed AccuracyResult instance.
        """
        return cls(**data)


def check_accuracy(
    achieved: float,
    target: float,
    *,
    metric_type: str = "accuracy",
    units: str = "relative",
) -> AccuracyResult:
    """Check whether an achieved value meets a target.

    Args:
        achieved: The measured value.
        target: The target threshold (achieved must be <= target to pass).
        metric_type: Label for the type of accuracy check.
        units: Units of measurement.

    Returns:
        AccuracyResult with pass/fail and margin.
    """
    return AccuracyResult(
        target=target,
        achieved=achieved,
        metric_type=metric_type,
        units=units,
        passed=achieved <= target,
        margin=target - achieved,
    )
