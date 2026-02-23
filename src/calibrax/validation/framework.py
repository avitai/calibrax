"""Generic validation report for benchmark validation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationReport:
    """Report of validation results against reference methods.

    Attributes:
        name: Benchmark or experiment name.
        reference: Name of reference method or dataset.
        accuracy_metrics: Metric name to achieved value.
        convergence_metrics: Convergence metric name to rate.
        violations: Tuple of violation descriptions (empty if none).
        passed: Whether validation passed overall.
        notes: Free-form notes or warnings.
    """

    name: str
    reference: str
    accuracy_metrics: dict[str, float]
    convergence_metrics: dict[str, float] = field(default_factory=dict)
    violations: tuple[str, ...] = ()
    passed: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "reference": self.reference,
            "accuracy_metrics": dict(self.accuracy_metrics),
            "convergence_metrics": dict(self.convergence_metrics),
            "violations": list(self.violations),
            "passed": self.passed,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationReport:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with validation report fields.

        Returns:
            Reconstructed ValidationReport instance.
        """
        return cls(
            name=data["name"],
            reference=data["reference"],
            accuracy_metrics=data["accuracy_metrics"],
            convergence_metrics=data.get("convergence_metrics", {}),
            violations=tuple(data.get("violations", ())),
            passed=data.get("passed", True),
            notes=data.get("notes", ""),
        )
