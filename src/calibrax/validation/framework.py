"""Generic validation report for benchmark validation results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from substrax.records import read_record
from substrax.typing import JsonValue


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

    def to_dict(self) -> dict[str, JsonValue]:
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
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> ValidationReport:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)
