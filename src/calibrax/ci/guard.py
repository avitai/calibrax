"""CI regression gate for automated performance checks.

Compares the latest (or specified) run against the stored baseline and
flags regressions that exceed a configured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from calibrax.analysis.regression import detect_regressions
from calibrax.core.models import Regression
from calibrax.storage.store import Store


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardResult:
    """Result of a CI regression check.

    Attributes:
        passed: True if no regressions were detected.
        regressions: Detected regressions (empty tuple if passed).
        threshold: Regression threshold used for the check.
        baseline_id: ID of the baseline run used for comparison.
        current_id: ID of the run being checked.
    """

    passed: bool
    regressions: tuple[Regression, ...]
    threshold: float
    baseline_id: str | None
    current_id: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "passed": self.passed,
            "regressions": [r.to_dict() for r in self.regressions],
            "threshold": float(self.threshold),
            "baseline_id": self.baseline_id,
            "current_id": self.current_id,
        }


class CIGuard:
    """Regression gate that compares runs against a stored baseline.

    Args:
        store: Storage backend containing runs and baselines.
        threshold: Relative change threshold (e.g. 0.05 = 5%).
    """

    def __init__(self, store: Store, threshold: float = 0.05) -> None:
        """Initialize the CI guard."""
        self._store = store
        self._threshold = threshold

    def check(self, run_id: str | None = None) -> GuardResult:
        """Check for regressions against the baseline.

        Args:
            run_id: Run to check. Defaults to latest run.

        Returns:
            GuardResult indicating pass/fail with regression details.

        Raises:
            FileNotFoundError: If no baseline is set or run not found.
        """
        baseline = self._store.get_baseline()
        if baseline is None:
            msg = "No baseline set. Use store.set_baseline() first."
            raise FileNotFoundError(msg)

        if run_id is not None:
            run = self._store.load(run_id)
        else:
            run = self._store.latest()

        regressions = detect_regressions(run, baseline, threshold=self._threshold)

        return GuardResult(
            passed=len(regressions) == 0,
            regressions=tuple(regressions),
            threshold=self._threshold,
            baseline_id=baseline.id,
            current_id=run.id,
        )
