"""Generic convergence analysis for benchmark validation.

Provides convergence rate computation and tolerance achievement tracking
using pure Python math (no numpy/jax dependency).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvergenceResult:
    """Analysis of convergence behavior.

    Attributes:
        rates: Metric name to convergence rate (log-reduction per step).
        achieved: Composite key (metric_tolerance) to whether convergence achieved.
        iterations: Composite key (metric_tolerance) to iteration count.
        optimal_tolerance: Best tolerance that was still achieved, or None.
    """

    rates: dict[str, float]
    achieved: dict[str, bool]
    iterations: dict[str, int] = field(default_factory=dict)
    optimal_tolerance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "rates": dict(self.rates),
            "achieved": dict(self.achieved),
            "iterations": dict(self.iterations),
            "optimal_tolerance": self.optimal_tolerance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvergenceResult:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with convergence result fields.

        Returns:
            Reconstructed ConvergenceResult instance.
        """
        return cls(
            rates=data["rates"],
            achieved=data["achieved"],
            iterations=data.get("iterations", {}),
            optimal_tolerance=data.get("optimal_tolerance"),
        )


def _compute_convergence_rate(values: Sequence[float]) -> float:
    """Compute average log-reduction rate per step.

    Args:
        values: Metric values at increasing resolution (must have len >= 2).

    Returns:
        Absolute average rate of log-reduction.
    """
    log_values = [math.log(max(v, 1e-12)) for v in values]
    diffs = [log_values[i + 1] - log_values[i] for i in range(len(log_values) - 1)]
    return abs(sum(diffs) / len(diffs))


def _check_tolerance_achievement(
    metric_name: str,
    values: Sequence[float],
    tolerances: Sequence[float],
    achieved: dict[str, bool],
    iterations: dict[str, int],
) -> None:
    """Check which tolerances are achieved for a single metric.

    Args:
        metric_name: Name of the metric.
        values: Metric values at increasing resolution.
        tolerances: Tolerance thresholds to check.
        achieved: Accumulator for achievement flags (mutated).
        iterations: Accumulator for iteration counts (mutated).
    """
    for tol in tolerances:
        key = f"{metric_name}_{tol}"
        for i, v in enumerate(values):
            if v <= tol:
                achieved[key] = True
                iterations[key] = i + 1
                break
        else:
            achieved[key] = False


def _find_optimal_tolerance(
    metric_series: Mapping[str, Sequence[float]],
    tolerances: Sequence[float],
    achieved: dict[str, bool],
) -> float | None:
    """Find the best tolerance achieved by all metrics.

    Args:
        metric_series: All metric series.
        tolerances: Tolerance thresholds sorted ascending.
        achieved: Achievement flags.

    Returns:
        Tightest tolerance met by all metrics, or None.
    """
    for tol in sorted(tolerances):
        if all(achieved.get(f"{m}_{tol}", False) for m in metric_series):
            return tol
    return None


def check_convergence(
    metric_series: Mapping[str, Sequence[float]],
    tolerances: Sequence[float],
) -> ConvergenceResult:
    """Check convergence across metrics at given tolerances.

    For each metric, computes the average log-reduction rate per step and
    checks whether the final value meets each tolerance.

    Args:
        metric_series: {metric_name: [values_at_increasing_resolution]}.
            Values should decrease toward zero for convergent metrics.
        tolerances: Tolerance thresholds to check against.

    Returns:
        ConvergenceResult with rates, achievement flags, and iteration counts.
    """
    rates: dict[str, float] = {}
    achieved: dict[str, bool] = {}
    iterations: dict[str, int] = {}

    for metric_name, values in metric_series.items():
        if len(values) < 2:
            rates[metric_name] = 0.0
            for tol in tolerances:
                achieved[f"{metric_name}_{tol}"] = bool(len(values) == 1 and values[0] <= tol)
            continue

        rates[metric_name] = _compute_convergence_rate(values)
        _check_tolerance_achievement(metric_name, values, tolerances, achieved, iterations)

    return ConvergenceResult(
        rates=rates,
        achieved=achieved,
        iterations=iterations,
        optimal_tolerance=_find_optimal_tolerance(metric_series, tolerances, achieved),
    )
