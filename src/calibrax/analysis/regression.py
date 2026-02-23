"""Regression detection for benchmark runs.

Compares a current run against a baseline to flag metrics that degraded
beyond a specified threshold.
"""

from __future__ import annotations

from calibrax.core.models import MetricDef, MetricDirection, Point, Regression, Run


def _point_key(p: Point) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Composite key for matching points across runs."""
    return (p.name, tuple(sorted(p.tags.items())))


def _check_metric_regression(
    metric_name: str,
    current_value: float,
    baseline_value: float,
    direction: MetricDirection,
    threshold: float,
) -> bool:
    """Determine whether a metric change constitutes a regression.

    Args:
        metric_name: Name of the metric (unused, for future logging).
        current_value: Current metric value.
        baseline_value: Baseline metric value.
        direction: Whether higher or lower is better.
        threshold: Relative change threshold.

    Returns:
        True if the change is a regression.
    """
    _ = metric_name
    delta_pct = ((current_value - baseline_value) / abs(baseline_value)) * 100.0
    if direction == MetricDirection.HIGHER:
        return delta_pct < -threshold * 100
    return delta_pct > threshold * 100


def _check_point_regressions(
    point: Point,
    baseline_point: Point,
    metric_defs: dict[str, MetricDef],
    threshold: float,
) -> list[Regression]:
    """Check all metrics on a single point for regressions.

    Args:
        point: Current point.
        baseline_point: Corresponding baseline point.
        metric_defs: Merged metric definitions.
        threshold: Relative change threshold.

    Returns:
        List of regressions found on this point.
    """
    regressions: list[Regression] = []
    for metric_name, current_metric in point.metrics.items():
        md = metric_defs.get(metric_name)
        if md is None or md.direction == MetricDirection.INFO:
            continue

        baseline_metric = baseline_point.metrics.get(metric_name)
        if baseline_metric is None or baseline_metric.value == 0:
            continue

        bv = baseline_metric.value
        cv = current_metric.value
        delta_pct = ((cv - bv) / abs(bv)) * 100.0

        if _check_metric_regression(metric_name, cv, bv, md.direction, threshold):
            regressions.append(
                Regression(
                    metric=metric_name,
                    point_name=point.name,
                    baseline_value=bv,
                    current_value=cv,
                    delta_pct=delta_pct,
                    direction=md.direction,
                )
            )
    return regressions


def detect_regressions(
    run: Run,
    baseline: Run,
    threshold: float = 0.05,
) -> list[Regression]:
    """Flag metrics that degraded beyond threshold.

    Uses MetricDef.direction: 'higher' metrics regress when they decrease,
    'lower' metrics regress when they increase. 'info' metrics are skipped.

    Args:
        run: Current benchmark run.
        baseline: Baseline run to compare against.
        threshold: Relative change threshold (e.g. 0.05 = 5%).

    Returns:
        List of detected regressions.
    """
    metric_defs = {**baseline.metric_defs, **run.metric_defs}
    baseline_lookup = {_point_key(p): p for p in baseline.points}

    regressions: list[Regression] = []
    for point in run.points:
        bp = baseline_lookup.get(_point_key(point))
        if bp is None:
            continue
        regressions.extend(_check_point_regressions(point, bp, metric_defs, threshold))

    return regressions
