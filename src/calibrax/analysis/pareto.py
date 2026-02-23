"""Pareto front identification for multi-objective benchmark analysis.

Identifies Pareto-optimal points for two metrics, respecting
MetricDef.direction for dominance checks.
"""

from __future__ import annotations

from calibrax.core.models import is_higher_better, MetricDef, Point


def _dominates(
    xj: float,
    yj: float,
    xi: float,
    yi: float,
    *,
    x_higher: bool,
    y_higher: bool,
) -> bool:
    """Check whether point (xj, yj) dominates point (xi, yi).

    A point dominates another if it is at least as good on both metrics
    and strictly better on at least one.

    Args:
        xj: X-metric value of the candidate dominator.
        yj: Y-metric value of the candidate dominator.
        xi: X-metric value of the point being tested.
        yi: Y-metric value of the point being tested.
        x_higher: Whether higher is better for the x-metric.
        y_higher: Whether higher is better for the y-metric.

    Returns:
        True if (xj, yj) dominates (xi, yi).
    """
    x_better = (xj > xi) if x_higher else (xj < xi)
    x_equal = xj == xi
    y_better = (yj > yi) if y_higher else (yj < yi)
    y_equal = yj == yi
    return (x_better or x_equal) and (y_better or y_equal) and (x_better or y_better)


def _extract_indexed_values(
    points: list[Point],
    x_metric: str,
    y_metric: str,
) -> list[tuple[int, float, float]]:
    """Extract (index, x_value, y_value) tuples for points with both metrics.

    Args:
        points: Benchmark points to scan.
        x_metric: First metric name.
        y_metric: Second metric name.

    Returns:
        List of (original_index, x_value, y_value) tuples.
    """
    indexed: list[tuple[int, float, float]] = []
    for i, p in enumerate(points):
        if x_metric in p.metrics and y_metric in p.metrics:
            indexed.append((i, p.metrics[x_metric].value, p.metrics[y_metric].value))
    return indexed


def _filter_dominated(
    indexed: list[tuple[int, float, float]],
    *,
    x_higher: bool,
    y_higher: bool,
) -> list[int]:
    """Return indices of non-dominated points.

    Args:
        indexed: List of (original_index, x_value, y_value) tuples.
        x_higher: Whether higher is better for the x-metric.
        y_higher: Whether higher is better for the y-metric.

    Returns:
        Original indices of Pareto-optimal points.
    """
    front_indices: list[int] = []
    for i, xi, yi in indexed:
        dominated = any(
            _dominates(xj, yj, xi, yi, x_higher=x_higher, y_higher=y_higher)
            for j, xj, yj in indexed
            if j != i
        )
        if not dominated:
            front_indices.append(i)
    return front_indices


def pareto_front(
    points: list[Point],
    x_metric: str,
    y_metric: str,
    *,
    metric_defs: dict[str, MetricDef] | None = None,
) -> list[Point]:
    """Identify Pareto-optimal points for two metrics.

    A point is Pareto-optimal if no other point is strictly better on both
    metrics. Uses MetricDef.direction to determine "better".

    Args:
        points: List of benchmark points to analyze.
        x_metric: First metric name.
        y_metric: Second metric name.
        metric_defs: Optional metric definitions for direction. If not provided,
            defaults to higher-is-better for both metrics.

    Returns:
        List of Pareto-optimal points (subset of input, same order).
    """
    if not points:
        return []

    defs = metric_defs or {}
    x_higher = is_higher_better(defs.get(x_metric))
    y_higher = is_higher_better(defs.get(y_metric))

    indexed = _extract_indexed_values(points, x_metric, y_metric)
    front_indices = _filter_dominated(indexed, x_higher=x_higher, y_higher=y_higher)

    return [points[i] for i in front_indices]
