"""Ranking and aggregate scoring for benchmark runs.

Ranks entries by metric value and computes weighted aggregate scores
across multiple metrics.
"""

from __future__ import annotations

from calibrax.core.models import (
    extract_framework_metrics,
    is_higher_better,
    MetricDef,
    RankEntry,
    Run,
)


def rank_table(
    run: Run,
    metric: str,
    group_by_tag: str = "framework",
) -> list[RankEntry]:
    """Rank entries by metric value, grouped by a tag.

    Uses MetricDef.direction for determining best-is-highest vs best-is-lowest.

    Args:
        run: Benchmark run with points and metric_defs.
        metric: Metric name to rank by.
        group_by_tag: Tag key used to group points (default "framework").

    Returns:
        Sorted list of RankEntry, rank 1 = best.
    """
    md = run.metric_defs.get(metric)
    higher_is_better = is_higher_better(md)

    groups: dict[str, float] = {}
    for point in run.points:
        label = point.tags.get(group_by_tag, point.name)
        if label not in groups and metric in point.metrics:
            groups[label] = point.metrics[metric].value

    sorted_items = sorted(
        groups.items(),
        key=lambda x: x[1],
        reverse=higher_is_better,
    )

    if not sorted_items:
        return []

    best_value = sorted_items[0][1]
    entries: list[RankEntry] = []
    for rank, (label, value) in enumerate(sorted_items, start=1):
        if best_value != 0:
            delta = abs((value - best_value) / best_value) * 100.0
        else:
            delta = 0.0

        entries.append(
            RankEntry(
                label=label,
                value=value,
                rank=rank,
                is_best=(rank == 1),
                delta_from_best=delta,
            )
        )

    return entries


def _compute_metric_ranges(
    frameworks: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> dict[str, tuple[float, float]]:
    """Compute (min, max) range for each metric across all frameworks.

    Args:
        frameworks: Per-framework metric values.
        weights: Metric names to compute ranges for.

    Returns:
        {metric_name: (min_value, max_value)}.
    """
    metric_ranges: dict[str, tuple[float, float]] = {}
    for metric_name in weights:
        values = [
            fw_metrics[metric_name]
            for fw_metrics in frameworks.values()
            if metric_name in fw_metrics
        ]
        if values:
            metric_ranges[metric_name] = (min(values), max(values))
    return metric_ranges


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to sum to 1.0.

    Args:
        weights: Raw weights.

    Returns:
        Weights normalized to unit sum, or unchanged if total is zero.
    """
    total = sum(weights.values())
    if total > 0:
        return {k: v / total for k, v in weights.items()}
    return weights


def _compute_weighted_scores(
    frameworks: dict[str, dict[str, float]],
    metric_ranges: dict[str, tuple[float, float]],
    norm_weights: dict[str, float],
    metric_defs: dict[str, MetricDef],
) -> dict[str, float]:
    """Compute weighted aggregate scores per framework.

    Args:
        frameworks: Per-framework metric values.
        metric_ranges: (min, max) ranges for normalization.
        norm_weights: Normalized weights per metric.
        metric_defs: Metric definitions for direction.

    Returns:
        {framework_label: aggregate_score}.
    """
    scores: dict[str, float] = {}
    for fw, fw_metrics in frameworks.items():
        score = 0.0
        for metric_name, weight in norm_weights.items():
            if metric_name not in fw_metrics or metric_name not in metric_ranges:
                continue
            val = fw_metrics[metric_name]
            mn, mx = metric_ranges[metric_name]
            if mx == mn:
                normalized = 1.0
            elif is_higher_better(metric_defs.get(metric_name)):
                normalized = (val - mn) / (mx - mn)
            else:
                normalized = (mx - val) / (mx - mn)
            score += weight * normalized
        scores[fw] = score
    return scores


def aggregate_score(run: Run, weights: dict[str, float]) -> dict[str, float]:
    """Weighted aggregate score across metrics.

    Normalizes each metric to [0, 1] range (best = 1.0, worst = 0.0),
    then computes a weighted sum. Uses MetricDef.direction for normalization.

    Args:
        run: Benchmark run with points and metric_defs.
        weights: {metric_name: weight} — weights are normalized to sum to 1.0.

    Returns:
        {framework_label: aggregate_score} where score is in [0, 1].
    """
    if not run.points:
        return {}

    frameworks = extract_framework_metrics(run, weights)
    if not frameworks:
        return {}

    metric_ranges = _compute_metric_ranges(frameworks, weights)
    norm_weights = _normalize_weights(weights)

    return _compute_weighted_scores(frameworks, metric_ranges, norm_weights, run.metric_defs)
