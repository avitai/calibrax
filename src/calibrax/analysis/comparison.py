"""Multi-configuration benchmark comparison.

Compares benchmark runs across different configurations (frameworks, hardware,
etc.) using MetricDef-aware direction logic and aggregate scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from calibrax.analysis.ranking import aggregate_score, rank_table
from calibrax.core.models import (
    is_higher_better,
    MetricDef,
    Point,
    RankEntry,
    Run,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricComparison:
    """Comparison results for a single metric across configurations.

    Attributes:
        metric_name: Name of the compared metric.
        values: Mapping of configuration label to metric value.
        rankings: Ranked entries for this metric.
        best_label: Label of the best-performing configuration.
        improvement_factors: How much better the best is vs each config.
    """

    metric_name: str
    values: dict[str, float]
    rankings: tuple[RankEntry, ...]
    best_label: str
    improvement_factors: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "metric_name": self.metric_name,
            "values": {k: float(v) for k, v in self.values.items()},
            "rankings": [r.to_dict() for r in self.rankings],
            "best_label": self.best_label,
            "improvement_factors": {k: float(v) for k, v in self.improvement_factors.items()},
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonReport:
    """Full comparison across multiple metrics and configurations.

    Attributes:
        name: Name of this comparison.
        labels_compared: Configuration labels included.
        metric_comparisons: Per-metric comparison results.
        winner_by_metric: Best label for each metric.
        overall_winner: Best label by aggregate score.
    """

    name: str
    labels_compared: tuple[str, ...]
    metric_comparisons: tuple[MetricComparison, ...]
    winner_by_metric: dict[str, str]
    overall_winner: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "labels_compared": list(self.labels_compared),
            "metric_comparisons": [mc.to_dict() for mc in self.metric_comparisons],
            "winner_by_metric": dict(self.winner_by_metric),
            "overall_winner": self.overall_winner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonReport:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with comparison report fields.

        Returns:
            Reconstructed ComparisonReport instance.
        """
        return cls(
            name=data["name"],
            labels_compared=tuple(data["labels_compared"]),
            metric_comparisons=tuple(
                MetricComparison(
                    metric_name=mc["metric_name"],
                    values=mc["values"],
                    rankings=tuple(RankEntry.from_dict(r) for r in mc["rankings"]),
                    best_label=mc["best_label"],
                    improvement_factors=mc["improvement_factors"],
                )
                for mc in data["metric_comparisons"]
            ),
            winner_by_metric=data["winner_by_metric"],
            overall_winner=data["overall_winner"],
        )


def _build_merged_run(
    runs: dict[str, Run],
    group_by_tag: str,
) -> tuple[Run, set[str]]:
    """Merge all runs into a single Run with configuration labels as tags.

    Args:
        runs: Mapping of configuration label to benchmark Run.
        group_by_tag: Tag key used for grouping.

    Returns:
        Tuple of (merged_run, available_metric_names).
    """
    all_metric_defs: dict[str, MetricDef] = {}
    all_points: list[Point] = []
    available_metrics: set[str] = set()

    for label, run in runs.items():
        all_metric_defs.update(run.metric_defs)
        for point in run.points:
            tagged_point = Point(
                name=point.name,
                scenario=point.scenario,
                tags={**point.tags, group_by_tag: label},
                metrics=point.metrics,
            )
            all_points.append(tagged_point)
            available_metrics.update(point.metrics.keys())

    merged_run = Run(points=tuple(all_points), metric_defs=all_metric_defs)
    return merged_run, available_metrics


def _compute_improvement_factors(
    rankings: list[RankEntry],
    *,
    higher: bool,
) -> dict[str, float]:
    """Compute improvement factor of best config vs each other config.

    Args:
        rankings: Ranked entries (rank 1 = best).
        higher: Whether higher values are better for this metric.

    Returns:
        {label: improvement_factor}.
    """
    best_value = rankings[0].value
    factors: dict[str, float] = {}
    for r in rankings:
        if r.value == 0 and best_value == 0:
            factors[r.label] = 1.0
        elif higher:
            factors[r.label] = best_value / r.value if r.value != 0 else float("inf")
        else:
            factors[r.label] = r.value / best_value if best_value != 0 else float("inf")
    return factors


def compare_configurations(
    runs: dict[str, Run],
    metrics: Sequence[str] | None = None,
    *,
    group_by_tag: str = "framework",
) -> ComparisonReport:
    """Compare benchmark runs across different configurations.

    Builds a merged Run from all provided runs, using configuration labels
    as framework tags, then leverages rank_table and aggregate_score.

    Args:
        runs: Mapping of configuration label to benchmark Run.
        metrics: Subset of metric names to compare. Defaults to all metrics
            found across all runs.
        group_by_tag: Tag key used for grouping (default "framework").

    Returns:
        ComparisonReport with per-metric comparisons and overall winner.

    Raises:
        ValueError: If fewer than 2 configurations are provided.
    """
    if len(runs) < 2:
        msg = "At least 2 configurations are required for comparison"
        raise ValueError(msg)

    merged_run, available_metrics = _build_merged_run(runs, group_by_tag)
    metric_names = list(metrics) if metrics is not None else sorted(available_metrics)
    labels = tuple(runs.keys())

    comparisons: list[MetricComparison] = []
    winner_by_metric: dict[str, str] = {}

    for metric_name in metric_names:
        rankings = rank_table(merged_run, metric_name, group_by_tag)
        if not rankings:
            continue

        higher = is_higher_better(merged_run.metric_defs.get(metric_name))
        improvement_factors = _compute_improvement_factors(rankings, higher=higher)

        comparisons.append(
            MetricComparison(
                metric_name=metric_name,
                values={r.label: r.value for r in rankings},
                rankings=tuple(rankings),
                best_label=rankings[0].label,
                improvement_factors=improvement_factors,
            )
        )
        winner_by_metric[metric_name] = rankings[0].label

    weights = {m: 1.0 for m in metric_names if m in available_metrics}
    scores = aggregate_score(merged_run, weights)
    overall_winner = max(scores, key=lambda k: scores[k]) if scores else labels[0]

    return ComparisonReport(
        name="comparison",
        labels_compared=labels,
        metric_comparisons=tuple(comparisons),
        winner_by_metric=winner_by_metric,
        overall_winner=overall_winner,
    )
