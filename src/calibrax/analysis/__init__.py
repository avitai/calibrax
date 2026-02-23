"""Analysis: regression detection, comparison, ranking, scaling, Pareto fronts."""

from calibrax.analysis.comparison import compare_configurations, ComparisonReport, MetricComparison
from calibrax.analysis.pareto import pareto_front
from calibrax.analysis.ranking import aggregate_score, rank_table
from calibrax.analysis.regression import detect_regressions
from calibrax.analysis.scaling import scaling_fit


__all__ = [
    "ComparisonReport",
    "MetricComparison",
    "aggregate_score",
    "compare_configurations",
    "detect_regressions",
    "pareto_front",
    "rank_table",
    "scaling_fit",
]

# Note: detect_change_points and ChangePoint are NOT re-exported here
# because they require the optional 'ruptures' dependency.
# Import directly: from calibrax.analysis.changepoint import detect_change_points
