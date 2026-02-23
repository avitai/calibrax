"""Tests for calibrax.analysis.comparison module."""

from __future__ import annotations

import pytest

from calibrax.analysis.comparison import (
    compare_configurations,
    ComparisonReport,
    MetricComparison,
)
from calibrax.core.models import (
    Metric,
    MetricDef,
    MetricDirection,
    Point,
    Run,
)


def _make_run(
    points: list[Point],
    metric_defs: dict[str, MetricDef] | None = None,
) -> Run:
    """Helper to create a Run with given points and metric_defs."""
    return Run(
        points=tuple(points),
        metric_defs=metric_defs or {},
    )


def _make_config_run(
    label: str,
    throughput: float,
    latency: float,
    metric_defs: dict[str, MetricDef],
) -> Run:
    """Helper to create a single-point run for a configuration."""
    return _make_run(
        [
            Point(
                name="bench1",
                scenario="default",
                tags={"framework": label},
                metrics={
                    "throughput": Metric(value=throughput),
                    "latency": Metric(value=latency),
                },
            )
        ],
        metric_defs,
    )


METRIC_DEFS = {
    "throughput": MetricDef(name="throughput", unit="ops/s", direction=MetricDirection.HIGHER),
    "latency": MetricDef(name="latency", unit="ms", direction=MetricDirection.LOWER),
}


class TestCompareConfigurations:
    """Tests for compare_configurations."""

    def test_basic_comparison(self) -> None:
        """Should compare two configurations and identify winners."""
        runs = {
            "jax": _make_config_run("jax", 200.0, 5.0, METRIC_DEFS),
            "pytorch": _make_config_run("pytorch", 100.0, 10.0, METRIC_DEFS),
        }
        report = compare_configurations(runs)
        assert report.winner_by_metric["throughput"] == "jax"
        assert report.winner_by_metric["latency"] == "jax"
        assert report.overall_winner == "jax"

    def test_mixed_winners(self) -> None:
        """Different metrics can have different winners."""
        runs = {
            "fast": _make_config_run("fast", 100.0, 2.0, METRIC_DEFS),
            "cheap": _make_config_run("cheap", 200.0, 10.0, METRIC_DEFS),
        }
        report = compare_configurations(runs)
        assert report.winner_by_metric["throughput"] == "cheap"
        assert report.winner_by_metric["latency"] == "fast"

    def test_metric_subset(self) -> None:
        """Should only compare requested metrics."""
        runs = {
            "a": _make_config_run("a", 100.0, 5.0, METRIC_DEFS),
            "b": _make_config_run("b", 200.0, 10.0, METRIC_DEFS),
        }
        report = compare_configurations(runs, metrics=["throughput"])
        assert len(report.metric_comparisons) == 1
        assert report.metric_comparisons[0].metric_name == "throughput"

    def test_labels_compared(self) -> None:
        """Report should list all compared configuration labels."""
        runs = {
            "x": _make_config_run("x", 100.0, 5.0, METRIC_DEFS),
            "y": _make_config_run("y", 200.0, 10.0, METRIC_DEFS),
        }
        report = compare_configurations(runs)
        assert set(report.labels_compared) == {"x", "y"}

    def test_improvement_factors_higher_is_better(self) -> None:
        """Improvement factor for higher-is-better: best/this."""
        runs = {
            "fast": _make_config_run("fast", 200.0, 5.0, METRIC_DEFS),
            "slow": _make_config_run("slow", 100.0, 5.0, METRIC_DEFS),
        }
        report = compare_configurations(runs)
        tp_comp = next(mc for mc in report.metric_comparisons if mc.metric_name == "throughput")
        assert tp_comp.improvement_factors["fast"] == pytest.approx(1.0)
        assert tp_comp.improvement_factors["slow"] == pytest.approx(2.0)

    def test_improvement_factors_lower_is_better(self) -> None:
        """Improvement factor for lower-is-better: this/best."""
        runs = {
            "fast": _make_config_run("fast", 100.0, 2.0, METRIC_DEFS),
            "slow": _make_config_run("slow", 100.0, 10.0, METRIC_DEFS),
        }
        report = compare_configurations(runs)
        lat_comp = next(mc for mc in report.metric_comparisons if mc.metric_name == "latency")
        assert lat_comp.improvement_factors["fast"] == pytest.approx(1.0)
        assert lat_comp.improvement_factors["slow"] == pytest.approx(5.0)

    def test_three_configurations(self) -> None:
        """Should handle three or more configurations."""
        runs = {
            "a": _make_config_run("a", 300.0, 3.0, METRIC_DEFS),
            "b": _make_config_run("b", 200.0, 5.0, METRIC_DEFS),
            "c": _make_config_run("c", 100.0, 10.0, METRIC_DEFS),
        }
        report = compare_configurations(runs)
        assert len(report.labels_compared) == 3
        assert report.winner_by_metric["throughput"] == "a"

    def test_too_few_configurations_raises(self) -> None:
        """Should raise ValueError with fewer than 2 configs."""
        runs = {"only": _make_config_run("only", 100.0, 5.0, METRIC_DEFS)}
        with pytest.raises(ValueError, match="At least 2"):
            compare_configurations(runs)

    def test_rankings_are_tuples(self) -> None:
        """Rankings in MetricComparison should be tuples."""
        runs = {
            "a": _make_config_run("a", 100.0, 5.0, METRIC_DEFS),
            "b": _make_config_run("b", 200.0, 10.0, METRIC_DEFS),
        }
        report = compare_configurations(runs)
        for mc in report.metric_comparisons:
            assert isinstance(mc.rankings, tuple)


class TestComparisonReportSerde:
    """Tests for ComparisonReport serialization."""

    def test_to_dict_roundtrip(self) -> None:
        """to_dict/from_dict should produce equivalent objects."""
        runs = {
            "a": _make_config_run("a", 100.0, 5.0, METRIC_DEFS),
            "b": _make_config_run("b", 200.0, 10.0, METRIC_DEFS),
        }
        report = compare_configurations(runs)
        d = report.to_dict()
        restored = ComparisonReport.from_dict(d)
        assert restored.name == report.name
        assert restored.overall_winner == report.overall_winner
        assert set(restored.labels_compared) == set(report.labels_compared)
        assert len(restored.metric_comparisons) == len(report.metric_comparisons)

    def test_to_dict_values_are_python_floats(self) -> None:
        """Serialized values should be Python floats."""
        runs = {
            "a": _make_config_run("a", 100.0, 5.0, METRIC_DEFS),
            "b": _make_config_run("b", 200.0, 10.0, METRIC_DEFS),
        }
        report = compare_configurations(runs)
        d = report.to_dict()
        for mc in d["metric_comparisons"]:
            for v in mc["values"].values():
                assert type(v) is float


class TestMetricComparison:
    """Tests for MetricComparison dataclass."""

    def test_to_dict(self) -> None:
        """to_dict should return all fields."""
        mc = MetricComparison(
            metric_name="throughput",
            values={"a": 100.0, "b": 200.0},
            rankings=(),
            best_label="b",
            improvement_factors={"a": 2.0, "b": 1.0},
        )
        d = mc.to_dict()
        assert d["metric_name"] == "throughput"
        assert d["best_label"] == "b"
        assert d["improvement_factors"]["a"] == 2.0
