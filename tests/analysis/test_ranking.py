"""Tests for calibrax.analysis.ranking module."""

from __future__ import annotations

import pytest

from calibrax.analysis.ranking import aggregate_score, rank_table
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


class TestRankTable:
    """Tests for rank_table."""

    def test_higher_is_better_ordering(self) -> None:
        """Higher-is-better should rank highest value first."""
        defs = {
            "throughput": MetricDef(
                name="throughput", unit="ops/s", direction=MetricDirection.HIGHER
            )
        }
        points = [
            Point(
                name="a",
                scenario="s1",
                tags={"framework": "jax"},
                metrics={"throughput": Metric(value=100.0)},
            ),
            Point(
                name="b",
                scenario="s1",
                tags={"framework": "pytorch"},
                metrics={"throughput": Metric(value=200.0)},
            ),
            Point(
                name="c",
                scenario="s1",
                tags={"framework": "numpy"},
                metrics={"throughput": Metric(value=50.0)},
            ),
        ]
        run = _make_run(points, defs)
        result = rank_table(run, "throughput")
        assert result[0].label == "pytorch"
        assert result[0].rank == 1
        assert result[-1].label == "numpy"
        assert result[-1].rank == 3

    def test_lower_is_better_ordering(self) -> None:
        """Lower-is-better should rank lowest value first."""
        defs = {"latency": MetricDef(name="latency", unit="ms", direction=MetricDirection.LOWER)}
        points = [
            Point(
                name="a",
                scenario="s1",
                tags={"framework": "jax"},
                metrics={"latency": Metric(value=5.0)},
            ),
            Point(
                name="b",
                scenario="s1",
                tags={"framework": "pytorch"},
                metrics={"latency": Metric(value=20.0)},
            ),
        ]
        run = _make_run(points, defs)
        result = rank_table(run, "latency")
        assert result[0].label == "jax"
        assert result[0].rank == 1

    def test_delta_from_best(self) -> None:
        """Delta from best should be computed correctly."""
        defs = {
            "throughput": MetricDef(
                name="throughput", unit="ops/s", direction=MetricDirection.HIGHER
            )
        }
        points = [
            Point(
                name="a",
                scenario="s1",
                tags={"framework": "fast"},
                metrics={"throughput": Metric(value=200.0)},
            ),
            Point(
                name="b",
                scenario="s1",
                tags={"framework": "slow"},
                metrics={"throughput": Metric(value=100.0)},
            ),
        ]
        run = _make_run(points, defs)
        result = rank_table(run, "throughput")
        assert result[0].delta_from_best == pytest.approx(0.0)
        assert result[1].delta_from_best == pytest.approx(50.0)

    def test_is_best_flag(self) -> None:
        """Only rank 1 should have is_best=True."""
        defs = {
            "throughput": MetricDef(
                name="throughput", unit="ops/s", direction=MetricDirection.HIGHER
            )
        }
        points = [
            Point(
                name="a",
                scenario="s1",
                tags={"framework": "a"},
                metrics={"throughput": Metric(value=100.0)},
            ),
            Point(
                name="b",
                scenario="s1",
                tags={"framework": "b"},
                metrics={"throughput": Metric(value=50.0)},
            ),
        ]
        run = _make_run(points, defs)
        result = rank_table(run, "throughput")
        assert result[0].is_best is True
        assert result[1].is_best is False

    def test_missing_metric_skipped(self) -> None:
        """Points without the metric should be skipped."""
        defs = {
            "throughput": MetricDef(
                name="throughput", unit="ops/s", direction=MetricDirection.HIGHER
            )
        }
        points = [
            Point(
                name="a",
                scenario="s1",
                tags={"framework": "has_it"},
                metrics={"throughput": Metric(value=100.0)},
            ),
            Point(
                name="b",
                scenario="s1",
                tags={"framework": "no_it"},
                metrics={"latency": Metric(value=5.0)},
            ),
        ]
        run = _make_run(points, defs)
        result = rank_table(run, "throughput")
        assert len(result) == 1
        assert result[0].label == "has_it"

    def test_empty_run(self) -> None:
        """Empty run should return empty list."""
        run = _make_run([])
        assert rank_table(run, "throughput") == []


class TestAggregateScore:
    """Tests for aggregate_score."""

    def test_equal_weights(self) -> None:
        """Equal weights should weight all metrics equally."""
        defs = {
            "throughput": MetricDef(
                name="throughput", unit="ops/s", direction=MetricDirection.HIGHER
            ),
            "latency": MetricDef(name="latency", unit="ms", direction=MetricDirection.LOWER),
        }
        points = [
            Point(
                name="a",
                scenario="s1",
                tags={"framework": "best"},
                metrics={
                    "throughput": Metric(value=200.0),
                    "latency": Metric(value=5.0),
                },
            ),
            Point(
                name="b",
                scenario="s1",
                tags={"framework": "worst"},
                metrics={
                    "throughput": Metric(value=100.0),
                    "latency": Metric(value=20.0),
                },
            ),
        ]
        run = _make_run(points, defs)
        scores = aggregate_score(run, {"throughput": 1.0, "latency": 1.0})
        assert scores["best"] == pytest.approx(1.0)
        assert scores["worst"] == pytest.approx(0.0)

    def test_normalization_range(self) -> None:
        """Scores should be in [0, 1]."""
        defs = {
            "throughput": MetricDef(
                name="throughput", unit="ops/s", direction=MetricDirection.HIGHER
            ),
        }
        points = [
            Point(
                name="a",
                scenario="s1",
                tags={"framework": "a"},
                metrics={"throughput": Metric(value=50.0)},
            ),
            Point(
                name="b",
                scenario="s1",
                tags={"framework": "b"},
                metrics={"throughput": Metric(value=100.0)},
            ),
            Point(
                name="c",
                scenario="s1",
                tags={"framework": "c"},
                metrics={"throughput": Metric(value=75.0)},
            ),
        ]
        run = _make_run(points, defs)
        scores = aggregate_score(run, {"throughput": 1.0})
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_empty_run(self) -> None:
        """Empty run should return empty dict."""
        run = _make_run([])
        assert aggregate_score(run, {"throughput": 1.0}) == {}

    def test_single_framework(self) -> None:
        """Single framework should get score 1.0 (best by default)."""
        defs = {
            "throughput": MetricDef(
                name="throughput", unit="ops/s", direction=MetricDirection.HIGHER
            ),
        }
        points = [
            Point(
                name="a",
                scenario="s1",
                tags={"framework": "only"},
                metrics={"throughput": Metric(value=100.0)},
            ),
        ]
        run = _make_run(points, defs)
        scores = aggregate_score(run, {"throughput": 1.0})
        assert scores["only"] == pytest.approx(1.0)
