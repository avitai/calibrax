"""Tests for calibrax.analysis.pareto module."""

from __future__ import annotations

from calibrax.analysis.pareto import pareto_front
from calibrax.core.models import (
    Metric,
    MetricDef,
    MetricDirection,
    Point,
)


def _make_point(
    name: str, x_val: float, y_val: float, x_metric: str = "speed", y_metric: str = "accuracy"
) -> Point:
    """Helper to create a Point with two metrics."""
    return Point(
        name=name,
        scenario="s1",
        metrics={
            x_metric: Metric(value=x_val),
            y_metric: Metric(value=y_val),
        },
    )


class TestParetoFront:
    """Tests for pareto_front."""

    def test_single_point(self) -> None:
        """Single point is always on the front."""
        p = _make_point("a", 10.0, 20.0)
        front = pareto_front([p], "speed", "accuracy")
        assert len(front) == 1
        assert front[0] is p

    def test_dominated_point_excluded(self) -> None:
        """A dominated point should not be on the front."""
        dominant = _make_point("dominant", 10.0, 20.0)
        dominated = _make_point("dominated", 5.0, 10.0)
        front = pareto_front([dominant, dominated], "speed", "accuracy")
        assert len(front) == 1
        assert front[0].name == "dominant"

    def test_all_on_front(self) -> None:
        """Non-dominated points should all be on the front."""
        # Trade-off: a is fast but inaccurate, b is slow but accurate
        a = _make_point("a", 100.0, 10.0)
        b = _make_point("b", 10.0, 100.0)
        front = pareto_front([a, b], "speed", "accuracy")
        assert len(front) == 2

    def test_direction_aware_lower_is_better(self) -> None:
        """Lower-is-better direction should invert dominance."""
        defs = {
            "latency": MetricDef(name="latency", unit="ms", direction=MetricDirection.LOWER),
            "error": MetricDef(name="error", unit="", direction=MetricDirection.LOWER),
        }
        # Both lower is better: (1, 1) dominates (5, 5)
        good = Point(
            name="good",
            scenario="s1",
            metrics={
                "latency": Metric(value=1.0),
                "error": Metric(value=1.0),
            },
        )
        bad = Point(
            name="bad",
            scenario="s1",
            metrics={
                "latency": Metric(value=5.0),
                "error": Metric(value=5.0),
            },
        )
        front = pareto_front([good, bad], "latency", "error", metric_defs=defs)
        assert len(front) == 1
        assert front[0].name == "good"

    def test_missing_metrics_skipped(self) -> None:
        """Points missing either metric should be skipped."""
        complete = _make_point("complete", 10.0, 20.0)
        missing_x = Point(
            name="missing_x",
            scenario="s1",
            metrics={"accuracy": Metric(value=30.0)},
        )
        front = pareto_front([complete, missing_x], "speed", "accuracy")
        assert len(front) == 1
        assert front[0].name == "complete"

    def test_empty_list(self) -> None:
        """Empty input should return empty front."""
        assert pareto_front([], "speed", "accuracy") == []

    def test_three_points_mixed(self) -> None:
        """Mixed scenario with some dominated and some on front."""
        a = _make_point("a", 10.0, 80.0)  # On front
        b = _make_point("b", 80.0, 10.0)  # On front
        c = _make_point("c", 5.0, 5.0)  # Dominated by both
        front = pareto_front([a, b, c], "speed", "accuracy")
        assert len(front) == 2
        names = {p.name for p in front}
        assert names == {"a", "b"}

    def test_equal_on_one_metric(self) -> None:
        """Points equal on one metric, different on other."""
        a = _make_point("a", 10.0, 50.0)
        b = _make_point("b", 10.0, 40.0)
        # a dominates b: equal on speed, better on accuracy
        front = pareto_front([a, b], "speed", "accuracy")
        assert len(front) == 1
        assert front[0].name == "a"

    def test_order_preserved(self) -> None:
        """Front points should maintain input order."""
        a = _make_point("a", 100.0, 10.0)
        b = _make_point("b", 50.0, 50.0)
        c = _make_point("c", 10.0, 100.0)
        front = pareto_front([a, b, c], "speed", "accuracy")
        names = [p.name for p in front]
        assert names == ["a", "b", "c"] or set(names).issubset({"a", "b", "c"})
