"""Tests for calibrax.analysis.regression module."""

from __future__ import annotations

from calibrax.analysis.regression import detect_regressions
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


def _throughput_def() -> MetricDef:
    return MetricDef(name="throughput", unit="ops/s", direction=MetricDirection.HIGHER)


def _latency_def() -> MetricDef:
    return MetricDef(name="latency", unit="ms", direction=MetricDirection.LOWER)


def _info_def() -> MetricDef:
    return MetricDef(name="version", unit="", direction=MetricDirection.INFO)


class TestDetectRegressions:
    """Tests for detect_regressions."""

    def test_no_regression_identical_values(self) -> None:
        """Identical values should produce no regressions."""
        defs = {"throughput": _throughput_def()}
        p = Point(
            name="bench1",
            scenario="s1",
            metrics={"throughput": Metric(value=100.0)},
        )
        run = _make_run([p], defs)
        baseline = _make_run([p], defs)
        assert detect_regressions(run, baseline) == []

    def test_higher_is_better_regression(self) -> None:
        """Higher-is-better metric that decreased should be flagged."""
        defs = {"throughput": _throughput_def()}
        baseline_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"throughput": Metric(value=100.0)},
        )
        current_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"throughput": Metric(value=90.0)},
        )
        run = _make_run([current_p], defs)
        baseline = _make_run([baseline_p], defs)
        regs = detect_regressions(run, baseline)
        assert len(regs) == 1
        assert regs[0].metric == "throughput"
        assert regs[0].point_name == "bench1"
        assert regs[0].delta_pct < 0

    def test_lower_is_better_regression(self) -> None:
        """Lower-is-better metric that increased should be flagged."""
        defs = {"latency": _latency_def()}
        baseline_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"latency": Metric(value=10.0)},
        )
        current_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"latency": Metric(value=20.0)},
        )
        run = _make_run([current_p], defs)
        baseline = _make_run([baseline_p], defs)
        regs = detect_regressions(run, baseline)
        assert len(regs) == 1
        assert regs[0].metric == "latency"
        assert regs[0].delta_pct > 0

    def test_info_direction_skipped(self) -> None:
        """Info metrics should never flag regressions."""
        defs = {"version": _info_def()}
        baseline_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"version": Metric(value=1.0)},
        )
        current_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"version": Metric(value=2.0)},
        )
        run = _make_run([current_p], defs)
        baseline = _make_run([baseline_p], defs)
        assert detect_regressions(run, baseline) == []

    def test_skips_points_not_in_baseline(self) -> None:
        """New points not in baseline should be skipped."""
        defs = {"throughput": _throughput_def()}
        current_p = Point(
            name="new_bench",
            scenario="s1",
            metrics={"throughput": Metric(value=50.0)},
        )
        baseline_p = Point(
            name="old_bench",
            scenario="s1",
            metrics={"throughput": Metric(value=100.0)},
        )
        run = _make_run([current_p], defs)
        baseline = _make_run([baseline_p], defs)
        assert detect_regressions(run, baseline) == []

    def test_threshold_boundary_no_regression(self) -> None:
        """Exactly at threshold should NOT be flagged."""
        defs = {"throughput": _throughput_def()}
        baseline_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"throughput": Metric(value=100.0)},
        )
        # 5% decrease = exactly at threshold=0.05
        current_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"throughput": Metric(value=95.0)},
        )
        run = _make_run([current_p], defs)
        baseline = _make_run([baseline_p], defs)
        assert detect_regressions(run, baseline, threshold=0.05) == []

    def test_threshold_boundary_just_beyond(self) -> None:
        """Just beyond threshold should be flagged."""
        defs = {"throughput": _throughput_def()}
        baseline_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"throughput": Metric(value=100.0)},
        )
        current_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"throughput": Metric(value=94.9)},
        )
        run = _make_run([current_p], defs)
        baseline = _make_run([baseline_p], defs)
        regs = detect_regressions(run, baseline, threshold=0.05)
        assert len(regs) == 1

    def test_multiple_regressions(self) -> None:
        """Multiple regressions in one run should all be reported."""
        defs = {
            "throughput": _throughput_def(),
            "latency": _latency_def(),
        }
        baseline_p = Point(
            name="bench1",
            scenario="s1",
            metrics={
                "throughput": Metric(value=100.0),
                "latency": Metric(value=10.0),
            },
        )
        current_p = Point(
            name="bench1",
            scenario="s1",
            metrics={
                "throughput": Metric(value=50.0),
                "latency": Metric(value=50.0),
            },
        )
        run = _make_run([current_p], defs)
        baseline = _make_run([baseline_p], defs)
        regs = detect_regressions(run, baseline)
        assert len(regs) == 2
        metric_names = {r.metric for r in regs}
        assert metric_names == {"throughput", "latency"}

    def test_empty_runs(self) -> None:
        """Empty runs should produce no regressions."""
        run = _make_run([])
        baseline = _make_run([])
        assert detect_regressions(run, baseline) == []

    def test_tag_matching(self) -> None:
        """Points with different tags should not match."""
        defs = {"throughput": _throughput_def()}
        baseline_p = Point(
            name="bench1",
            scenario="s1",
            tags={"framework": "jax"},
            metrics={"throughput": Metric(value=100.0)},
        )
        current_p = Point(
            name="bench1",
            scenario="s1",
            tags={"framework": "pytorch"},
            metrics={"throughput": Metric(value=50.0)},
        )
        run = _make_run([current_p], defs)
        baseline = _make_run([baseline_p], defs)
        assert detect_regressions(run, baseline) == []

    def test_zero_baseline_skipped(self) -> None:
        """Zero baseline value should be skipped (avoid div-by-zero)."""
        defs = {"throughput": _throughput_def()}
        baseline_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"throughput": Metric(value=0.0)},
        )
        current_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"throughput": Metric(value=100.0)},
        )
        run = _make_run([current_p], defs)
        baseline = _make_run([baseline_p], defs)
        assert detect_regressions(run, baseline) == []

    def test_improvement_not_flagged(self) -> None:
        """Improvements should not be flagged as regressions."""
        defs = {"throughput": _throughput_def()}
        baseline_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"throughput": Metric(value=100.0)},
        )
        current_p = Point(
            name="bench1",
            scenario="s1",
            metrics={"throughput": Metric(value=200.0)},
        )
        run = _make_run([current_p], defs)
        baseline = _make_run([baseline_p], defs)
        assert detect_regressions(run, baseline) == []

    def test_metric_defs_merged(self) -> None:
        """Metric defs from both runs should be merged."""
        baseline_defs = {"throughput": _throughput_def()}
        run_defs = {"latency": _latency_def()}
        baseline_p = Point(
            name="bench1",
            scenario="s1",
            metrics={
                "throughput": Metric(value=100.0),
                "latency": Metric(value=10.0),
            },
        )
        current_p = Point(
            name="bench1",
            scenario="s1",
            metrics={
                "throughput": Metric(value=50.0),
                "latency": Metric(value=50.0),
            },
        )
        run = _make_run([current_p], run_defs)
        baseline = _make_run([baseline_p], baseline_defs)
        regs = detect_regressions(run, baseline)
        assert len(regs) == 2
