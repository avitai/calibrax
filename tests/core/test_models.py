"""Tests for calibrax core data models.

Covers MetricDirection, MetricPriority, MetricDef, Metric, Point, Run,
Regression, RankEntry, SignificanceResult, ScalingLaw, TrendPoint, TrendSeries,
and the is_higher_better helper function.
"""

import dataclasses
import json
from datetime import datetime

import jax.numpy as jnp
import pytest

from calibrax.core.models import (
    is_higher_better,
    Metric,
    MetricDef,
    MetricDirection,
    MetricPriority,
    Point,
    RankEntry,
    Regression,
    Run,
    ScalingLaw,
    SignificanceResult,
    TrendPoint,
    TrendSeries,
)


class TestMetricDirection:
    """Tests for MetricDirection StrEnum."""

    def test_enum_values(self) -> None:
        assert MetricDirection.HIGHER == "higher"
        assert MetricDirection.LOWER == "lower"
        assert MetricDirection.INFO == "info"

    def test_string_comparison(self) -> None:
        assert MetricDirection.HIGHER == "higher"
        assert MetricDirection("lower") == MetricDirection.LOWER


class TestMetricPriority:
    """Tests for MetricPriority StrEnum."""

    def test_enum_values(self) -> None:
        assert MetricPriority.PRIMARY == "primary"
        assert MetricPriority.SECONDARY == "secondary"


class TestIsHigherBetter:
    """Tests for is_higher_better helper."""

    def test_none_returns_true(self) -> None:
        assert is_higher_better(None) is True

    def test_higher_returns_true(self) -> None:
        md = MetricDef(
            name="throughput",
            unit="elem/s",
            direction=MetricDirection.HIGHER,
        )
        assert is_higher_better(md) is True

    def test_info_returns_true(self) -> None:
        md = MetricDef(
            name="version",
            unit="",
            direction=MetricDirection.INFO,
        )
        assert is_higher_better(md) is True

    def test_lower_returns_false(self) -> None:
        md = MetricDef(
            name="latency",
            unit="ms",
            direction=MetricDirection.LOWER,
        )
        assert is_higher_better(md) is False


class TestMetricDef:
    """Tests for MetricDef frozen dataclass."""

    def test_basic_construction(self) -> None:
        md = MetricDef(
            name="throughput",
            unit="elem/s",
            direction=MetricDirection.HIGHER,
        )
        assert md.name == "throughput"
        assert md.unit == "elem/s"
        assert md.direction == MetricDirection.HIGHER
        assert md.group == ""
        assert md.priority == MetricPriority.SECONDARY
        assert md.description == ""

    def test_full_construction(self) -> None:
        md = MetricDef(
            name="latency_p50",
            unit="ms",
            direction=MetricDirection.LOWER,
            group="Latency",
            priority=MetricPriority.PRIMARY,
            description="Median latency",
        )
        assert md.group == "Latency"
        assert md.priority == MetricPriority.PRIMARY
        assert md.description == "Median latency"

    def test_frozen_immutability(self) -> None:
        md = MetricDef(
            name="x",
            unit="",
            direction=MetricDirection.HIGHER,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            md.name = "y"  # type: ignore[misc]

    def test_kw_only_enforcement(self) -> None:
        with pytest.raises(TypeError):
            MetricDef("throughput", "elem/s", MetricDirection.HIGHER)  # type: ignore[misc]

    def test_serde_round_trip(self) -> None:
        md = MetricDef(
            name="throughput",
            unit="elem/s",
            direction=MetricDirection.HIGHER,
            group="Throughput",
            priority=MetricPriority.PRIMARY,
            description="Elements per second",
        )
        data = md.to_dict()
        restored = MetricDef.from_dict(data)
        assert restored.name == md.name
        assert restored.unit == md.unit
        assert restored.direction == md.direction
        assert restored.group == md.group
        assert restored.priority == md.priority
        assert restored.description == md.description


class TestMetric:
    """Tests for Metric frozen dataclass."""

    def test_value_only(self) -> None:
        m = Metric(value=42.0)
        assert m.value == 42.0
        assert m.lower is None
        assert m.upper is None
        assert m.samples is None

    def test_with_ci_bounds(self) -> None:
        m = Metric(value=100.0, lower=95.0, upper=105.0)
        assert m.lower == 95.0
        assert m.upper == 105.0

    def test_with_samples_as_tuple(self) -> None:
        m = Metric(value=100.0, samples=(95.0, 100.0, 105.0))
        assert isinstance(m.samples, tuple)
        assert len(m.samples) == 3

    def test_frozen_immutability(self) -> None:
        m = Metric(value=42.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.value = 99.0  # type: ignore[misc]

    def test_kw_only_enforcement(self) -> None:
        with pytest.raises(TypeError):
            Metric(42.0)  # type: ignore[misc]

    def test_serde_round_trip(self) -> None:
        m = Metric(
            value=42.0,
            lower=40.0,
            upper=44.0,
            samples=(40.0, 42.0, 44.0),
        )
        data = m.to_dict()
        restored = Metric.from_dict(data)
        assert restored.value == m.value
        assert restored.lower == m.lower
        assert restored.upper == m.upper
        assert restored.samples == m.samples

    def test_serde_without_optionals(self) -> None:
        m = Metric(value=42.0)
        data = m.to_dict()
        restored = Metric.from_dict(data)
        assert restored.value == 42.0
        assert restored.lower is None
        assert restored.samples is None


class TestPoint:
    """Tests for Point frozen dataclass."""

    def test_basic_construction(self) -> None:
        p = Point(
            name="CV-1/small",
            scenario="CV-1",
            tags={"framework": "Datarax"},
            metrics={"throughput": Metric(value=5000.0)},
        )
        assert p.name == "CV-1/small"
        assert p.scenario == "CV-1"
        assert p.tags["framework"] == "Datarax"
        assert p.metrics["throughput"].value == 5000.0

    def test_frozen_immutability(self) -> None:
        p = Point(
            name="x",
            scenario="s",
            tags={},
            metrics={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.name = "y"  # type: ignore[misc]

    def test_kw_only_enforcement(self) -> None:
        with pytest.raises(TypeError):
            Point("CV-1/small", "CV-1", {}, {})  # type: ignore[misc]

    def test_serde_round_trip(self) -> None:
        p = Point(
            name="CV-1/small",
            scenario="CV-1",
            tags={"framework": "Datarax", "variant": "small"},
            metrics={
                "throughput": Metric(value=5000.0),
                "latency_p50": Metric(value=12.0, lower=10.0, upper=14.0),
            },
        )
        data = p.to_dict()
        restored = Point.from_dict(data)
        assert restored.name == p.name
        assert restored.scenario == p.scenario
        assert restored.tags == p.tags
        assert restored.metrics["throughput"].value == 5000.0
        assert restored.metrics["latency_p50"].lower == 10.0


class TestRun:
    """Tests for Run frozen dataclass."""

    def test_minimal_construction(self) -> None:
        run = Run(points=())
        assert run.points == ()
        assert isinstance(run.id, str)
        assert len(run.id) == 12
        assert isinstance(run.timestamp, datetime)
        assert run.commit is None
        assert run.branch is None

    def test_full_construction(self) -> None:
        now = datetime(2026, 2, 10, 12, 0, 0)
        run = Run(
            points=(
                Point(
                    name="CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(value=5000.0)},
                ),
            ),
            id="abc123def456",
            timestamp=now,
            commit="abc123",
            branch="main",
            environment={"cpu": "AMD Ryzen"},
            metadata={"runner": "full"},
            metric_defs={
                "throughput": MetricDef(
                    name="throughput",
                    unit="elem/s",
                    direction=MetricDirection.HIGHER,
                ),
            },
        )
        assert run.id == "abc123def456"
        assert run.timestamp == now
        assert run.commit == "abc123"
        assert len(run.points) == 1

    def test_points_is_tuple(self) -> None:
        run = Run(points=())
        assert isinstance(run.points, tuple)

    def test_frozen_immutability(self) -> None:
        run = Run(points=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            run.id = "new"  # type: ignore[misc]

    def test_kw_only_enforcement(self) -> None:
        with pytest.raises(TypeError):
            Run(())  # type: ignore[misc]

    def test_auto_generated_id_unique(self) -> None:
        r1 = Run(points=())
        r2 = Run(points=())
        assert r1.id != r2.id

    def test_serde_round_trip(self) -> None:
        run = Run(
            points=(
                Point(
                    name="CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(value=5000.0)},
                ),
            ),
            commit="abc123",
            branch="main",
            environment={"cpu": "AMD Ryzen"},
            metric_defs={
                "throughput": MetricDef(
                    name="throughput",
                    unit="elem/s",
                    direction=MetricDirection.HIGHER,
                    group="Throughput",
                    priority=MetricPriority.PRIMARY,
                ),
            },
        )
        data = run.to_dict()
        restored = Run.from_dict(data)
        assert restored.commit == run.commit
        assert restored.branch == run.branch
        assert len(restored.points) == 1
        assert restored.points[0].metrics["throughput"].value == 5000.0
        assert restored.metric_defs["throughput"].direction == MetricDirection.HIGHER


class TestRegression:
    """Tests for Regression frozen dataclass."""

    def test_construction(self) -> None:
        r = Regression(
            metric="throughput",
            point_name="CV-1/small",
            baseline_value=5000.0,
            current_value=4500.0,
            delta_pct=-10.0,
            direction=MetricDirection.HIGHER,
        )
        assert r.delta_pct == -10.0

    def test_frozen_immutability(self) -> None:
        r = Regression(
            metric="throughput",
            point_name="CV-1/small",
            baseline_value=5000.0,
            current_value=4500.0,
            delta_pct=-10.0,
            direction=MetricDirection.HIGHER,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.metric = "latency"  # type: ignore[misc]

    def test_serde_round_trip(self) -> None:
        r = Regression(
            metric="throughput",
            point_name="CV-1/small",
            baseline_value=5000.0,
            current_value=4500.0,
            delta_pct=-10.0,
            direction=MetricDirection.HIGHER,
        )
        data = r.to_dict()
        restored = Regression.from_dict(data)
        assert restored.metric == r.metric
        assert restored.delta_pct == r.delta_pct
        assert restored.direction == r.direction


class TestRankEntry:
    """Tests for RankEntry frozen dataclass."""

    def test_construction(self) -> None:
        entry = RankEntry(
            label="Datarax",
            value=5000.0,
            rank=1,
            is_best=True,
            delta_from_best=0.0,
        )
        assert entry.is_best is True
        assert entry.rank == 1

    def test_frozen_immutability(self) -> None:
        entry = RankEntry(
            label="x",
            value=1.0,
            rank=1,
            is_best=True,
            delta_from_best=0.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.rank = 2  # type: ignore[misc]

    def test_serde_round_trip(self) -> None:
        entry = RankEntry(
            label="Datarax",
            value=5000.0,
            rank=1,
            is_best=True,
            delta_from_best=0.0,
        )
        data = entry.to_dict()
        restored = RankEntry.from_dict(data)
        assert restored.label == entry.label
        assert restored.value == entry.value
        assert restored.is_best == entry.is_best


class TestSignificanceResult:
    """Tests for SignificanceResult frozen dataclass."""

    def test_construction(self) -> None:
        sr = SignificanceResult(
            p_value=0.03,
            statistic=2.5,
            effect_size=0.8,
            significant=True,
            method="wilcoxon",
        )
        assert sr.significant is True

    def test_frozen_immutability(self) -> None:
        sr = SignificanceResult(
            p_value=0.03,
            statistic=2.5,
            effect_size=0.8,
            significant=True,
            method="wilcoxon",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sr.p_value = 0.5  # type: ignore[misc]

    def test_serde_round_trip(self) -> None:
        sr = SignificanceResult(
            p_value=0.03,
            statistic=2.5,
            effect_size=0.8,
            significant=True,
            method="wilcoxon",
        )
        data = sr.to_dict()
        restored = SignificanceResult.from_dict(data)
        assert restored.p_value == sr.p_value
        assert restored.significant == sr.significant
        assert restored.method == sr.method


class TestScalingLaw:
    """Tests for ScalingLaw frozen dataclass."""

    def test_construction(self) -> None:
        sl = ScalingLaw(
            coefficient=1.5,
            exponent=0.9,
            r_squared=0.98,
            complexity="O(n)",
        )
        assert sl.r_squared == 0.98

    def test_frozen_immutability(self) -> None:
        sl = ScalingLaw(
            coefficient=1.5,
            exponent=0.9,
            r_squared=0.98,
            complexity="O(n)",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sl.exponent = 1.0  # type: ignore[misc]

    def test_serde_round_trip(self) -> None:
        sl = ScalingLaw(
            coefficient=1.5,
            exponent=0.9,
            r_squared=0.98,
            complexity="O(n)",
        )
        data = sl.to_dict()
        restored = ScalingLaw.from_dict(data)
        assert restored.coefficient == sl.coefficient
        assert restored.exponent == sl.exponent
        assert restored.complexity == sl.complexity


class TestTrendPoint:
    """Tests for TrendPoint frozen dataclass."""

    def test_construction(self) -> None:
        tp = TrendPoint(
            run_id="abc123",
            timestamp=datetime(2026, 2, 10, 12, 0, 0),
            value=5000.0,
            commit="def456",
        )
        assert tp.value == 5000.0
        assert tp.commit == "def456"

    def test_optional_fields(self) -> None:
        tp = TrendPoint(
            run_id="abc123",
            timestamp=datetime(2026, 2, 10),
            value=5000.0,
        )
        assert tp.commit is None
        assert tp.lower is None
        assert tp.upper is None

    def test_frozen_immutability(self) -> None:
        tp = TrendPoint(
            run_id="abc123",
            timestamp=datetime(2026, 2, 10),
            value=5000.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            tp.value = 99.0  # type: ignore[misc]

    def test_serde_round_trip(self) -> None:
        tp = TrendPoint(
            run_id="abc123",
            timestamp=datetime(2026, 2, 10, 12, 0, 0),
            value=5000.0,
            commit="def456",
            lower=4800.0,
            upper=5200.0,
        )
        data = tp.to_dict()
        restored = TrendPoint.from_dict(data)
        assert restored.run_id == tp.run_id
        assert restored.timestamp == tp.timestamp
        assert restored.value == tp.value
        assert restored.commit == tp.commit
        assert restored.lower == tp.lower

    def test_serde_without_optionals(self) -> None:
        tp = TrendPoint(
            run_id="abc123",
            timestamp=datetime(2026, 2, 10),
            value=5000.0,
        )
        data = tp.to_dict()
        restored = TrendPoint.from_dict(data)
        assert restored.value == 5000.0
        assert restored.commit is None
        assert "commit" not in data


class TestTrendSeries:
    """Tests for TrendSeries frozen dataclass."""

    def test_construction(self) -> None:
        ts = TrendSeries(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
            points=(
                TrendPoint(
                    run_id="r1",
                    timestamp=datetime(2026, 2, 1),
                    value=4800.0,
                ),
                TrendPoint(
                    run_id="r2",
                    timestamp=datetime(2026, 2, 2),
                    value=5000.0,
                ),
            ),
        )
        assert ts.metric == "throughput"
        assert len(ts.points) == 2

    def test_points_is_tuple(self) -> None:
        ts = TrendSeries(
            metric="throughput",
            point_name="CV-1/small",
            tags={},
            points=(),
        )
        assert isinstance(ts.points, tuple)

    def test_frozen_immutability(self) -> None:
        ts = TrendSeries(
            metric="throughput",
            point_name="CV-1/small",
            tags={},
            points=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ts.metric = "latency"  # type: ignore[misc]

    def test_serde_round_trip(self) -> None:
        ts = TrendSeries(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
            points=(
                TrendPoint(
                    run_id="r1",
                    timestamp=datetime(2026, 2, 1),
                    value=4800.0,
                    commit="aaa",
                ),
                TrendPoint(
                    run_id="r2",
                    timestamp=datetime(2026, 2, 2),
                    value=5000.0,
                    commit="bbb",
                ),
            ),
        )
        data = ts.to_dict()
        restored = TrendSeries.from_dict(data)
        assert restored.metric == ts.metric
        assert restored.point_name == ts.point_name
        assert len(restored.points) == 2
        assert restored.points[0].value == 4800.0

    def test_empty_points(self) -> None:
        ts = TrendSeries(
            metric="latency",
            point_name="NLP-1",
            tags={},
            points=(),
        )
        assert ts.points == ()
        data = ts.to_dict()
        restored = TrendSeries.from_dict(data)
        assert restored.points == ()


class TestJAXScalarSerialization:
    """All to_dict() methods must produce JSON-serializable output for JAX scalars.

    When users compute metrics with JAX (e.g., jnp.mean()), results are JAX scalars
    (jnp.float32, jnp.int32) which json.dumps() cannot serialize. Every to_dict()
    method must convert numeric fields to Python primitives.
    """

    def test_metric_with_jax_scalars(self) -> None:
        """Metric fields from JAX operations must serialize to JSON."""
        m = Metric(
            value=jnp.float32(42.0),
            lower=jnp.float32(40.0),
            upper=jnp.float32(44.0),
            samples=(jnp.float32(40.0), jnp.float32(42.0), jnp.float32(44.0)),
        )
        data = m.to_dict()
        json_str = json.dumps(data)
        assert '"value": 42.0' in json_str

        restored = Metric.from_dict(data)
        assert restored.value == pytest.approx(42.0)

    def test_metric_value_is_python_float_in_dict(self) -> None:
        """to_dict() must return Python float, not JAX scalar."""
        m = Metric(value=jnp.float32(3.14))
        data = m.to_dict()
        assert type(data["value"]) is float

    def test_regression_with_jax_scalars(self) -> None:
        """Regression numeric fields from JAX must serialize."""
        r = Regression(
            metric="throughput",
            point_name="CV-1",
            baseline_value=jnp.float32(5000.0),
            current_value=jnp.float32(4500.0),
            delta_pct=jnp.float32(-10.0),
            direction=MetricDirection.HIGHER,
        )
        data = r.to_dict()
        json_str = json.dumps(data)
        assert "5000.0" in json_str

    def test_rank_entry_with_jax_scalars(self) -> None:
        """RankEntry with JAX value and rank must serialize."""
        entry = RankEntry(
            label="Model-A",
            value=jnp.float32(95.0),
            rank=int(jnp.int32(1)),
            is_best=True,
            delta_from_best=jnp.float32(0.0),
        )
        data = entry.to_dict()
        json_str = json.dumps(data)
        assert "95.0" in json_str

    def test_significance_result_with_jax_scalars(self) -> None:
        """SignificanceResult with JAX p_value, statistic, effect_size."""
        sr = SignificanceResult(
            p_value=jnp.float32(0.03),
            statistic=jnp.float32(2.5),
            effect_size=jnp.float32(0.8),
            significant=True,
            method="wilcoxon",
        )
        data = sr.to_dict()
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["p_value"] == pytest.approx(0.03, abs=1e-5)

    def test_scaling_law_with_jax_scalars(self) -> None:
        """ScalingLaw with JAX coefficient, exponent, r_squared."""
        sl = ScalingLaw(
            coefficient=jnp.float32(1.5),
            exponent=jnp.float32(0.9),
            r_squared=jnp.float32(0.98),
            complexity="O(n)",
        )
        data = sl.to_dict()
        json_str = json.dumps(data)
        assert "1.5" in json_str

    def test_trend_point_with_jax_scalars(self) -> None:
        """TrendPoint with JAX value, lower, upper."""
        tp = TrendPoint(
            run_id="abc",
            timestamp=datetime(2026, 2, 10),
            value=jnp.float32(5000.0),
            lower=jnp.float32(4800.0),
            upper=jnp.float32(5200.0),
        )
        data = tp.to_dict()
        json_str = json.dumps(data)
        assert "5000.0" in json_str

    def test_point_with_jax_metric_values(self) -> None:
        """Point containing Metric with JAX scalars must serialize."""
        p = Point(
            name="test",
            scenario="s1",
            metrics={"acc": Metric(value=jnp.float32(0.95))},
        )
        data = p.to_dict()
        json_str = json.dumps(data)
        assert "0.95" in json_str or "0.949" in json_str

    def test_full_round_trip_with_jax_scalars(self) -> None:
        """Full serde round-trip with JAX scalars throughout."""
        m = Metric(
            value=jnp.float32(42.0),
            lower=jnp.float32(40.0),
            upper=jnp.float32(44.0),
            samples=(jnp.float32(40.0), jnp.float32(42.0), jnp.float32(44.0)),
        )
        data = m.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        restored = Metric.from_dict(parsed)
        assert restored.value == pytest.approx(42.0)
        assert restored.lower == pytest.approx(40.0)
        assert restored.samples is not None
        assert len(restored.samples) == 3
