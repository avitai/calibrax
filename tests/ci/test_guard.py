"""Tests for calibrax.ci.guard module."""

from __future__ import annotations

import pytest

from calibrax.ci.guard import CIGuard, GuardResult
from calibrax.core.models import (
    MetricDef,
    MetricDirection,
    Run,
)
from calibrax.storage.store import Store
from tests.factories import make_throughput_only_run


def _make_run(throughput: float, run_id: str = "run1") -> Run:
    """Helper to create a single-point run."""
    return make_throughput_only_run(
        throughput=throughput,
        run_id=run_id,
        metric_defs={
            "throughput": MetricDef(
                name="throughput",
                unit="ops/s",
                direction=MetricDirection.HIGHER,
            )
        },
    )


class TestCIGuard:
    """Tests for CIGuard."""

    def test_no_regression_passes(self, tmp_path: str) -> None:
        """Check should pass when no regressions detected."""
        store = Store(tmp_path)
        baseline = _make_run(100.0, run_id="baseline")
        current = _make_run(100.0, run_id="current")
        store.save(baseline)
        store.save(current)
        store.set_baseline("baseline")

        guard = CIGuard(store, threshold=0.05)
        result = guard.check("current")
        assert result.passed is True
        assert len(result.regressions) == 0

    def test_regression_detected(self, tmp_path: str) -> None:
        """Check should fail when regression exceeds threshold."""
        store = Store(tmp_path)
        baseline = _make_run(100.0, run_id="baseline")
        current = _make_run(80.0, run_id="current")  # 20% regression
        store.save(baseline)
        store.save(current)
        store.set_baseline("baseline")

        guard = CIGuard(store, threshold=0.05)
        result = guard.check("current")
        assert result.passed is False
        assert len(result.regressions) > 0

    def test_no_baseline_raises(self, tmp_path: str) -> None:
        """Check should raise when no baseline is set."""
        store = Store(tmp_path)
        current = _make_run(100.0, run_id="current")
        store.save(current)

        guard = CIGuard(store)
        with pytest.raises(FileNotFoundError, match="No baseline"):
            guard.check("current")

    def test_uses_latest_when_no_run_id(self, tmp_path: str) -> None:
        """Check without run_id should use the latest run."""
        store = Store(tmp_path)
        baseline = _make_run(100.0, run_id="baseline")
        current = _make_run(100.0, run_id="latest")
        store.save(baseline)
        store.save(current)
        store.set_baseline("baseline")

        guard = CIGuard(store)
        result = guard.check()
        assert result.current_id == "latest"

    def test_custom_threshold(self, tmp_path: str) -> None:
        """Higher threshold should tolerate larger regressions."""
        store = Store(tmp_path)
        baseline = _make_run(100.0, run_id="baseline")
        current = _make_run(85.0, run_id="current")  # 15% regression
        store.save(baseline)
        store.save(current)
        store.set_baseline("baseline")

        guard_strict = CIGuard(store, threshold=0.05)
        guard_lenient = CIGuard(store, threshold=0.20)
        assert guard_strict.check("current").passed is False
        assert guard_lenient.check("current").passed is True

    def test_result_to_dict(self, tmp_path: str) -> None:
        """GuardResult.to_dict should produce JSON-compatible output."""
        store = Store(tmp_path)
        baseline = _make_run(100.0, run_id="baseline")
        current = _make_run(100.0, run_id="current")
        store.save(baseline)
        store.save(current)
        store.set_baseline("baseline")

        guard = CIGuard(store)
        result = guard.check("current")
        d = result.to_dict()
        assert d["passed"] is True
        assert isinstance(d["threshold"], float)
        assert d["current_id"] == "current"

    def test_improvement_does_not_trigger(self, tmp_path: str) -> None:
        """Improvements (better performance) should not be regressions."""
        store = Store(tmp_path)
        baseline = _make_run(100.0, run_id="baseline")
        current = _make_run(120.0, run_id="current")  # 20% improvement
        store.save(baseline)
        store.save(current)
        store.set_baseline("baseline")

        guard = CIGuard(store)
        result = guard.check("current")
        assert result.passed is True


class TestGuardResult:
    """Tests for GuardResult dataclass."""

    def test_frozen(self) -> None:
        """GuardResult should be immutable."""
        result = GuardResult(
            passed=True,
            regressions=(),
            threshold=0.05,
            baseline_id="b1",
            current_id="c1",
        )
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]
