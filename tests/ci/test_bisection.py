"""Tests for calibrax.ci.bisection module."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from calibrax.ci.bisection import BisectionEngine, BisectionResult
from calibrax.core.models import Metric, Point, Run


def _make_run(commit: str, throughput: float = 100.0) -> Run:
    """Helper to create a Run for a specific commit."""
    return Run(
        points=(
            Point(
                name="bench1",
                scenario="default",
                metrics={"throughput": Metric(value=throughput)},
            ),
        ),
        commit=commit,
    )


def _make_mock_subprocess(
    commits: list[str],
    *,
    symbolic_ref: str | None = None,
) -> MagicMock:
    """Create a mock subprocess.run that simulates git operations."""
    mock = MagicMock()

    def side_effect(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        if cmd[1] == "symbolic-ref":
            if symbolic_ref is None:
                raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
            result.stdout = f"{symbolic_ref}\n"
        elif cmd[1] == "rev-parse":
            result.stdout = "original_head_hash\n"
        elif cmd[1] == "rev-list":
            result.stdout = "\n".join(commits) + "\n"
        elif cmd[1] == "checkout":
            result.stdout = ""
        return result

    mock.side_effect = side_effect
    return mock


class TestBisectionEngine:
    """Tests for BisectionEngine."""

    def test_finds_culprit_in_middle(self) -> None:
        """Should identify the regression-causing commit."""
        commits = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]
        # Regression starts at c4
        regression_at = {"c4", "c5", "c6", "c7"}

        def benchmark_fn(commit: str) -> Run:
            return _make_run(commit, 50.0 if commit in regression_at else 100.0)

        def regression_fn(run: Run) -> bool:
            return run.points[0].metrics["throughput"].value < 80.0

        engine = BisectionEngine("/fake/repo", benchmark_fn, regression_fn)

        with patch("calibrax.ci.bisection.subprocess.run", _make_mock_subprocess(commits)):
            result = engine.bisect("c0", "c7")

        assert result.is_regression_found is True
        assert result.culprit_commit == "c4"
        assert result.total_steps > 0

    def test_adjacent_commits(self) -> None:
        """Two adjacent commits should identify the bad one."""
        commits = ["good", "bad"]

        def benchmark_fn(commit: str) -> Run:
            return _make_run(commit)

        def regression_fn(run: Run) -> bool:
            return False

        engine = BisectionEngine("/fake/repo", benchmark_fn, regression_fn)

        with patch("calibrax.ci.bisection.subprocess.run", _make_mock_subprocess(commits)):
            result = engine.bisect("good", "bad")

        assert result.is_regression_found is True
        assert result.culprit_commit == "bad"
        assert result.total_steps == 0

    def test_single_commit(self) -> None:
        """Single commit should not find regression."""
        commits = ["only"]

        def benchmark_fn(commit: str) -> Run:
            return _make_run(commit)

        def regression_fn(run: Run) -> bool:
            return False

        engine = BisectionEngine("/fake/repo", benchmark_fn, regression_fn)

        with patch("calibrax.ci.bisection.subprocess.run", _make_mock_subprocess(commits)):
            result = engine.bisect("only", "only")

        assert result.is_regression_found is False
        assert result.culprit_commit is None

    def test_restores_original_head(self) -> None:
        """Should restore original HEAD after bisection."""
        commits = ["c0", "c1", "c2", "c3"]
        checkout_calls: list[str] = []

        def benchmark_fn(commit: str) -> Run:
            return _make_run(commit)

        def regression_fn(run: Run) -> bool:
            return run.commit in {"c2", "c3"}

        mock_sub = _make_mock_subprocess(commits)
        original_side_effect = mock_sub.side_effect

        def tracking_effect(cmd, **kwargs):
            if cmd[1] == "checkout":
                checkout_calls.append(cmd[2])
            return original_side_effect(cmd, **kwargs)

        mock_sub.side_effect = tracking_effect

        engine = BisectionEngine("/fake/repo", benchmark_fn, regression_fn)

        with patch("calibrax.ci.bisection.subprocess.run", mock_sub):
            engine.bisect("c0", "c3")

        assert checkout_calls[-1] == "original_head_hash"

    def test_tested_commits_recorded(self) -> None:
        """All tested commits should be recorded in the result."""
        commits = [f"c{i}" for i in range(8)]

        def benchmark_fn(commit: str) -> Run:
            return _make_run(commit)

        def regression_fn(run: Run) -> bool:
            return run.commit in {"c6", "c7"}

        engine = BisectionEngine("/fake/repo", benchmark_fn, regression_fn)

        with patch("calibrax.ci.bisection.subprocess.run", _make_mock_subprocess(commits)):
            result = engine.bisect("c0", "c7")

        assert len(result.tested_commits) == result.total_steps
        assert all(isinstance(c, str) for c in result.tested_commits)

    def test_restores_original_branch_when_available(self) -> None:
        """Should restore symbolic branch ref instead of detached commit hash."""
        commits = ["c0", "c1", "c2", "c3"]
        checkout_calls: list[str] = []

        def benchmark_fn(commit: str) -> Run:
            return _make_run(commit)

        def regression_fn(run: Run) -> bool:
            return run.commit in {"c2", "c3"}

        mock_sub = _make_mock_subprocess(commits, symbolic_ref="main")
        original_side_effect = mock_sub.side_effect

        def tracking_effect(cmd, **kwargs):
            if cmd[1] == "checkout":
                checkout_calls.append(cmd[2])
            return original_side_effect(cmd, **kwargs)

        mock_sub.side_effect = tracking_effect
        engine = BisectionEngine("/fake/repo", benchmark_fn, regression_fn)

        with patch("calibrax.ci.bisection.subprocess.run", mock_sub):
            engine.bisect("c0", "c3")

        assert checkout_calls[-1] == "main"


class TestBisectionResult:
    """Tests for BisectionResult dataclass."""

    def test_to_dict(self) -> None:
        """to_dict should produce JSON-compatible output."""
        result = BisectionResult(
            culprit_commit="abc123",
            total_steps=5,
            tested_commits=("c1", "c2", "c3"),
            is_regression_found=True,
        )
        d = result.to_dict()
        assert d["culprit_commit"] == "abc123"
        assert d["total_steps"] == 5
        assert isinstance(d["tested_commits"], list)

    def test_frozen(self) -> None:
        """BisectionResult should be immutable."""
        result = BisectionResult(
            culprit_commit=None,
            total_steps=0,
            tested_commits=(),
            is_regression_found=False,
        )
        with pytest.raises(AttributeError):
            result.total_steps = 10  # type: ignore[misc]

    def test_no_regression(self) -> None:
        """Result with no regression found."""
        result = BisectionResult(
            culprit_commit=None,
            total_steps=0,
            tested_commits=(),
            is_regression_found=False,
        )
        d = result.to_dict()
        assert d["culprit_commit"] is None
        assert d["is_regression_found"] is False
