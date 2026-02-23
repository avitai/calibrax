"""Git bisection engine for performance regression root-cause analysis.

Binary-searches through git history to find the commit that introduced
a performance regression, using user-provided benchmark and regression
detection functions.
"""

from __future__ import annotations

import logging
import subprocess  # nosec B404
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrax.core.models import Run


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class BisectionResult:
    """Result of a git bisection for performance regression.

    Attributes:
        culprit_commit: Hash of the commit that introduced the regression,
            or None if no regression was found.
        total_steps: Number of bisection steps performed.
        tested_commits: All commit hashes that were tested.
        is_regression_found: Whether a regression-causing commit was identified.
    """

    culprit_commit: str | None
    total_steps: int
    tested_commits: tuple[str, ...]
    is_regression_found: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "culprit_commit": self.culprit_commit,
            "total_steps": int(self.total_steps),
            "tested_commits": list(self.tested_commits),
            "is_regression_found": self.is_regression_found,
        }


class BisectionEngine:
    """Binary search through git history to find regression-causing commit.

    Args:
        repo_path: Path to the git repository.
        benchmark_fn: Function that checks out a commit and runs benchmarks.
            Receives a commit hash string, returns a Run.
        regression_fn: Function that checks if a Run exhibits regression.
            Returns True if the run shows regression.
    """

    def __init__(
        self,
        repo_path: Path | str,
        benchmark_fn: Callable[[str], Run],
        regression_fn: Callable[[Run], bool],
    ) -> None:
        """Initialize the bisection engine."""
        self._repo_path = Path(repo_path)
        self._benchmark_fn = benchmark_fn
        self._regression_fn = regression_fn

    def bisect(self, good_commit: str, bad_commit: str) -> BisectionResult:
        """Binary search between a known good and bad commit.

        Restores the original HEAD after bisection completes.

        Args:
            good_commit: Commit hash known to be regression-free.
            bad_commit: Commit hash known to have the regression.

        Returns:
            BisectionResult with the culprit commit (if found).
        """
        original_head = self._get_original_ref()
        tested: list[str] = []

        try:
            commits = self._get_commit_range(good_commit, bad_commit)
            if len(commits) <= 2:
                return BisectionResult(
                    culprit_commit=bad_commit if len(commits) == 2 else None,
                    total_steps=0,
                    tested_commits=(),
                    is_regression_found=len(commits) == 2,
                )

            left = 0
            right = len(commits) - 1

            while right - left > 1:
                mid = (left + right) // 2
                commit = commits[mid]
                tested.append(commit)

                logger.info(
                    "Bisecting: testing commit %s (step %d, range %d-%d)",
                    commit[:8],
                    len(tested),
                    left,
                    right,
                )

                self._checkout(commit)
                run = self._benchmark_fn(commit)

                if self._regression_fn(run):
                    right = mid
                else:
                    left = mid

            culprit = commits[right]
            return BisectionResult(
                culprit_commit=culprit,
                total_steps=len(tested),
                tested_commits=tuple(tested),
                is_regression_found=True,
            )

        finally:
            self._restore_head(original_head)

    def _get_current_head(self) -> str:
        """Get the current HEAD commit hash."""
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            cwd=self._repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def _get_original_ref(self) -> str:
        """Get a ref that can restore the user's original git state.

        Prefers a symbolic branch name when HEAD is attached; falls back
        to the current commit hash when detached.
        """
        try:
            result = subprocess.run(  # nosec B603 B607
                ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
                cwd=self._repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            ref = result.stdout.strip()
            if ref:
                return ref
        except subprocess.CalledProcessError:
            # Detached HEAD: restore by commit hash instead.
            pass

        return self._get_current_head()

    def _get_commit_range(self, good: str, bad: str) -> list[str]:
        """Get list of commits from good to bad (oldest first).

        Args:
            good: Good (older) commit hash.
            bad: Bad (newer) commit hash.

        Returns:
            List of commit hashes from good to bad inclusive.
        """
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-list", "--reverse", f"{good}^..{bad}"],
            cwd=self._repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]

    def _checkout(self, commit: str) -> None:
        """Checkout a specific commit."""
        subprocess.run(  # nosec B603 B607
            ["git", "checkout", commit],
            cwd=self._repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

    def _restore_head(self, original_head: str) -> None:
        """Restore the original HEAD after bisection."""
        subprocess.run(  # nosec B603 B607
            ["git", "checkout", original_head],
            cwd=self._repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
