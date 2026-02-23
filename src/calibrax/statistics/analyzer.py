"""Statistical analysis for benchmark measurements.

Provides summary statistics with bootstrap confidence intervals,
outlier detection via modified Z-scores, and stability assessment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


# Coefficient of variation threshold for measurement stability.
# CV < this value means "stable" measurement (low noise).
STABILITY_CV_THRESHOLD: float = 0.10

# Bootstrap confidence interval significance level.
# alpha=0.05 gives a 95% CI: [2.5th percentile, 97.5th percentile].
BOOTSTRAP_CI_ALPHA: float = 0.05

# Modified Z-score threshold for outlier detection (Iglewicz & Hoaglin).
OUTLIER_Z_THRESHOLD: float = 3.5

# MAD consistency constant: 1 / inverse_normal_cdf(3/4) ~ 0.6745.
# Scales MAD to be a consistent estimator of sigma for normal distributions.
_MAD_CONSISTENCY_CONSTANT: float = 0.6745


@dataclass(frozen=True, slots=True, kw_only=True)
class StatisticalResult:
    """Summary statistics with confidence intervals.

    Attributes:
        mean: Arithmetic mean.
        median: Median value.
        std: Sample standard deviation (ddof=1).
        min: Minimum value.
        max: Maximum value.
        cv: Coefficient of variation (std / mean).
        ci_lower: 95% bootstrap CI lower bound.
        ci_upper: 95% bootstrap CI upper bound.
        n: Number of samples.
        is_stable: True when CV < STABILITY_CV_THRESHOLD.
    """

    mean: float
    median: float
    std: float
    min: float
    max: float
    cv: float
    ci_lower: float
    ci_upper: float
    n: int
    is_stable: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "cv": self.cv,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n": self.n,
            "is_stable": self.is_stable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StatisticalResult:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with statistical result fields.

        Returns:
            Reconstructed StatisticalResult instance.
        """
        return cls(
            mean=data["mean"],
            median=data["median"],
            std=data["std"],
            min=data["min"],
            max=data["max"],
            cv=data["cv"],
            ci_lower=data["ci_lower"],
            ci_upper=data["ci_upper"],
            n=data["n"],
            is_stable=data["is_stable"],
        )


class StatisticalAnalyzer:
    """Statistical analysis for benchmark measurements.

    Provides summary statistics with bootstrap confidence intervals,
    modified Z-score outlier detection, and stability assessment.

    Args:
        bootstrap_resamples: Number of bootstrap resamples for CI computation.
        seed: Random seed for reproducible bootstrap sampling.
    """

    def __init__(self, bootstrap_resamples: int = 1000, seed: int = 42) -> None:
        """Initialize with bootstrap parameters.

        Args:
            bootstrap_resamples: Number of bootstrap resamples for CI computation.
            seed: Random seed for reproducible bootstrap sampling.
        """
        self._bootstrap_resamples = bootstrap_resamples
        self._rng = np.random.default_rng(seed)

    def summarize(self, samples: Sequence[float]) -> StatisticalResult:
        """Compute summary statistics with bootstrap CI.

        Args:
            samples: Sequence of measurement values (at least 1).

        Returns:
            StatisticalResult with all computed statistics.
        """
        arr = np.array(samples, dtype=np.float64)
        n = len(arr)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        cv = std / mean if mean != 0 else 0.0

        ci_lower, ci_upper = self.bootstrap_ci(list(samples))

        return StatisticalResult(
            mean=mean,
            median=median,
            std=std,
            min=min_val,
            max=max_val,
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n=n,
            is_stable=cv < STABILITY_CV_THRESHOLD,
        )

    def bootstrap_ci(
        self, samples: Sequence[float], confidence: float = 0.95
    ) -> tuple[float, float]:
        """Percentile bootstrap confidence interval.

        Args:
            samples: Sequence of measurement values.
            confidence: Confidence level (default 0.95 for 95% CI).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        arr = np.array(samples, dtype=np.float64)
        n = len(arr)

        if n <= 1:
            val = float(arr[0])
            return (val, val)

        bootstrap_means = np.array(
            [
                float(np.mean(self._rng.choice(arr, size=n, replace=True)))
                for _ in range(self._bootstrap_resamples)
            ]
        )
        alpha = 1.0 - confidence
        lo = (alpha / 2) * 100
        hi = (1 - alpha / 2) * 100
        return (
            float(np.percentile(bootstrap_means, lo)),
            float(np.percentile(bootstrap_means, hi)),
        )

    def detect_outliers(
        self, samples: Sequence[float], threshold: float = OUTLIER_Z_THRESHOLD
    ) -> list[int]:
        """Modified Z-score outlier detection.

        Uses median absolute deviation (MAD) instead of standard deviation
        for robustness against the outliers themselves.

        Args:
            samples: Sequence of values to check.
            threshold: Modified Z-score threshold (default 3.5).

        Returns:
            List of indices where outliers are detected.
        """
        arr = np.array(samples, dtype=np.float64)
        if len(arr) < 3:
            return []
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad == 0:
            return []
        modified_z = _MAD_CONSISTENCY_CONSTANT * (arr - median) / mad
        return [int(i) for i in np.where(np.abs(modified_z) > threshold)[0]]
