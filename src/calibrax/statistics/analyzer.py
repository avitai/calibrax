"""Statistical analysis for benchmark measurements.

Provides summary statistics with bootstrap confidence intervals,
outlier detection via modified Z-scores, and stability assessment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from substrax.records import read_record
from substrax.rng import key_from
from substrax.typing import JsonValue

from calibrax.statistics.bootstrap import bootstrap_interval, DEFAULT_RESAMPLES


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


# A median absolute deviation from fewer than three samples says nothing about outliers.
_MIN_SAMPLES_FOR_MAD = 3


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

    def to_dict(self) -> dict[str, JsonValue]:
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
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> StatisticalResult:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


class StatisticalAnalyzer:
    """Statistical analysis for benchmark measurements.

    Provides summary statistics with bootstrap confidence intervals,
    modified Z-score outlier detection, and stability assessment.
    """

    def __init__(
        self, *, key: jax.Array | nnx.Rngs, bootstrap_resamples: int = DEFAULT_RESAMPLES
    ) -> None:
        """Initialize with the bootstrap's key and resample count.

        Args:
            key: The key bootstrap resampling starts from, or an ``nnx.Rngs`` whose ``sample``
                or ``default`` stream supplies it; each call splits it, so successive calls
                draw fresh resamples and the same key reproduces the sequence.
            bootstrap_resamples: Number of bootstrap resamples for CI computation.
        """
        self._bootstrap_resamples = bootstrap_resamples
        self._key = key_from(key, streams=("sample", "default"), context="StatisticalAnalyzer")

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
        self._key, key = jax.random.split(self._key)
        interval = bootstrap_interval(
            jnp.mean,
            jnp.asarray(samples),
            key=key,
            num_resamples=self._bootstrap_resamples,
            confidence=confidence,
        )
        return float(interval.lower), float(interval.upper)

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
        if len(arr) < _MIN_SAMPLES_FOR_MAD:
            return []
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad == 0:
            return []
        modified_z = _MAD_CONSISTENCY_CONSTANT * (arr - median) / mad
        return [int(i) for i in np.where(np.abs(modified_z) > threshold)[0]]
