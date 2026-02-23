"""Change point detection for benchmark time series.

Uses the ``ruptures`` library to detect significant changes in metric
trends, enabling automated identification of performance regressions
or improvements over time. Requires the optional ``ruptures`` dependency
(``uv pip install "calibrax[changepoint]"``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from calibrax.core.models import TrendSeries


try:
    import ruptures

    RUPTURES_AVAILABLE = True
except ImportError:
    ruptures = None  # type: ignore[assignment]
    RUPTURES_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChangePoint:
    """A detected change point in a benchmark trend series.

    Attributes:
        index: Index in the trend series where the change was detected.
        timestamp: Timestamp of the change point, if available.
        run_id: Run ID at the change point, if available.
        magnitude: Absolute difference in mean values before/after the change.
    """

    index: int
    timestamp: datetime | None = None
    run_id: str | None = None
    magnitude: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d: dict[str, Any] = {
            "index": int(self.index),
            "magnitude": float(self.magnitude),
        }
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp.isoformat()
        if self.run_id is not None:
            d["run_id"] = self.run_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangePoint:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with change point fields.

        Returns:
            Reconstructed ChangePoint instance.
        """
        ts = data.get("timestamp")
        return cls(
            index=data["index"],
            timestamp=datetime.fromisoformat(ts) if ts else None,
            run_id=data.get("run_id"),
            magnitude=data.get("magnitude", 0.0),
        )


def detect_change_points(
    trend: TrendSeries,
    *,
    method: str = "pelt",
    min_size: int = 3,
    penalty: float | None = None,
) -> list[ChangePoint]:
    """Detect change points in a benchmark trend series.

    Uses the ``ruptures`` library for change point detection with
    configurable algorithms.

    Args:
        trend: TrendSeries containing the metric values over time.
        method: Detection method ("pelt", "binseg", or "window").
        min_size: Minimum segment size between change points.
        penalty: Penalty value for PELT/BinSeg. Auto-calibrated if None.

    Returns:
        List of detected ChangePoint instances, ordered by index.

    Raises:
        ImportError: If ruptures is not installed.
        ValueError: If the trend has fewer points than min_size.
    """
    if not RUPTURES_AVAILABLE:
        msg = (
            "ruptures is required for change point detection: "
            'uv pip install "calibrax[changepoint]"'
        )
        raise ImportError(msg)

    if len(trend.points) < min_size:
        msg = f"Need at least {min_size} points, got {len(trend.points)}"
        raise ValueError(msg)

    values = np.array([p.value for p in trend.points])

    if penalty is None:
        penalty = _auto_penalty(values)

    algo = _get_algorithm(method, min_size)
    algo.fit(values.reshape(-1, 1))

    # predict returns breakpoints including the final index (len)
    breakpoints = algo.predict(pen=penalty)
    # Remove the final index (always equals len(values))
    change_indices = [bp for bp in breakpoints if bp < len(values)]

    result: list[ChangePoint] = []
    for idx in change_indices:
        tp = trend.points[idx]
        before = values[max(0, idx - min_size) : idx]
        after = values[idx : min(len(values), idx + min_size)]
        if len(before) > 0 and len(after) > 0:
            magnitude = abs(float(np.mean(after) - np.mean(before)))
        else:
            magnitude = 0.0

        result.append(
            ChangePoint(
                index=idx,
                timestamp=tp.timestamp,
                run_id=tp.run_id,
                magnitude=magnitude,
            )
        )

    return result


def _get_algorithm(method: str, min_size: int) -> Any:
    """Get a ruptures algorithm instance.

    Args:
        method: Algorithm name ("pelt", "binseg", or "window").
        min_size: Minimum segment size.

    Returns:
        Configured ruptures algorithm instance.

    Raises:
        ValueError: If the method is not recognized.
    """
    if method == "pelt":
        return ruptures.Pelt(model="l2", min_size=min_size)  # type: ignore[union-attr]
    if method == "binseg":
        return ruptures.Binseg(model="l2", min_size=min_size)  # type: ignore[union-attr]
    if method == "window":
        return ruptures.Window(  # type: ignore[union-attr]
            model="l2", min_size=min_size, width=min_size * 2
        )

    msg = f"Unknown method: {method!r}. Use 'pelt', 'binseg', or 'window'."
    raise ValueError(msg)


def _auto_penalty(values: np.ndarray) -> float:
    """Auto-calibrate penalty based on signal variance.

    Args:
        values: Array of metric values.

    Returns:
        Penalty value scaled to the signal's variance.
    """
    variance = float(np.var(values))
    # Penalty proportional to variance avoids over/under-segmentation
    return max(variance * 2.0, 1.0)
