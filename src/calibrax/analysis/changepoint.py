"""Change point detection for benchmark time series.

Uses the ``ruptures`` library to detect significant changes in metric
trends, enabling automated identification of performance regressions
or improvements over time. This module is the ruptures integration: it needs the
``changepoint`` extra, importing it without ruptures raises ``ImportError`` naming the extra,
and it is not re-exported from ``calibrax.analysis``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Self, SupportsInt

import numpy as np
from substrax.records import read_record
from substrax.typing import JsonValue


try:
    import ruptures
except ImportError as error:
    msg = 'calibrax.analysis.changepoint needs ruptures: uv pip install "calibrax[changepoint]"'
    raise ImportError(msg) from error

from calibrax.core.models import TrendSeries
from calibrax.core.record_values import aware


class _ChangePointAlgorithm(Protocol):
    """The part of a ruptures detector the analysis uses.

    ``pen`` is keyword-only: ``Binseg`` and ``Window`` take ``n_bkps`` first, so a
    positional penalty would be read as a breakpoint count.
    """

    def fit(self, signal: np.ndarray) -> Self: ...

    def predict(self, *, pen: float) -> list[SupportsInt]: ...


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

    def __post_init__(self) -> None:
        """Make the timestamp aware; a naive one is local time."""
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", aware(self.timestamp))

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        d: dict[str, JsonValue] = {
            "index": int(self.index),
            "magnitude": float(self.magnitude),
        }
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp.isoformat()
        if self.run_id is not None:
            d["run_id"] = self.run_id
        return d

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> ChangePoint:
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
        ValueError: If the trend has fewer points than min_size.
    """
    if len(trend.points) < min_size:
        msg = f"Need at least {min_size} points, got {len(trend.points)}"
        raise ValueError(msg)

    values = np.array([p.value for p in trend.points])

    if penalty is None:
        penalty = _auto_penalty(values)

    algo = _get_algorithm(method, min_size)
    algo.fit(values.reshape(-1, 1))

    # predict returns breakpoints including the final index (len)
    breakpoints = [int(bp) for bp in algo.predict(pen=penalty)]
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


def _get_algorithm(method: str, min_size: int) -> _ChangePointAlgorithm:
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
        return ruptures.Pelt(model="l2", min_size=min_size)
    if method == "binseg":
        return ruptures.Binseg(model="l2", min_size=min_size)
    if method == "window":
        return ruptures.Window(model="l2", min_size=min_size, width=min_size * 2)

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
