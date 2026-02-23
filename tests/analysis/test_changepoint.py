"""Tests for calibrax.analysis.changepoint module."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from unittest.mock import patch

import pytest

from calibrax.analysis.changepoint import (
    ChangePoint,
    detect_change_points,
    RUPTURES_AVAILABLE,
)
from calibrax.core.models import TrendPoint, TrendSeries


_skip_no_ruptures = pytest.mark.skipif(
    not RUPTURES_AVAILABLE,
    reason="ruptures not installed",
)


def _make_trend(values: list[float], metric: str = "throughput") -> TrendSeries:
    """Create a TrendSeries from a list of values for testing."""
    points = tuple(
        TrendPoint(
            run_id=f"r{i}",
            timestamp=datetime(2024, 1, i + 1),
            value=v,
        )
        for i, v in enumerate(values)
    )
    return TrendSeries(metric=metric, point_name="test", points=points)


class TestChangePoint:
    """Tests for ChangePoint frozen dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Should create ChangePoint with default optional fields."""
        cp = ChangePoint(index=3)
        assert cp.index == 3
        assert cp.timestamp is None
        assert cp.run_id is None
        assert cp.magnitude == 0.0

    def test_creation_with_all_fields(self) -> None:
        """Should create ChangePoint with all fields populated."""
        ts = datetime(2024, 6, 15)
        cp = ChangePoint(index=5, timestamp=ts, run_id="run99", magnitude=3.14)
        assert cp.index == 5
        assert cp.timestamp == ts
        assert cp.run_id == "run99"
        assert cp.magnitude == 3.14

    def test_frozen_immutability(self) -> None:
        """Should raise FrozenInstanceError on attribute mutation."""
        cp = ChangePoint(index=1, magnitude=2.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cp.index = 10  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            cp.magnitude = 0.0  # type: ignore[misc]

    def test_to_dict_minimal(self) -> None:
        """to_dict should omit None optional fields."""
        cp = ChangePoint(index=2, magnitude=1.5)
        d = cp.to_dict()
        assert d == {"index": 2, "magnitude": 1.5}
        assert "timestamp" not in d
        assert "run_id" not in d

    def test_to_dict_full(self) -> None:
        """to_dict should include all populated fields."""
        ts = datetime(2024, 3, 10, 12, 0, 0)
        cp = ChangePoint(index=4, timestamp=ts, run_id="abc", magnitude=5.0)
        d = cp.to_dict()
        assert d["index"] == 4
        assert d["timestamp"] == ts.isoformat()
        assert d["run_id"] == "abc"
        assert d["magnitude"] == 5.0

    def test_from_dict_minimal(self) -> None:
        """from_dict should handle minimal data with defaults."""
        cp = ChangePoint.from_dict({"index": 7})
        assert cp.index == 7
        assert cp.timestamp is None
        assert cp.run_id is None
        assert cp.magnitude == 0.0

    def test_from_dict_full(self) -> None:
        """from_dict should restore all fields."""
        ts = datetime(2024, 8, 20, 14, 30)
        cp = ChangePoint.from_dict(
            {
                "index": 3,
                "timestamp": ts.isoformat(),
                "run_id": "xyz",
                "magnitude": 2.7,
            }
        )
        assert cp.index == 3
        assert cp.timestamp == ts
        assert cp.run_id == "xyz"
        assert cp.magnitude == 2.7

    def test_to_dict_from_dict_round_trip(self) -> None:
        """to_dict/from_dict should produce an equivalent object."""
        ts = datetime(2024, 5, 1, 8, 0, 0)
        original = ChangePoint(index=6, timestamp=ts, run_id="r42", magnitude=10.0)
        reconstructed = ChangePoint.from_dict(original.to_dict())
        assert reconstructed.index == original.index
        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.run_id == original.run_id
        assert reconstructed.magnitude == original.magnitude

    def test_to_dict_from_dict_round_trip_minimal(self) -> None:
        """Round-trip should work for ChangePoint with only index."""
        original = ChangePoint(index=0, magnitude=0.5)
        reconstructed = ChangePoint.from_dict(original.to_dict())
        assert reconstructed.index == original.index
        assert reconstructed.magnitude == original.magnitude
        assert reconstructed.timestamp is None


class TestDetectChangePoints:
    """Tests for detect_change_points function."""

    def test_raises_import_error_when_unavailable(self) -> None:
        """Should raise ImportError when ruptures is not installed."""
        trend = _make_trend([1.0, 2.0, 3.0, 4.0])
        with patch("calibrax.analysis.changepoint.RUPTURES_AVAILABLE", False):
            with pytest.raises(ImportError, match="ruptures is required"):
                detect_change_points(trend)

    @_skip_no_ruptures
    def test_raises_value_error_too_few_points(self) -> None:
        """Should raise ValueError when trend has fewer points than min_size."""
        trend = _make_trend([1.0, 2.0])
        with pytest.raises(ValueError, match="Need at least 3 points, got 2"):
            detect_change_points(trend)

    @_skip_no_ruptures
    def test_raises_value_error_custom_min_size(self) -> None:
        """Should raise ValueError with custom min_size."""
        trend = _make_trend([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="Need at least 5 points, got 4"):
            detect_change_points(trend, min_size=5)

    @_skip_no_ruptures
    def test_detects_obvious_step_change(self) -> None:
        """Should detect a clear step change in the data."""
        values = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]
        trend = _make_trend(values)
        change_points = detect_change_points(trend)
        assert len(change_points) > 0
        # The change point should be near index 4 (the transition)
        indices = [cp.index for cp in change_points]
        assert any(3 <= idx <= 5 for idx in indices)

    @_skip_no_ruptures
    def test_change_point_has_positive_magnitude(self) -> None:
        """Detected change points should have positive magnitude for step change."""
        values = [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0]
        trend = _make_trend(values)
        change_points = detect_change_points(trend)
        assert len(change_points) > 0
        for cp in change_points:
            assert cp.magnitude > 0.0

    @_skip_no_ruptures
    def test_change_point_has_metadata(self) -> None:
        """Detected change points should carry timestamp and run_id from trend."""
        values = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]
        trend = _make_trend(values)
        change_points = detect_change_points(trend)
        for cp in change_points:
            assert cp.timestamp is not None
            assert cp.run_id is not None
            assert cp.run_id.startswith("r")

    @_skip_no_ruptures
    def test_constant_signal_no_change_points(self) -> None:
        """A constant signal should produce no change points."""
        values = [5.0] * 10
        trend = _make_trend(values)
        change_points = detect_change_points(trend)
        assert len(change_points) == 0

    @_skip_no_ruptures
    def test_method_binseg(self) -> None:
        """Should work with binseg method."""
        values = [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0]
        trend = _make_trend(values)
        change_points = detect_change_points(trend, method="binseg")
        assert len(change_points) > 0

    @_skip_no_ruptures
    def test_invalid_method_raises(self) -> None:
        """Should raise ValueError for unknown detection method."""
        values = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]
        trend = _make_trend(values)
        with pytest.raises(ValueError, match="Unknown method"):
            detect_change_points(trend, method="invalid_method")

    @_skip_no_ruptures
    def test_returns_list_of_change_points(self) -> None:
        """Return type should be a list of ChangePoint instances."""
        values = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]
        trend = _make_trend(values)
        result = detect_change_points(trend)
        assert isinstance(result, list)
        for cp in result:
            assert isinstance(cp, ChangePoint)
