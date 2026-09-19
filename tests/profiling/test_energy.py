"""Tests for EnergyMonitor, EnergySample, and EnergySummary.

The GPU is a fixed-reading fake and RAPL is patched, so no hardware is needed.
"""

import dataclasses
import time
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

import pytest

from calibrax.profiling.energy import (
    EnergyMonitor,
    EnergySample,
    EnergySummary,
)
from calibrax.profiling.resources import GpuPower
from tests.factories import (
    assert_monitor_collects_samples_twice,
    assert_monitor_thread_lifecycle,
    FakeGpu,
    make_empty_energy_summary,
)


class _GpuQueryError(Exception):
    pass


def _gpu_drawing(watts: float) -> FakeGpu:
    return FakeGpu(power_reading=GpuPower(draw_w=watts, limit_w=450.0))


def _readings(*first: int, then: int) -> Callable[[], int]:
    """A RAPL counter reading ``first`` in order, then ``then`` on every later read."""
    values: Iterator[int] = iter(first)
    return lambda: next(values, then)


class TestEnergySample:
    """Tests for EnergySample frozen dataclass."""

    def test_construction(self) -> None:
        sample = EnergySample(
            timestamp=1.0,
            gpu_power_watts=250.0,
            cpu_energy_joules=10.0,
            gpu_energy_joules=5.0,
        )
        assert sample.gpu_power_watts == 250.0
        assert sample.cpu_energy_joules == 10.0

    def test_frozen_immutability(self) -> None:
        sample = EnergySample(
            timestamp=1.0,
            gpu_power_watts=250.0,
            cpu_energy_joules=None,
            gpu_energy_joules=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample.gpu_power_watts = 0.0  # type: ignore[misc]


class TestEnergySummary:
    """Tests for EnergySummary frozen dataclass."""

    def test_construction(self) -> None:
        summary = EnergySummary(
            total_gpu_energy_joules=100.0,
            total_cpu_energy_joules=50.0,
            total_combined_energy_joules=150.0,
            mean_gpu_power_watts=200.0,
            peak_gpu_power_watts=300.0,
            duration_sec=5.0,
            num_samples=50,
        )
        assert summary.total_combined_energy_joules == 150.0

    def test_frozen_immutability(self) -> None:
        summary = make_empty_energy_summary()
        with pytest.raises(dataclasses.FrozenInstanceError):
            summary.num_samples = 10  # type: ignore[misc]


class TestEnergyMonitor:
    """Tests for EnergyMonitor context manager."""

    def test_context_manager_starts_and_stops(self) -> None:
        mon = EnergyMonitor(sample_interval_sec=0.05)
        assert_monitor_thread_lifecycle(mon)

    def test_thread_is_daemon(self) -> None:
        mon = EnergyMonitor(sample_interval_sec=0.05)
        with mon:
            assert mon._sampling_thread._thread is not None
            assert mon._sampling_thread._thread.daemon is True

    def test_multiple_enter_exit_cycles(self) -> None:
        mon = EnergyMonitor(sample_interval_sec=0.05)
        assert_monitor_collects_samples_twice(mon)

    @patch("calibrax.profiling.energy._read_rapl_energy_uj", new=MagicMock(return_value=None))
    def test_without_gpu_or_rapl_every_energy_is_none(self) -> None:
        mon = EnergyMonitor(sample_interval_sec=0.05)
        with mon:
            time.sleep(0.2)

        summary = mon.summary
        assert summary.total_gpu_energy_joules is None
        assert summary.total_cpu_energy_joules is None
        assert summary.total_combined_energy_joules is None

    @patch("calibrax.profiling.energy._read_rapl_energy_uj", new=MagicMock(return_value=None))
    def test_gpu_energy_is_power_times_time(self) -> None:
        mon = EnergyMonitor(sample_interval_sec=0.05, gpu=_gpu_drawing(100.0))
        with mon:
            time.sleep(0.3)

        summary = mon.summary
        assert summary.mean_gpu_power_watts == pytest.approx(100.0)
        assert summary.peak_gpu_power_watts == pytest.approx(100.0)
        # Constant draw: the rectangular integral over the readings is exactly draw x duration.
        assert summary.total_gpu_energy_joules == pytest.approx(100.0 * summary.duration_sec)
        assert summary.total_combined_energy_joules == summary.total_gpu_energy_joules

    def test_a_gpu_without_a_power_reading_leaves_gpu_energy_none(self) -> None:
        mon = EnergyMonitor(sample_interval_sec=0.05, gpu=FakeGpu())
        with mon:
            time.sleep(0.15)

        assert mon.summary.total_gpu_energy_joules is None
        assert mon.summary.mean_gpu_power_watts is None

    @patch("calibrax.profiling.energy._read_rapl_range_uj", new=MagicMock(return_value=None))
    @patch("calibrax.profiling.energy._read_rapl_energy_uj")
    def test_cpu_energy_is_the_rapl_increase(self, mock_rapl: MagicMock) -> None:
        mock_rapl.side_effect = _readings(1_000_000, 1_500_000, then=3_000_000)

        mon = EnergyMonitor(sample_interval_sec=0.05)
        with mon:
            time.sleep(0.15)

        assert mon.summary.total_cpu_energy_joules == pytest.approx(2.0)

    @patch("calibrax.profiling.energy._read_rapl_range_uj", new=MagicMock(return_value=10_000_000))
    @patch("calibrax.profiling.energy._read_rapl_energy_uj")
    def test_rapl_wraparound_adds_the_counter_range(self, mock_rapl: MagicMock) -> None:
        # 9 J at start, then the counter wraps at 10 J and reads 1 J: 2 J were used.
        mock_rapl.side_effect = _readings(9_000_000, then=1_000_000)

        mon = EnergyMonitor(sample_interval_sec=0.05)
        with mon:
            time.sleep(0.15)

        assert mon.summary.total_cpu_energy_joules == pytest.approx(2.0)

    @patch("calibrax.profiling.energy._read_rapl_range_uj", new=MagicMock(return_value=None))
    @patch("calibrax.profiling.energy._read_rapl_energy_uj")
    def test_rapl_wraparound_without_a_range_leaves_cpu_energy_unknown(
        self, mock_rapl: MagicMock
    ) -> None:
        mock_rapl.side_effect = _readings(9_000_000, then=1_000_000)

        mon = EnergyMonitor(sample_interval_sec=0.05)
        with mon:
            time.sleep(0.15)

        assert mon.summary.total_cpu_energy_joules is None

    @patch("calibrax.profiling.energy._read_rapl_range_uj", new=MagicMock(return_value=None))
    @patch("calibrax.profiling.energy._read_rapl_energy_uj")
    def test_gpu_and_cpu_energy_combine(self, mock_rapl: MagicMock) -> None:
        mock_rapl.side_effect = _readings(0, then=500_000)

        mon = EnergyMonitor(sample_interval_sec=0.05, gpu=_gpu_drawing(200.0))
        with mon:
            time.sleep(0.2)

        summary = mon.summary
        assert summary.total_gpu_energy_joules is not None
        assert summary.total_cpu_energy_joules == pytest.approx(0.5)
        assert summary.total_combined_energy_joules == pytest.approx(
            summary.total_gpu_energy_joules + 0.5
        )

    def test_a_gpu_error_is_raised_on_exit(self) -> None:
        class _FailingGpu(FakeGpu):
            def power(self) -> GpuPower | None:
                raise _GpuQueryError

        with (
            pytest.raises(_GpuQueryError),
            EnergyMonitor(sample_interval_sec=0.05, gpu=_FailingGpu()),
        ):
            time.sleep(0.1)

    def test_empty_summary(self) -> None:
        mon = EnergyMonitor(sample_interval_sec=0.05)
        summary = mon.summary
        assert summary.num_samples == 0
        assert summary.duration_sec == 0.0
