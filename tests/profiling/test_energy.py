"""Tests for EnergyMonitor, EnergySample, and EnergySummary.

All NVML/RAPL access is fully mocked — no hardware dependency.
"""

import dataclasses
import sys
import time
import types
from unittest.mock import MagicMock, patch

import pytest

from calibrax.profiling.energy import (
    _get_nvml_power_mw,
    EnergyMonitor,
    EnergySample,
    EnergySummary,
)
from tests.factories import (
    assert_monitor_collects_samples_twice,
    assert_monitor_thread_lifecycle,
    make_empty_energy_summary,
)


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

    @patch("calibrax.profiling.energy._read_rapl_energy_uj", return_value=None)
    @patch("calibrax.profiling.energy._get_nvml_power_mw", return_value=None)
    def test_graceful_degradation_no_nvml_no_rapl(
        self,
        _mock_nvml: MagicMock,
        _mock_rapl: MagicMock,
    ) -> None:
        mon = EnergyMonitor(sample_interval_sec=0.05)
        with mon:
            time.sleep(0.2)

        summary = mon.summary
        assert summary.total_gpu_energy_joules is None
        assert summary.total_cpu_energy_joules is None
        assert summary.total_combined_energy_joules is None

    @patch("calibrax.profiling.energy._get_nvml_power_mw")
    def test_nvml_only(self, mock_nvml: MagicMock) -> None:
        mock_nvml.return_value = 250_000  # 250W in milliwatts

        mon = EnergyMonitor(sample_interval_sec=0.05)
        with mon:
            time.sleep(0.2)

        summary = mon.summary
        assert summary.mean_gpu_power_watts is not None
        assert summary.mean_gpu_power_watts == pytest.approx(250.0)
        assert summary.peak_gpu_power_watts is not None
        assert summary.total_gpu_energy_joules is not None
        assert summary.total_gpu_energy_joules > 0

    @patch("calibrax.profiling.energy._read_rapl_energy_uj")
    def test_rapl_only(self, mock_rapl: MagicMock) -> None:
        counter = [0]

        def rapl_counter() -> int:
            counter[0] += 1_000_000  # 1 joule per call
            return counter[0]

        mock_rapl.side_effect = rapl_counter

        mon = EnergyMonitor(sample_interval_sec=0.05)
        with mon:
            time.sleep(0.2)

        summary = mon.summary
        assert summary.total_cpu_energy_joules is not None
        assert summary.total_cpu_energy_joules > 0

    @patch("calibrax.profiling.energy._read_rapl_energy_uj")
    def test_rapl_wraparound(self, mock_rapl: MagicMock) -> None:
        calls = [0]
        max_uj = 10_000_000  # 10J max before wraparound

        def rapl_with_wrap() -> int:
            calls[0] += 1
            if calls[0] <= 2:
                return max_uj - 1_000_000  # 9J
            return 1_000_000  # wrapped around to 1J

        mock_rapl.side_effect = rapl_with_wrap

        mon = EnergyMonitor(sample_interval_sec=0.05)
        with mon:
            time.sleep(0.15)

        summary = mon.summary
        # Should handle wraparound gracefully
        assert summary.total_cpu_energy_joules is not None

    @patch("calibrax.profiling.energy._get_nvml_power_mw")
    @patch("calibrax.profiling.energy._read_rapl_energy_uj")
    def test_both_nvml_and_rapl(
        self,
        mock_rapl: MagicMock,
        mock_nvml: MagicMock,
    ) -> None:
        mock_nvml.return_value = 200_000  # 200W
        counter = [0]

        def rapl_counter() -> int:
            counter[0] += 500_000  # 0.5J per call
            return counter[0]

        mock_rapl.side_effect = rapl_counter

        mon = EnergyMonitor(sample_interval_sec=0.05)
        with mon:
            time.sleep(0.2)

        summary = mon.summary
        assert summary.total_gpu_energy_joules is not None
        assert summary.total_cpu_energy_joules is not None
        assert summary.total_combined_energy_joules is not None

    def test_gpu_power_integration(self) -> None:
        """Verify GPU energy = sum(power * delta_t)."""
        with patch(
            "calibrax.profiling.energy._get_nvml_power_mw",
        ) as mock_nvml:
            mock_nvml.return_value = 100_000  # 100W constant

            mon = EnergyMonitor(sample_interval_sec=0.05)
            with mon:
                time.sleep(0.3)

            summary = mon.summary
            assert summary.total_gpu_energy_joules is not None
            # Energy should be roughly power * time
            expected = 100.0 * summary.duration_sec
            assert summary.total_gpu_energy_joules == pytest.approx(
                expected,
                rel=0.5,
            )

    def test_empty_summary(self) -> None:
        mon = EnergyMonitor(sample_interval_sec=0.05)
        summary = mon.summary
        assert summary.num_samples == 0
        assert summary.duration_sec == 0.0


class TestNVMLPowerReader:
    """Tests for NVML power sampling helper."""

    def test_known_nvml_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeNVMLException(Exception):
            pass

        def raise_nvml_error(_handle: object) -> int:
            raise FakeNVMLException("nvml query failed")

        fake_nvml = types.SimpleNamespace(
            NVMLError=FakeNVMLException,
            nvmlInit=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda _: object(),
            nvmlDeviceGetPowerUsage=raise_nvml_error,
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

        assert _get_nvml_power_mw() is None

    def test_unexpected_error_is_not_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeNVMLException(Exception):
            pass

        class CatastrophicNVMLFailure(Exception):
            pass

        def raise_catastrophic_error(_handle: object) -> int:
            raise CatastrophicNVMLFailure("unexpected failure")

        fake_nvml = types.SimpleNamespace(
            NVMLError=FakeNVMLException,
            nvmlInit=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda _: object(),
            nvmlDeviceGetPowerUsage=raise_catastrophic_error,
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

        with pytest.raises(CatastrophicNVMLFailure):
            _get_nvml_power_mw()
