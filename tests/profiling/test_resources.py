"""Tests for ResourceMonitor, ResourceSample, ResourceSummary, GPUProfilerProtocol.

Verifies context manager protocol, background thread sampling,
summary computation, GPU field handling (including clock/power),
and daemon thread behavior.
"""

import dataclasses
import time
from unittest.mock import MagicMock

import pytest

from calibrax.profiling.resources import (
    GPUProfilerProtocol,
    ResourceMonitor,
    ResourceSample,
    ResourceSummary,
)
from tests.factories import (
    assert_monitor_collects_samples_twice,
    assert_monitor_thread_lifecycle,
    make_default_resource_summary,
)


class TestResourceSample:
    """Tests for ResourceSample frozen dataclass."""

    def test_creation_cpu_only(self) -> None:
        sample = ResourceSample(
            timestamp=1.0,
            cpu_percent=50.0,
            rss_mb=256.0,
            gpu_util=None,
            gpu_mem_mb=None,
        )
        assert sample.cpu_percent == 50.0
        assert sample.rss_mb == 256.0
        assert sample.gpu_util is None

    def test_creation_with_gpu(self) -> None:
        sample = ResourceSample(
            timestamp=1.0,
            cpu_percent=30.0,
            rss_mb=512.0,
            gpu_util=75.0,
            gpu_mem_mb=4096.0,
        )
        assert sample.gpu_util == 75.0
        assert sample.gpu_mem_mb == 4096.0

    def test_frozen_immutability(self) -> None:
        sample = ResourceSample(
            timestamp=1.0,
            cpu_percent=50.0,
            rss_mb=256.0,
            gpu_util=None,
            gpu_mem_mb=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample.cpu_percent = 99.0  # type: ignore[misc]

    def test_gpu_clock_power_defaults_to_none(self) -> None:
        sample = ResourceSample(
            timestamp=1.0,
            cpu_percent=50.0,
            rss_mb=256.0,
            gpu_util=None,
            gpu_mem_mb=None,
        )
        assert sample.gpu_clock_mhz is None
        assert sample.gpu_power_w is None

    def test_gpu_clock_power_fields(self) -> None:
        sample = ResourceSample(
            timestamp=1.0,
            cpu_percent=30.0,
            rss_mb=512.0,
            gpu_util=75.0,
            gpu_mem_mb=4096.0,
            gpu_clock_mhz=1500.0,
            gpu_power_w=250.0,
        )
        assert sample.gpu_clock_mhz == 1500.0
        assert sample.gpu_power_w == 250.0


class TestResourceSummary:
    """Tests for ResourceSummary frozen dataclass."""

    def test_creation(self) -> None:
        summary = make_default_resource_summary()
        assert summary.peak_rss_mb == 512.0
        assert summary.num_samples == 50

    def test_frozen_immutability(self) -> None:
        summary = make_default_resource_summary()
        with pytest.raises(dataclasses.FrozenInstanceError):
            summary.peak_rss_mb = 0.0  # type: ignore[misc]

    def test_gpu_clock_power_defaults_to_none(self) -> None:
        summary = make_default_resource_summary()
        assert summary.mean_gpu_clock_mhz is None
        assert summary.mean_gpu_power_w is None

    def test_gpu_clock_power_fields(self) -> None:
        summary = make_default_resource_summary(mean_gpu_clock_mhz=1400.0, mean_gpu_power_w=200.0)
        assert summary.mean_gpu_clock_mhz == 1400.0
        assert summary.mean_gpu_power_w == 200.0


class TestGPUProfilerProtocol:
    """Tests for GPUProfilerProtocol structural subtyping."""

    def test_conforming_class_satisfies_protocol(self) -> None:
        class GoodProfiler:
            def get_utilization(self) -> float:
                return 75.0

            def get_memory_usage(self) -> dict[str, float]:
                return {"gpu_memory_used_mb": 2048.0}

            def get_clock_info(self) -> dict[str, float]:
                return {"gpu_clock_mhz": 1500.0, "mem_clock_mhz": 900.0}

            def get_power_info(self) -> dict[str, float]:
                return {"power_draw_w": 250.0, "power_limit_w": 350.0}

        assert isinstance(GoodProfiler(), GPUProfilerProtocol)

    def test_non_conforming_class_fails(self) -> None:
        class BadProfiler:
            def get_utilization(self) -> float:
                return 0.0

        assert not isinstance(BadProfiler(), GPUProfilerProtocol)


class TestResourceMonitor:
    """Tests for ResourceMonitor context manager."""

    def test_context_manager_starts_and_stops_thread(self) -> None:
        mon = ResourceMonitor(sample_interval_sec=0.05)
        assert_monitor_thread_lifecycle(mon)

    def test_samples_grow_during_monitoring(self) -> None:
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.3)

        assert len(mon.samples) > 0

    def test_summary_returns_resource_summary(self) -> None:
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.3)

        summary = mon.summary
        assert isinstance(summary, ResourceSummary)
        assert summary.num_samples > 0

    def test_peak_rss_at_least_mean_rss(self) -> None:
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.3)

        summary = mon.summary
        assert summary.peak_rss_mb >= summary.mean_rss_mb

    def test_gpu_fields_none_without_profiler(self) -> None:
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.2)

        summary = mon.summary
        assert summary.peak_gpu_mem_mb is None
        assert summary.mean_gpu_util is None
        assert summary.mean_gpu_clock_mhz is None
        assert summary.mean_gpu_power_w is None

    def test_mock_gpu_profiler_provides_data(self) -> None:
        mock_gpu = MagicMock()
        mock_gpu.get_utilization.return_value = 65.0
        mock_gpu.get_memory_usage.return_value = {
            "gpu_memory_used_mb": 2048.0,
        }
        mock_gpu.get_clock_info.return_value = {
            "gpu_clock_mhz": 1500.0,
            "mem_clock_mhz": 900.0,
        }
        mock_gpu.get_power_info.return_value = {
            "power_draw_w": 250.0,
            "power_limit_w": 350.0,
        }

        with ResourceMonitor(
            sample_interval_sec=0.05,
            gpu_profiler=mock_gpu,
        ) as mon:
            time.sleep(0.25)

        gpu_samples = [s for s in mon.samples if s.gpu_util is not None]
        assert len(gpu_samples) > 0
        assert gpu_samples[0].gpu_util == 65.0

        summary = mon.summary
        assert summary.peak_gpu_mem_mb is not None
        assert summary.mean_gpu_util is not None
        assert summary.mean_gpu_clock_mhz is not None
        assert summary.mean_gpu_power_w is not None

    def test_mock_gpu_profiler_clock_power_in_samples(self) -> None:
        mock_gpu = MagicMock()
        mock_gpu.get_utilization.return_value = 50.0
        mock_gpu.get_memory_usage.return_value = {"gpu_memory_used_mb": 1024.0}
        mock_gpu.get_clock_info.return_value = {
            "gpu_clock_mhz": 1200.0,
            "mem_clock_mhz": 800.0,
        }
        mock_gpu.get_power_info.return_value = {
            "power_draw_w": 180.0,
            "power_limit_w": 300.0,
        }

        with ResourceMonitor(
            sample_interval_sec=0.05,
            gpu_profiler=mock_gpu,
        ) as mon:
            time.sleep(0.2)

        clock_samples = [s for s in mon.samples if s.gpu_clock_mhz is not None]
        power_samples = [s for s in mon.samples if s.gpu_power_w is not None]
        assert len(clock_samples) > 0
        assert clock_samples[0].gpu_clock_mhz == 1200.0
        assert len(power_samples) > 0
        assert power_samples[0].gpu_power_w == 180.0

    def test_gpu_profiler_exception_graceful_degradation(self) -> None:
        mock_gpu = MagicMock()
        mock_gpu.get_utilization.side_effect = RuntimeError("GPU error")
        mock_gpu.get_memory_usage.side_effect = RuntimeError("GPU error")
        mock_gpu.get_clock_info.side_effect = RuntimeError("GPU error")
        mock_gpu.get_power_info.side_effect = RuntimeError("GPU error")

        with ResourceMonitor(
            sample_interval_sec=0.05,
            gpu_profiler=mock_gpu,
        ) as mon:
            time.sleep(0.25)

        for sample in mon.samples:
            assert sample.gpu_util is None
            assert sample.gpu_mem_mb is None
            assert sample.gpu_clock_mhz is None
            assert sample.gpu_power_w is None

        summary = mon.summary
        assert summary.peak_gpu_mem_mb is None
        assert summary.mean_gpu_util is None
        assert summary.mean_gpu_clock_mhz is None
        assert summary.mean_gpu_power_w is None

    def test_gpu_profiler_malformed_memory_payload_graceful_degradation(self) -> None:
        mock_gpu = MagicMock()
        mock_gpu.get_utilization.return_value = 42.0
        mock_gpu.get_memory_usage.return_value = 123.0  # Not a dict
        mock_gpu.get_clock_info.return_value = {}
        mock_gpu.get_power_info.return_value = {}

        with ResourceMonitor(
            sample_interval_sec=0.05,
            gpu_profiler=mock_gpu,
        ) as mon:
            time.sleep(0.2)

        assert len(mon.samples) > 0
        assert any(sample.gpu_util == 42.0 for sample in mon.samples)
        assert all(sample.gpu_mem_mb is None for sample in mon.samples)

    def test_thread_is_daemon(self) -> None:
        mon = ResourceMonitor(sample_interval_sec=0.05)
        with mon:
            assert mon._sampling_thread._thread is not None
            assert mon._sampling_thread._thread.daemon is True

    def test_multiple_enter_exit_cycles(self) -> None:
        mon = ResourceMonitor(sample_interval_sec=0.05)
        assert_monitor_collects_samples_twice(mon)

    def test_empty_summary_when_no_samples(self) -> None:
        mon = ResourceMonitor(sample_interval_sec=0.05)
        summary = mon.summary
        assert summary.num_samples == 0
        assert summary.peak_rss_mb == 0
        assert summary.duration_sec == 0

    def test_rss_values_positive(self) -> None:
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.2)

        for sample in mon.samples:
            assert sample.rss_mb > 0
