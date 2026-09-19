"""Tests for ResourceMonitor, ResourceSample, ResourceSummary, GPUProfilerProtocol, GpuMemory.

Verifies context manager protocol, background thread sampling,
summary computation, GPU field handling (including clock/power),
and daemon thread behavior.
"""

import dataclasses
import time

import pytest

from calibrax.profiling.resources import (
    GpuClocks,
    GpuMemory,
    GpuPower,
    GPUProfilerProtocol,
    ResourceMonitor,
    ResourceSample,
    ResourceSummary,
)
from tests.factories import (
    assert_monitor_collects_samples_twice,
    assert_monitor_thread_lifecycle,
    FakeGpu,
    make_default_resource_summary,
)


class _GpuQueryError(Exception):
    pass


_ALL_READINGS = FakeGpu(
    memory_reading=GpuMemory(used_mb=2048.0, total_mb=8192.0),
    utilization_reading=65.0,
    clocks_reading=GpuClocks(graphics_mhz=1500.0, memory_mhz=900.0),
    power_reading=GpuPower(draw_w=250.0, limit_w=350.0),
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
        assert isinstance(FakeGpu(), GPUProfilerProtocol)

    def test_non_conforming_class_fails(self) -> None:
        class MemoryOnly:
            def memory(self) -> GpuMemory | None:
                return None

        assert not isinstance(MemoryOnly(), GPUProfilerProtocol)


class TestGpuMemory:
    """Tests for the GpuMemory reading."""

    def test_occupancy_is_the_fraction_in_use(self) -> None:
        assert GpuMemory(used_mb=2048.0, total_mb=8192.0).occupancy == 0.25

    def test_occupancy_without_a_total_is_zero(self) -> None:
        assert GpuMemory(used_mb=10.0, total_mb=0.0).occupancy == 0.0


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

    def test_gpu_readings_fill_samples_and_summary(self) -> None:
        with ResourceMonitor(sample_interval_sec=0.05, gpu_profiler=_ALL_READINGS) as mon:
            time.sleep(0.2)

        sample = mon.samples[0]
        assert sample.gpu_util == 65.0
        assert sample.gpu_mem_mb == 2048.0
        assert sample.gpu_clock_mhz == 1500.0
        assert sample.gpu_power_w == 250.0

        summary = mon.summary
        assert summary.peak_gpu_mem_mb == 2048.0
        assert summary.mean_gpu_util == pytest.approx(65.0)
        assert summary.mean_gpu_clock_mhz == pytest.approx(1500.0)
        assert summary.mean_gpu_power_w == pytest.approx(250.0)

    def test_a_reading_not_taken_stays_none(self) -> None:
        with ResourceMonitor(
            sample_interval_sec=0.05, gpu_profiler=FakeGpu(utilization_reading=42.0)
        ) as mon:
            time.sleep(0.15)

        assert all(sample.gpu_util == 42.0 for sample in mon.samples)
        assert all(sample.gpu_mem_mb is None for sample in mon.samples)
        assert mon.summary.mean_gpu_power_w is None

    def test_a_gpu_error_is_raised_on_exit(self) -> None:
        class _FailingGpu(FakeGpu):
            def memory(self) -> GpuMemory | None:
                raise _GpuQueryError

        with (
            pytest.raises(_GpuQueryError),
            ResourceMonitor(sample_interval_sec=0.05, gpu_profiler=_FailingGpu()),
        ):
            time.sleep(0.1)

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
