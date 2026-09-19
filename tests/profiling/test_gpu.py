"""Tests for GPU profiling: GPUMemoryProfiler, memory patterns and pipeline memory analysis.

All GPU/hardware access is mocked — no hardware dependency.
"""

import dataclasses
from collections.abc import Mapping
from unittest.mock import MagicMock, patch

import pytest

from calibrax.profiling.gpu import (
    analyze_memory_pattern,
    GPUMemoryProfiler,
    MemoryAnalysis,
    MemoryOptimizer,
)
from calibrax.profiling.resources import GpuMemory, GPUProfilerProtocol


_MB = 1024 * 1024


class _Device:
    """A device answering fixed memory statistics."""

    def __init__(self, stats: Mapping[str, int] | None) -> None:
        self._stats = stats

    def memory_stats(self) -> Mapping[str, int] | None:
        return self._stats


def _reading(used_mb: float, occupancy: float) -> GpuMemory:
    return GpuMemory(used_mb=used_mb, total_mb=used_mb / occupancy)


class TestGPUMemoryProfiler:
    """Tests for GPUMemoryProfiler."""

    def test_memory_is_read_from_the_device_statistics(self) -> None:
        device = _Device({"bytes_in_use": 2048 * _MB, "bytes_limit": 8192 * _MB})

        assert GPUMemoryProfiler(device).memory() == GpuMemory(used_mb=2048.0, total_mb=8192.0)

    @pytest.mark.parametrize("stats", [None, {}, {"bytes_in_use": 1}, {"bytes_limit": 1}], ids=str)
    def test_statistics_without_both_figures_are_no_reading(
        self, stats: Mapping[str, int] | None
    ) -> None:
        assert GPUMemoryProfiler(_Device(stats)).memory() is None

    @patch("calibrax.profiling.gpu.jax")
    def test_without_a_gpu_backend_there_is_no_reading(self, mock_jax: MagicMock) -> None:
        mock_jax.devices.side_effect = RuntimeError("Unknown backend gpu")

        assert GPUMemoryProfiler().memory() is None

    @patch("calibrax.profiling.gpu.jax")
    def test_the_first_gpu_is_read_by_default(self, mock_jax: MagicMock) -> None:
        mock_jax.devices.return_value = [
            _Device({"bytes_in_use": _MB, "bytes_limit": 4 * _MB}),
            _Device(None),
        ]

        assert GPUMemoryProfiler().memory() == GpuMemory(used_mb=1.0, total_mb=4.0)
        mock_jax.devices.assert_called_once_with("gpu")

    def test_an_unexpected_backend_error_propagates(self) -> None:
        class CatastrophicMemoryQueryError(Exception):
            pass

        with patch("calibrax.profiling.gpu.jax") as mock_jax:
            mock_jax.devices.side_effect = CatastrophicMemoryQueryError("catastrophic")
            with pytest.raises(CatastrophicMemoryQueryError):
                GPUMemoryProfiler()

    def test_jax_reports_no_utilization_clocks_or_power(self) -> None:
        profiler = GPUMemoryProfiler(_Device({}))

        assert profiler.utilization() is None
        assert profiler.clocks() is None
        assert profiler.power() is None

    def test_satisfies_gpu_profiler_protocol(self) -> None:
        assert isinstance(GPUMemoryProfiler(_Device(None)), GPUProfilerProtocol)

    def test_a_cpu_run_reads_nothing(self) -> None:
        # The suite runs under JAX_PLATFORMS=cpu: no GPU backend, so no reading.
        assert GPUMemoryProfiler().memory() is None


class TestAnalyzeMemoryPattern:
    """Tests for analyze_memory_pattern."""

    def test_no_readings_no_suggestions(self) -> None:
        assert analyze_memory_pattern([]) == []

    def test_a_rising_trend_suggests_a_leak(self) -> None:
        readings = [_reading(100, 0.3), _reading(200, 0.3), _reading(300, 0.3)]

        assert any("memory leak" in s.lower() for s in analyze_memory_pattern(readings))

    def test_a_peak_over_ninety_percent_is_reported(self) -> None:
        readings = [_reading(100, 0.95), _reading(100, 0.5), _reading(100, 0.5)]

        assert any("90%" in s for s in analyze_memory_pattern(readings))

    def test_steady_low_usage_has_no_suggestions(self) -> None:
        readings = [_reading(50, 0.1), _reading(51, 0.1), _reading(50, 0.1)]

        assert analyze_memory_pattern(readings) == []

    def test_a_sustained_eighty_percent_is_reported_without_the_peak_warning(self) -> None:
        readings = [_reading(100, 0.82), _reading(102, 0.85), _reading(101, 0.83)]

        suggestions = analyze_memory_pattern(readings)

        assert any("80%" in suggestion for suggestion in suggestions)
        assert all("90%" not in suggestion for suggestion in suggestions)

    def test_a_short_series_skips_the_leak_trend(self) -> None:
        readings = [_reading(100, 0.2), _reading(300, 0.2)]

        assert all("memory leak" not in s.lower() for s in analyze_memory_pattern(readings))


class TestMemoryAnalysis:
    """Tests for MemoryAnalysis frozen dataclass."""

    def test_construction(self) -> None:
        analysis = MemoryAnalysis(
            baseline_memory_mb=100.0,
            peak_memory_mb=200.0,
            peak_usage_mb=100.0,
            retained_memory_mb=10.0,
            memory_efficiency=0.9,
            suggestions=("Use smaller batches",),
        )
        assert analysis.memory_efficiency == 0.9

    def test_frozen_immutability(self) -> None:
        analysis = MemoryAnalysis(
            baseline_memory_mb=100.0,
            peak_memory_mb=200.0,
            peak_usage_mb=100.0,
            retained_memory_mb=10.0,
            memory_efficiency=0.9,
            suggestions=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            analysis.peak_memory_mb = 0.0  # type: ignore[misc]


class TestMemoryOptimizer:
    """Tests for MemoryOptimizer."""

    def test_analyze_pipeline_returns_memory_analysis(self) -> None:
        optimizer = MemoryOptimizer()
        result = optimizer.analyze_pipeline_memory(
            lambda _: None,
            sample_data=None,
        )
        assert isinstance(result, MemoryAnalysis)

    def test_analyze_pipeline_exception_returns_error(self) -> None:
        optimizer = MemoryOptimizer()

        def failing_fn(_: object) -> None:
            msg = "pipeline exploded"
            raise ValueError(msg)

        result = optimizer.analyze_pipeline_memory(
            failing_fn,
            sample_data=None,
        )
        assert result is None

    def test_analyze_pipeline_unexpected_exception_propagates(self) -> None:
        class CatastrophicPipelineError(Exception):
            pass

        optimizer = MemoryOptimizer()

        def catastrophic_fn(_: object) -> None:
            raise CatastrophicPipelineError("unexpected pipeline failure")

        with pytest.raises(CatastrophicPipelineError):
            optimizer.analyze_pipeline_memory(
                catastrophic_fn,
                sample_data=None,
            )

    def test_generate_suggestions_includes_high_peak_and_low_efficiency(self) -> None:
        optimizer = MemoryOptimizer()

        suggestions = optimizer._generate_suggestions(peak_usage=1500.0, retained_memory=1200.0)

        assert any("High memory usage detected" in suggestion for suggestion in suggestions)
        assert any("Low memory efficiency" in suggestion for suggestion in suggestions)
