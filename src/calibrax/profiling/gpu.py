"""GPU memory profiling and memory usage analysis.

Provides GPU memory readings from JAX's device statistics (satisfying GPUProfilerProtocol),
suggestions from a series of readings, and pipeline memory analysis.
GPU clocks, power and compute utilization come from ``calibrax.profiling.nvml``.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import jax
import psutil

from calibrax.profiling.resources import GpuClocks, GpuMemory, GpuPower


_BYTES_PER_MB = 1024 * 1024


# Thresholds behind the memory suggestions.
_MIN_SAMPLES_FOR_TREND = 3
_LEAK_TREND_MB_PER_SAMPLE = 10
_HIGH_MEMORY_UTILIZATION = 0.9
_SUSTAINED_MEMORY_UTILIZATION = 0.8
_HIGH_PEAK_USAGE_MB = 1000
_LOW_MEMORY_EFFICIENCY = 0.7


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryAnalysis:
    """Result of pipeline memory analysis.

    Attributes:
        baseline_memory_mb: Memory usage before pipeline execution.
        peak_memory_mb: Memory usage at peak during execution.
        peak_usage_mb: Peak usage above baseline.
        retained_memory_mb: Memory retained after GC.
        memory_efficiency: Ratio of freed memory to peak usage.
        suggestions: Optimization suggestions.
    """

    baseline_memory_mb: float
    peak_memory_mb: float
    peak_usage_mb: float
    retained_memory_mb: float
    memory_efficiency: float
    suggestions: tuple[str, ...] = ()


class _MemoryStatsDevice(Protocol):
    """The part of a ``jax.Device`` the profiler reads."""

    def memory_stats(self) -> Mapping[str, int] | None: ...


class GPUMemoryProfiler:
    """GPU memory from JAX's device memory statistics; satisfies ``GPUProfilerProtocol``.

    JAX reports a device's memory, not its compute utilization, clocks or power, so those
    readings are ``None``; ``calibrax.profiling.nvml.NvmlDevice`` takes them all through NVML.
    """

    def __init__(self, device: _MemoryStatsDevice | None = None) -> None:
        """Read ``device``, or the first GPU JAX sees; with neither, every reading is ``None``.

        Args:
            device: The JAX device to read.
        """
        if device is None:
            try:
                gpus = jax.devices("gpu")
            except RuntimeError:  # jax has no GPU backend
                gpus = []
            device = gpus[0] if gpus else None
        self._device = device

    def memory(self) -> GpuMemory | None:
        """The device's memory in use and its limit, from ``memory_stats()``.

        ``None`` without a device, or when its statistics lack either figure.
        """
        if self._device is None:
            return None
        stats = self._device.memory_stats() or {}
        used, limit = stats.get("bytes_in_use"), stats.get("bytes_limit")
        if used is None or limit is None:
            return None
        return GpuMemory(used_mb=used / _BYTES_PER_MB, total_mb=limit / _BYTES_PER_MB)

    def utilization(self) -> float | None:
        """``None``: JAX does not report compute utilization."""
        return None

    def clocks(self) -> GpuClocks | None:
        """``None``: JAX does not report clocks."""
        return None

    def power(self) -> GpuPower | None:
        """``None``: JAX does not report power."""
        return None


def analyze_memory_pattern(readings: Sequence[GpuMemory]) -> list[str]:
    """Suggestions from a series of memory readings: a leak trend, and high occupancy.

    Args:
        readings: Memory readings in the order they were taken.

    Returns:
        Optimization suggestion strings.
    """
    if not readings:
        return []

    suggestions: list[str] = []
    usage = [reading.used_mb for reading in readings]
    occupancy = [reading.occupancy for reading in readings]

    if len(usage) >= _MIN_SAMPLES_FOR_TREND:
        trend = (usage[-1] - usage[0]) / (len(usage) - 1)
        if trend > _LEAK_TREND_MB_PER_SAMPLE:
            suggestions.append(
                "Potential memory leak detected. Consider using JAX's "
                "garbage collection or clearing unused variables."
            )

    if max(occupancy) > _HIGH_MEMORY_UTILIZATION:
        suggestions.append(
            "High GPU memory utilization (>90%). Consider reducing "
            "batch size or using gradient checkpointing."
        )
    elif sum(occupancy) / len(occupancy) > _SUSTAINED_MEMORY_UTILIZATION:
        suggestions.append(
            "Consistently high GPU memory usage (>80%). Monitor for potential out-of-memory errors."
        )

    return suggestions


class MemoryOptimizer:
    """Memory optimization analysis for pipeline functions."""

    def analyze_pipeline_memory[SampleT](
        self,
        pipeline_fn: Callable[[SampleT], object],
        sample_data: SampleT,
    ) -> MemoryAnalysis | None:
        """Analyze memory usage of a pipeline function.

        Args:
            pipeline_fn: Function to analyze.
            sample_data: Sample input data.

        Returns:
            MemoryAnalysis with measurements and suggestions,
            or None if the pipeline raises an exception.
        """
        baseline = self._get_rss_mb()
        gc.collect()

        try:
            pipeline_fn(sample_data)
        except (OSError, RuntimeError, TypeError, ValueError):
            logger.warning("Pipeline function raised during memory analysis")
            return None

        peak = self._get_rss_mb()
        gc.collect()
        post_gc = self._get_rss_mb()

        peak_usage = peak - baseline
        retained = post_gc - baseline
        efficiency = (peak_usage - retained) / peak_usage if peak_usage > 0 else 1.0

        suggestions = self._generate_suggestions(peak_usage, retained)

        return MemoryAnalysis(
            baseline_memory_mb=baseline,
            peak_memory_mb=peak,
            peak_usage_mb=peak_usage,
            retained_memory_mb=retained,
            memory_efficiency=efficiency,
            suggestions=tuple(suggestions),
        )

    def _get_rss_mb(self) -> float:
        """Get current process RSS in MB."""
        return psutil.Process().memory_info().rss / (1024 * 1024)

    def _generate_suggestions(
        self,
        peak_usage: float,
        retained_memory: float,
    ) -> list[str]:
        """Generate memory optimization suggestions.

        Args:
            peak_usage: Peak memory usage in MB.
            retained_memory: Retained memory after GC in MB.

        Returns:
            List of suggestion strings.
        """
        suggestions: list[str] = []

        if peak_usage > _HIGH_PEAK_USAGE_MB:
            suggestions.append(
                "High memory usage detected. Consider processing data "
                "in smaller batches or using sharding."
            )

        efficiency = (peak_usage - retained_memory) / peak_usage if peak_usage > 0 else 1.0
        if efficiency < _LOW_MEMORY_EFFICIENCY:
            suggestions.append(
                "Low memory efficiency. Consider explicit del statements "
                "for large temporary arrays."
            )

        return suggestions
