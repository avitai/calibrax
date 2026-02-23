"""GPU memory profiling and hardware-adaptive operations.

Provides hardware detection, shape optimization, GPU memory profiling
(satisfying GPUProfilerProtocol), and memory usage analysis.
Includes NVML-based GPU clock and power monitoring when pynvml is available.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import psutil  # pyright: ignore[reportMissingModuleSource]


try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None  # type: ignore[assignment]
    PYNVML_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class HardwareConfig:
    """Hardware-specific optimization configuration.

    Attributes:
        platform: Detected platform ("cpu", "tpu", "gpu_modern", "gpu_legacy").
        precision: Recommended floating-point precision string.
        tile_size: Tile size for matrix operation alignment.
        critical_batch_size: Optimal batch size for the platform.
        memory_layout: Memory layout preference.
        use_vmem_optimization: Whether VMEM optimization is available.
    """

    platform: str
    precision: str
    tile_size: int
    critical_batch_size: int
    memory_layout: str
    use_vmem_optimization: bool


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


_CPU_DEFAULT = HardwareConfig(
    platform="cpu",
    precision="float32",
    tile_size=64,
    critical_batch_size=32,
    memory_layout="row_major",
    use_vmem_optimization=False,
)


class AdaptiveOperation:
    """Hardware-adaptive operations with auto-detection.

    Detects the current JAX backend (CPU/GPU/TPU) and provides
    optimized configuration and shape padding.
    """

    def __init__(self) -> None:
        """Initialize with auto-detected hardware configuration."""
        self.config = self._detect_hardware()

    def _detect_hardware(self) -> HardwareConfig:
        """Detect hardware and return optimal configuration.

        Returns:
            HardwareConfig for the detected platform.
        """
        backend = jax.default_backend()

        if backend == "tpu":
            return HardwareConfig(
                platform="tpu",
                precision="bfloat16",
                tile_size=128,
                critical_batch_size=240,
                memory_layout="row_major",
                use_vmem_optimization=True,
            )

        if backend == "gpu":
            return self._detect_gpu_config()

        return _CPU_DEFAULT

    def _detect_gpu_config(self) -> HardwareConfig:
        """Detect GPU variant and return config.

        Returns:
            HardwareConfig for the detected GPU, or CPU default on failure.
        """
        try:
            devices = jax.devices()
        except RuntimeError:
            return _CPU_DEFAULT
        if not devices:
            return _CPU_DEFAULT
        device_kind = getattr(
            devices[0],
            "device_kind",
            "unknown",
        ).lower()
        if "h100" in device_kind or "a100" in device_kind:
            return HardwareConfig(
                platform="gpu_modern",
                precision="bfloat16",
                tile_size=16,
                critical_batch_size=298,
                memory_layout="row_major",
                use_vmem_optimization=False,
            )
        return HardwareConfig(
            platform="gpu_legacy",
            precision="float32",
            tile_size=32,
            critical_batch_size=128,
            memory_layout="row_major",
            use_vmem_optimization=False,
        )

    def optimize_shapes(
        self,
        *shapes: tuple[int, ...],
    ) -> list[tuple[int, ...]]:
        """Pad tensor shapes to align with hardware tile size.

        Args:
            *shapes: Variable number of tensor shapes to optimize.

        Returns:
            List of optimized shapes padded to tile_size multiples.
        """
        tile = self.config.tile_size
        result: list[tuple[int, ...]] = []

        for shape in shapes:
            opt = list(shape)
            for i in [-2, -1]:
                if len(opt) >= abs(i):
                    dim = opt[i]
                    if dim % tile != 0:
                        opt[i] = ((dim + tile - 1) // tile) * tile
            result.append(tuple(opt))

        return result


class GPUMemoryProfiler:
    """GPU memory profiling satisfying GPUProfilerProtocol.

    Uses multi-fallback strategy: memory_stats -> xla_bridge -> zeros.
    """

    def __init__(self) -> None:
        """Initialize GPU memory profiler with GPU detection."""
        try:
            self.has_gpu = len(jax.devices("gpu")) > 0
        except (RuntimeError, ValueError):
            self.has_gpu = False

    def get_memory_usage(self) -> dict[str, float]:
        """Get current GPU memory usage statistics.

        Returns:
            Dictionary with gpu_memory_used_mb, gpu_memory_total_mb,
            and optionally gpu_memory_utilization.
        """
        if not self.has_gpu:
            return {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}

        try:
            device = jax.devices("gpu")[0]
            if hasattr(device, "memory_stats"):
                stats = device.memory_stats()
                if stats:
                    used = stats.get("bytes_in_use", 0) / (1024 * 1024)
                    limit = stats.get("bytes_limit", 0) / (1024 * 1024)
                    return {
                        "gpu_memory_used_mb": used,
                        "gpu_memory_total_mb": limit,
                        "gpu_memory_utilization": (used / limit if limit > 0 else 0.0),
                    }
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
            logger.debug("GPU memory stats query failed, returning zeros")

        return {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}

    def get_utilization(self) -> float:
        """Get GPU utilization percentage for ResourceMonitor.

        Returns:
            GPU memory utilization as percentage (0-100), or 0.0.
        """
        mem = self.get_memory_usage()
        return mem.get("gpu_memory_utilization", 0.0) * 100

    def _safe_nvml_query(
        self,
        query_fn: Callable[[Any], dict[str, float]],
        fallback: dict[str, float],
    ) -> dict[str, float]:
        """Execute an NVML query with init and fallback on failure.

        Args:
            query_fn: Function that takes an NVML handle and returns metrics.
            fallback: Default dict to return on failure.

        Returns:
            Query result or fallback on any error.
        """
        if not PYNVML_AVAILABLE or not self.has_gpu:
            return fallback
        try:
            pynvml.nvmlInit()  # type: ignore[union-attr]
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # type: ignore[union-attr]
            return query_fn(handle)
        except pynvml.NVMLError:  # type: ignore[union-attr]
            return fallback

    def get_clock_info(self) -> dict[str, float]:
        """Get current GPU clock frequencies via NVML.

        Returns:
            Dictionary with 'gpu_clock_mhz' and 'mem_clock_mhz' keys.
            Returns zeros if NVML is unavailable or query fails.
        """

        def _query(handle: Any) -> dict[str, float]:
            """Query GPU and memory clock frequencies."""
            gpu_clock = pynvml.nvmlDeviceGetClockInfo(  # type: ignore[union-attr]
                handle,
                pynvml.NVML_CLOCK_GRAPHICS,  # type: ignore[union-attr]
            )
            mem_clock = pynvml.nvmlDeviceGetClockInfo(  # type: ignore[union-attr]
                handle,
                pynvml.NVML_CLOCK_MEM,  # type: ignore[union-attr]
            )
            return {"gpu_clock_mhz": float(gpu_clock), "mem_clock_mhz": float(mem_clock)}

        return self._safe_nvml_query(_query, {"gpu_clock_mhz": 0.0, "mem_clock_mhz": 0.0})

    def get_power_info(self) -> dict[str, float]:
        """Get current GPU power draw and limit via NVML.

        Returns:
            Dictionary with 'power_draw_w' and 'power_limit_w' keys.
            Returns zeros if NVML is unavailable or query fails.
        """

        def _query(handle: Any) -> dict[str, float]:
            """Query GPU power draw and management limit."""
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # type: ignore[union-attr]
            limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(  # type: ignore[union-attr]
                handle
            )
            return {
                "power_draw_w": float(power_mw) / 1000.0,
                "power_limit_w": float(limit_mw) / 1000.0,
            }

        return self._safe_nvml_query(_query, {"power_draw_w": 0.0, "power_limit_w": 0.0})

    def analyze_memory_pattern(
        self,
        measurements: list[dict[str, float]],
    ) -> list[str]:
        """Analyze memory usage patterns and suggest optimizations.

        Args:
            measurements: List of memory usage dictionaries.

        Returns:
            List of optimization suggestion strings.
        """
        if not measurements:
            return []

        suggestions: list[str] = []
        usage_values = [m.get("gpu_memory_used_mb", 0) for m in measurements]
        utilization_values = [m.get("gpu_memory_utilization", 0) for m in measurements]

        if len(usage_values) >= 3:
            trend = (usage_values[-1] - usage_values[0]) / (len(usage_values) - 1)
            if trend > 10:
                suggestions.append(
                    "Potential memory leak detected. Consider using JAX's "
                    "garbage collection or clearing unused variables."
                )

        max_util = max(utilization_values) if utilization_values else 0
        avg_util = sum(utilization_values) / len(utilization_values) if utilization_values else 0

        if max_util > 0.9:
            suggestions.append(
                "High GPU memory utilization (>90%). Consider reducing "
                "batch size or using gradient checkpointing."
            )
        elif avg_util > 0.8:
            suggestions.append(
                "Consistently high GPU memory usage (>80%). Monitor for "
                "potential out-of-memory errors."
            )

        return suggestions


class MemoryOptimizer:
    """Memory optimization analysis for pipeline functions."""

    def analyze_pipeline_memory(
        self,
        pipeline_fn: Callable[[Any], Any],
        sample_data: Any,
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

        if peak_usage > 1000:
            suggestions.append(
                "High memory usage detected. Consider processing data "
                "in smaller batches or using sharding."
            )

        efficiency = (peak_usage - retained_memory) / peak_usage if peak_usage > 0 else 1.0
        if efficiency < 0.7:
            suggestions.append(
                "Low memory efficiency. Consider explicit del statements "
                "for large temporary arrays."
            )

        return suggestions
