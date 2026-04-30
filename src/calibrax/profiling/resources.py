"""Background resource monitoring with 10Hz sampling.

Provides ResourceMonitor context manager for tracking CPU, memory,
and optional GPU utilization during benchmark execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import psutil  # pyright: ignore[reportMissingModuleSource]

from calibrax.profiling._sampling import SamplingThread


@runtime_checkable
class GPUProfilerProtocol(Protocol):
    """Protocol for GPU profilers providing utilization and memory data."""

    def get_utilization(self) -> float:
        """Get current GPU utilization percentage.

        Returns:
            GPU utilization as a percentage (0-100).
        """
        ...

    def get_memory_usage(self) -> dict[str, float]:
        """Get current GPU memory usage statistics.

        Returns:
            Dictionary with at least 'gpu_memory_used_mb' key.
        """
        ...

    def get_clock_info(self) -> dict[str, float]:
        """Get current GPU clock frequencies.

        Returns:
            Dictionary with 'gpu_clock_mhz' and 'mem_clock_mhz' keys.
        """
        ...

    def get_power_info(self) -> dict[str, float]:
        """Get current GPU power draw and limits.

        Returns:
            Dictionary with 'power_draw_w' and 'power_limit_w' keys.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceSample:
    """Single resource measurement at a point in time.

    Attributes:
        timestamp: Time of measurement (perf_counter).
        cpu_percent: CPU utilization percentage.
        rss_mb: Resident set size in MB.
        gpu_util: GPU utilization percentage (None if no GPU).
        gpu_mem_mb: GPU memory used in MB (None if no GPU).
    """

    timestamp: float
    cpu_percent: float
    rss_mb: float
    gpu_util: float | None
    gpu_mem_mb: float | None
    gpu_clock_mhz: float | None = None
    gpu_power_w: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceSummary:
    """Aggregated resource usage over a monitoring period.

    Attributes:
        peak_rss_mb: Maximum RSS observed.
        mean_rss_mb: Average RSS across all samples.
        peak_gpu_mem_mb: Maximum GPU memory (None if no GPU).
        mean_gpu_util: Average GPU utilization (None if no GPU).
        memory_growth_mb: Last RSS minus first RSS (positive = growth).
        num_samples: Total samples collected.
        duration_sec: Time span of monitoring.
    """

    peak_rss_mb: float
    mean_rss_mb: float
    peak_gpu_mem_mb: float | None
    mean_gpu_util: float | None
    memory_growth_mb: float
    num_samples: int
    duration_sec: float
    mean_gpu_clock_mhz: float | None = None
    mean_gpu_power_w: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Optional GPU fields are included only when not None.
        Numeric values are converted to Python primitives for JAX scalar safety.

        Returns:
            Dictionary representation with all resource summary fields.
        """
        d: dict[str, Any] = {
            "peak_rss_mb": float(self.peak_rss_mb),
            "mean_rss_mb": float(self.mean_rss_mb),
            "peak_gpu_mem_mb": (
                float(self.peak_gpu_mem_mb) if self.peak_gpu_mem_mb is not None else None
            ),
            "mean_gpu_util": (
                float(self.mean_gpu_util) if self.mean_gpu_util is not None else None
            ),
            "memory_growth_mb": float(self.memory_growth_mb),
            "num_samples": int(self.num_samples),
            "duration_sec": float(self.duration_sec),
        }
        if self.mean_gpu_clock_mhz is not None:
            d["mean_gpu_clock_mhz"] = float(self.mean_gpu_clock_mhz)
        if self.mean_gpu_power_w is not None:
            d["mean_gpu_power_w"] = float(self.mean_gpu_power_w)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceSummary:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with resource summary fields.

        Returns:
            Reconstructed ResourceSummary instance.
        """
        return cls(
            peak_rss_mb=data["peak_rss_mb"],
            mean_rss_mb=data["mean_rss_mb"],
            peak_gpu_mem_mb=data.get("peak_gpu_mem_mb"),
            mean_gpu_util=data.get("mean_gpu_util"),
            memory_growth_mb=data["memory_growth_mb"],
            num_samples=data["num_samples"],
            duration_sec=data["duration_sec"],
            mean_gpu_clock_mhz=data.get("mean_gpu_clock_mhz"),
            mean_gpu_power_w=data.get("mean_gpu_power_w"),
        )


class ResourceMonitor:
    """Background 10Hz resource sampling via context manager.

    Usage:

    ```python
    with ResourceMonitor() as mon:
        # ... run benchmark ...
    summary = mon.summary
    ```

    Args:
        sample_interval_sec: Seconds between samples (default 0.1 = 10Hz).
        gpu_profiler: Optional profiler satisfying GPUProfilerProtocol.
    """

    def __init__(
        self,
        sample_interval_sec: float = 0.1,
        gpu_profiler: GPUProfilerProtocol | None = None,
    ) -> None:
        """Initialize ResourceMonitor.

        Args:
            sample_interval_sec: Seconds between resource samples.
            gpu_profiler: Optional GPU profiler for GPU metrics.
        """
        self._interval = sample_interval_sec
        self._gpu_profiler = gpu_profiler
        self._samples: list[ResourceSample] = []
        self._sampling_thread = SamplingThread(target=self._sample_loop)
        self._process = psutil.Process()

    def __enter__(self) -> ResourceMonitor:
        """Start background sampling thread."""
        self._samples.clear()
        self._sampling_thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop background sampling thread."""
        self._sampling_thread.stop()

    def _sample_loop(self) -> None:
        """Collect samples at the configured interval until stopped."""
        while not self._sampling_thread.stop_event.is_set():
            clock_info = self._get_gpu_clock()
            power_info = self._get_gpu_power()
            sample = ResourceSample(
                timestamp=time.perf_counter(),
                cpu_percent=self._process.cpu_percent(),
                rss_mb=self._process.memory_info().rss / (1024 * 1024),
                gpu_util=self._get_gpu_util(),
                gpu_mem_mb=self._get_gpu_mem(),
                gpu_clock_mhz=clock_info,
                gpu_power_w=power_info,
            )
            self._samples.append(sample)
            self._sampling_thread.stop_event.wait(timeout=self._interval)

    def _safe_gpu_call(
        self,
        method: str,
        key: str | None = None,
    ) -> float | None:
        """Safely call a GPU profiler method, returning None on failure.

        Args:
            method: Method name on the GPU profiler to call.
            key: If the method returns a dict, extract this key.

        Returns:
            Float value, or None if profiler is absent or call fails.
        """
        if self._gpu_profiler is None:
            return None
        try:
            result = getattr(self._gpu_profiler, method)()
            if key is not None:
                return result.get(key)  # type: ignore[union-attr]
            return result  # type: ignore[return-value]
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None

    def _get_gpu_util(self) -> float | None:
        """Get GPU utilization, returning None on failure or no profiler."""
        return self._safe_gpu_call("get_utilization")

    def _get_gpu_mem(self) -> float | None:
        """Get GPU memory usage in MB, returning None on failure."""
        return self._safe_gpu_call("get_memory_usage", key="gpu_memory_used_mb")

    def _get_gpu_clock(self) -> float | None:
        """Get GPU clock frequency in MHz, returning None on failure."""
        return self._safe_gpu_call("get_clock_info", key="gpu_clock_mhz")

    def _get_gpu_power(self) -> float | None:
        """Get GPU power draw in watts, returning None on failure."""
        return self._safe_gpu_call("get_power_info", key="power_draw_w")

    @property
    def samples(self) -> list[ResourceSample]:
        """Return a copy of all collected samples."""
        return list(self._samples)

    @property
    def summary(self) -> ResourceSummary:
        """Compute aggregated summary from collected samples.

        Returns:
            ResourceSummary with aggregated metrics, or zeroed summary
            if no samples were collected.
        """
        if not self._samples:
            return ResourceSummary(
                peak_rss_mb=0,
                mean_rss_mb=0,
                peak_gpu_mem_mb=None,
                mean_gpu_util=None,
                memory_growth_mb=0,
                num_samples=0,
                duration_sec=0,
            )

        rss_values = [s.rss_mb for s in self._samples]
        duration = (
            self._samples[-1].timestamp - self._samples[0].timestamp
            if len(self._samples) > 1
            else 0.0
        )

        return ResourceSummary(
            peak_rss_mb=max(rss_values),
            mean_rss_mb=sum(rss_values) / len(rss_values),
            peak_gpu_mem_mb=self._compute_gpu_peak_mem(),
            mean_gpu_util=self._compute_gpu_mean(attr="gpu_util"),
            memory_growth_mb=rss_values[-1] - rss_values[0],
            num_samples=len(self._samples),
            duration_sec=duration,
            mean_gpu_clock_mhz=self._compute_gpu_mean(attr="gpu_clock_mhz"),
            mean_gpu_power_w=self._compute_gpu_mean(attr="gpu_power_w"),
        )

    def _compute_gpu_peak_mem(self) -> float | None:
        """Compute peak GPU memory from samples.

        Returns:
            Peak GPU memory in MB, or None if no GPU data.
        """
        values = [s.gpu_mem_mb for s in self._samples if s.gpu_mem_mb is not None]
        return max(values) if values else None

    def _compute_gpu_mean(self, *, attr: str) -> float | None:
        """Compute mean of a GPU metric from samples.

        Args:
            attr: Sample attribute name to average.

        Returns:
            Mean value, or None if no data.
        """
        values = [getattr(s, attr) for s in self._samples if getattr(s, attr) is not None]
        return sum(values) / len(values) if values else None
