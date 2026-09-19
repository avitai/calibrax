"""Background resource monitoring with 10Hz sampling.

Provides ResourceMonitor context manager for tracking CPU, memory,
and optional GPU utilization during benchmark execution.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import psutil
from substrax.records import read_record
from substrax.typing import JsonValue

from calibrax.profiling._sampling import SamplingThread


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuMemory:
    """A GPU's memory in use and in total, in MB."""

    used_mb: float
    total_mb: float

    @property
    def occupancy(self) -> float:
        """The fraction of the memory in use, 0 when the total is unknown."""
        return self.used_mb / self.total_mb if self.total_mb > 0 else 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuClocks:
    """A GPU's current graphics (SM) and memory clocks, in MHz."""

    graphics_mhz: float
    memory_mhz: float


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuPower:
    """A GPU's current power draw and its management limit, in watts."""

    draw_w: float
    limit_w: float


@runtime_checkable
class GPUProfilerProtocol(Protocol):
    """A source of GPU readings; a reading it cannot take is ``None``."""

    def memory(self) -> GpuMemory | None:
        """The GPU's memory in use."""
        ...

    def utilization(self) -> float | None:
        """The GPU's compute utilization, in percent."""
        ...

    def clocks(self) -> GpuClocks | None:
        """The GPU's current clocks."""
        ...

    def power(self) -> GpuPower | None:
        """The GPU's current power draw."""
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
        gpu_clock_mhz: GPU SM clock in MHz (None if unavailable).
        gpu_power_w: GPU power draw in watts (None if unavailable).
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
        mean_gpu_clock_mhz: Average GPU SM clock in MHz (None if unavailable).
        mean_gpu_power_w: Average GPU power draw in watts (None if unavailable).
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

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary.

        Optional GPU fields are included only when not None.
        Numeric values are converted to Python primitives for JAX scalar safety.

        Returns:
            Dictionary representation with all resource summary fields.
        """
        d: dict[str, JsonValue] = {
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
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> ResourceSummary:
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


class ResourceMonitor:
    """Background 10Hz resource sampling via context manager.

    Usage:

    ```python
    with ResourceMonitor() as mon:
        # ... run benchmark ...
    summary = mon.summary
    ```
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
        gpu = self._gpu_profiler
        while not self._sampling_thread.stop_event.is_set():
            memory = gpu.memory() if gpu is not None else None
            clocks = gpu.clocks() if gpu is not None else None
            power = gpu.power() if gpu is not None else None
            sample = ResourceSample(
                timestamp=time.perf_counter(),
                cpu_percent=self._process.cpu_percent(),
                rss_mb=self._process.memory_info().rss / (1024 * 1024),
                gpu_util=gpu.utilization() if gpu is not None else None,
                gpu_mem_mb=memory.used_mb if memory is not None else None,
                gpu_clock_mhz=clocks.graphics_mhz if clocks is not None else None,
                gpu_power_w=power.draw_w if power is not None else None,
            )
            self._samples.append(sample)
            self._sampling_thread.stop_event.wait(timeout=self._interval)

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
            mean_gpu_util=_mean(s.gpu_util for s in self._samples),
            memory_growth_mb=rss_values[-1] - rss_values[0],
            num_samples=len(self._samples),
            duration_sec=duration,
            mean_gpu_clock_mhz=_mean(s.gpu_clock_mhz for s in self._samples),
            mean_gpu_power_w=_mean(s.gpu_power_w for s in self._samples),
        )

    def _compute_gpu_peak_mem(self) -> float | None:
        """Compute peak GPU memory from samples.

        Returns:
            Peak GPU memory in MB, or None if no GPU data.
        """
        values = [s.gpu_mem_mb for s in self._samples if s.gpu_mem_mb is not None]
        return max(values) if values else None


def _mean(values: Iterable[float | None]) -> float | None:
    """The mean of the readings taken, or ``None`` when none was."""
    taken = [value for value in values if value is not None]
    return sum(taken) / len(taken) if taken else None
