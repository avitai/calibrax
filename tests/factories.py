"""Shared test data builders used across test modules."""

from __future__ import annotations

import time
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from calibrax.core.models import Metric, MetricDef, MetricDirection, Point, Run
from calibrax.profiling._sampling import SamplingThread
from calibrax.profiling.energy import EnergySummary
from calibrax.profiling.gpu import HardwareConfig
from calibrax.profiling.resources import GpuClocks, GpuMemory, GpuPower, ResourceSummary
from calibrax.profiling.timing_records import TimingSample


def _run(
    points: tuple[Point, ...],
    *,
    run_id: str,
    commit: str | None = None,
    branch: str | None = None,
    metric_defs: dict[str, MetricDef] | None = None,
    timestamp: datetime | None = None,
    environment: Mapping[str, str] | None = None,
) -> Run:
    """A Run of ``points``; a field left None keeps Run's own default."""
    run = Run(points=points, id=run_id, commit=commit, branch=branch)
    if metric_defs is not None:
        run = replace(run, metric_defs=metric_defs)
    if timestamp is not None:
        run = replace(run, timestamp=timestamp)
    if environment is not None:
        run = replace(run, environment=dict(environment))
    return run


def make_throughput_latency_defs() -> dict[str, MetricDef]:
    """Return standard throughput/latency metric definitions."""
    return {
        "throughput": MetricDef(name="throughput", unit="ops/s", direction=MetricDirection.HIGHER),
        "latency": MetricDef(name="latency", unit="ms", direction=MetricDirection.LOWER),
    }


def make_single_framework_run(
    *,
    run_id: str = "run1",
    point_name: str = "bench1",
    scenario: str = "default",
    framework: str = "jax",
    throughput: float = 100.0,
    latency: float = 5.0,
    commit: str | None = None,
    branch: str | None = None,
    metric_defs: dict[str, MetricDef] | None = None,
    timestamp: datetime | None = None,
    environment: Mapping[str, str] | None = None,
) -> Run:
    """Build a single-point run with throughput and latency metrics."""
    return _run(
        (
            Point(
                name=point_name,
                scenario=scenario,
                tags={"framework": framework},
                metrics={
                    "throughput": Metric(value=throughput),
                    "latency": Metric(value=latency),
                },
            ),
        ),
        run_id=run_id,
        commit=commit,
        branch=branch,
        metric_defs=metric_defs,
        timestamp=timestamp,
        environment=environment,
    )


def make_dual_framework_run(
    *,
    run_id: str = "run1",
    point_name: str = "bench1",
    scenario: str = "default",
    first_framework: str = "jax",
    second_framework: str = "pytorch",
    first_throughput: float = 200.0,
    first_latency: float = 5.0,
    second_throughput: float = 100.0,
    second_latency: float = 10.0,
    commit: str | None = None,
    branch: str | None = None,
    metric_defs: dict[str, MetricDef] | None = None,
) -> Run:
    """Build a two-framework run used in exporter/comparison tests."""
    return _run(
        (
            Point(
                name=point_name,
                scenario=scenario,
                tags={"framework": first_framework},
                metrics={
                    "throughput": Metric(value=first_throughput),
                    "latency": Metric(value=first_latency),
                },
            ),
            Point(
                name=point_name,
                scenario=scenario,
                tags={"framework": second_framework},
                metrics={
                    "throughput": Metric(value=second_throughput),
                    "latency": Metric(value=second_latency),
                },
            ),
        ),
        run_id=run_id,
        commit=commit,
        branch=branch,
        metric_defs=metric_defs,
    )


def make_throughput_only_run(
    *,
    throughput: float,
    run_id: str = "run1",
    point_name: str = "bench1",
    scenario: str = "default",
    framework: str = "jax",
    commit: str | None = None,
    branch: str | None = None,
    metric_defs: dict[str, MetricDef] | None = None,
) -> Run:
    """Build a single-point run with throughput only."""
    return _run(
        (
            Point(
                name=point_name,
                scenario=scenario,
                tags={"framework": framework},
                metrics={"throughput": Metric(value=throughput)},
            ),
        ),
        run_id=run_id,
        commit=commit,
        branch=branch,
        metric_defs=metric_defs,
    )


def make_matmul_run(
    *,
    run_id: str = "run1",
    commit: str | None = None,
    branch: str | None = None,
    throughput: float = 100.0,
    latency: float = 5.0,
    metric_defs: dict[str, MetricDef] | None = None,
    timestamp: datetime | None = None,
    environment: Mapping[str, str] | None = None,
) -> Run:
    """Build a standard ``matmul/perf`` run used by exporter/store tests."""
    return make_single_framework_run(
        run_id=run_id,
        point_name="matmul",
        scenario="perf",
        throughput=throughput,
        latency=latency,
        commit=commit,
        branch=branch,
        metric_defs=metric_defs,
        timestamp=timestamp,
        environment=environment,
    )


def make_default_timing_sample(
    *,
    wall_clock_sec: float = 1.5,
    per_batch_times: tuple[float, ...] = (0.1, 0.2, 0.3),
    first_batch_time: float = 0.15,
    num_batches: int = 3,
    num_elements: int = 96,
    compilation_time_sec: float | None = None,
    warmup_batches_excluded: int = 0,
) -> TimingSample:
    """Build a representative TimingSample used across tests."""
    return TimingSample(
        wall_clock_sec=wall_clock_sec,
        per_batch_times=per_batch_times,
        first_batch_time=first_batch_time,
        num_batches=num_batches,
        num_elements=num_elements,
        compilation_time_sec=compilation_time_sec,
        warmup_batches_excluded=warmup_batches_excluded,
    )


def make_default_resource_summary(
    *,
    peak_rss_mb: float = 512.0,
    mean_rss_mb: float = 400.0,
    peak_gpu_mem_mb: float | None = None,
    mean_gpu_util: float | None = None,
    memory_growth_mb: float = 10.0,
    num_samples: int = 50,
    duration_sec: float = 5.0,
    mean_gpu_clock_mhz: float | None = None,
    mean_gpu_power_w: float | None = None,
) -> ResourceSummary:
    """Build a representative ResourceSummary used across tests."""
    return ResourceSummary(
        peak_rss_mb=peak_rss_mb,
        mean_rss_mb=mean_rss_mb,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        mean_gpu_util=mean_gpu_util,
        memory_growth_mb=memory_growth_mb,
        num_samples=num_samples,
        duration_sec=duration_sec,
        mean_gpu_clock_mhz=mean_gpu_clock_mhz,
        mean_gpu_power_w=mean_gpu_power_w,
    )


def make_empty_energy_summary() -> EnergySummary:
    """Build an empty EnergySummary with all optional metrics unset."""
    return EnergySummary(
        total_gpu_energy_joules=None,
        total_cpu_energy_joules=None,
        total_combined_energy_joules=None,
        mean_gpu_power_watts=None,
        peak_gpu_power_watts=None,
        duration_sec=0.0,
        num_samples=0,
    )


def make_cpu_hardware_config() -> HardwareConfig:
    """Build a canonical CPU HardwareConfig."""
    return HardwareConfig(
        platform="cpu",
        precision="float32",
        tile_size=64,
        critical_batch_size=32,
        memory_layout="row_major",
        use_vmem_optimization=False,
    )


class _SamplingMonitor(Protocol):
    """``ResourceMonitor`` and ``EnergyMonitor``: a context manager sampling in a thread."""

    def __enter__(self) -> object: ...

    def __exit__(self, *args: object) -> None: ...

    @property
    def samples(self) -> Sequence[object]: ...

    @property
    def _sampling_thread(self) -> SamplingThread: ...


def assert_monitor_collects_samples_twice(
    monitor: _SamplingMonitor,
    *,
    sleep_seconds: float = 0.15,
) -> None:
    """Assert that a monitor collects samples across two context cycles."""
    with monitor:
        time.sleep(sleep_seconds)
    first_count = len(monitor.samples)
    with monitor:
        time.sleep(sleep_seconds)
    second_count = len(monitor.samples)
    assert first_count > 0
    assert second_count > 0


def assert_monitor_thread_lifecycle(monitor: _SamplingMonitor) -> None:
    """Assert a monitor starts and stops its sampling thread in a context block."""
    assert monitor._sampling_thread._thread is None
    with monitor:
        assert monitor._sampling_thread._thread is not None
        assert monitor._sampling_thread._thread.is_alive()
    assert not monitor._sampling_thread._thread.is_alive()


@dataclass(frozen=True, slots=True, kw_only=True)
class FakeGpu:
    """A GPU profiler answering fixed readings; a field left ``None`` is a reading not taken."""

    memory_reading: GpuMemory | None = None
    utilization_reading: float | None = None
    clocks_reading: GpuClocks | None = None
    power_reading: GpuPower | None = None

    def memory(self) -> GpuMemory | None:
        """The fixed memory reading."""
        return self.memory_reading

    def utilization(self) -> float | None:
        """The fixed utilization reading."""
        return self.utilization_reading

    def clocks(self) -> GpuClocks | None:
        """The fixed clocks reading."""
        return self.clocks_reading

    def power(self) -> GpuPower | None:
        """The fixed power reading."""
        return self.power_reading


NVML_NOT_SUPPORTED = 3


class FakeNvmlError(Exception):
    """``pynvml.NVMLError``: carries the NVML return code as ``value``."""

    def __init__(self, value: int) -> None:
        super().__init__(value)
        self.value = value


def make_fake_nvml() -> types.SimpleNamespace:
    """The NVML calls ``NvmlDevice`` makes, answering for one GPU; ``calls`` records init/shutdown.

    Patch it over ``calibrax.profiling.nvml.pynvml`` to run NVML code without a GPU or driver.
    """
    calls: list[str] = []
    return types.SimpleNamespace(
        calls=calls,
        NVMLError=FakeNvmlError,
        NVML_ERROR_NOT_SUPPORTED=NVML_NOT_SUPPORTED,
        NVML_CLOCK_GRAPHICS=0,
        NVML_CLOCK_MEM=2,
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByIndex=lambda index: f"gpu{index}",
        nvmlDeviceGetMemoryInfo=lambda _h: types.SimpleNamespace(used=2 * 2**30, total=8 * 2**30),
        nvmlDeviceGetUtilizationRates=lambda _h: types.SimpleNamespace(gpu=73, memory=40),
        nvmlDeviceGetClockInfo=lambda _h, kind: {0: 1500, 2: 10_000}[kind],
        nvmlDeviceGetPowerUsage=lambda _h: 240_000,
        nvmlDeviceGetPowerManagementLimit=lambda _h: 450_000,
    )
