"""Shared test data builders used across test modules."""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from calibrax.core.models import Metric, MetricDef, MetricDirection, Point, Run
from calibrax.profiling.energy import EnergySummary
from calibrax.profiling.gpu import HardwareConfig
from calibrax.profiling.resources import ResourceSummary
from calibrax.profiling.timing import TimingSample


_CPU_HARDWARE_KWARGS: dict[str, str | int | bool] = {
    "platform": "cpu",
    "precision": "float32",
    "tile_size": 64,
    "critical_batch_size": 32,
    "memory_layout": "row_major",
    "use_vmem_optimization": False,
}


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
    run_kwargs: dict[str, object] = {"id": run_id}
    if commit is not None:
        run_kwargs["commit"] = commit
    if branch is not None:
        run_kwargs["branch"] = branch
    if timestamp is not None:
        run_kwargs["timestamp"] = timestamp
    if environment is not None:
        run_kwargs["environment"] = dict(environment)
    if metric_defs is not None:
        run_kwargs["metric_defs"] = metric_defs

    return Run(
        points=(
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
        **run_kwargs,
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
    run_kwargs: dict[str, object] = {"id": run_id}
    if commit is not None:
        run_kwargs["commit"] = commit
    if branch is not None:
        run_kwargs["branch"] = branch
    if metric_defs is not None:
        run_kwargs["metric_defs"] = metric_defs

    return Run(
        points=(
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
        **run_kwargs,
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
    run_kwargs: dict[str, object] = {"id": run_id}
    if commit is not None:
        run_kwargs["commit"] = commit
    if branch is not None:
        run_kwargs["branch"] = branch
    if metric_defs is not None:
        run_kwargs["metric_defs"] = metric_defs

    return Run(
        points=(
            Point(
                name=point_name,
                scenario=scenario,
                tags={"framework": framework},
                metrics={"throughput": Metric(value=throughput)},
            ),
        ),
        **run_kwargs,
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
    empty_values: dict[str, float | None] = {
        "total_gpu_energy_joules": None,
        "total_cpu_energy_joules": None,
        "total_combined_energy_joules": None,
        "mean_gpu_power_watts": None,
        "peak_gpu_power_watts": None,
    }
    return EnergySummary(duration_sec=0.0, num_samples=0, **empty_values)


def make_cpu_hardware_config() -> HardwareConfig:
    """Build a canonical CPU HardwareConfig."""
    return HardwareConfig(**_CPU_HARDWARE_KWARGS)


def assert_monitor_collects_samples_twice(
    monitor: Any,
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


def assert_monitor_thread_lifecycle(monitor: Any) -> None:
    """Assert a monitor starts and stops its sampling thread in a context block."""
    assert monitor._sampling_thread._thread is None
    with monitor:
        assert monitor._sampling_thread._thread is not None
        assert monitor._sampling_thread._thread.is_alive()
    assert not monitor._sampling_thread._thread.is_alive()
