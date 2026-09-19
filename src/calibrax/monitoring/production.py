"""Production-grade monitoring with pipeline health tracking.

Extends AdvancedMonitor with performance baselines, pipeline execution
tracking, and health report generation.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from calibrax.core.record_values import Metadata
from calibrax.monitoring.monitor import AdvancedMonitor, AlertManager, AlertSeverity
from calibrax.profiling.resources import GPUProfilerProtocol, ResourceMonitor


# Error-rate thresholds for the health level and the pipeline alert.
_CRITICAL_ERROR_RATE = 0.5
_DEGRADED_ERROR_RATE = 0.2
_ALERT_ERROR_RATE = 0.3
_MIN_EXECUTIONS_FOR_ERROR_RATE = 3
# The error-rate alert looks at a pipeline's own most recent executions.
_ERROR_RATE_WINDOW = 20


logger = logging.getLogger(__name__)


class PipelineHealth(StrEnum):
    """A pipeline's health by its error rate."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineExecution:
    """One recorded pipeline execution.

    Attributes:
        pipeline_name: The pipeline that ran.
        execution_time: Wall-clock time in seconds.
        success: Whether it succeeded.
        timestamp: When it was recorded, in seconds since the epoch.
        metadata: The caller's context for the run.
    """

    pipeline_name: str
    execution_time: float
    success: bool
    timestamp: float
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineStats:
    """One pipeline's executions summarised.

    Attributes:
        total_executions: Executions recorded.
        success_rate: Fraction that succeeded.
        error_rate: Fraction that failed.
        mean_execution_time: Mean wall-clock time in seconds.
        min_execution_time: Shortest wall-clock time in seconds.
        max_execution_time: Longest wall-clock time in seconds.
        health: The health level the error rate places it at.
    """

    total_executions: int
    success_rate: float
    error_rate: float
    mean_execution_time: float
    min_execution_time: float
    max_execution_time: float
    health: PipelineHealth


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineHealthReport:
    """Every tracked pipeline's statistics and the overall health.

    Attributes:
        pipelines: Each pipeline's statistics by name.
        overall_health: ``DEGRADED`` when any pipeline is not healthy.
        baselines: The performance baseline of each metric.
        total_executions: Executions recorded across pipelines.
    """

    pipelines: dict[str, PipelineStats]
    overall_health: PipelineHealth
    baselines: dict[str, float]
    total_executions: int


def _classify_health(error_rate: float) -> PipelineHealth:
    """Classify pipeline health based on error rate.

    Args:
        error_rate: Fraction of failed executions.

    Returns:
        The health level.
    """
    if error_rate > _CRITICAL_ERROR_RATE:
        return PipelineHealth.CRITICAL
    if error_rate > _DEGRADED_ERROR_RATE:
        return PipelineHealth.DEGRADED
    return PipelineHealth.HEALTHY


@dataclass(slots=True)
class _PipelineTally:
    """One pipeline's running statistics over every execution, and its recent outcomes."""

    count: int = 0
    successes: int = 0
    total_time: float = 0.0
    min_time: float = math.inf
    max_time: float = -math.inf
    recent_successes: deque[bool] = field(default_factory=lambda: deque(maxlen=_ERROR_RATE_WINDOW))

    def add(self, execution_time: float, *, success: bool) -> None:
        """Count one execution."""
        self.count += 1
        self.successes += success
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.recent_successes.append(success)

    def stats(self) -> PipelineStats:
        """The statistics over every execution counted, at least one."""
        error_rate = 1.0 - self.successes / self.count
        return PipelineStats(
            total_executions=self.count,
            success_rate=self.successes / self.count,
            error_rate=error_rate,
            mean_execution_time=self.total_time / self.count,
            min_execution_time=self.min_time,
            max_execution_time=self.max_time,
            health=_classify_health(error_rate),
        )


class ProductionMonitor(AdvancedMonitor):
    """Extended monitor with pipeline health tracking and performance baselines.

    Tracks pipeline execution times, success rates, and detects performance degradation
    against configured baselines. Memory is bounded for a long-running process: the health
    report's statistics are running tallies over every execution, ``executions`` holds the
    most recent ``history_maxlen``, and the error-rate alert reads each pipeline's own most
    recent executions.
    """

    def __init__(
        self,
        alert_manager: AlertManager | None = None,
        gpu_profiler: GPUProfilerProtocol | None = None,
        resource_monitor: ResourceMonitor | None = None,
        *,
        history_maxlen: int = 100,
    ) -> None:
        """Initialize the production monitor.

        Args:
            alert_manager: Alert manager for dispatching alerts. Created if not provided.
            gpu_profiler: Optional GPU profiler for GPU metrics.
            resource_monitor: Optional ResourceMonitor for background sampling.
            history_maxlen: Most recent executions, and values per metric, kept.
        """
        super().__init__(
            alert_manager=alert_manager,
            gpu_profiler=gpu_profiler,
            resource_monitor=resource_monitor,
            history_maxlen=history_maxlen,
        )
        self._baselines: dict[str, float] = {}
        self._pipeline_executions: deque[PipelineExecution] = deque(maxlen=history_maxlen)
        self._tallies: dict[str, _PipelineTally] = {}
        self._degradation_threshold = 0.2

    @property
    def executions(self) -> tuple[PipelineExecution, ...]:
        """The most recent ``history_maxlen`` executions, oldest first."""
        with self._state_lock:
            return tuple(self._pipeline_executions)

    def set_performance_baseline(self, metric_name: str, baseline_value: float) -> None:
        """Set a performance baseline for degradation detection.

        Args:
            metric_name: Metric to track against baseline.
            baseline_value: Expected baseline value.
        """
        with self._state_lock:
            self._baselines[metric_name] = baseline_value

    def record_pipeline_execution(
        self,
        pipeline_name: str,
        execution_time: float,
        success: bool,
        metadata: Metadata | None = None,
    ) -> None:
        """Record a pipeline execution for health tracking.

        Args:
            pipeline_name: Name of the pipeline that executed.
            execution_time: Wall-clock execution time in seconds.
            success: Whether the execution succeeded.
            metadata: Optional additional context.
        """
        record = PipelineExecution(
            pipeline_name=pipeline_name,
            execution_time=float(execution_time),
            success=success,
            timestamp=time.time(),
            metadata=dict(metadata or {}),
        )
        with self._state_lock:
            self._pipeline_executions.append(record)
            self._tallies.setdefault(pipeline_name, _PipelineTally()).add(
                record.execution_time, success=success
            )

        self._check_performance_degradation(pipeline_name, execution_time)
        if not success:
            self._check_error_rate(pipeline_name)

    def get_pipeline_health_report(self) -> PipelineHealthReport:
        """Generate a health report across all tracked pipelines.

        Returns:
            Each pipeline's statistics, the overall health, the baselines and the total count.
        """
        with self._state_lock:
            pipelines = {name: tally.stats() for name, tally in self._tallies.items()}
            baselines_snapshot = dict(self._baselines)

        unhealthy = any(stats.health != PipelineHealth.HEALTHY for stats in pipelines.values())
        return PipelineHealthReport(
            pipelines=pipelines,
            overall_health=PipelineHealth.DEGRADED if unhealthy else PipelineHealth.HEALTHY,
            baselines=baselines_snapshot,
            total_executions=sum(stats.total_executions for stats in pipelines.values()),
        )

    def _check_performance_degradation(self, pipeline_name: str, execution_time: float) -> None:
        """Alert if execution time exceeds baseline by degradation threshold."""
        with self._state_lock:
            baseline = self._baselines.get(pipeline_name)
        if baseline is None or baseline == 0:
            return
        degradation = (execution_time - baseline) / baseline
        if degradation > self._degradation_threshold:
            self.alert_manager.trigger_alert(
                message=(
                    f"Pipeline '{pipeline_name}' degraded: "
                    f"{execution_time:.2f}s vs baseline {baseline:.2f}s "
                    f"({degradation:.0%} slower)"
                ),
                severity=AlertSeverity.WARNING,
                metric_name=f"pipeline_{pipeline_name}_time",
                metric_value=execution_time,
                threshold=baseline * (1 + self._degradation_threshold),
            )

    def _check_error_rate(self, pipeline_name: str) -> None:
        """Alert if recent error rate is too high for a pipeline."""
        with self._state_lock:
            recent = list(self._tallies[pipeline_name].recent_successes)
        if len(recent) < _MIN_EXECUTIONS_FOR_ERROR_RATE:
            return
        failures = recent.count(False)
        error_rate = failures / len(recent)
        if error_rate > _ALERT_ERROR_RATE:
            self.alert_manager.trigger_alert(
                message=(
                    f"Pipeline '{pipeline_name}' high error rate: "
                    f"{error_rate:.0%} ({failures}/{len(recent)} recent runs failed)"
                ),
                severity=AlertSeverity.ERROR,
                metric_name=f"pipeline_{pipeline_name}_error_rate",
                metric_value=error_rate,
                threshold=_ALERT_ERROR_RATE,
            )
