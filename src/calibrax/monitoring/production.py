"""Production-grade monitoring with pipeline health tracking.

Extends AdvancedMonitor with performance baselines, pipeline execution
tracking, and health report generation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from calibrax.monitoring.monitor import AdvancedMonitor, AlertSeverity


logger = logging.getLogger(__name__)


def _classify_health(error_rate: float) -> str:
    """Classify pipeline health based on error rate.

    Args:
        error_rate: Fraction of failed executions.

    Returns:
        Health level string: "critical", "degraded", or "healthy".
    """
    if error_rate > 0.5:
        return "critical"
    if error_rate > 0.2:
        return "degraded"
    return "healthy"


def _compute_pipeline_stats(executions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute statistics for a single pipeline's executions.

    Args:
        executions: List of execution records for one pipeline.

    Returns:
        Dictionary with success rate, timing stats, and health status.
    """
    times = [e["execution_time"] for e in executions]
    successes = sum(1 for e in executions if e["success"])
    total = len(executions)
    error_rate = 1.0 - (successes / total) if total > 0 else 0.0

    return {
        "total_executions": total,
        "success_rate": float(successes / total) if total > 0 else 0.0,
        "error_rate": float(error_rate),
        "mean_execution_time": float(sum(times) / len(times)) if times else 0.0,
        "min_execution_time": float(min(times)) if times else 0.0,
        "max_execution_time": float(max(times)) if times else 0.0,
        "health": _classify_health(error_rate),
    }


class ProductionMonitor(AdvancedMonitor):
    """Extended monitor with pipeline health tracking and performance baselines.

    Tracks pipeline execution times, success rates, and detects performance
    degradation against configured baselines.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the production monitor."""
        super().__init__(**kwargs)
        self._baselines: dict[str, float] = {}
        self._pipeline_executions: list[dict[str, Any]] = []
        self._degradation_threshold = 0.2

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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a pipeline execution for health tracking.

        Args:
            pipeline_name: Name of the pipeline that executed.
            execution_time: Wall-clock execution time in seconds.
            success: Whether the execution succeeded.
            metadata: Optional additional context.
        """
        record: dict[str, Any] = {
            "pipeline_name": pipeline_name,
            "execution_time": float(execution_time),
            "success": success,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        with self._state_lock:
            self._pipeline_executions.append(record)

        self._check_performance_degradation(pipeline_name, execution_time)
        if not success:
            self._check_error_rate(pipeline_name)

    def get_pipeline_health_report(self) -> dict[str, Any]:
        """Generate a health report across all tracked pipelines.

        Returns:
            Dictionary with per-pipeline statistics and overall health status.
        """
        with self._state_lock:
            executions_snapshot = list(self._pipeline_executions)
            baselines_snapshot = dict(self._baselines)

        pipelines: dict[str, list[dict[str, Any]]] = {}
        for execution in executions_snapshot:
            name = execution["pipeline_name"]
            if name not in pipelines:
                pipelines[name] = []
            pipelines[name].append(execution)

        report: dict[str, Any] = {"pipelines": {}, "overall_health": "healthy"}
        has_unhealthy = False

        for name, executions in pipelines.items():
            stats = _compute_pipeline_stats(executions)
            report["pipelines"][name] = stats
            if stats["health"] != "healthy":
                has_unhealthy = True

        if has_unhealthy:
            report["overall_health"] = "degraded"
        report["baselines"] = baselines_snapshot
        report["total_executions"] = len(executions_snapshot)

        return report

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
            recent_window = self._pipeline_executions[-20:]
            recent = [e for e in recent_window if e["pipeline_name"] == pipeline_name]
        if len(recent) < 3:
            return
        failures = sum(1 for e in recent if not e["success"])
        error_rate = failures / len(recent)
        if error_rate > 0.3:
            self.alert_manager.trigger_alert(
                message=(
                    f"Pipeline '{pipeline_name}' high error rate: "
                    f"{error_rate:.0%} ({failures}/{len(recent)} recent runs failed)"
                ),
                severity=AlertSeverity.ERROR,
                metric_name=f"pipeline_{pipeline_name}_error_rate",
                metric_value=error_rate,
                threshold=0.3,
            )
