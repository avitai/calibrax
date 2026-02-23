"""Alert management and background metric monitoring.

Provides threshold-based alerting with configurable handlers and background
monitoring of system resources via daemon thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import psutil  # pyright: ignore[reportMissingModuleSource]

from calibrax.profiling.resources import GPUProfilerProtocol, ResourceMonitor


logger = logging.getLogger(__name__)


class AlertSeverity(StrEnum):
    """Severity levels for monitoring alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True, kw_only=True)
class Alert:
    """A single monitoring alert triggered by a threshold violation.

    Attributes:
        message: Human-readable description of the alert.
        severity: Alert severity level.
        metric_name: Name of the metric that triggered the alert.
        metric_value: Observed value that triggered the alert.
        threshold: Threshold that was exceeded.
        timestamp: When the alert was triggered.
        metadata: Additional context about the alert.
    """

    message: str
    severity: AlertSeverity
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "message": self.message,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "metric_value": float(self.metric_value),
            "threshold": float(self.threshold),
            "timestamp": float(self.timestamp),
            "metadata": dict(self.metadata),
        }


class AlertManager:
    """Thread-safe alert storage with callback handlers.

    Args:
        max_alerts: Maximum number of alerts to retain (oldest dropped first).
    """

    def __init__(self, max_alerts: int = 1000) -> None:
        """Initialize the alert manager."""
        self._alerts: deque[Alert] = deque(maxlen=max_alerts)
        self._handlers: list[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a callback invoked on each new alert.

        Args:
            handler: Callable that receives an Alert instance.
        """
        self._handlers.append(handler)

    def trigger_alert(
        self,
        message: str,
        severity: AlertSeverity,
        metric_name: str,
        metric_value: float,
        threshold: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create and store an alert, notifying all registered handlers.

        Args:
            message: Human-readable alert description.
            severity: Severity level.
            metric_name: Metric that triggered the alert.
            metric_value: Observed metric value.
            threshold: Threshold that was exceeded.
            metadata: Optional additional context.
        """
        alert = Alert(
            message=message,
            severity=severity,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            metadata=metadata or {},
        )
        with self._lock:
            self._alerts.append(alert)

        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:  # noqa: BLE001 - isolate third-party handlers from monitor core
                logger.exception("Alert handler raised an exception")

    def get_recent_alerts(self, count: int = 10) -> list[Alert]:
        """Return the most recent alerts.

        Args:
            count: Maximum number of alerts to return.

        Returns:
            List of recent alerts, newest first.
        """
        with self._lock:
            alerts = list(self._alerts)
        return list(reversed(alerts[-count:]))

    def get_alerts_by_severity(self, severity: AlertSeverity) -> list[Alert]:
        """Return all alerts matching the given severity.

        Args:
            severity: Severity level to filter by.

        Returns:
            List of matching alerts.
        """
        with self._lock:
            return [a for a in self._alerts if a.severity == severity]

    def clear_alerts(self) -> None:
        """Remove all stored alerts."""
        with self._lock:
            self._alerts.clear()


class AdvancedMonitor:
    """Background resource monitor with threshold-based alerting.

    Collects CPU, memory, and optional GPU metrics on a daemon thread.
    Triggers alerts when thresholds are exceeded.

    Args:
        alert_manager: Alert manager for dispatching alerts. Created if not provided.
        gpu_profiler: Optional GPU profiler for GPU metrics.
        resource_monitor: Optional ResourceMonitor for background sampling.
    """

    def __init__(
        self,
        alert_manager: AlertManager | None = None,
        gpu_profiler: GPUProfilerProtocol | None = None,
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        """Initialize the monitor."""
        self._alert_manager = alert_manager or AlertManager()
        self._gpu_profiler = gpu_profiler
        self._resource_monitor = resource_monitor
        self._thresholds: dict[str, float] = {}
        self._metric_history: dict[str, deque[float]] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._history_maxlen = 100
        self._state_lock = threading.RLock()

    @property
    def alert_manager(self) -> AlertManager:
        """Access the underlying alert manager."""
        return self._alert_manager

    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """Set an alerting threshold for a metric.

        Args:
            metric_name: Name of the metric to monitor.
            threshold: Value above which an alert is triggered.
        """
        with self._state_lock:
            self._thresholds[metric_name] = threshold

    def start_monitoring(self, interval: float = 5.0) -> None:
        """Start background monitoring on a daemon thread.

        Args:
            interval: Seconds between metric collection cycles.
        """
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True,
            )
            self._thread = thread
        thread.start()
        logger.info("Monitoring started with %.1fs interval", interval)

    def stop_monitoring(self) -> None:
        """Stop background monitoring and wait for the thread to finish."""
        self._stop_event.set()
        with self._state_lock:
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=5.0)
        logger.info("Monitoring stopped")

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Return a summary of current monitoring state.

        Returns:
            Dictionary with thresholds, alert counts, and metric history summaries.
        """
        alerts = self._alert_manager.get_recent_alerts(count=100)
        with self._state_lock:
            thresholds_snapshot = dict(self._thresholds)
            history_snapshot = {name: list(values) for name, values in self._metric_history.items()}
            is_monitoring = self._thread is not None and self._thread.is_alive()

        history_summary: dict[str, dict[str, float]] = {}
        for name, values in history_snapshot.items():
            if values:
                history_summary[name] = {
                    "latest": float(values[-1]),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "mean": float(sum(values) / len(values)),
                    "samples": len(values),
                }
        return {
            "thresholds": thresholds_snapshot,
            "alert_count": len(alerts),
            "metric_history": history_summary,
            "is_monitoring": is_monitoring,
        }

    def _monitor_loop(self, interval: float) -> None:
        """Collect metrics and check thresholds on a loop until stopped."""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                self._check_thresholds(metrics)
            except Exception:  # noqa: BLE001 - loop boundary must not crash monitoring thread
                logger.exception("Error during monitoring cycle")
            self._stop_event.wait(timeout=interval)

    def _collect_metrics(self) -> dict[str, float]:
        """Collect system metrics from available sources.

        Returns:
            Dictionary of metric names to current values.
        """
        metrics: dict[str, float] = {}

        try:
            process = psutil.Process()
            metrics["cpu_percent"] = process.cpu_percent()
            metrics["memory_rss_mb"] = process.memory_info().rss / (1024 * 1024)
        except psutil.Error:
            logger.debug("Failed to collect process metrics")

        if self._gpu_profiler is not None:
            try:
                metrics["gpu_utilization"] = self._gpu_profiler.get_utilization()
                mem = self._gpu_profiler.get_memory_usage()
                gpu_mem = mem.get("gpu_memory_used_mb")
                if gpu_mem is not None:
                    metrics["gpu_memory_mb"] = gpu_mem
            except (AttributeError, TypeError, ValueError, RuntimeError):
                logger.debug("Failed to collect GPU metrics")

        with self._state_lock:
            for name, value in metrics.items():
                if name not in self._metric_history:
                    self._metric_history[name] = deque(maxlen=self._history_maxlen)
                self._metric_history[name].append(value)

        return metrics

    def _check_thresholds(self, metrics: dict[str, float]) -> None:
        """Check collected metrics against configured thresholds."""
        with self._state_lock:
            thresholds = list(self._thresholds.items())

        for name, threshold in thresholds:
            value = metrics.get(name)
            if value is None:
                continue
            if value > threshold:
                severity = self._determine_alert_severity(value, threshold)
                trend = self._analyze_trend(name)
                self._alert_manager.trigger_alert(
                    message=(
                        f"{name} exceeded threshold: {value:.2f} > {threshold:.2f} (trend: {trend})"
                    ),
                    severity=severity,
                    metric_name=name,
                    metric_value=value,
                    threshold=threshold,
                    metadata={"trend": trend},
                )

    def _determine_alert_severity(self, value: float, threshold: float) -> AlertSeverity:
        """Determine alert severity based on how far value exceeds threshold."""
        if threshold == 0:
            return AlertSeverity.CRITICAL
        ratio = value / threshold
        if ratio > 2.0:
            return AlertSeverity.CRITICAL
        if ratio > 1.5:
            return AlertSeverity.ERROR
        if ratio > 1.2:
            return AlertSeverity.WARNING
        return AlertSeverity.INFO

    def _analyze_trend(self, metric_name: str) -> str:
        """Analyze recent trend direction for a metric.

        Args:
            metric_name: Metric to analyze.

        Returns:
            One of "increasing", "decreasing", or "stable".
        """
        with self._state_lock:
            history = self._metric_history.get(metric_name)
            if not history or len(history) < 3:
                return "stable"
            values = list(history)
        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator
        relative_slope = slope / (abs(y_mean) + 1e-8)
        if relative_slope > 0.01:
            return "increasing"
        if relative_slope < -0.01:
            return "decreasing"
        return "stable"
