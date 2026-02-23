"""Tests for calibrax.monitoring.monitor module."""

from __future__ import annotations

import logging
import time
from collections import deque
from unittest.mock import MagicMock

import psutil  # pyright: ignore[reportMissingModuleSource]
import pytest

from calibrax.monitoring.monitor import (
    AdvancedMonitor,
    Alert,
    AlertManager,
    AlertSeverity,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_values(self) -> None:
        """All severity levels should have string values."""
        assert AlertSeverity.INFO == "info"
        assert AlertSeverity.WARNING == "warning"
        assert AlertSeverity.ERROR == "error"
        assert AlertSeverity.CRITICAL == "critical"

    def test_is_str(self) -> None:
        """AlertSeverity should be usable as a string."""
        assert isinstance(AlertSeverity.INFO, str)


class TestAlert:
    """Tests for Alert dataclass."""

    def test_creation(self) -> None:
        """Alert should store all provided fields."""
        alert = Alert(
            message="test",
            severity=AlertSeverity.WARNING,
            metric_name="cpu",
            metric_value=95.0,
            threshold=80.0,
        )
        assert alert.message == "test"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.metric_value == 95.0

    def test_to_dict(self) -> None:
        """to_dict should produce JSON-serializable output."""
        alert = Alert(
            message="high cpu",
            severity=AlertSeverity.ERROR,
            metric_name="cpu_percent",
            metric_value=99.5,
            threshold=80.0,
            metadata={"node": "gpu-1"},
        )
        d = alert.to_dict()
        assert d["severity"] == "error"
        assert d["metric_value"] == 99.5
        assert d["metadata"]["node"] == "gpu-1"
        assert isinstance(d["timestamp"], float)

    def test_frozen(self) -> None:
        """Alert should be immutable."""
        alert = Alert(
            message="test",
            severity=AlertSeverity.INFO,
            metric_name="mem",
            metric_value=50.0,
            threshold=80.0,
        )
        with pytest.raises(AttributeError):
            alert.message = "changed"  # type: ignore[misc]


class TestAlertManager:
    """Tests for AlertManager."""

    def test_trigger_and_retrieve(self) -> None:
        """Triggered alerts should be retrievable."""
        manager = AlertManager()
        manager.trigger_alert(
            message="test alert",
            severity=AlertSeverity.WARNING,
            metric_name="cpu",
            metric_value=90.0,
            threshold=80.0,
        )
        alerts = manager.get_recent_alerts()
        assert len(alerts) == 1
        assert alerts[0].message == "test alert"

    def test_max_alerts_respected(self) -> None:
        """Alert count should not exceed max_alerts."""
        manager = AlertManager(max_alerts=5)
        for i in range(10):
            manager.trigger_alert(
                message=f"alert {i}",
                severity=AlertSeverity.INFO,
                metric_name="cpu",
                metric_value=float(i),
                threshold=0.0,
            )
        alerts = manager.get_recent_alerts(count=100)
        assert len(alerts) == 5

    def test_handler_called(self) -> None:
        """Registered handlers should receive alerts."""
        manager = AlertManager()
        received: list[Alert] = []
        manager.add_alert_handler(received.append)
        manager.trigger_alert(
            message="callback test",
            severity=AlertSeverity.INFO,
            metric_name="mem",
            metric_value=50.0,
            threshold=80.0,
        )
        assert len(received) == 1
        assert received[0].message == "callback test"

    def test_handler_exception_does_not_crash(self) -> None:
        """A failing handler should not prevent alert storage."""
        manager = AlertManager()

        def bad_handler(alert: Alert) -> None:
            msg = "handler error"
            raise RuntimeError(msg)

        manager.add_alert_handler(bad_handler)
        manager.trigger_alert(
            message="survives handler error",
            severity=AlertSeverity.INFO,
            metric_name="test",
            metric_value=1.0,
            threshold=0.0,
        )
        assert len(manager.get_recent_alerts()) == 1

    def test_get_alerts_by_severity(self) -> None:
        """Should filter alerts by severity."""
        manager = AlertManager()
        manager.trigger_alert(
            message="info",
            severity=AlertSeverity.INFO,
            metric_name="a",
            metric_value=1.0,
            threshold=0.0,
        )
        manager.trigger_alert(
            message="error",
            severity=AlertSeverity.ERROR,
            metric_name="b",
            metric_value=2.0,
            threshold=0.0,
        )
        errors = manager.get_alerts_by_severity(AlertSeverity.ERROR)
        assert len(errors) == 1
        assert errors[0].message == "error"

    def test_clear_alerts(self) -> None:
        """clear_alerts should remove all alerts."""
        manager = AlertManager()
        manager.trigger_alert(
            message="to be cleared",
            severity=AlertSeverity.INFO,
            metric_name="a",
            metric_value=1.0,
            threshold=0.0,
        )
        manager.clear_alerts()
        assert len(manager.get_recent_alerts()) == 0

    def test_recent_alerts_ordered_newest_first(self) -> None:
        """get_recent_alerts should return newest first."""
        manager = AlertManager()
        for i in range(5):
            manager.trigger_alert(
                message=f"alert-{i}",
                severity=AlertSeverity.INFO,
                metric_name="a",
                metric_value=float(i),
                threshold=0.0,
            )
        alerts = manager.get_recent_alerts(count=3)
        assert alerts[0].message == "alert-4"
        assert alerts[2].message == "alert-2"


class TestAdvancedMonitor:
    """Tests for AdvancedMonitor."""

    def test_set_threshold(self) -> None:
        """set_threshold should register thresholds."""
        monitor = AdvancedMonitor()
        monitor.set_threshold("cpu_percent", 80.0)
        summary = monitor.get_monitoring_summary()
        assert summary["thresholds"]["cpu_percent"] == 80.0

    def test_start_stop_monitoring(self) -> None:
        """Start and stop should manage daemon thread lifecycle."""
        monitor = AdvancedMonitor()
        monitor.start_monitoring(interval=0.1)
        time.sleep(0.3)
        summary = monitor.get_monitoring_summary()
        assert summary["is_monitoring"] is True
        monitor.stop_monitoring()
        summary = monitor.get_monitoring_summary()
        assert summary["is_monitoring"] is False

    def test_threshold_triggers_alert(self) -> None:
        """Exceeding a threshold should produce an alert."""
        manager = AlertManager()
        monitor = AdvancedMonitor(alert_manager=manager)
        # memory_rss_mb is always > 0, so threshold of 0.0 guarantees trigger
        monitor.set_threshold("memory_rss_mb", 0.0001)
        monitor.start_monitoring(interval=0.1)
        time.sleep(0.5)
        monitor.stop_monitoring()
        alerts = manager.get_recent_alerts(count=100)
        mem_alerts = [a for a in alerts if a.metric_name == "memory_rss_mb"]
        assert len(mem_alerts) > 0

    def test_monitoring_summary_structure(self) -> None:
        """Summary should contain expected keys."""
        monitor = AdvancedMonitor()
        summary = monitor.get_monitoring_summary()
        assert "thresholds" in summary
        assert "alert_count" in summary
        assert "metric_history" in summary
        assert "is_monitoring" in summary

    def test_double_start_is_idempotent(self) -> None:
        """Starting twice should not create duplicate threads."""
        monitor = AdvancedMonitor()
        monitor.start_monitoring(interval=0.1)
        monitor.start_monitoring(interval=0.1)
        time.sleep(0.2)
        monitor.stop_monitoring()

    def test_stop_monitoring_without_active_thread_is_noop(self) -> None:
        """Stopping before start should not raise and should remain stopped."""
        monitor = AdvancedMonitor()
        monitor.stop_monitoring()

        summary = monitor.get_monitoring_summary()
        assert summary["is_monitoring"] is False

    def test_gpu_profiler_integration(self) -> None:
        """GPU profiler metrics should be collected when available."""
        gpu_mock = MagicMock()
        gpu_mock.get_utilization.return_value = 75.0
        gpu_mock.get_memory_usage.return_value = {"gpu_memory_used_mb": 4096.0}

        monitor = AdvancedMonitor(gpu_profiler=gpu_mock)
        monitor.start_monitoring(interval=0.1)
        time.sleep(0.3)
        monitor.stop_monitoring()

        summary = monitor.get_monitoring_summary()
        assert "gpu_utilization" in summary["metric_history"]

    def test_gpu_profiler_malformed_memory_payload_is_ignored(self) -> None:
        """Malformed GPU memory payload should not break metrics collection."""
        gpu_mock = MagicMock()
        gpu_mock.get_utilization.return_value = 75.0
        gpu_mock.get_memory_usage.return_value = 4096.0  # Not a dict

        monitor = AdvancedMonitor(gpu_profiler=gpu_mock)
        metrics = monitor._collect_metrics()

        assert metrics["gpu_utilization"] == 75.0
        assert "gpu_memory_mb" not in metrics

    def test_gpu_profiler_missing_memory_key_does_not_emit_gpu_memory(self) -> None:
        """Missing gpu_memory_used_mb should not add gpu_memory_mb metric."""
        gpu_mock = MagicMock()
        gpu_mock.get_utilization.return_value = 40.0
        gpu_mock.get_memory_usage.return_value = {}

        monitor = AdvancedMonitor(gpu_profiler=gpu_mock)
        metrics = monitor._collect_metrics()

        assert metrics["gpu_utilization"] == 40.0
        assert "gpu_memory_mb" not in metrics

    def test_determine_alert_severity_levels(self) -> None:
        """Severity should escalate based on value/threshold ratio."""
        monitor = AdvancedMonitor()
        assert monitor._determine_alert_severity(85.0, 80.0) == AlertSeverity.INFO
        assert monitor._determine_alert_severity(100.0, 80.0) == AlertSeverity.WARNING
        assert monitor._determine_alert_severity(130.0, 80.0) == AlertSeverity.ERROR
        assert monitor._determine_alert_severity(200.0, 80.0) == AlertSeverity.CRITICAL

    def test_analyze_trend_stable(self) -> None:
        """Constant values should give 'stable' trend."""
        monitor = AdvancedMonitor()
        monitor._metric_history["test"] = deque([50.0] * 10, maxlen=100)
        assert monitor._analyze_trend("test") == "stable"

    def test_analyze_trend_increasing(self) -> None:
        """Linearly increasing values should give 'increasing' trend."""
        monitor = AdvancedMonitor()
        monitor._metric_history["test"] = deque([float(i) for i in range(20)], maxlen=100)
        assert monitor._analyze_trend("test") == "increasing"

    def test_analyze_trend_decreasing(self) -> None:
        """Linearly decreasing values should give 'decreasing' trend."""
        monitor = AdvancedMonitor()
        monitor._metric_history["test"] = deque([float(20 - i) for i in range(20)], maxlen=100)
        assert monitor._analyze_trend("test") == "decreasing"

    def test_analyze_trend_insufficient_data(self) -> None:
        """Fewer than 3 samples should give 'stable'."""
        monitor = AdvancedMonitor()
        monitor._metric_history["test"] = deque([1.0, 2.0], maxlen=100)
        assert monitor._analyze_trend("test") == "stable"

    def test_monitoring_summary_skips_empty_history_deques(self) -> None:
        """Summary should not include metrics that have empty history."""
        monitor = AdvancedMonitor()
        monitor._metric_history["empty"] = deque(maxlen=100)

        summary = monitor.get_monitoring_summary()

        assert "empty" not in summary["metric_history"]

    def test_check_thresholds_is_stable_when_thresholds_mutate(self) -> None:
        """Threshold iteration should be robust to runtime threshold updates."""
        monitor = AdvancedMonitor()
        monitor.set_threshold("cpu_percent", 10.0)

        def _mutating_analyze_trend(_metric_name: str) -> str:
            monitor.set_threshold("memory_rss_mb", 10.0)
            return "stable"

        monitor._analyze_trend = _mutating_analyze_trend  # type: ignore[method-assign]

        monitor._check_thresholds({"cpu_percent": 99.0})

    def test_check_thresholds_does_not_alert_when_value_below_threshold(self) -> None:
        """Values at/below threshold should not emit alerts."""
        manager = AlertManager()
        monitor = AdvancedMonitor(alert_manager=manager)
        monitor.set_threshold("cpu_percent", 80.0)

        monitor._check_thresholds({"cpu_percent": 20.0})

        assert manager.get_recent_alerts() == []

    def test_collect_metrics_handles_psutil_errors(self) -> None:
        """psutil collection errors should be handled without raising."""
        monitor = AdvancedMonitor()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "calibrax.monitoring.monitor.psutil.Process",
                lambda: (_ for _ in ()).throw(psutil.Error("boom")),
            )
            metrics = monitor._collect_metrics()

        assert "cpu_percent" not in metrics
        assert "memory_rss_mb" not in metrics

    def test_monitor_loop_logs_collection_exceptions(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Monitor loop should catch cycle exceptions and continue safely."""
        monitor = AdvancedMonitor()

        def _raise_once() -> dict[str, float]:
            monitor._stop_event.set()
            raise RuntimeError("cycle failed")

        monitor._collect_metrics = _raise_once  # type: ignore[method-assign]

        with caplog.at_level(logging.ERROR):
            monitor._monitor_loop(interval=0.0)

        assert "Error during monitoring cycle" in caplog.text

    def test_determine_alert_severity_zero_threshold(self) -> None:
        """Zero threshold should always classify as CRITICAL."""
        monitor = AdvancedMonitor()
        assert monitor._determine_alert_severity(1.0, 0.0) == AlertSeverity.CRITICAL
