"""Tests for calibrax.monitoring.production module."""

from __future__ import annotations

from calibrax.monitoring.monitor import AlertSeverity
from calibrax.monitoring.production import ProductionMonitor


class TestProductionMonitor:
    """Tests for ProductionMonitor."""

    def test_set_performance_baseline(self) -> None:
        """Baselines should be stored for later degradation checks."""
        monitor = ProductionMonitor()
        monitor.set_performance_baseline("pipeline_a", 2.0)
        report = monitor.get_pipeline_health_report()
        assert report["baselines"]["pipeline_a"] == 2.0

    def test_record_pipeline_execution(self) -> None:
        """Recorded executions should appear in health report."""
        monitor = ProductionMonitor()
        monitor.record_pipeline_execution("train", 5.0, success=True)
        monitor.record_pipeline_execution("train", 6.0, success=True)
        report = monitor.get_pipeline_health_report()
        assert report["pipelines"]["train"]["total_executions"] == 2
        assert report["pipelines"]["train"]["success_rate"] == 1.0

    def test_degradation_alert(self) -> None:
        """Performance degradation beyond threshold should trigger alert."""
        monitor = ProductionMonitor()
        monitor.set_performance_baseline("train", 2.0)
        monitor.record_pipeline_execution("train", 10.0, success=True)
        alerts = monitor.alert_manager.get_recent_alerts()
        degradation_alerts = [a for a in alerts if "degraded" in a.message]
        assert len(degradation_alerts) > 0

    def test_no_degradation_alert_within_threshold(self) -> None:
        """Normal execution should not trigger degradation alert."""
        monitor = ProductionMonitor()
        monitor.set_performance_baseline("train", 2.0)
        monitor.record_pipeline_execution("train", 2.1, success=True)
        alerts = monitor.alert_manager.get_recent_alerts()
        degradation_alerts = [a for a in alerts if "degraded" in a.message]
        assert len(degradation_alerts) == 0

    def test_error_rate_alert(self) -> None:
        """High failure rate should trigger error rate alert."""
        monitor = ProductionMonitor()
        for _ in range(3):
            monitor.record_pipeline_execution("infer", 1.0, success=False)
        alerts = monitor.alert_manager.get_recent_alerts()
        error_alerts = [a for a in alerts if "error rate" in a.message]
        assert len(error_alerts) > 0

    def test_pipeline_health_status_critical(self) -> None:
        """Pipeline with >50% error rate should be marked critical."""
        monitor = ProductionMonitor()
        monitor.record_pipeline_execution("bad", 1.0, success=False)
        monitor.record_pipeline_execution("bad", 1.0, success=False)
        monitor.record_pipeline_execution("bad", 1.0, success=True)
        report = monitor.get_pipeline_health_report()
        assert report["pipelines"]["bad"]["health"] == "critical"

    def test_pipeline_health_status_healthy(self) -> None:
        """All-success pipeline should be healthy."""
        monitor = ProductionMonitor()
        for _ in range(5):
            monitor.record_pipeline_execution("good", 1.0, success=True)
        report = monitor.get_pipeline_health_report()
        assert report["pipelines"]["good"]["health"] == "healthy"

    def test_overall_health_degrades(self) -> None:
        """Overall health should reflect worst pipeline status."""
        monitor = ProductionMonitor()
        for _ in range(5):
            monitor.record_pipeline_execution("good", 1.0, success=True)
        for _ in range(3):
            monitor.record_pipeline_execution("bad", 1.0, success=False)
        report = monitor.get_pipeline_health_report()
        assert report["overall_health"] == "degraded"

    def test_empty_report(self) -> None:
        """Health report should be valid with no executions."""
        monitor = ProductionMonitor()
        report = monitor.get_pipeline_health_report()
        assert report["total_executions"] == 0
        assert report["overall_health"] == "healthy"

    def test_execution_metadata(self) -> None:
        """Metadata should be stored with execution records."""
        monitor = ProductionMonitor()
        monitor.record_pipeline_execution("train", 3.0, success=True, metadata={"batch_size": 32})
        report = monitor.get_pipeline_health_report()
        assert report["total_executions"] == 1

    def test_degradation_severity_is_warning(self) -> None:
        """Degradation alerts should have WARNING severity."""
        monitor = ProductionMonitor()
        monitor.set_performance_baseline("train", 1.0)
        monitor.record_pipeline_execution("train", 5.0, success=True)
        alerts = monitor.alert_manager.get_recent_alerts()
        degradation_alerts = [a for a in alerts if "degraded" in a.message]
        assert degradation_alerts[0].severity == AlertSeverity.WARNING

    def test_error_rate_severity_is_error(self) -> None:
        """Error rate alerts should have ERROR severity."""
        monitor = ProductionMonitor()
        for _ in range(5):
            monitor.record_pipeline_execution("infer", 1.0, success=False)
        alerts = monitor.alert_manager.get_recent_alerts()
        error_alerts = [a for a in alerts if "error rate" in a.message]
        assert error_alerts[0].severity == AlertSeverity.ERROR

    def test_inherits_from_advanced_monitor(self) -> None:
        """ProductionMonitor should have all AdvancedMonitor capabilities."""
        monitor = ProductionMonitor()
        monitor.set_threshold("cpu_percent", 90.0)
        summary = monitor.get_monitoring_summary()
        assert summary["thresholds"]["cpu_percent"] == 90.0
