# Production Monitoring

Calibrax provides a monitoring stack for tracking metrics at runtime, triggering
alerts when thresholds are exceeded, and generating pipeline health reports.

## Alert Management

`AlertManager` collects alerts and dispatches them to registered handlers:

```python
from calibrax.monitoring.monitor import AlertManager, AlertSeverity

manager = AlertManager(max_alerts=1000)

# Register a handler (any callable that accepts an Alert)
def log_alert(alert):
    print(f"[{alert.severity.value}] {alert.message}: "
          f"{alert.metric_name}={alert.metric_value:.2f} "
          f"(threshold={alert.threshold:.2f})")

manager.add_alert_handler(log_alert)

# Trigger an alert
manager.trigger_alert(
    message="Throughput below minimum",
    severity=AlertSeverity.WARNING,
    metric_name="throughput",
    metric_value=850.0,
    threshold=1000.0,
)
```

```text
[warning] Throughput below minimum: throughput=850.00 (threshold=1000.00)
```

### Alert Severity Levels

| Severity | Use Case |
|----------|----------|
| `INFO` | Informational — metric crossed a soft threshold |
| `WARNING` | Performance degraded but within tolerance |
| `ERROR` | Significant regression requiring investigation |
| `CRITICAL` | Service-impacting issue requiring immediate action |

### Querying Alerts

```python
# Most recent alerts
recent = manager.get_recent_alerts(count=5)

# Filter by severity
errors = manager.get_alerts_by_severity(AlertSeverity.ERROR)

# Clear all alerts
manager.clear_alerts()
```

## Threshold-Based Monitoring

`AdvancedMonitor` runs a background thread that periodically checks metric values
against configured thresholds:

```python
from calibrax.monitoring.monitor import AdvancedMonitor

monitor = AdvancedMonitor()

# Set thresholds
monitor.set_threshold("latency", 1.5)       # alert if latency > 1.5
monitor.set_threshold("throughput", 500.0)   # alert if throughput > 500

# Start background monitoring (checks every 5 seconds)
monitor.start_monitoring(interval=5.0)

# ... run workloads ...

# Stop monitoring and get summary
monitor.stop_monitoring()
summary = monitor.get_monitoring_summary()
print(f"Thresholds: {summary['thresholds']}")
print(f"Alert count: {summary['alert_count']}")
for metric, history in summary["metric_history"].items():
    print(f"  {metric}: latest={history['latest']:.2f}, "
          f"min={history['min']:.2f}, max={history['max']:.2f}")
```

`AdvancedMonitor` optionally accepts a `GPUProfilerProtocol` and a
`ResourceMonitor` to include GPU and resource metrics in the monitoring loop.

## Production Monitor

`ProductionMonitor` extends `AdvancedMonitor` with pipeline execution tracking
and health reporting:

```python
from calibrax.monitoring.production import ProductionMonitor

monitor = ProductionMonitor()

# Set performance baselines
monitor.set_performance_baseline("inference_latency", 0.05)

# Record pipeline executions
monitor.record_pipeline_execution(
    pipeline_name="inference",
    execution_time=0.048,
    success=True,
)

monitor.record_pipeline_execution(
    pipeline_name="inference",
    execution_time=0.150,
    success=False,
    metadata={"error": "timeout"},
)

# Get health report
report = monitor.get_pipeline_health_report()
print(f"Overall health: {report['overall_health']}")
for name, stats in report["pipelines"].items():
    print(f"\n{name}:")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Mean time: {stats['mean_execution_time']:.4f}s")
    print(f"  Health: {stats['health']}")
```

```text
Overall health: degraded

inference:
  Total executions: 2
  Success rate: 50.0%
  Mean time: 0.0990s
  Health: degraded
```

### Health Status Values

| Status | Meaning |
|--------|---------|
| `healthy` | All pipelines have high success rate and normal execution times |
| `degraded` | Some pipelines have elevated error rates or slow execution |
| `critical` | One or more pipelines are failing consistently |

## Best Practices

- Use `AlertManager` directly when monitoring is event-driven (you push metrics)
- Use `AdvancedMonitor` when monitoring is poll-based (periodic checks)
- Use `ProductionMonitor` for full pipeline health tracking with execution
  history
- Set conservative thresholds initially and tighten them as you understand your
  workload's normal variance
- Register multiple handlers (e.g., logging + Slack webhook) for alert fan-out

## Next Steps

<div class="grid cards" markdown>

-   :material-shield-check:{ .lg .middle } **CI Integration**

    ---

    Use monitoring thresholds to gate CI pipelines

    [:octicons-arrow-right-24: CI integration](ci-integration.md)

-   :material-puzzle:{ .lg .middle } **Writing Adapters**

    ---

    Wrap external models for monitoring with Calibrax

    [:octicons-arrow-right-24: Adapters](adapters.md)

</div>
