# Quick Start

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~5 min |
| **Prerequisites** | Python 3.12+, JAX installed |

This guide walks through four progressive examples covering the core Calibrax
workflow: defining results, profiling workloads, detecting regressions, and
using the CLI.

## 1. Define and Store Results

Create metric definitions, record benchmark data points, and save them to a store.

```python
from pathlib import Path
from calibrax.core.models import (
    MetricDef, MetricDirection, Metric, Point, Run,
)
from calibrax.storage.store import Store

# Define metrics with direction (higher or lower is better)
throughput_def = MetricDef(
    name="throughput", unit="samples/sec", direction=MetricDirection.HIGHER
)
latency_def = MetricDef(
    name="latency", unit="ms", direction=MetricDirection.LOWER
)

# Record a benchmark data point
point = Point(
    name="forward_pass",
    scenario="training",
    tags={"framework": "flax"},
    metrics={
        "throughput": Metric(value=1250.0, lower=1200.0, upper=1300.0),
        "latency": Metric(value=0.8, lower=0.75, upper=0.85),
    },
)

# Create a run and save it
run = Run(
    points=(point,),
    metric_defs={"throughput": throughput_def, "latency": latency_def},
)

store = Store(Path("/tmp/calibrax-quickstart"))
store.save(run)
store.set_baseline(run.id)
print(f"Saved run {run.id} as baseline")
```

The run ID is generated automatically — output will look like:

```text
Saved run 2ca7dade29a7 as baseline
```

## 2. Profile a JAX Workload

Use `TimingCollector` to measure wall-clock time with GPU synchronization,
and `ResourceMonitor` to track CPU and memory usage.

```python
import jax
import jax.numpy as jnp
from calibrax.profiling.timing import TimingCollector
from calibrax.profiling.resources import ResourceMonitor

# A simple JAX workload
def make_batches(n: int):
    for _ in range(n):
        x = jnp.ones((64, 128))
        yield jax.nn.relu(x)

# Measure timing with JAX GPU sync
collector = TimingCollector(sync_fn=lambda batch: jax.block_until_ready(batch))
sample = collector.measure_iteration(
    make_batches(50),
    num_batches=50,
    count_fn=lambda batch: batch.shape[0],  # 64 elements per batch
)

print(f"Wall clock: {sample.wall_clock_sec:.3f}s")
print(f"Batches: {sample.num_batches}")
print(f"Elements: {sample.num_elements}")
print(f"First batch: {sample.first_batch_time:.4f}s (includes JIT)")

# Monitor resources during a heavier workload
with ResourceMonitor(sample_interval_sec=0.05) as monitor:
    for _ in range(200):
        x = jnp.ones((256, 256))
        y = jnp.dot(x, x)
        jax.block_until_ready(y)

summary = monitor.summary
print(f"Peak RSS: {summary.peak_rss_mb:.1f} MB")
print(f"Duration: {summary.duration_sec:.2f}s")
```

Timing and memory values vary by hardware. Example output on a CUDA GPU:

```text
Wall clock: 0.259s
Batches: 50
Elements: 3200
First batch: 0.2452s (includes JIT)
Peak RSS: 709.9 MB
Duration: 0.46s
```

## 3. Detect Regressions

Load a baseline from the store and compare a new run against it.

```python
from calibrax.analysis.regression import detect_regressions

# Simulate a new run with slightly degraded throughput
new_point = Point(
    name="forward_pass",
    scenario="training",
    tags={"framework": "flax"},
    metrics={
        "throughput": Metric(value=1100.0),  # dropped from 1250
        "latency": Metric(value=0.82),       # slightly worse
    },
)
new_run = Run(
    points=(new_point,),
    metric_defs={"throughput": throughput_def, "latency": latency_def},
)

# Compare against baseline (5% threshold)
baseline = store.get_baseline()
if baseline is not None:
    regressions = detect_regressions(new_run, baseline, threshold=0.05)
    if regressions:
        print("Regressions detected:")
        for r in regressions:
            print(f"  {r.metric}: {r.baseline_value} -> {r.current_value}"
                  f" ({r.delta_pct:+.1f}%)")
    else:
        print("No regressions detected")
```

```text
Regressions detected:
  throughput: 1250.0 -> 1100.0 (-12.0%)
```

## 4. CLI Workflow

Calibrax provides a command-line interface for common operations.

```bash
# Ingest external benchmark results
calibrax ingest --data ./benchmark-data --input results.json

# Set the latest run as baseline
calibrax baseline --data ./benchmark-data --run latest

# Run regression check (exits with code 1 on failure — suitable for CI)
calibrax check --data ./benchmark-data --threshold 0.05

# Show metric trends over time
calibrax trend --data ./benchmark-data --metric throughput \
    --point forward_pass --framework flax

# Show a run summary
calibrax summary --data ./benchmark-data
```

The `summary` command outputs run metadata and metrics:

```text
Run: 2ca7dade29a7
  Timestamp: 2026-02-20 11:34:24.346877
  Points: 1

Scenario: training
  flax: latency=0.8000, throughput=1250.0000
```

## Next Steps

<div class="grid cards" markdown>

-   :material-book-open:{ .lg .middle } **Core Concepts**

    ---

    Understand the data model, direction-aware metrics, and benchmark lifecycle

    [:octicons-arrow-right-24: Concepts](concepts.md)

-   :material-timer:{ .lg .middle } **Profiling Guide**

    ---

    Timing, resource monitoring, GPU memory, energy, and FLOP counting

    [:octicons-arrow-right-24: Profiling](../user-guide/profiling.md)

-   :material-database:{ .lg .middle } **Storage Guide**

    ---

    JSON store layout, baselines, trends, and data ingestion

    [:octicons-arrow-right-24: Storage](../user-guide/storage.md)

</div>
