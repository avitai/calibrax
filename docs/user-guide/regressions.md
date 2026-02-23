# Regression Detection

Calibrax detects performance regressions by comparing a new run against a stored
baseline, respecting each metric's direction to determine what counts as a
degradation.

## How It Works

`detect_regressions()` compares matching metrics between two runs. For each
metric present in both the current run and the baseline:

1. It looks up the `MetricDef.direction` from the run's `metric_defs`
2. For `HIGHER` metrics, a regression occurs when the value drops below
   `baseline * (1 - threshold)`
3. For `LOWER` metrics, a regression occurs when the value rises above
   `baseline * (1 + threshold)`
4. `INFO` metrics are always skipped

```python
from calibrax.analysis.regression import detect_regressions
from calibrax.core.models import (
    MetricDef, MetricDirection, Metric, Point, Run,
)

# Two runs with the same metric definitions
metric_defs = {
    "throughput": MetricDef(
        name="throughput", unit="samples/sec", direction=MetricDirection.HIGHER
    ),
    "latency": MetricDef(
        name="latency", unit="ms", direction=MetricDirection.LOWER
    ),
}

baseline = Run(
    points=(Point(
        name="forward_pass", scenario="training",
        metrics={"throughput": Metric(value=1000.0), "latency": Metric(value=1.0)},
    ),),
    metric_defs=metric_defs,
)

current = Run(
    points=(Point(
        name="forward_pass", scenario="training",
        metrics={"throughput": Metric(value=920.0), "latency": Metric(value=1.12)},
    ),),
    metric_defs=metric_defs,
)

regressions = detect_regressions(current, baseline, threshold=0.05)
for r in regressions:
    print(f"{r.metric} ({r.direction.value}): "
          f"{r.baseline_value} -> {r.current_value} ({r.delta_pct:+.1f}%)")
```

```text
throughput (higher): 1000.0 -> 920.0 (-8.0%)
latency (lower): 1.0 -> 1.12 (+12.0%)
```

## Using Baselines from a Store

In practice, baselines are managed through the `Store`:

```python
from calibrax.storage.store import Store
from calibrax.analysis.regression import detect_regressions
from calibrax.core.models import MetricDef, MetricDirection, Metric, Point, Run

store = Store("/tmp/calibrax-regression-demo")

baseline_run = Run(
    points=(Point(name="fwd", scenario="train",
                  metrics={"throughput": Metric(value=1000.0)}),),
    metric_defs={"throughput": MetricDef(
        name="throughput", unit="samples/sec", direction=MetricDirection.HIGHER)},
)
new_run = Run(
    points=(Point(name="fwd", scenario="train",
                  metrics={"throughput": Metric(value=900.0)}),),
    metric_defs={"throughput": MetricDef(
        name="throughput", unit="samples/sec", direction=MetricDirection.HIGHER)},
)

# Set a baseline
store.save(baseline_run)
store.set_baseline(baseline_run.id)

# Later, detect regressions against the stored baseline
baseline = store.get_baseline()
if baseline is not None:
    regressions = detect_regressions(new_run, baseline, threshold=0.05)
```

## Interpreting Regression Objects

Each `Regression` object contains:

| Field | Description |
|-------|-------------|
| `metric` | Metric name (e.g., `"throughput"`) |
| `point_name` | Name of the measurement point |
| `baseline_value` | Value from the baseline run |
| `current_value` | Value from the current run |
| `delta_pct` | Percentage change (negative for drops, positive for increases) |
| `direction` | `MetricDirection` — how this metric is compared |

## Scaling Law Fitting

`scaling_fit()` fits a power law `y = a * x^b` to a series of measurements,
useful for predicting how performance scales with input size:

```python
from calibrax.analysis.scaling import scaling_fit

sizes = [100, 500, 1000, 5000, 10000]
times = [0.01, 0.05, 0.10, 0.52, 1.05]

law = scaling_fit(sizes, times)
print(f"Coefficient: {law.coefficient:.4f}")
print(f"Exponent: {law.exponent:.2f}")
print(f"R-squared: {law.r_squared:.4f}")
print(f"Complexity: {law.complexity}")  # e.g., "O(n)"
```

```text
Coefficient: 0.0001
Exponent: 1.01
R-squared: 1.0000
Complexity: O(n)
```

The `complexity` field maps common exponents to Big-O notation:
`O(1)`, `O(sqrt(n))`, `O(n)`, `O(n^1.5)`, `O(n^2)`, `O(n^3)`.

## Next Steps

<div class="grid cards" markdown>

-   :material-shield-check:{ .lg .middle } **CI Integration**

    ---

    Gate CI pipelines on regression detection results

    [:octicons-arrow-right-24: CI integration](ci-integration.md)

-   :material-compare:{ .lg .middle } **Comparing Configurations**

    ---

    Rank multiple configurations and find Pareto-optimal tradeoffs

    [:octicons-arrow-right-24: Comparison](comparison.md)

</div>
