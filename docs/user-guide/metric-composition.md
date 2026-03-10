# Metric Composition

Calibrax provides composition and wrapper classes that enhance individual
metrics without modifying them. Group related metrics into collections,
combine them with weights, set CI quality gates, add confidence intervals,
or track values over time.

## MetricCollection

Group multiple metric functions and compute them in a single call.

```python
from calibrax.metrics.functional.regression import mse, mae, rmse
from calibrax.metrics.composition import MetricCollection

collection = MetricCollection({
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
})

results = collection.compute_functional(predictions, targets)
# {"mse": 0.012, "mae": 0.08, "rmse": 0.11}
```

### Building from the Registry

Create a collection from all registered Tier 0 metrics in a domain:

```python
from calibrax.metrics.composition import MetricCollection

# All classification metrics
clf_metrics = MetricCollection.from_registry(domain="classification")
results = clf_metrics.compute_functional(predictions, targets)

# All metrics across all domains
all_metrics = MetricCollection.from_registry()
```

### Dynamic Modification

Add or remove metrics after creation:

```python
from calibrax.metrics.functional.regression import r_squared

collection.add("r_squared", r_squared)
collection.remove("rmse")
print(collection.names)  # ["mse", "mae", "r_squared"]
```

## WeightedMetric

Combine multiple metric values into a single scalar score for
multi-objective optimization or model selection.

```python
from calibrax.metrics.composition import WeightedMetric

weighted = WeightedMetric({"mse": 0.7, "mae": 0.3})

# Compute from pre-computed values
metric_values = {"mse": 0.012, "mae": 0.08}
score = weighted.compute(metric_values)
# 0.7 * 0.012 + 0.3 * 0.08 = 0.0324
```

Weights do not need to sum to 1. Use `normalized_weights` if you need them:

```python
print(weighted.weights)             # {"mse": 0.7, "mae": 0.3}
print(weighted.normalized_weights)  # {"mse": 0.7, "mae": 0.3} (already sums to 1)
```

### Combining with MetricCollection

A typical pattern: compute metrics in bulk, then reduce to a single score.

```python
from calibrax.metrics.composition import MetricCollection, WeightedMetric
from calibrax.metrics.functional.regression import mse, mae

collection = MetricCollection({"mse": mse, "mae": mae})
weighted = WeightedMetric({"mse": 0.6, "mae": 0.4})

results = collection.compute_functional(predictions, targets)
score = weighted.compute(results)
```

## MetricSuite

Organize metrics into named groups for structured evaluation. Each group
is a list of registered metric names.

```python
from calibrax.metrics.composition import MetricSuite

suite = MetricSuite()
suite.add_group("regression", ["mse", "mae", "rmse"])
suite.add_group("correlation", ["pearson_correlation", "spearman_rank_correlation"])

results = suite.compute_all(predictions, targets)
# {
#     "regression": {"mse": 0.012, "mae": 0.08, "rmse": 0.11},
#     "correlation": {"pearson_correlation": 0.95, "spearman_rank_correlation": 0.93},
# }
```

### Auto-Populating from Domains

Build a suite with one group per registry domain:

```python
suite = MetricSuite.from_registry_domains()
print(suite.list_groups())
# ["audio", "calibration", "classification", "clustering", "distance", ...]

# Use domain-specific suites for compute_all (metrics have different signatures)
regression_suite = MetricSuite()
regression_suite.add_group("regression", ["mse", "mae", "rmse"])
results = regression_suite.compute_all(predictions, targets)
```

!!! warning "Signature Mismatch"

    `compute_all` passes `(predictions, targets)` to every metric. Metrics
    with non-standard signatures (`SINGLE_INPUT`, `CUSTOM`, `FEATURES_LABELS`)
    will fail if called this way. Group only `PREDICTIONS_TARGETS` metrics
    together, or compute others separately.

## ThresholdMetric

Wrap a registered metric with a pass/fail threshold for CI quality gates.

```python
from calibrax.metrics.composition import ThresholdMetric

# Regression gate: MSE must not exceed 0.05
gate = ThresholdMetric("mse", max_value=0.05)
result = gate.evaluate(predictions, targets)
# {"value": 0.012, "passed": True, "threshold": 0.05, "metric_name": "mse"}

# Classification gate: accuracy must be at least 0.90
gate = ThresholdMetric("accuracy", min_value=0.90)
result = gate.evaluate(predictions, targets)
# {"value": 0.94, "passed": True, "threshold": 0.90, "metric_name": "accuracy"}
```

You can set both bounds for metrics that should fall in a range:

```python
gate = ThresholdMetric("r_squared", min_value=0.85, max_value=1.0)
```

### CI Integration

Use `ThresholdMetric` alongside `CIGuard` for metric-specific quality gates.
See [CI Integration](ci-integration.md) for the full workflow.

## BootstrapMetric

Wrap any metric with bootstrap confidence interval estimation. A measurement
without uncertainty is incomplete.

```python
from calibrax.metrics.functional.regression import mse
from calibrax.metrics.wrappers import BootstrapMetric

bootstrap = BootstrapMetric(mse, num_bootstraps=1000, confidence=0.95, seed=42)
result = bootstrap.compute(predictions, targets)

print(f"MSE: {result['value']:.4f}")
print(f"95% CI: [{result['lower']:.4f}, {result['upper']:.4f}]")
print(f"Bootstrap samples: {len(result['samples'])}")
```

The wrapper resamples `(predictions, targets)` pairs with replacement,
computes the metric on each resample, and returns percentile-based confidence
bounds.

!!! tip "Choosing Bootstrap Count"

    1000 resamples is adequate for 95% CIs. For 99% CIs or when reporting
    narrow intervals, use 5000-10000.

## ClasswiseWrapper

Break down any metric by class to identify where a model struggles.

```python
import jax.numpy as jnp
from calibrax.metrics.functional.regression import mse
from calibrax.metrics.wrappers import ClasswiseWrapper

classwise = ClasswiseWrapper(mse, class_names=["cat", "dog", "bird"])

predictions = jnp.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
targets = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
labels = jnp.array([0, 0, 1, 1, 2, 2])

result = classwise.compute(predictions, targets, labels)
# {"cat": 0.025, "dog": 0.445, "bird": 0.305, "mean": 0.258}
```

If `class_names` is not provided, integer label indices are used as keys.

## MetricTracker

Track a metric across evaluation epochs with automatic best-value detection.

```python
from calibrax.metrics.functional.regression import mse
from calibrax.metrics.wrappers import MetricTracker

tracker = MetricTracker(mse, direction="lower")

# Simulate evaluation after each training epoch
preds = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
tgts = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
eval_batches = [
    (preds + jnp.array([0.1, -0.1, 0.0, 0.05, -0.05]) * i, tgts)
    for i in range(5)
]
for epoch_preds, epoch_targets in eval_batches:
    value = tracker.increment(epoch_preds, epoch_targets)
    print(f"Epoch MSE: {value:.4f}")

print(f"Best MSE: {tracker.best():.4f}")
print(f"Best epoch: {tracker.best_epoch}")
print(f"Full history: {tracker.history}")

# Start fresh for a new experiment
tracker.reset()
```

The `direction` parameter determines what "best" means:

- `"lower"`: best = minimum value (MSE, MAE, loss)
- `"higher"`: best = maximum value (accuracy, R-squared)

## MinMaxTracker

Track running extrema without storing full history. Useful for monitoring
metric ranges during training with minimal memory overhead.

```python
from calibrax.metrics.functional.regression import mse
from calibrax.metrics.wrappers import MinMaxTracker

tracker = MinMaxTracker(mse)

preds = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
tgts = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
training_batches = [
    (preds + jnp.array([0.1, -0.1, 0.0, 0.05, -0.05]) * i, tgts)
    for i in range(5)
]
for batch_preds, batch_targets in training_batches:
    value = tracker.update(batch_preds, batch_targets)

print(f"Current: {tracker.current:.4f}")
print(f"Min seen: {tracker.min:.4f}")
print(f"Max seen: {tracker.max:.4f}")

tracker.reset()  # Clear tracking state
```

## Which Composition Tool Should I Use?

| Goal | Tool | Key method |
|------|------|------------|
| Compute several metrics at once | `MetricCollection` | `compute_functional()` |
| Combine metrics into a single score | `WeightedMetric` | `compute()` |
| Organize metrics into named domain groups | `MetricSuite` | `compute_all()` |
| Fail a CI pipeline if a metric regresses | `ThresholdMetric` | `evaluate()` |
| Report confidence intervals | `BootstrapMetric` | `compute()` |
| Break down a metric by class | `ClasswiseWrapper` | `compute()` |
| Track best value across epochs | `MetricTracker` | `increment()` / `best()` |
| Monitor running min/max without storing history | `MinMaxTracker` | `update()` |

!!! tip "Combining Tools"

    These tools compose freely. A common pipeline is:
    `MetricCollection` (compute) -> `WeightedMetric` (reduce) ->
    `ThresholdMetric` (gate) -> `MetricTracker` (track over time).

## Composition Patterns

### Multi-Metric Evaluation Pipeline

Combine several composition tools for a full evaluation workflow:

```python
from calibrax.metrics.composition import (
    MetricCollection,
    MetricSuite,
    ThresholdMetric,
    WeightedMetric,
)
from calibrax.metrics.wrappers import BootstrapMetric, MetricTracker
from calibrax.metrics.functional.regression import mse, mae

# 1. Compute metrics with confidence intervals
bootstrap_mse = BootstrapMetric(mse, num_bootstraps=500, seed=0)
bootstrap_mae = BootstrapMetric(mae, num_bootstraps=500, seed=0)

mse_result = bootstrap_mse.compute(predictions, targets)
mae_result = bootstrap_mae.compute(predictions, targets)

# 2. Check quality gates
mse_gate = ThresholdMetric("mse", max_value=0.05)
gate_result = mse_gate.evaluate(predictions, targets)

# 3. Compute weighted composite score
weighted = WeightedMetric({"mse": 0.6, "mae": 0.4})
score = weighted.compute({
    "mse": mse_result["value"],
    "mae": mae_result["value"],
})
```

### Epoch-Level Tracking with Multiple Metrics

```python
from calibrax.metrics.wrappers import MetricTracker
from calibrax.metrics.functional.regression import mse, mae
from calibrax.metrics.functional.statistical import pearson_correlation

trackers = {
    "mse": MetricTracker(mse, direction="lower"),
    "mae": MetricTracker(mae, direction="lower"),
    "pearson": MetricTracker(pearson_correlation, direction="higher"),
}

preds = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
tgts = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
eval_epochs = [
    (preds + jnp.array([0.1, -0.1, 0.0, 0.05, -0.05]) * i, tgts)
    for i in range(5)
]
for epoch_preds, epoch_targets in eval_epochs:
    for name, tracker in trackers.items():
        value = tracker.increment(epoch_preds, epoch_targets)

for name, tracker in trackers.items():
    print(f"{name}: best={tracker.best():.4f} at epoch {tracker.best_epoch}")
```

## Next Steps

<div class="grid cards" markdown>

-   **Metrics Overview**

    ---

    Registry, tiers, and domain reference

    [:octicons-arrow-right-24: Metrics Overview](metrics-overview.md)

-   **Stateful Metrics**

    ---

    Frozen backbone, learned, and metric learning losses

    [:octicons-arrow-right-24: Stateful Metrics](stateful-metrics.md)

</div>
