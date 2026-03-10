# Model Evaluation with Composition

| | |
|---|---|
| **Level** | Intermediate |
| **Time** | ~15 minutes |
| **Prerequisites** | [Quickstart](quickstart.md), [Regression Metrics](regression-metrics.md) |
| **Format** | Python + Jupyter |

## Overview

Real evaluation pipelines rarely use a single metric. Calibrax provides a composition framework that lets you group metrics into collections, assign weights for multi-objective scoring, enforce quality gates, estimate confidence intervals via bootstrapping, and track metric history across training epochs.

This example walks through six composable building blocks: `MetricCollection`, `WeightedMetric`, `MetricSuite`, `ThresholdMetric`, `BootstrapMetric`, and `MetricTracker`. Each can be used independently or combined into a full evaluation pipeline.

## What You'll Learn

1. Group metrics into a `MetricCollection` for batch computation
2. Produce a single weighted score from multiple metrics with `WeightedMetric`
3. Organise metrics into domain-based groups with `MetricSuite`
4. Set pass/fail quality gates using `ThresholdMetric`
5. Estimate confidence intervals with `BootstrapMetric`
6. Track metric history across epochs with `MetricTracker`

## Files

- **Python Script**: [`examples/metrics/05_composition.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/05_composition.py)
- **Jupyter Notebook**: [`examples/metrics/05_composition.ipynb`](https://github.com/avitai/calibrax/blob/main/examples/metrics/05_composition.ipynb)

## Quick Start

```bash
source activate.sh && python examples/metrics/05_composition.py
```

## Key Concepts

### MetricCollection

A `MetricCollection` groups related metric functions and computes them all in one call. You can build one manually from a dict of functions or automatically from the registry.

```python
from calibrax.metrics import MetricCollection
from calibrax.metrics.functional.regression import mae, mse, r_squared, rmse

collection = MetricCollection({
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
    "r_squared": r_squared,
})
results = collection.compute_functional(predictions, targets)
# results = {"mse": 0.04, "mae": 0.16, "rmse": 0.20, "r_squared": 0.99}

# Or build from registry by domain
registry_collection = MetricCollection.from_registry(domain="general")
```

### WeightedMetric

When you need a single scalar score from multiple metrics, `WeightedMetric` applies normalised weights to a results dict.

```python
from calibrax.metrics import WeightedMetric

weighted = WeightedMetric({"mse": 0.7, "mae": 0.3})
score = weighted.compute(results)
# score = 0.7 * results["mse"] + 0.3 * results["mae"]
```

Weights are automatically normalised to sum to 1.0.

### MetricSuite

A `MetricSuite` organises metrics into named groups. Each group is evaluated independently, producing a nested results dict. This is useful for separating error metrics from fit-quality metrics from robustness metrics.

```python
from calibrax.metrics import MetricSuite

suite = MetricSuite()
suite.add_group("error_metrics", ["mse", "mae", "rmse"])
suite.add_group("fit_quality", ["r_squared", "explained_variance"])
suite.add_group("robust_metrics", ["huber_loss", "log_cosh_loss"])

suite_results = suite.compute_all(predictions, targets)
# suite_results["error_metrics"]["mse"] -> 0.04
# suite_results["fit_quality"]["r_squared"] -> 0.99

# Auto-create groups from registry domains
auto_suite = MetricSuite.from_registry_domains()
```

### ThresholdMetric (Quality Gate)

`ThresholdMetric` wraps a single metric with a pass/fail threshold. Use `max_value` for lower-is-better metrics and `min_value` for higher-is-better metrics.

```python
from calibrax.metrics import ThresholdMetric

# MSE must be below 0.1
mse_gate = ThresholdMetric("mse", max_value=0.1)
result = mse_gate.evaluate(predictions, targets)
# result["value"] = 0.04, result["passed"] = True

# R-squared must be above 0.95
r2_gate = ThresholdMetric("r_squared", min_value=0.95)
result_r2 = r2_gate.evaluate(predictions, targets)
```

This integrates naturally into CI pipelines -- see the [CI Integration](../../user-guide/ci-integration.md) guide.

### BootstrapMetric

`BootstrapMetric` wraps any metric function and computes confidence intervals by resampling.

```python
from calibrax.metrics import BootstrapMetric

bootstrap = BootstrapMetric(mse, num_bootstraps=200, confidence=0.95, seed=42)
boot_result = bootstrap.compute(predictions, targets)
# boot_result["value"]  = point estimate
# boot_result["lower"]  = 2.5th percentile
# boot_result["upper"]  = 97.5th percentile
# boot_result["samples"] = all 200 bootstrap values
```

### MetricTracker

`MetricTracker` records a metric's value at each call and tracks the best value seen so far. Useful for logging during training.

```python
from calibrax.metrics import MetricTracker

num_epochs = 5
tracker = MetricTracker(mse, direction="lower")

for epoch in range(num_epochs):
    value = tracker.increment(predictions, targets)
    print(f"Epoch {epoch}: MSE = {value:.6f}")

print(f"Best MSE: {tracker.best():.6f} at epoch {tracker.best_epoch}")
print(f"Full history: {tracker.history}")
```

The `direction` parameter (`"lower"` or `"higher"`) determines how "best" is defined.

## Example Code

The script ties all six components together. Here is the quality-gate section:

```python
# MSE must be below 0.1 (lower-is-better)
mse_gate = ThresholdMetric("mse", max_value=0.1)
result = mse_gate.evaluate(predictions, targets)
print(f"Value: {result['value']:.6f}, Passed: {result['passed']}")

# R-squared must be above 0.95 (higher-is-better)
r2_gate = ThresholdMetric("r_squared", min_value=0.95)
result_r2 = r2_gate.evaluate(predictions, targets)
print(f"Value: {result_r2['value']:.6f}, Passed: {result_r2['passed']}")
```

## Next Steps

- [Image Quality Metrics](image-quality.md) -- PSNR, SSIM, MS-SSIM, BLEU, ROUGE, FID
- [Metric Learning Losses](metric-learning.md) -- contrastive, triplet, NTXent, and ArcFace losses
- [API Reference: `calibrax.metrics`](../../api-reference/metrics/index.md) -- full composition API
