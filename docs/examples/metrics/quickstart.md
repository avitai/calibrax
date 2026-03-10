# Metrics Quickstart

| | |
|---|---|
| **Level** | Beginner |
| **Time** | ~5 minutes |
| **Prerequisites** | [Installation guide](../../getting-started/installation.md) |
| **Format** | Python + Jupyter |

## Overview

This example introduces the core metrics API in calibrax. You will compute individual regression metrics, evaluate an entire metric suite in a single call with `calculate_all()`, and explore the `MetricRegistry` to discover what metrics are available, filter them by domain or tier, and inspect their metadata.

Start here if you are new to calibrax. Every subsequent example builds on the patterns shown below.

## What You'll Learn

1. Compute individual metrics (MSE, MAE, R-squared) on JAX arrays
2. Batch-evaluate all registered metrics with `calculate_all()`
3. Query the `MetricRegistry` by name, domain, tier, and invariance
4. Inspect `MetricEntry` metadata (direction, differentiability, invariances)

## Files

- **Python Script**: [`examples/metrics/01_quickstart.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/01_quickstart.py)
- **Jupyter Notebook**: [`examples/metrics/01_quickstart.ipynb`](https://github.com/avitai/calibrax/blob/main/examples/metrics/01_quickstart.ipynb)

## Quick Start

```bash
source activate.sh && python examples/metrics/01_quickstart.py
```

## Key Concepts

### Individual Metric Computation

Every metric in calibrax is a pure JAX function with the signature `(predictions, targets) -> scalar`. These functions are JIT-compatible and differentiable.

```python
import jax.numpy as jnp
from calibrax.metrics.functional.regression import mae, mse, r_squared

targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
predictions = jnp.array([1.1, 2.3, 2.8, 4.2, 4.7])

mse_val = mse(predictions, targets)
mae_val = mae(predictions, targets)
r2_val = r_squared(predictions, targets)
```

### Batch Computation with `calculate_all`

Rather than calling metrics one at a time, `calculate_all()` evaluates every registered general-domain metric in a single call. You can also pass a `metrics` list to select a subset.

```python
from calibrax.metrics import calculate_all

# All general metrics
all_results = calculate_all(predictions, targets)

# Selected subset
selected = calculate_all(
    predictions, targets,
    metrics=["mse", "mae", "rmse", "r_squared"],
)
```

### MetricRegistry

The registry is the central catalogue of all metrics. Each entry stores the metric function alongside rich metadata -- tier, domain, direction (lower/higher is better), mathematical properties, and invariances.

```python
from calibrax.metrics import MetricRegistry, MetricTier

registry = MetricRegistry()

# List all names
all_names = registry.list_names()

# Filter by domain
general_metrics = registry.list_by_domain("general")
classification_metrics = registry.list_by_domain("classification")

# Filter by tier (pure functions, stateful, plugins)
tier0 = registry.list_by_tier(MetricTier.PURE_FUNCTION)

# Filter by mathematical property
true_metrics = registry.list_true_metrics()
rotation_invariant = registry.list_by_invariance("rotation")
```

### Inspecting a MetricEntry

Each registry entry exposes fields that describe the metric's behaviour, which is useful for automated pipeline configuration.

```python
entry = registry.get("mse")
entry.name              # "mse"
entry.tier              # MetricTier.PURE_FUNCTION
entry.domain            # "general"
entry.direction         # "lower" (lower is better)
entry.properties.is_true_metric    # whether it satisfies metric space axioms
entry.properties.is_symmetric      # d(x,y) == d(y,x)
entry.properties.is_differentiable # smooth gradient everywhere
entry.properties.is_jit_compatible # safe inside jax.jit
entry.properties.invariances       # e.g., ["translation"]
```

## Example Code

The full script walks through five sections. Here is the registry inspection section that ties everything together:

```python
registry = MetricRegistry()

# How many metrics are available?
print(f"Total registered: {len(registry.list_names())}")

# Group by domain
for domain in ("general", "classification", "distance", "image"):
    entries = registry.list_by_domain(domain)
    names = [e.name for e in entries]
    print(f"  {domain} ({len(names)}): {names}")

# Find all metrics that satisfy the triangle inequality
true_metrics = registry.list_true_metrics()
print(f"True metrics: {[e.name for e in true_metrics[:8]]}...")
```

## Next Steps

- [Regression Metrics Deep Dive](regression-metrics.md) -- explore all 12 regression metrics with outlier and quantile analysis
- [API Reference: `calibrax.metrics`](../../api-reference/metrics/index.md) -- full function signatures and parameters
- [User Guide: Geometric Metrics](../../user-guide/geometric-metrics.md) -- choosing the right metric for your task
