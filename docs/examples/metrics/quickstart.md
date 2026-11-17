# Metrics Quickstart

> Ran in 2.7 s on 1 cpu (python 3.12.14, jax 0.11.1, substrax 0.1.27, calibrax 0.1.19), from [`examples/metrics/quickstart.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/quickstart.py) ([notebook](https://github.com/avitai/calibrax/blob/main/examples/metrics/quickstart.ipynb)). Run it from the repository root:

```bash
source ./activate.sh
uv run python examples/metrics/quickstart.py
```

| Metadata | Value |
| --- | --- |
| **Tier** | Tutorial |
| **Devices** | CPU-compatible |
| **Prerequisites** | Basic Python and JAX arrays |

## Overview

This example introduces calibrax's core metrics API: it computes individual regression metrics,
evaluates a whole metric suite in one call with `calculate_all()`, and uses the `MetricRegistry`
to discover the metrics calibrax ships, filter them by domain, tier and property, and read their
metadata. Start here if you are new to calibrax; the other examples build on these patterns.

## Learning Goals

1. Compute individual metrics (MSE, MAE, R-squared) on JAX arrays
2. Evaluate every general metric, or a chosen subset, with `calculate_all()`
3. Query the `MetricRegistry` by domain, tier and invariance
4. Inspect a `MetricEntry`'s metadata: direction, differentiability, invariances

## Setup


```python
import jax
import jax.numpy as jnp

from calibrax.metrics import calculate_all, MetricEntry, MetricRegistry, MetricTier
from calibrax.metrics.functional.regression import mae, mse, r_squared
```

## Core Concepts

Every metric in calibrax is a pure JAX function with the signature
`(predictions, targets) -> scalar`, so it is JIT-compatible and differentiable. The
`MetricRegistry` is the catalogue of those functions: each `MetricEntry` stores the function with
its tier (pure function, stateful or plugin), its domain, its direction (whether lower or higher
is better), its mathematical properties and its invariances, which is what an automated
evaluation pipeline reads to configure itself.

## Implementation

### Step 1: Compute individual metrics

Each metric takes the predictions first and the targets second.


```python
TARGETS = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
PREDICTIONS = jnp.array([1.1, 2.3, 2.8, 4.2, 4.7])


def individual_metrics(predictions: jax.Array, targets: jax.Array) -> dict[str, float]:
    """Compute MSE, MAE and R-squared one function at a time.

    Args:
        predictions: The model's predictions.
        targets: The true values.

    Returns:
        Each metric's value by name.
    """
    return {
        "mse": float(mse(predictions, targets)),
        "mae": float(mae(predictions, targets)),
        "r_squared": float(r_squared(predictions, targets)),
    }
```

    An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.


### Step 2: Evaluate every general metric at once

`calculate_all()` evaluates every registered general-domain metric in one call; a `metrics` list
selects a subset.


```python
def all_general_metrics(predictions: jax.Array, targets: jax.Array) -> dict[str, float]:
    """Evaluate every registered general-domain metric.

    Args:
        predictions: The model's predictions.
        targets: The true values.

    Returns:
        Each metric's value by name, sorted by name.
    """
    values = calculate_all(predictions, targets)
    return {name: float(value) for name, value in sorted(values.items())}


def selected_metrics(predictions: jax.Array, targets: jax.Array) -> dict[str, float]:
    """Evaluate a chosen subset of the general metrics.

    Args:
        predictions: The model's predictions.
        targets: The true values.

    Returns:
        The four selected metrics' values by name.
    """
    chosen = calculate_all(predictions, targets, metrics=["mse", "mae", "rmse", "r_squared"])
    return {name: float(value) for name, value in chosen.items()}
```

### Step 3: Query the registry

The registry lists every metric by name, and filters them by domain and by tier.


```python
DOMAINS = ("general", "classification", "distance", "image")


def metrics_by_domain(registry: MetricRegistry) -> dict[str, list[str]]:
    """List the metrics registered in each of a few domains.

    Args:
        registry: The metric registry.

    Returns:
        The metric names of each domain in `DOMAINS`.
    """
    return {domain: [entry.name for entry in registry.list_by_domain(domain)] for domain in DOMAINS}
```

### Step 4: Inspect a MetricEntry

An entry describes how its metric behaves.


```python
def describe(entry: MetricEntry) -> dict[str, str]:
    """Read the fields of a registry entry.

    Args:
        entry: The registry entry.

    Returns:
        Each field's value, as text.
    """
    properties = entry.properties
    return {
        "name": entry.name,
        "tier": str(entry.tier),
        "domain": entry.domain,
        "direction": str(entry.direction),
        "description": entry.description,
        "signature": str(entry.signature),
        "is_true_metric": str(properties.is_true_metric),
        "is_symmetric": str(properties.is_symmetric),
        "is_differentiable": str(properties.is_differentiable),
        "is_jit_compatible": str(properties.is_jit_compatible),
        "invariances": str(properties.invariances),
    }
```

### Step 5: Find metrics by their properties

The registry also answers questions about properties: which metrics satisfy the metric-space
axioms, and which are invariant to a transformation.


```python
def true_metrics(registry: MetricRegistry) -> list[str]:
    """List the metrics that satisfy the metric-space axioms.

    Args:
        registry: The metric registry.

    Returns:
        Their names.
    """
    return [entry.name for entry in registry.list_true_metrics()]


def rotation_invariant(registry: MetricRegistry) -> list[str]:
    """List the metrics invariant to rotation.

    Args:
        registry: The metric registry.

    Returns:
        Their names.
    """
    return [entry.name for entry in registry.list_by_invariance("rotation")]
```

## Results

### Individual metrics, the whole suite and a subset

The first three tables print each metric's value on the same five predictions.

### What the registry holds

The rest of the run lists the registry by domain and by tier, describes the `mse` entry, and
names the true metrics and the rotation-invariant ones.


```python
def print_values(title: str, values: dict[str, float]) -> None:
    """Print one table of metric values.

    Args:
        title: The table's title.
        values: Each metric's value by name.
    """
    print(f"=== {title} ===")
    for name, value in values.items():
        print(f"  {name:25s} = {value:.6f}")


def print_registry(registry: MetricRegistry) -> dict[str, int]:
    """Print what the registry holds.

    Args:
        registry: The metric registry.

    Returns:
        The number of metrics registered, of pure functions and of true metrics.
    """
    names = sorted(registry.list_names())
    print("=== Registry: all registered metric names ===")
    print(f"  Total registered: {len(names)}")
    print(f"  First 10: {names[:10]}")
    print("=== Registry: metrics by domain ===")
    for domain, domain_names in metrics_by_domain(registry).items():
        print(f"  {domain:20s} ({len(domain_names)}): {domain_names}")
    tier0 = registry.list_by_tier(MetricTier.PURE_FUNCTION)
    print("=== Registry: Tier 0 pure functions ===")
    print(f"  Count: {len(tier0)}")
    print("=== MetricEntry fields for 'mse' ===")
    for field, value in describe(registry.get("mse")).items():
        print(f"  {field + ':':18s} {value}")
    metrics = true_metrics(registry)
    print("=== Registry: true metrics (satisfy metric space axioms) ===")
    print(f"  Count: {len(metrics)}")
    print(f"  Names: {metrics[:8]}...")
    print("=== Registry: rotation-invariant metrics ===")
    print(f"  Names: {rotation_invariant(registry)}")
    return {"registered": len(names), "tier0": len(tier0), "true_metrics": len(metrics)}


def main() -> dict[str, float | int]:
    """Run the example and return its results.

    Returns:
        The three individual metrics and the registry's counts.
    """
    individual = individual_metrics(PREDICTIONS, TARGETS)
    print_values("Individual metrics", individual)
    print_values("calculate_all (all general metrics)", all_general_metrics(PREDICTIONS, TARGETS))
    print_values("calculate_all (selected subset)", selected_metrics(PREDICTIONS, TARGETS))
    return {**individual, **print_registry(MetricRegistry())}


if __name__ == "__main__":
    main()
```

    === Individual metrics ===
      mse                       = 0.054000
      mae                       = 0.220000
      r_squared                 = 0.973000
    === calculate_all (all general metrics) ===
      charbonnier_loss          = 0.220003
      explained_variance        = 0.973200
      huber_loss                = 0.027000
      log_cosh_loss             = 0.026682
      mae                       = 0.220000
      mape                      = 0.085333
      max_error                 = 0.300000
      mean_error                = 0.020000
      mse                       = 0.054000
      quantile_loss             = 0.110000
      r_squared                 = 0.973000
      relative_error            = 0.070065
      relative_l2_error         = 0.085333
      rmse                      = 0.232379
      smape                     = 0.082875
    === calculate_all (selected subset) ===
      mse                       = 0.054000
      mae                       = 0.220000
      rmse                      = 0.232379
      r_squared                 = 0.973000
    === Registry: all registered metric names ===
      Total registered: 156
      First 10: ['accuracy', 'adaptive_calibration_error', 'adjusted_mutual_information', 'adjusted_rand_index', 'anees', 'average_precision', 'balanced_accuracy', 'bedroc', 'bleu', 'bregman_divergence']
    === Registry: metrics by domain ===
      general              (17): ['mse', 'mae', 'rmse', 'mean_error', 'r_squared', 'r_squared_adjusted', 'mape', 'relative_error', 'relative_l2_error', 'explained_variance', 'max_error', 'huber_loss', 'charbonnier_loss', 'quantile_loss', 'log_cosh_loss', 'smape', 'crps']
      classification       (13): ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision', 'log_loss', 'matthews_corrcoef', 'cohen_kappa', 'balanced_accuracy', 'specificity', 'sensitivity', 'softmax_cross_entropy']
      distance             (11): ['cosine_distance', 'euclidean_distance', 'manhattan_distance', 'chebyshev_distance', 'mahalanobis_distance', 'hamming_distance', 'minkowski_distance', 'jaccard_distance', 'poincare_distance', 'lorentz_distance', 'randers_distance']
      image                (4): ['psnr', 'ssim', 'ms_ssim', 'vendi_score']
    === Registry: Tier 0 pure functions ===
      Count: 156
    === MetricEntry fields for 'mse' ===
      name:              mse
      tier:              pure_function
      domain:            general
      direction:         lower
      description:       Mean squared error
      signature:         predictions_targets
      is_true_metric:    False
      is_symmetric:      True
      is_differentiable: True
      is_jit_compatible: True
      invariances:       ()
    === Registry: true metrics (satisfy metric space axioms) ===
      Count: 27
      Names: ['mae', 'rmse', 'euclidean_distance', 'manhattan_distance', 'chebyshev_distance', 'mahalanobis_distance', 'hamming_distance', 'minkowski_distance']...
    === Registry: rotation-invariant metrics ===
      Names: ['euclidean_distance']


## Next Steps

- [Regression metrics deep dive](regression_deep_dive.md): regression metrics with outlier,
  quantile and CRPS analysis
- [Metrics API reference](../../api-reference/metrics/index.md): every function's signature and
  parameters
- [Geometric metrics guide](../../user-guide/geometric-metrics.md): choosing the right metric for
  your task

<!-- substrax-notebooks: {"page": {"body": "49da4fbcd955c4e71ee39dd9564e089586f9a8cd86663df7f85839dd9e2b1c51", "figures": {}, "record": "4553716bc6ca9af0eb819aeac6ee933e4e9703ce84744b2ad386c4abc62279ed"}, "provenance": {"device": "1 cpu", "outputs": [[], [], [], [], [], [], ["text"]], "seconds": 2.7, "sha256": "f4c7b78a30d02e2b4bc80b49324e655dec0ce0ff5e6f2f4d4c90e428dc98fa28", "source": "examples/metrics/quickstart.py", "versions": {"calibrax": "0.1.19", "jax": "0.11.1", "python": "3.12.14", "substrax": "0.1.27"}}} -->
