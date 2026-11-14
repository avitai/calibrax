# Metric Composition and Wrappers

> Ran in 2.4 s on 1 cpu (python 3.12.14, jax 0.11.1, substrax 0.1.27, calibrax 0.1.17), from [`examples/metrics/composition.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/composition.py) ([notebook](https://github.com/avitai/calibrax/blob/main/examples/metrics/composition.ipynb)). Run it from the repository root:

```bash
source ./activate.sh
uv run python examples/metrics/composition.py
```

| Metadata | Value |
| --- | --- |
| **Tier** | Advanced Guide |
| **Devices** | CPU-compatible |
| **Prerequisites** | The quickstart and regression deep dive examples |

## Overview

Real evaluation pipelines rarely use a single metric. calibrax's composition framework groups
metrics into collections, weighs them into one multi-objective score, organises them into suites
of named groups, gates a run on thresholds, estimates confidence intervals by bootstrapping, and
tracks a metric across training epochs. This example walks through the six building blocks,
`MetricCollection`, `WeightedMetric`, `MetricSuite`, `ThresholdMetric`, `BootstrapMetric` and
`MetricTracker`, each usable alone or combined into one evaluation pipeline.

## Learning Goals

1. Group metrics into a `MetricCollection` and compute them in one call
2. Combine several metrics into one weighted score with `WeightedMetric`
3. Organise metrics into named groups with `MetricSuite`
4. Set pass or fail quality gates with `ThresholdMetric`
5. Estimate confidence intervals with `BootstrapMetric`
6. Track a metric across epochs with `MetricTracker`

## Setup


```python
import jax
import jax.numpy as jnp

from calibrax.metrics import (
    BootstrapMetric,
    MetricCollection,
    MetricSuite,
    MetricTracker,
    ThresholdMetric,
    WeightedMetric,
)
from calibrax.metrics.functional.regression import mae, mse, r_squared, rmse
```

## Core Concepts

A collection computes a fixed set of metric functions together, built by hand or from the
registry's domains. A weighted metric normalises its weights to sum to 1 and applies them to
a dict of results. A suite holds named groups of registered metrics and returns one nested dict.
A threshold metric wraps one metric with a bound: `max_value` for a lower-is-better metric,
`min_value` for a higher-is-better one, which is how a CI pipeline gates a model (see the
[CI integration guide](../../user-guide/ci-integration.md)). A bootstrap metric resamples the
predictions and targets to put a confidence interval on any metric, drawing from the key it is
given, and a tracker records a metric at each call with the best value under its `direction`.

## Implementation

### Step 1: Group metrics into a collection


```python
TARGETS = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
PREDICTIONS = jnp.array([1.2, 1.8, 3.3, 3.9, 5.2, 5.7, 7.1, 8.3])


def collection_results() -> tuple[dict[str, float], list[str]]:
    """Compute a hand-built collection, and list a collection built from the registry.

    Returns:
        The hand-built collection's results, and the names the general domain holds.
    """
    collection = MetricCollection({"mse": mse, "mae": mae, "rmse": rmse, "r_squared": r_squared})
    from_registry = MetricCollection.from_registry(domain="general")
    return collection.compute_functional(PREDICTIONS, TARGETS), sorted(from_registry.names)
```

### Step 2: Weigh two metrics into one score

70% of the score comes from MSE and 30% from MAE.


```python
def weighted_score(results: dict[str, float]) -> tuple[WeightedMetric, float]:
    """Combine MSE and MAE into one weighted score.

    Args:
        results: The collection's results.

    Returns:
        The weighted metric and its score.
    """
    weighted = WeightedMetric({"mse": 0.7, "mae": 0.3})
    return weighted, weighted.compute(results)
```

### Step 3: Organise metrics into a suite

Three groups separate the error metrics from the fit-quality and the robust ones;
`MetricSuite.from_registry_domains()` builds one group per registry domain instead.


```python
def suite_results() -> tuple[dict[str, dict[str, float]], list[str]]:
    """Compute a three-group suite, and list the groups of a suite built from the registry.

    Returns:
        Each group's results, and the registry suite's groups.
    """
    suite = MetricSuite()
    suite.add_group("error_metrics", ["mse", "mae", "rmse"])
    suite.add_group("fit_quality", ["r_squared", "explained_variance"])
    suite.add_group("robust_metrics", ["huber_loss", "log_cosh_loss"])
    groups = MetricSuite.from_registry_domains().list_groups()
    return suite.compute_all(PREDICTIONS, TARGETS), groups
```

### Step 4: Gate on thresholds

MSE must stay below 0.1 and R-squared above 0.95.


```python
def quality_gates() -> dict[str, tuple[float, bool]]:
    """Evaluate the two quality gates.

    Returns:
        Each gate's value and whether it passed.
    """
    gates = {
        "mse below 0.1": ThresholdMetric("mse", max_value=0.1),
        "r_squared above 0.95": ThresholdMetric("r_squared", min_value=0.95),
    }
    results = {name: gate.evaluate(PREDICTIONS, TARGETS) for name, gate in gates.items()}
    return {name: (float(result.value), bool(result.passed)) for name, result in results.items()}
```

### Step 5: Put confidence intervals on two metrics

Each interval draws its resamples from its own key, split from one.


```python
def confidence_intervals(key: jax.Array) -> dict[str, tuple[float, float, float]]:
    """Bootstrap a 95% interval on MSE and a 90% interval on R-squared.

    Args:
        key: The key the resamples are drawn from.

    Returns:
        Each metric's point estimate, lower bound and upper bound.
    """
    mse_key, r2_key = jax.random.split(key)
    mse_interval = BootstrapMetric(mse, num_resamples=200, confidence=0.95).compute(
        PREDICTIONS, TARGETS, key=mse_key
    )
    r2_interval = BootstrapMetric(r_squared, num_resamples=200, confidence=0.90).compute(
        PREDICTIONS, TARGETS, key=r2_key
    )
    return {
        "MSE, 95%": (
            float(mse_interval.value),
            float(mse_interval.lower),
            float(mse_interval.upper),
        ),
        "R-squared, 90%": (
            float(r2_interval.value),
            float(r2_interval.lower),
            float(r2_interval.upper),
        ),
    }
```

### Step 6: Track a metric across epochs

Five simulated epochs shrink the same pattern of errors, as training would.


```python
ERROR_PATTERN = jnp.array([0.2, -0.3, 0.4, -0.1, 0.3, -0.2, 0.1, -0.4])


def track_epochs(epochs: int = 5) -> MetricTracker:
    """Record MSE over simulated epochs of shrinking error.

    Args:
        epochs: The number of epochs.

    Returns:
        The tracker, holding the history.
    """
    tracker = MetricTracker(mse, direction="lower")
    for epoch in range(epochs):
        tracker.increment(TARGETS + (0.5 - epoch * 0.08) * ERROR_PATTERN, TARGETS)
    return tracker
```

## Results

### Collections, weights and suites

The collection and the suite's groups report the same predictions' errors, and the weighted
score is 0.7 times the MSE plus 0.3 times the MAE.

### Gates, intervals and history

Both gates pass, each interval brackets its point estimate, and the tracker's best epoch is the
last, where the simulated error is smallest.


```python
def print_building_blocks() -> dict[str, float]:
    """Print the collection, the weighted score and the suite.

    Returns:
        The collection's results and the weighted score.
    """
    results, general = collection_results()
    print("=== MetricCollection ===")
    for name, value in results.items():
        print(f"  {name:12s} = {value:.6f}")
    print(f"  Registry (general domain): {general}")
    weighted, score = weighted_score(results)
    print("=== WeightedMetric ===")
    print(f"  Normalized weights: {weighted.normalized_weights}")
    print(f"  Weighted score: {score:.6f}")
    groups, registry_groups = suite_results()
    print("=== MetricSuite ===")
    for group_name, group_results in groups.items():
        print(f"  {group_name}:")
        for metric_name, value in group_results.items():
            print(f"    {metric_name:25s} = {value:.6f}")
    print(f"  Groups from the registry's domains: {registry_groups}")
    return {**results, "weighted score": score}


def main() -> dict[str, float | bool | int]:
    """Run the example and return its results.

    Returns:
        The collection's results, the weighted score, each gate, each interval and the best epoch.
    """
    values: dict[str, float | bool | int] = dict(print_building_blocks())
    print("=== ThresholdMetric (Quality Gate) ===")
    for name, (value, passed) in quality_gates().items():
        print(f"  {name:22s} value = {value:.6f}  passed = {passed}")
        values[f"{name} passed"] = passed
    print("=== BootstrapMetric (Confidence Intervals) ===")
    for name, (estimate, lower, upper) in confidence_intervals(jax.random.key(42)).items():
        print(f"  {name:15s} {estimate:.6f}  CI [{lower:.6f}, {upper:.6f}]")
        values[f"{name} lower"], values[f"{name} upper"] = lower, upper
    tracker = track_epochs()
    print("=== MetricTracker (Training History) ===")
    for epoch, value in enumerate(tracker.history):
        print(f"  Epoch {epoch}: MSE = {value:.6f}")
    print(f"  Best MSE: {tracker.best():.6f} at epoch {tracker.best_epoch}")
    values["best epoch"] = tracker.best_epoch
    return values


if __name__ == "__main__":
    main()
```

    === MetricCollection ===
      mse          = 0.051250
      mae          = 0.212500
      rmse         = 0.226385
      r_squared    = 0.990238
      Registry (general domain): ['charbonnier_loss', 'explained_variance', 'huber_loss', 'log_cosh_loss', 'mae', 'mape', 'max_error', 'mse', 'quantile_loss', 'r_squared', 'relative_error', 'relative_l2_error', 'rmse', 'smape']
    === WeightedMetric ===
      Normalized weights: {'mse': 0.7, 'mae': 0.3}
      Weighted score: 0.099625
    === MetricSuite ===
      error_metrics:
        mse                       = 0.051250
        mae                       = 0.212500
        rmse                      = 0.226385
      fit_quality:
        r_squared                 = 0.990238
        explained_variance        = 0.990982
      robust_metrics:
        huber_loss                = 0.025625
        log_cosh_loss             = 0.025326
      Groups from the registry's domains: ['audio', 'calibration', 'classification', 'clustering', 'distance', 'divergence', 'fairness', 'forecasting', 'general', 'generative', 'geometric', 'graph', 'image', 'information', 'manifold', 'ranking', 'segmentation', 'statistical', 'text', 'uncertainty']
    === ThresholdMetric (Quality Gate) ===
      mse below 0.1          value = 0.051250  passed = True
      r_squared above 0.95   value = 0.990238  passed = True
    === BootstrapMetric (Confidence Intervals) ===
      MSE, 95%        0.051250  CI [0.028719, 0.073750]
      R-squared, 90%  0.990238  CI [0.972522, 0.993658]
    === MetricTracker (Training History) ===
      Epoch 0: MSE = 0.018750
      Epoch 1: MSE = 0.013230
      Epoch 2: MSE = 0.008670
      Epoch 3: MSE = 0.005070
      Epoch 4: MSE = 0.002430
      Best MSE: 0.002430 at epoch 4


## Next Steps

- [Image quality and text metrics](image_quality.md): PSNR, SSIM, MS-SSIM, BLEU and ROUGE
- [Metric learning losses](metric-learning.md): contrastive, triplet, NTXent and ArcFace losses
- [Metrics API reference](../../api-reference/metrics/index.md): the composition API

<!-- substrax-notebooks: {"page": {"body": "81df10072d420018fa619fccdc72f350a56868f2f1a55947a7edd8794030aa5e", "figures": {}, "record": "0dd792734f6ac2fd8e3b14b492382b5e373f64417dd6d90ad6006351ef2d504d"}, "provenance": {"device": "1 cpu", "outputs": [[], [], [], [], [], [], [], ["text"]], "seconds": 2.4, "sha256": "03786f09cefee80fc05fe4ac8bb9daa14ec51930170b05b0cefed4a1f40647a3", "source": "examples/metrics/composition.py", "versions": {"calibrax": "0.1.17", "jax": "0.11.1", "python": "3.12.14", "substrax": "0.1.27"}}} -->
