# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Metric Composition and Wrappers
#
# | | |
# |---|---|
# | **Level** | Tier 2: Tutorial |
# | **Time** | ~20 minutes |
# | **Prerequisites** | `01_quickstart.py`, `02_regression_deep_dive.py` |
# | **Metrics covered** | MetricCollection, WeightedMetric, MetricSuite, ThresholdMetric |
# | **Key concepts** | Grouping, weighting, quality gates, confidence intervals, tracking |

# %%
"""Composition framework: collections, weighted metrics, suites, and wrappers.

Demonstrates:
- MetricCollection for grouping regression metrics
- WeightedMetric for multi-objective evaluation
- MetricSuite with domain-based groups
- ThresholdMetric for quality gates
- BootstrapMetric for confidence intervals
- MetricTracker for tracking across evaluations
"""

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


def main() -> None:
    """Run composition framework examples."""
    # -- Sample data -------------------------------------------------------
    targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    predictions = jnp.array([1.2, 1.8, 3.3, 3.9, 5.2, 5.7, 7.1, 8.3])

    # -- 1. MetricCollection -----------------------------------------------
    print("=== MetricCollection ===")
    collection = MetricCollection(
        {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r_squared": r_squared,
        }
    )
    results = collection.compute_functional(predictions, targets)
    for name, value in results.items():
        print(f"  {name:12s} = {value:.6f}")

    print(f"\n  Metric names: {collection.names}")

    # Build from registry
    registry_collection = MetricCollection.from_registry(domain="general")
    print(f"  Registry (general domain): {sorted(registry_collection.names)}")

    # -- 2. WeightedMetric -------------------------------------------------
    print("\n=== WeightedMetric ===")
    # Multi-objective: 70% weight on MSE, 30% on MAE
    weighted = WeightedMetric({"mse": 0.7, "mae": 0.3})
    score = weighted.compute(results)
    print(f"  Weights: {weighted.weights}")
    print(f"  Normalized weights: {weighted.normalized_weights}")
    print(f"  Weighted score: {score:.6f}")
    print(f"  Breakdown: 0.7 * {results['mse']:.4f} + 0.3 * {results['mae']:.4f} = {score:.6f}")

    # -- 3. MetricSuite ---------------------------------------------------
    print("\n=== MetricSuite ===")
    suite = MetricSuite()
    suite.add_group("error_metrics", ["mse", "mae", "rmse"])
    suite.add_group("fit_quality", ["r_squared", "explained_variance"])
    suite.add_group("robust_metrics", ["huber_loss", "log_cosh_loss"])

    suite_results = suite.compute_all(predictions, targets)
    for group_name, group_results in suite_results.items():
        print(f"  {group_name}:")
        for metric_name, value in group_results.items():
            print(f"    {metric_name:25s} = {value:.6f}")

    print(f"\n  Groups: {suite.list_groups()}")

    # Auto-create from registry domains
    auto_suite = MetricSuite.from_registry_domains()
    print(f"  Auto-created groups: {auto_suite.list_groups()}")

    # -- 4. ThresholdMetric ------------------------------------------------
    print("\n=== ThresholdMetric (Quality Gate) ===")
    # MSE must be below 0.1 (max_value for lower-is-better metric)
    mse_gate = ThresholdMetric("mse", max_value=0.1)
    result = mse_gate.evaluate(predictions, targets)
    print(f"  MSE threshold: max_value={mse_gate.max_value}")
    print(f"  Value:  {result['value']:.6f}")
    print(f"  Passed: {result['passed']}")

    # R-squared must be above 0.95 (min_value for higher-is-better metric)
    r2_gate = ThresholdMetric("r_squared", min_value=0.95)
    result_r2 = r2_gate.evaluate(predictions, targets)
    print(f"\n  R-squared threshold: min_value={r2_gate.min_value}")
    print(f"  Value:  {result_r2['value']:.6f}")
    print(f"  Passed: {result_r2['passed']}")

    # -- 5. BootstrapMetric ------------------------------------------------
    print("\n=== BootstrapMetric (Confidence Intervals) ===")
    bootstrap = BootstrapMetric(mse, num_bootstraps=200, confidence=0.95, seed=42)
    boot_result = bootstrap.compute(predictions, targets)
    print(f"  MSE point estimate: {boot_result['value']:.6f}")
    print(f"  95% CI: [{boot_result['lower']:.6f}, {boot_result['upper']:.6f}]")
    print(f"  Bootstrap samples:  {len(boot_result['samples'])}")

    bootstrap_r2 = BootstrapMetric(r_squared, num_bootstraps=200, confidence=0.90, seed=42)
    boot_r2 = bootstrap_r2.compute(predictions, targets)
    print(f"\n  R-squared point estimate: {boot_r2['value']:.6f}")
    print(f"  90% CI: [{boot_r2['lower']:.6f}, {boot_r2['upper']:.6f}]")

    # -- 6. MetricTracker --------------------------------------------------
    print("\n=== MetricTracker (Training History) ===")
    tracker = MetricTracker(mse, direction="lower")

    # Simulate improving predictions over 5 epochs
    for epoch in range(5):
        noise_scale = 0.5 - epoch * 0.08
        noisy_preds = targets + noise_scale * jnp.array(
            [0.2, -0.3, 0.4, -0.1, 0.3, -0.2, 0.1, -0.4]
        )
        value = tracker.increment(noisy_preds, targets)
        print(f"  Epoch {epoch}: MSE = {value:.6f}")

    print(f"\n  Best MSE:   {tracker.best():.6f}")
    print(f"  Best epoch: {tracker.best_epoch}")
    print(f"  History:    {[f'{v:.4f}' for v in tracker.history]}")


if __name__ == "__main__":
    main()
