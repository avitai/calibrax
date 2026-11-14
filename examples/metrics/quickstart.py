# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# # Metrics Quickstart
#
# | | |
# |---|---|
# | **Level** | Tier 1: Quick Reference |
# | **Time** | ~5 minutes |
# | **Prerequisites** | Basic Python, JAX arrays |
# | **Metrics covered** | MSE, MAE, R-squared, calculate_all, MetricRegistry |
# | **Key concepts** | Individual computation, batch evaluation, registry queries |

# %%
"""Quickstart: basic metric computation, batch evaluation, and registry queries.

Demonstrates:
- Computing individual regression metrics (MSE, MAE, R-squared)
- Using calculate_all() for batch computation
- Querying MetricRegistry: list_names(), list_by_domain(), list_by_tier()
- Inspecting MetricEntry fields (tier, domain, direction, invariances)
"""

import jax.numpy as jnp

from calibrax.metrics import (
    calculate_all,
    MetricRegistry,
    MetricTier,
)
from calibrax.metrics.functional.regression import mae, mse, r_squared


def main() -> None:
    """Run quickstart examples for the calibrax metrics module."""
    # -- Sample data -------------------------------------------------------
    targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = jnp.array([1.1, 2.3, 2.8, 4.2, 4.7])

    # -- 1. Individual metric computation ----------------------------------
    print("=== Individual Metrics ===")
    mse_val = mse(predictions, targets)
    mae_val = mae(predictions, targets)
    r2_val = r_squared(predictions, targets)
    print(f"  MSE:       {mse_val:.6f}")
    print(f"  MAE:       {mae_val:.6f}")
    print(f"  R-squared: {r2_val:.6f}")

    # -- 2. Batch computation with calculate_all ----------------------------
    print("\n=== calculate_all (all general metrics) ===")
    all_results = calculate_all(predictions, targets)
    for name, value in sorted(all_results.items()):
        print(f"  {name:25s} = {value:.6f}")

    # -- 3. Selective batch computation ------------------------------------
    print("\n=== calculate_all (selected subset) ===")
    selected = calculate_all(
        predictions,
        targets,
        metrics=["mse", "mae", "rmse", "r_squared"],
    )
    for name, value in selected.items():
        print(f"  {name:12s} = {value:.6f}")

    # -- 4. Registry queries -----------------------------------------------
    registry = MetricRegistry()

    print("\n=== Registry: all registered metric names ===")
    all_names = sorted(registry.list_names())
    print(f"  Total registered: {len(all_names)}")
    print(f"  First 10: {all_names[:10]}")

    print("\n=== Registry: metrics by domain ===")
    for domain in ("general", "classification", "distance", "image"):
        entries = registry.list_by_domain(domain)
        names = [e.name for e in entries]
        print(f"  {domain:20s} ({len(names)}): {names}")

    print("\n=== Registry: Tier 0 pure functions ===")
    tier0 = registry.list_by_tier(MetricTier.PURE_FUNCTION)
    print(f"  Count: {len(tier0)}")

    # -- 5. Inspecting a MetricEntry ----------------------------------------
    print("\n=== MetricEntry fields for 'mse' ===")
    entry = registry.get("mse")
    print(f"  name:              {entry.name}")
    print(f"  tier:              {entry.tier}")
    print(f"  domain:            {entry.domain}")
    print(f"  direction:         {entry.direction}")
    print(f"  description:       {entry.description}")
    print(f"  signature:         {entry.signature}")
    print(f"  is_true_metric:    {entry.properties.is_true_metric}")
    print(f"  is_symmetric:      {entry.properties.is_symmetric}")
    print(f"  is_differentiable: {entry.properties.is_differentiable}")
    print(f"  is_jit_compatible: {entry.properties.is_jit_compatible}")
    print(f"  invariances:       {entry.properties.invariances}")

    # -- 6. Specialized queries ---------------------------------------------
    print("\n=== Registry: true metrics (satisfy metric space axioms) ===")
    true_metrics = registry.list_true_metrics()
    print(f"  Count: {len(true_metrics)}")
    print(f"  Names: {[e.name for e in true_metrics[:8]]}...")

    print("\n=== Registry: rotation-invariant metrics ===")
    rot_inv = registry.list_by_invariance("rotation")
    print(f"  Names: {[e.name for e in rot_inv]}")


if __name__ == "__main__":
    main()
