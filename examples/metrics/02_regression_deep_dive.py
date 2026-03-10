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
# # Regression Metrics Deep Dive
#
# | | |
# |---|---|
# | **Level** | Tier 1: Quick Reference |
# | **Time** | ~10 minutes |
# | **Prerequisites** | `01_quickstart.py`, basic statistics |
# | **Metrics covered** | MSE, MAE, RMSE, R-squared, MAPE, SMAPE, Huber, quantile loss, log-cosh |
# | **Key concepts** | Outlier sensitivity, robust losses, percentage errors |

# %%
"""Regression metrics deep dive: all 12 regression metrics with interpretation.

Demonstrates:
- Computing each of the 12 regression metrics
- Comparing MSE vs MAE vs Huber loss on data with outliers
- Quantile loss at different quantile levels
- SMAPE vs MAPE: symmetric vs asymmetric percentage errors
"""

import jax.numpy as jnp

from calibrax.metrics.functional.regression import (
    explained_variance,
    huber_loss,
    log_cosh_loss,
    mae,
    mape,
    max_error,
    mse,
    quantile_loss,
    r_squared,
    relative_error,
    rmse,
    smape,
)


def main() -> None:
    """Run regression metrics deep dive."""
    # -- Clean data --------------------------------------------------------
    targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    predictions = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.3, 7.8])

    print("=== All 12 Regression Metrics (clean data) ===")
    metrics = {
        "MSE": mse(predictions, targets),
        "MAE": mae(predictions, targets),
        "RMSE": rmse(predictions, targets),
        "R-squared": r_squared(predictions, targets),
        "MAPE": mape(predictions, targets),
        "SMAPE": smape(predictions, targets),
        "Relative Error": relative_error(predictions, targets),
        "Explained Variance": explained_variance(predictions, targets),
        "Max Error": max_error(predictions, targets),
        "Huber Loss (delta=1.0)": huber_loss(predictions, targets, delta=1.0),
        "Quantile Loss (q=0.5)": quantile_loss(predictions, targets, quantile=0.5),
        "Log-Cosh Loss": log_cosh_loss(predictions, targets),
    }
    for name, value in metrics.items():
        print(f"  {name:30s} = {value:.6f}")

    # -- Outlier comparison ------------------------------------------------
    print("\n=== Outlier Sensitivity: MSE vs MAE vs Huber ===")
    targets_clean = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    preds_clean = jnp.array([1.1, 2.1, 3.1, 4.1, 5.1])

    # Inject a single large outlier in the predictions
    preds_outlier = jnp.array([1.1, 2.1, 3.1, 4.1, 15.0])

    print("  Without outlier:")
    print(f"    MSE   = {mse(preds_clean, targets_clean):.6f}")
    print(f"    MAE   = {mae(preds_clean, targets_clean):.6f}")
    print(f"    Huber = {huber_loss(preds_clean, targets_clean, delta=1.0):.6f}")

    print("  With outlier (prediction[4] = 15.0, target[4] = 5.0):")
    print(f"    MSE   = {mse(preds_outlier, targets_clean):.6f}")
    print(f"    MAE   = {mae(preds_outlier, targets_clean):.6f}")
    print(f"    Huber = {huber_loss(preds_outlier, targets_clean, delta=1.0):.6f}")
    print("  MSE is heavily influenced by outliers; MAE is more robust.")
    print("  Huber provides a smooth transition: quadratic near zero, linear for large errors.")

    # -- Quantile loss at different levels ---------------------------------
    print("\n=== Quantile Loss at Different Quantiles ===")
    targets_q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # Predictions slightly below targets (under-predicting)
    preds_low = jnp.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    # Predictions slightly above targets (over-predicting)
    preds_high = jnp.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        loss_low = quantile_loss(preds_low, targets_q, quantile=q)
        loss_high = quantile_loss(preds_high, targets_q, quantile=q)
        print(f"  q={q:.2f}  under-predict={loss_low:.4f}  over-predict={loss_high:.4f}")
    print("  Higher quantiles penalize under-prediction more;")
    print("  lower quantiles penalize over-prediction more.")

    # -- SMAPE vs MAPE comparison ------------------------------------------
    print("\n=== SMAPE vs MAPE: Symmetric vs Asymmetric ===")
    targets_pct = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    preds_pct = jnp.array([12.0, 18.0, 35.0, 38.0, 55.0])

    print(f"  MAPE  = {mape(preds_pct, targets_pct):.6f}")
    print(f"  SMAPE = {smape(preds_pct, targets_pct):.6f}")

    # Swap predictions and targets to show asymmetry
    print("  After swapping predictions <-> targets:")
    print(f"  MAPE  = {mape(targets_pct, preds_pct):.6f}  (changes -- asymmetric)")
    print(f"  SMAPE = {smape(targets_pct, preds_pct):.6f}  (same -- symmetric)")

    # -- Log-cosh: smooth approximation to MAE -----------------------------
    print("\n=== Log-Cosh vs MAE: Smoothness ===")
    targets_s = jnp.array([0.0, 0.0, 0.0, 0.0])
    for delta in [0.01, 0.1, 1.0, 5.0, 10.0]:
        preds_s = jnp.array([delta, delta, delta, delta])
        lc = log_cosh_loss(preds_s, targets_s)
        ma = mae(preds_s, targets_s)
        print(f"  error={delta:6.2f}  log_cosh={lc:.6f}  mae={ma:.6f}")
    print("  Log-cosh approximates 0.5*MSE for small errors and MAE for large errors.")
    print("  It is twice-differentiable everywhere (unlike MAE).")


if __name__ == "__main__":
    main()
