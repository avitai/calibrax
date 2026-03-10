# Regression Metrics Deep Dive

| | |
|---|---|
| **Level** | Beginner |
| **Time** | ~10 minutes |
| **Prerequisites** | [Quickstart](quickstart.md) |
| **Format** | Python + Jupyter |

## Overview

Calibrax provides 12 regression metrics that span common error measures, robust alternatives, and percentage-based losses. This example computes all 12 on clean data, then shows how outliers affect MSE, MAE, and Huber loss differently. It also demonstrates quantile loss at various levels and compares symmetric vs asymmetric percentage errors (SMAPE vs MAPE).

Understanding when each metric is appropriate is essential for model evaluation. Squared-error metrics amplify large deviations; robust alternatives like Huber and log-cosh provide smoother behaviour near outliers while remaining differentiable.

## What You'll Learn

1. Compute all 12 regression metrics on a single dataset
2. Compare outlier sensitivity across MSE, MAE, and Huber loss
3. Use quantile loss to penalize under- vs over-prediction asymmetrically
4. Distinguish SMAPE (symmetric) from MAPE (asymmetric) percentage errors
5. Choose log-cosh as a twice-differentiable alternative to MAE

## Files

- **Python Script**: [`examples/metrics/02_regression_deep_dive.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/02_regression_deep_dive.py)
- **Jupyter Notebook**: [`examples/metrics/02_regression_deep_dive.ipynb`](https://github.com/avitai/calibrax/blob/main/examples/metrics/02_regression_deep_dive.ipynb)

## Quick Start

```bash
source activate.sh && python examples/metrics/02_regression_deep_dive.py
```

## Key Concepts

### The 12 Regression Metrics

| Metric | Description | Outlier Sensitivity |
|---|---|---|
| `mse` | Mean Squared Error | High -- squares amplify large errors |
| `mae` | Mean Absolute Error | Low -- linear in error magnitude |
| `rmse` | Root Mean Squared Error | High -- same as MSE, in original units |
| `r_squared` | Coefficient of determination | High -- based on MSE |
| `mape` | Mean Absolute Percentage Error | Moderate -- relative to target magnitude |
| `smape` | Symmetric MAPE | Moderate -- symmetric under prediction/target swap |
| `relative_error` | Mean relative error | Moderate |
| `explained_variance` | Variance of residuals vs targets | High |
| `max_error` | Worst-case absolute error | Extreme -- driven by single worst point |
| `huber_loss` | Quadratic near zero, linear far away | Configurable via `delta` |
| `quantile_loss` | Asymmetric loss for quantile regression | Low |
| `log_cosh_loss` | Smooth approximation to MAE | Low |

```python
from calibrax.metrics.functional.regression import (
    explained_variance, huber_loss, log_cosh_loss, mae, mape,
    max_error, mse, quantile_loss, r_squared, relative_error, rmse, smape,
)
```

### Outlier Sensitivity

When data contains outliers, squared-error metrics (MSE, RMSE) can be dominated by a single bad prediction. The example injects one outlier and compares the effect:

```python
preds_clean = jnp.array([1.1, 2.1, 3.1, 4.1, 5.1])
preds_outlier = jnp.array([1.1, 2.1, 3.1, 4.1, 15.0])  # outlier at index 4

# MSE jumps dramatically; MAE grows linearly; Huber caps the contribution
mse(preds_outlier, targets)    # large increase
mae(preds_outlier, targets)    # moderate increase
huber_loss(preds_outlier, targets, delta=1.0)  # bounded increase
```

Huber loss transitions from quadratic (for errors smaller than `delta`) to linear (for errors larger than `delta`), providing a tunable trade-off.

### Quantile Loss

Quantile loss penalizes under-prediction and over-prediction asymmetrically. At quantile `q`, under-prediction is penalized by factor `q` and over-prediction by factor `1-q`.

```python
# q=0.9: heavily penalizes under-prediction (useful for safety margins)
quantile_loss(predictions, targets, quantile=0.9)

# q=0.1: heavily penalizes over-prediction
quantile_loss(predictions, targets, quantile=0.1)

# q=0.5: equivalent to MAE (symmetric)
quantile_loss(predictions, targets, quantile=0.5)
```

### SMAPE vs MAPE

MAPE is asymmetric: swapping predictions and targets changes the result. SMAPE normalizes by the average of prediction and target, producing a symmetric measure.

```python
mape(predictions, targets)   # changes if you swap arguments
smape(predictions, targets)  # same value regardless of argument order
```

### Log-Cosh: Smooth MAE Alternative

Log-cosh behaves like `0.5 * MSE` for small errors and like `MAE` for large errors. Unlike MAE, it is twice-differentiable everywhere, which makes it well-suited for gradient-based optimisation.

```python
# For small errors: log_cosh ≈ 0.5 * error^2
# For large errors: log_cosh ≈ |error| - log(2)
log_cosh_loss(predictions, targets)
```

## Example Code

The script starts by computing all 12 metrics on clean data:

```python
targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
predictions = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.3, 7.8])

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
```

## Next Steps

- [Classification Metrics](classification.md) -- binary classification, calibration, and segmentation
- [API Reference: `calibrax.metrics.functional.regression`](../../api-reference/metrics/regression.md) -- full signatures for all 12 functions
- [Quickstart](quickstart.md) -- if you skipped the basics
