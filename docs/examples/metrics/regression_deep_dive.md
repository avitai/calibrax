# Regression Metrics Deep Dive

> Ran in 2.4 s on 1 cpu (python 3.12.14, jax 0.11.1, substrax 0.1.27, calibrax 0.1.17), from [`examples/metrics/regression_deep_dive.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/regression_deep_dive.py) ([notebook](https://github.com/avitai/calibrax/blob/main/examples/metrics/regression_deep_dive.ipynb)). Run it from the repository root:

```bash
source ./activate.sh
uv run python examples/metrics/regression_deep_dive.py
```

| Metadata | Value |
| --- | --- |
| **Tier** | Tutorial |
| **Devices** | CPU-compatible |
| **Prerequisites** | The quickstart example; basic statistics |

## Overview

calibrax's regression metrics span common error measures, robust alternatives, percentage-based
losses and probabilistic ensemble scoring. This example computes the same-shape regression
metrics on clean data, scores an ensemble forecast with CRPS, and then shows how an outlier moves
MSE, MAE and Huber loss differently, how quantile loss weighs under- against over-prediction, why
SMAPE is symmetric where MAPE is not, and where log-cosh stands between MSE and MAE. Squared-error
metrics amplify large deviations; the robust alternatives stay smooth near outliers while
remaining differentiable, and knowing which is which decides how a model is judged.

## Learning Goals

1. Compute the same-shape regression metrics on one dataset
2. Score an ensemble forecast with CRPS
3. Compare how one outlier moves MSE, MAE and Huber loss
4. Use quantile loss to penalise under- and over-prediction asymmetrically
5. Distinguish SMAPE (symmetric) from MAPE (asymmetric)
6. Choose log-cosh as a twice-differentiable alternative to MAE

## Setup


```python
import jax
import jax.numpy as jnp

from calibrax.metrics.functional.regression import (
    crps,
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
```

## Core Concepts

| Metric | Description | Outlier sensitivity |
| --- | --- | --- |
| `mse` | Mean squared error | High: squares amplify large errors |
| `mae` | Mean absolute error | Low: linear in the error |
| `rmse` | Root mean squared error | High: MSE in the original units |
| `r_squared` | Coefficient of determination | High: based on MSE |
| `mape` | Mean absolute percentage error | Moderate: relative to the target |
| `smape` | Symmetric MAPE | Moderate: symmetric under a prediction and target swap |
| `relative_error` | Mean relative error | Moderate |
| `explained_variance` | Variance of the residuals against the targets | High |
| `max_error` | Worst-case absolute error | Extreme: one point decides it |
| `huber_loss` | Quadratic near zero, linear far away | Set by `delta` |
| `quantile_loss` | Asymmetric loss for quantile regression | Low |
| `log_cosh_loss` | Smooth approximation to MAE | Low |
| `crps` | Continuous ranked probability score of an ensemble forecast | Depends on the spread |

Huber loss is quadratic for errors smaller than `delta` and linear beyond it. At quantile `q`,
quantile loss weighs under-prediction by `q` and over-prediction by `1 - q`, so `q = 0.5` is half
the MAE. Log-cosh behaves like `0.5 * MSE` for small errors and like `MAE - log(2)` for large ones,
and unlike MAE it is twice-differentiable everywhere.

## Implementation

### Step 1: Compute the same-shape metrics

Each metric takes predictions and targets of one shape.


```python
TARGETS = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
PREDICTIONS = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.3, 7.8])


def same_shape_metrics(predictions: jax.Array, targets: jax.Array) -> dict[str, float]:
    """Compute every same-shape regression metric.

    Args:
        predictions: The model's predictions.
        targets: The true values.

    Returns:
        Each metric's value by name.
    """
    return {
        "MSE": float(mse(predictions, targets)),
        "MAE": float(mae(predictions, targets)),
        "RMSE": float(rmse(predictions, targets)),
        "R-squared": float(r_squared(predictions, targets)),
        "MAPE": float(mape(predictions, targets)),
        "SMAPE": float(smape(predictions, targets)),
        "Relative Error": float(relative_error(predictions, targets)),
        "Explained Variance": float(explained_variance(predictions, targets)),
        "Max Error": float(max_error(predictions, targets)),
        "Huber Loss (delta=1.0)": float(huber_loss(predictions, targets, delta=1.0)),
        "Quantile Loss (q=0.5)": float(quantile_loss(predictions, targets, quantile=0.5)),
        "Log-Cosh Loss": float(log_cosh_loss(predictions, targets)),
    }
```

### Step 2: Score an ensemble forecast with CRPS

CRPS takes the ensemble members along a dimension of their own: one row of members per target.
A lower CRPS means a sharper, better calibrated ensemble.


```python
ENSEMBLE = jnp.array([[0.8, 1.0, 1.2], [1.8, 2.0, 2.2], [2.6, 3.0, 3.4]])
ENSEMBLE_TARGETS = jnp.array([1.0, 2.0, 3.0])


def ensemble_score(ensemble: jax.Array, targets: jax.Array) -> float:
    """Score an ensemble forecast.

    Args:
        ensemble: The members' forecasts, one row per target.
        targets: The true values.

    Returns:
        The forecast's CRPS.
    """
    return float(crps(ensemble, targets))
```

### Step 3: Compare outlier sensitivity

One prediction is moved from 5.1 to 15.0. MSE jumps, MAE grows linearly, and Huber loss bounds
the outlier's contribution.


```python
CLEAN_TARGETS = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
CLEAN_PREDICTIONS = jnp.array([1.1, 2.1, 3.1, 4.1, 5.1])
OUTLIER_PREDICTIONS = jnp.array([1.1, 2.1, 3.1, 4.1, 15.0])


def outlier_sensitivity(predictions: jax.Array, targets: jax.Array) -> dict[str, float]:
    """Compute the three losses an outlier moves differently.

    Args:
        predictions: The model's predictions.
        targets: The true values.

    Returns:
        MSE, MAE and Huber loss by name.
    """
    return {
        "MSE": float(mse(predictions, targets)),
        "MAE": float(mae(predictions, targets)),
        "Huber": float(huber_loss(predictions, targets, delta=1.0)),
    }
```

### Step 4: Sweep the quantile of quantile loss

The same targets are scored against predictions half a unit below and half a unit above them, at
five quantiles.


```python
QUANTILE_TARGETS = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
UNDER_PREDICTIONS = QUANTILE_TARGETS - 0.5
OVER_PREDICTIONS = QUANTILE_TARGETS + 0.5
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def quantile_sweep() -> dict[float, tuple[float, float]]:
    """Score under- and over-prediction at each quantile.

    Returns:
        The under-prediction and over-prediction losses at each quantile.
    """
    return {
        q: (
            float(quantile_loss(UNDER_PREDICTIONS, QUANTILE_TARGETS, quantile=q)),
            float(quantile_loss(OVER_PREDICTIONS, QUANTILE_TARGETS, quantile=q)),
        )
        for q in QUANTILES
    }
```

### Step 5: Swap predictions and targets in MAPE and SMAPE

MAPE divides by the target, so swapping the arguments changes it; SMAPE divides by their average
and does not change.


```python
PERCENT_TARGETS = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
PERCENT_PREDICTIONS = jnp.array([12.0, 18.0, 35.0, 38.0, 55.0])


def percentage_symmetry() -> dict[str, float]:
    """Compute MAPE and SMAPE both ways round.

    Returns:
        Each metric before and after the swap.
    """
    return {
        "MAPE": float(mape(PERCENT_PREDICTIONS, PERCENT_TARGETS)),
        "SMAPE": float(smape(PERCENT_PREDICTIONS, PERCENT_TARGETS)),
        "MAPE swapped": float(mape(PERCENT_TARGETS, PERCENT_PREDICTIONS)),
        "SMAPE swapped": float(smape(PERCENT_TARGETS, PERCENT_PREDICTIONS)),
    }
```

### Step 6: Compare log-cosh with MAE across error sizes

Four equal errors of each size are scored against zero targets.


```python
ERRORS = (0.01, 0.1, 1.0, 5.0, 10.0)


def log_cosh_against_mae() -> dict[float, tuple[float, float]]:
    """Score one error size at a time with log-cosh and MAE.

    Returns:
        The log-cosh loss and the MAE at each error size.
    """
    zeros = jnp.zeros(4)
    return {
        error: (
            float(log_cosh_loss(jnp.full(4, error), zeros)),
            float(mae(jnp.full(4, error), zeros)),
        )
        for error in ERRORS
    }
```

## Results

### The same-shape metrics and CRPS

The clean predictions sit within 0.3 of every target, so R-squared is close to 1 and the
squared-error metrics are small.

### How each loss responds

The outlier, quantile, percentage and log-cosh comparisons show each loss's behaviour in numbers:
MSE grows with the square of the outlier, the quantile sweep tilts with `q`, MAPE changes under the
swap while SMAPE does not, and log-cosh approaches MAE as the error grows.


```python
def print_comparisons() -> dict[str, float]:
    """Print the outlier, quantile, percentage and log-cosh comparisons.

    Returns:
        The losses with the outlier, and MAPE and SMAPE after the swap.
    """
    clean = outlier_sensitivity(CLEAN_PREDICTIONS, CLEAN_TARGETS)
    outlier = outlier_sensitivity(OUTLIER_PREDICTIONS, CLEAN_TARGETS)
    print("=== Outlier Sensitivity: MSE vs MAE vs Huber ===")
    for name in clean:
        print(f"  {name:6s} without outlier = {clean[name]:.6f}  with = {outlier[name]:.6f}")
    print("=== Quantile Loss at Different Quantiles ===")
    for q, (under, over) in quantile_sweep().items():
        print(f"  q={q:.2f}  under-predict={under:.4f}  over-predict={over:.4f}")
    percentages = percentage_symmetry()
    print("=== SMAPE vs MAPE: Symmetric vs Asymmetric ===")
    for name, value in percentages.items():
        print(f"  {name:14s} = {value:.6f}")
    print("=== Log-Cosh vs MAE: Smoothness ===")
    for error, (log_cosh, absolute) in log_cosh_against_mae().items():
        print(f"  error={error:6.2f}  log_cosh={log_cosh:.6f}  mae={absolute:.6f}")
    return {
        "mse_with_outlier": outlier["MSE"],
        "mae_with_outlier": outlier["MAE"],
        "huber_with_outlier": outlier["Huber"],
        "mape_swapped": percentages["MAPE swapped"],
        "smape_swapped": percentages["SMAPE swapped"],
    }


def main() -> dict[str, float]:
    """Run the example and return its results.

    Returns:
        The clean-data metrics, the ensemble's CRPS and the comparisons' key values.
    """
    metrics = same_shape_metrics(PREDICTIONS, TARGETS)
    print("=== Same-Shape Regression Metrics (clean data) ===")
    for name, value in metrics.items():
        print(f"  {name:30s} = {value:.6f}")
    score = ensemble_score(ENSEMBLE, ENSEMBLE_TARGETS)
    print("=== CRPS: Ensemble Forecast Quality ===")
    print(f"  CRPS = {score:.6f}")
    return {**metrics, "CRPS": score, **print_comparisons()}


if __name__ == "__main__":
    main()
```

    === Same-Shape Regression Metrics (clean data) ===
      MSE                            = 0.031250
      MAE                            = 0.162500
      RMSE                           = 0.176777
      R-squared                      = 0.994048
      MAPE                           = 0.046399
      SMAPE                          = 0.045775
      Relative Error                 = 0.035007
      Explained Variance             = 0.994077
      Max Error                      = 0.300000
      Huber Loss (delta=1.0)         = 0.015625
      Quantile Loss (q=0.5)          = 0.081250
      Log-Cosh Loss                  = 0.015489
    === CRPS: Ensemble Forecast Quality ===
      CRPS = 0.059259
    === Outlier Sensitivity: MSE vs MAE vs Huber ===
      MSE    without outlier = 0.010000  with = 20.008001
      MAE    without outlier = 0.100000  with = 2.080000
      Huber  without outlier = 0.005000  with = 1.904000
    === Quantile Loss at Different Quantiles ===
      q=0.10  under-predict=0.0500  over-predict=0.4500
      q=0.25  under-predict=0.1250  over-predict=0.3750
      q=0.50  under-predict=0.2500  over-predict=0.2500
      q=0.75  under-predict=0.3750  over-predict=0.1250
      q=0.90  under-predict=0.4500  over-predict=0.0500
    === SMAPE vs MAPE: Symmetric vs Asymmetric ===
      MAPE           = 0.123333
      SMAPE          = 0.117490
      MAPE swapped   = 0.112835
      SMAPE swapped  = 0.117490
    === Log-Cosh vs MAE: Smoothness ===
      error=  0.01  log_cosh=0.000050  mae=0.010000
      error=  0.10  log_cosh=0.004992  mae=0.100000
      error=  1.00  log_cosh=0.433781  mae=1.000000
      error=  5.00  log_cosh=4.306898  mae=5.000000
      error= 10.00  log_cosh=9.306852  mae=10.000000


## Next Steps

- [Classification metrics](classification.md): binary classification, calibration and
  segmentation
- [Regression API reference](../../api-reference/metrics/regression.md): every regression
  function's signature
- [Metrics quickstart](quickstart.md): the basics, if you skipped them

<!-- substrax-notebooks: {"page": {"body": "8d4ff96d778e6dabcc998ceab4a3f3173fd5a0b3e929a9339e846b2473a1ffeb", "figures": {}, "record": "361794e14bf21f47357892ca2521567cc332014305767870245fa26e80d71c7b"}, "provenance": {"device": "1 cpu", "outputs": [[], [], [], [], [], [], [], ["text"]], "seconds": 2.4, "sha256": "db4056bed932e470ab3bc7d967e695a3b03d8f07ec8aa0d6583012e8f2ed0be4", "source": "examples/metrics/regression_deep_dive.py", "versions": {"calibrax": "0.1.17", "jax": "0.11.1", "python": "3.12.14", "substrax": "0.1.27"}}} -->
