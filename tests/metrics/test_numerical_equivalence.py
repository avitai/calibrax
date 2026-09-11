"""Numerical-equivalence checks against established reference libraries."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import sqrtm
from scipy.spatial.distance import cosine as scipy_cosine, jensenshannon
from scipy.special import rel_entr
from sklearn.metrics import (
    accuracy_score,
    f1_score as sklearn_f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from calibrax.metrics.functional.classification import accuracy, f1_score, roc_auc
from calibrax.metrics.functional.distance import cosine_distance
from calibrax.metrics.functional.divergence import js_divergence, kl_divergence
from calibrax.metrics.functional.regression import crps, mae, mse, r_squared, rmse
from calibrax.metrics.plugins.image import FIDMetric


ABS_TOL = 1e-6


def _assert_close(actual: object, expected: float) -> None:
    """Assert JAX/Python scalar numerical equivalence."""
    assert float(actual) == pytest.approx(expected, abs=ABS_TOL)


def _numpy_ensemble_crps(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute empirical ensemble CRPS using NumPy."""
    forecast_error = np.mean(np.abs(predictions - targets[:, None]), axis=1)
    pairwise = np.abs(predictions[:, :, None] - predictions[:, None, :])
    ensemble_spread = 0.5 * np.mean(pairwise, axis=(1, 2))
    return float(np.mean(forecast_error - ensemble_spread))


def _scipy_frechet_distance(real: np.ndarray, generated: np.ndarray) -> float:
    """Fréchet distance between fitted Gaussians in float64 with SciPy's ``sqrtm``."""
    real, generated = real.astype(np.float64), generated.astype(np.float64)
    cov_real = np.cov(real, rowvar=False)
    cov_generated = np.cov(generated, rowvar=False)
    cross = sqrtm(cov_real @ cov_generated).real
    diff = real.mean(axis=0) - generated.mean(axis=0)
    return float(diff @ diff + np.trace(cov_real) + np.trace(cov_generated) - 2.0 * np.trace(cross))


def test_regression_metrics_match_sklearn_references() -> None:
    """Regression metrics match scikit-learn definitions."""
    targets = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    predictions = np.array([1.1, 1.9, 3.2, 3.8], dtype=np.float32)

    _assert_close(mse(predictions, targets), mean_squared_error(targets, predictions))
    _assert_close(mae(predictions, targets), mean_absolute_error(targets, predictions))
    _assert_close(rmse(predictions, targets), mean_squared_error(targets, predictions) ** 0.5)
    _assert_close(r_squared(predictions, targets), r2_score(targets, predictions))


def test_crps_matches_independent_numpy_reference() -> None:
    """CRPS matches an independent empirical ensemble implementation."""
    targets = np.array([1.0, 0.5], dtype=np.float32)
    predictions = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.5, 1.0],
        ],
        dtype=np.float32,
    )

    _assert_close(crps(predictions, targets), _numpy_ensemble_crps(predictions, targets))


def test_classification_metrics_match_sklearn_references() -> None:
    """Classification metrics match scikit-learn definitions."""
    targets = np.array([0, 1, 1, 0, 1, 0], dtype=np.int32)
    predictions = np.array([0, 1, 0, 0, 1, 1], dtype=np.int32)
    scores = np.array([0.05, 0.91, 0.40, 0.20, 0.85, 0.62], dtype=np.float32)

    _assert_close(accuracy(predictions, targets), accuracy_score(targets, predictions))
    _assert_close(f1_score(predictions, targets), sklearn_f1_score(targets, predictions))
    _assert_close(roc_auc(scores, targets), roc_auc_score(targets, scores))


def test_distance_and_divergence_metrics_match_scipy_references() -> None:
    """Distance and divergence metrics match SciPy definitions."""
    first = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    second = np.array([2.0, 0.0, 1.0], dtype=np.float32)
    p = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    q = np.array([0.1, 0.7, 0.2], dtype=np.float32)

    _assert_close(cosine_distance(first, second), scipy_cosine(first, second))
    _assert_close(kl_divergence(p, q), float(np.sum(rel_entr(p, q))))
    _assert_close(js_divergence(p, q), float(jensenshannon(p, q, base=np.e) ** 2))


@pytest.mark.parametrize("dim", [16, 64])
def test_fid_metric_matches_scipy_reference(dim: int) -> None:
    """FIDMetric matches the float64 SciPy Fréchet distance on non-commuting covariances.

    The functional form is within 1.43e-5 relative of this reference over dims 16 and 64 and
    seeds 0 to 2; the tolerance sits above that bound. Covariances that do not commute are the
    case an eigendecomposition of their plain product gets wrong.
    """
    rng = np.random.default_rng(0)
    real = (rng.standard_normal((400, dim)) @ rng.standard_normal((dim, dim))).astype(np.float32)
    generated = (rng.standard_normal((300, dim)) @ rng.standard_normal((dim, dim)) + 0.5).astype(
        np.float32
    )
    fid = FIDMetric(feature_dim=dim)
    fid.update(real=real, generated=generated)
    assert fid.compute()["fid"] == pytest.approx(_scipy_frechet_distance(real, generated), rel=1e-4)
