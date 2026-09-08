"""Probabilistic forecast verification for ensemble and distributional predictions.

Every function is a pure ``jax.numpy`` computation and traces under ``jax.jit``,
``jax.grad`` and ``jax.vmap``. Scoring rules return the mean over samples, like the
rest of the Tier 0 surface; the histograms return bin counts.

References:
    * Gneiting & Raftery 2007, "Strictly proper scoring rules, prediction, and
      estimation" (JASA): CRPS, energy score, ranked probability score.
    * Ferro 2014, "Fair scores for ensemble forecasts": the finite-ensemble
      bias-corrected CRPS.
    * Hamill 2001, "Interpretation of rank histograms".
    * Fortin et al. 2014, "Why should ensemble spread match the RMSE of the ensemble
      mean?": the unbiased spread-skill ratio.
    * Diebold, Gunther & Tay 1998: the probability integral transform histogram.
    * Epstein 1969 and Murphy 1971: ranked probability score and skill score.
    * Murphy 1973: the reliability component of the Brier decomposition.
    * Ferro, Richardson & Weigel 2008: the fair ensemble ranked probability score.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import cdf as _norm_cdf

from calibrax.metrics._utils import (
    _prepare_arrays,
    _prepare_ensemble_arrays,
    _prepare_multivariate_ensemble_arrays,
)


def fair_crps(predictions: Any, targets: Any) -> Any:  # noqa: DOC502  # raised by _prepare_ensemble_arrays
    """Fair (finite-ensemble bias-corrected) CRPS per Ferro 2014.

    Replaces the ``1/M^2`` averaging of the pairwise spread in the empirical CRPS
    with the unbiased ``1/(M(M-1))`` estimator, so the score no longer shrinks
    with the ensemble size; it recovers the population CRPS as ``M`` grows.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: (-inf, inf); the unbiased spread can exceed the error term.
        Proper scoring rule for probabilistic forecasts.

    Args:
        predictions: Forecast ensemble with shape ``(n_samples, n_members)``.
        targets: Observed targets with shape ``(n_samples,)``.

    Returns:
        Mean fair CRPS as a scalar JAX array.

    Raises:
        ValueError: If inputs do not have compatible ensemble forecast shapes.
    """
    pred, target = _prepare_ensemble_arrays(predictions, targets)
    n_members = pred.shape[1]
    forecast_error = jnp.mean(jnp.abs(pred - target[:, None]), axis=1)
    pairwise = jnp.abs(pred[:, :, None] - pred[:, None, :])
    # The diagonal is zero, so summing all pairs and dividing by M(M-1) is the
    # unbiased mean over the M(M-1) distinct pairs.
    spread = jnp.sum(pairwise, axis=(1, 2)) / (n_members * (n_members - 1))
    return jnp.mean(forecast_error - 0.5 * spread)


def energy_score(predictions: Any, targets: Any) -> Any:  # noqa: DOC502  # raised by _prepare_multivariate_ensemble_arrays
    """Energy score of a multivariate ensemble forecast (Gneiting & Raftery 2007).

    ``ES = mean_i ||X_i - y|| - 0.5 * mean_{i,j} ||X_i - X_j||`` with the Euclidean
    norm, averaged over samples. Reduces to the empirical CRPS for one output.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Proper scoring rule for multivariate probabilistic forecasts.

    Args:
        predictions: Ensemble with shape ``(n_samples, n_members, n_outputs)``.
        targets: Observed targets with shape ``(n_samples, n_outputs)``.

    Returns:
        Mean energy score as a scalar JAX array.

    Raises:
        ValueError: If inputs do not have compatible multivariate ensemble shapes.
    """
    pred, target = _prepare_multivariate_ensemble_arrays(predictions, targets)
    forecast_error = jnp.mean(jnp.linalg.norm(pred - target[:, None, :], axis=-1), axis=1)
    pairwise = jnp.linalg.norm(pred[:, :, None, :] - pred[:, None, :, :], axis=-1)
    return jnp.mean(forecast_error - 0.5 * jnp.mean(pairwise, axis=(1, 2)))


def rank_histogram(predictions: Any, targets: Any) -> Any:  # noqa: DOC502  # raised by _prepare_ensemble_arrays
    """Rank histogram of the targets within their ensembles (Hamill 2001).

    For each target, the rank is the number of ensemble members strictly below it,
    so a calibrated ensemble spreads the counts evenly over ``n_members + 1`` bins.

    Note:
        Returns bin counts, not a scalar, so it is not a registered metric.

    Args:
        predictions: Forecast ensemble with shape ``(n_samples, n_members)``.
        targets: Observed targets with shape ``(n_samples,)``.

    Returns:
        Integer counts with shape ``(n_members + 1,)``.

    Raises:
        ValueError: If inputs do not have compatible ensemble forecast shapes.
    """
    pred, target = _prepare_ensemble_arrays(predictions, targets)
    ranks = jnp.sum((pred < target[:, None]).astype(jnp.int32), axis=1)
    return jnp.bincount(ranks, length=pred.shape[1] + 1)


def spread_skill_ratio(predictions: Any, targets: Any) -> Any:  # noqa: DOC502  # raised by _prepare_ensemble_arrays
    """Unbiased spread-skill ratio (Fortin et al. 2014).

    The root mean unbiased ensemble variance over the bias-corrected RMSE of the
    ensemble mean, ``sqrt(mean(Var(X, ddof=1)) / mean((mean(X) - y)^2 - Var/M))``,
    the estimators WeatherBenchX calls ``EnsembleRootMeanVariance`` and
    ``UnbiasedEnsembleMeanSquaredError``.

    Note:
        Direction: NEUTRAL; 1.0 is calibrated dispersion, below 1 is
        under-dispersed, above 1 over-dispersed.
        Range: [0, inf), or ``nan`` when the bias-corrected error is not positive,
        which the unbiased estimator does not rule out for tiny samples.

    Args:
        predictions: Forecast ensemble with shape ``(n_samples, n_members)``.
        targets: Observed targets with shape ``(n_samples,)``.

    Returns:
        Scalar ratio as a JAX array.

    Raises:
        ValueError: If inputs do not have compatible ensemble forecast shapes.
    """
    pred, target = _prepare_ensemble_arrays(predictions, targets)
    n_members = pred.shape[1]
    per_sample_variance = jnp.var(pred, axis=1, ddof=1)
    squared_error = (jnp.mean(pred, axis=1) - target) ** 2
    unbiased_mse = jnp.mean(squared_error - per_sample_variance / n_members)
    return jnp.sqrt(jnp.mean(per_sample_variance) / unbiased_mse)


def pit_histogram(means: Any, variances: Any, targets: Any, *, num_bins: int = 10) -> Any:  # noqa: DOC502  # raised by _prepare_arrays
    """Histogram of probability integral transform values under a Gaussian predictive.

    Bins ``F(y | mean, sqrt(variance))`` into ``num_bins`` equal-width bins over
    ``[0, 1]``; a calibrated predictive distribution gives a flat histogram.

    Note:
        Returns bin counts, not a scalar, so it is not a registered metric.

    Args:
        means: Predictive means with shape ``(n_samples,)``.
        variances: Predictive variances, same shape, strictly positive.
        targets: Observed values, same shape.
        num_bins: Number of equal-width bins over ``[0, 1]``.

    Returns:
        Integer counts with shape ``(num_bins,)``.

    Raises:
        ValueError: If the shapes do not match.
    """
    mean, target = _prepare_arrays(means, targets)
    variance, _ = _prepare_arrays(variances, targets)
    pit_values = _norm_cdf(target, loc=mean, scale=jnp.sqrt(variance))
    bin_index = jnp.minimum((pit_values * num_bins).astype(jnp.int32), num_bins - 1)
    return jnp.bincount(bin_index.reshape(-1), length=num_bins)


def ranked_probability_score(probabilities: Any, targets: Any) -> Any:
    """Ranked probability score for ordered categories (Epstein 1969).

    ``RPS = sum_k (F_k - O_k)^2`` where ``F_k`` is the cumulative predicted
    probability through class ``k`` and ``O_k`` the cumulative observation, averaged
    over samples.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, num_classes - 1].
        Proper scoring rule for ordinal categorical forecasts.

    Args:
        probabilities: Per-class probabilities with shape ``(n_samples, n_classes)``.
        targets: Integer class indices with shape ``(n_samples,)``.

    Returns:
        Mean RPS as a scalar JAX array.
    """
    probs = jnp.asarray(probabilities)
    target = jnp.asarray(targets).astype(jnp.int32)
    cumulative_probs = jnp.cumsum(probs, axis=-1)
    cumulative_obs = jnp.cumsum(jax.nn.one_hot(target, probs.shape[-1]), axis=-1)
    return jnp.mean(jnp.sum((cumulative_probs - cumulative_obs) ** 2, axis=-1))


def event_reliability(probabilities: Any, events: Any, *, num_bins: int = 10) -> Any:  # noqa: DOC502  # raised by _prepare_arrays
    """Reliability component of the Brier decomposition (Murphy 1973).

    ``REL = (1/n) sum_k n_k (f_k - o_k)^2`` over ``num_bins`` equal-width bins of
    the forecast probability, where ``f_k`` is the mean forecast and ``o_k`` the
    observed event frequency in bin ``k``.

    Note:
        Direction: LOWER (0.0 = perfectly reliable).
        Range: [0, 1].

    Args:
        probabilities: Forecast event probabilities in ``[0, 1]``, shape ``(n,)``.
        events: Observed binary indicators, same shape.
        num_bins: Number of equal-width bins over ``[0, 1]``.

    Returns:
        Scalar reliability as a JAX array.

    Raises:
        ValueError: If the shapes do not match.
    """
    probs, observed = _prepare_arrays(probabilities, events)
    probs = probs.reshape(-1)
    observed = observed.reshape(-1).astype(probs.dtype)
    bin_index = jnp.minimum((probs * num_bins).astype(jnp.int32), num_bins - 1)
    one_hot = jax.nn.one_hot(bin_index, num_bins)
    counts = jnp.sum(one_hot, axis=0)
    safe_counts = jnp.maximum(counts, 1.0)
    mean_probs = jnp.sum(one_hot * probs[:, None], axis=0) / safe_counts
    mean_events = jnp.sum(one_hot * observed[:, None], axis=0) / safe_counts
    return jnp.sum(counts * (mean_probs - mean_events) ** 2) / probs.shape[0]


def ensemble_ranked_probability_score(  # noqa: DOC502  # raised by _prepare_ensemble_arrays
    predictions: Any,
    targets: Any,
    *,
    thresholds: Any,
    fair: bool = True,
) -> Any:
    """Ranked probability score of a continuous ensemble at fixed thresholds.

    The empirical CDF of the ensemble and the step CDF of the target are compared
    at each threshold and the squared gaps summed. With ``fair=True`` the
    Ferro, Richardson & Weigel 2008 correction ``Var(F_k, ddof=1) / M`` is
    subtracted per threshold, so the expected score does not depend on the
    ensemble size.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, num_thresholds] for the biased form; the fair form can be
        slightly negative.

    Args:
        predictions: Forecast ensemble with shape ``(n_samples, n_members)``.
        targets: Observed targets with shape ``(n_samples,)``.
        thresholds: Increasing threshold values with shape ``(num_thresholds,)``.
        fair: Apply the finite-ensemble debiasing.

    Returns:
        Mean RPS as a scalar JAX array.

    Raises:
        ValueError: If inputs do not have compatible ensemble forecast shapes.
    """
    pred, target = _prepare_ensemble_arrays(predictions, targets)
    levels = jnp.asarray(thresholds)
    below = (pred[:, :, None] <= levels[None, None, :]).astype(jnp.float32)
    predicted_cdf = jnp.mean(below, axis=1)
    target_cdf = (target[:, None] <= levels[None, :]).astype(jnp.float32)
    squared_gap = (predicted_cdf - target_cdf) ** 2
    if fair:
        squared_gap = squared_gap - jnp.var(below, axis=1, ddof=1) / pred.shape[1]
    return jnp.mean(jnp.sum(squared_gap, axis=-1))


def ranked_probability_skill_score(rps: Any, rps_reference: Any) -> Any:
    """Ranked probability skill score against a reference forecast (Murphy 1971).

    ``RPSS = 1 - RPS / RPS_reference``, elementwise over broadcastable inputs.

    Note:
        Direction: HIGHER (1.0 = perfect; 0.0 = no better than the reference).
        Range: (-inf, 1].

    Args:
        rps: Ranked probability score of the forecast.
        rps_reference: Ranked probability score of the reference, for example
            climatology.

    Returns:
        Skill score with the broadcast shape of the inputs.
    """
    return 1.0 - jnp.asarray(rps) / jnp.asarray(rps_reference)
