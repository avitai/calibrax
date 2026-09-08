"""Uncertainty quantification: interval coverage, proper scores and credibility.

Every function is a pure ``jax.numpy`` computation. Shape checks happen on the
static shapes only, so all of them trace under ``jax.jit``, ``jax.grad`` and
``jax.vmap``; data-dependent conditions such as inverted intervals are not
checked.

References:
    * Gneiting & Raftery 2007: the interval (Winkler) score.
    * Kuleshov, Fenner & Ermon 2018, "Accurate uncertainties for deep learning using
      calibrated regression": regression calibration error.
    * Gal & Ghahramani 2016 and Houlsby et al. 2011: predictive entropy and the
      BALD mutual-information decomposition.
    * Bar-Shalom, Li & Kirubarajan 2002: the average normalised estimation error
      squared (ANEES).
    * Li & Zhao 2006, "Measuring estimator's credibility": the non-credibility index.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import cdf as _norm_cdf

from calibrax.metrics._utils import _EPSILON, _prepare_arrays


def _prepare_interval_arrays(lower: Any, upper: Any, targets: Any) -> tuple[Any, Any, Any]:  # noqa: DOC502  # raised by _prepare_arrays
    """Validate an interval and its targets share one shape and convert them.

    Args:
        lower: Lower interval bounds.
        upper: Upper interval bounds, same shape.
        targets: Observed values, same shape.

    Returns:
        Tuple of JAX arrays ``(lower, upper, targets)``.

    Raises:
        ValueError: If the shapes differ.
    """
    low, target = _prepare_arrays(lower, targets)
    high, _ = _prepare_arrays(upper, targets)
    return low, high, target


def picp(lower: Any, upper: Any, targets: Any) -> Any:  # noqa: DOC502  # raised by _prepare_arrays
    """Prediction interval coverage probability.

    The fraction of targets inside ``[lower, upper]``. Compare it with the nominal
    level ``1 - alpha`` the intervals were built for.

    Note:
        Direction: INFO; the goal is the nominal coverage, not 1.0.
        Range: [0, 1].

    Args:
        lower: Lower interval bounds, any shape.
        upper: Upper interval bounds, same shape.
        targets: Observed values, same shape.

    Returns:
        Scalar coverage as a JAX array.

    Raises:
        ValueError: If the shapes differ.
    """
    low, high, target = _prepare_interval_arrays(lower, upper, targets)
    covered = (target >= low) & (target <= high)
    return jnp.mean(covered.astype(jnp.float32))


def mpiw(lower: Any, upper: Any) -> Any:  # noqa: DOC502  # raised by _prepare_arrays
    """Mean prediction interval width, ``mean(upper - lower)``.

    Note:
        Direction: LOWER, at a fixed coverage; sharper intervals are better.
        Range: (-inf, inf); negative for inverted intervals.

    Args:
        lower: Lower interval bounds, any shape.
        upper: Upper interval bounds, same shape.

    Returns:
        Scalar mean width as a JAX array.

    Raises:
        ValueError: If the shapes differ.
    """
    low, high = _prepare_arrays(lower, upper)
    return jnp.mean(high - low)


def interval_score(lower: Any, upper: Any, targets: Any, *, alpha: float) -> Any:  # noqa: DOC502  # raised by _prepare_arrays
    """Interval score of central ``(1 - alpha)`` prediction intervals.

    ``IS = (u - l) + (2 / alpha) (l - y)_+ + (2 / alpha) (y - u)_+``, averaged over
    the elements: the width plus a ``2 / alpha`` multiple of the miscoverage gap.

    Note:
        Direction: LOWER.
        Range: [0, inf).
        Proper scoring rule for central prediction intervals (Gneiting & Raftery).

    Args:
        lower: Lower interval bounds, any shape.
        upper: Upper interval bounds, same shape.
        targets: Observed values, same shape.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Scalar mean interval score as a JAX array.

    Raises:
        ValueError: If the shapes differ.
    """
    low, high, target = _prepare_interval_arrays(lower, upper, targets)
    below = jnp.maximum(0.0, low - target)
    above = jnp.maximum(0.0, target - high)
    return jnp.mean((high - low) + (2.0 / alpha) * (below + above))


def winkler_score(lower: Any, upper: Any, targets: Any, *, alpha: float) -> Any:
    """Winkler's 1972 name for :func:`interval_score`; same value.

    Args:
        lower: Lower interval bounds, any shape.
        upper: Upper interval bounds, same shape.
        targets: Observed values, same shape.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Scalar mean interval score as a JAX array.
    """
    return interval_score(lower, upper, targets, alpha=alpha)


def gaussian_nll(means: Any, variances: Any, targets: Any) -> Any:  # noqa: DOC502  # raised by _prepare_arrays
    """Mean negative log-likelihood of the targets under a diagonal Gaussian.

    ``mean(0.5 (log(2 pi variance) + (target - mean)^2 / variance))``.

    Note:
        Direction: LOWER.
        Range: (-inf, inf).
        Proper scoring rule; non-positive variances give non-finite values.

    Args:
        means: Predictive means, any shape.
        variances: Predictive variances, same shape, strictly positive.
        targets: Observed values, same shape.

    Returns:
        Scalar mean NLL as a JAX array.

    Raises:
        ValueError: If the shapes differ.
    """
    mean, target = _prepare_arrays(means, targets)
    variance, _ = _prepare_arrays(variances, targets)
    diff = target - mean
    return jnp.mean(0.5 * (jnp.log(2.0 * jnp.pi * variance) + diff * diff / variance))


def regression_calibration_error(  # noqa: DOC502  # raised by _prepare_arrays
    means: Any,
    variances: Any,
    targets: Any,
    *,
    quantile_levels: Any,
) -> Any:
    """Regression calibration error of a Gaussian predictive (Kuleshov et al. 2018).

    For each nominal level ``q`` the empirical fraction of targets below the
    predictive ``q``-quantile is compared with ``q``; the metric is the mean
    absolute gap over the levels, the regression analogue of ECE.

    Note:
        Direction: LOWER (0.0 = calibrated).
        Range: [0, 1].

    Args:
        means: Predictive means, any shape.
        variances: Predictive variances, same shape, strictly positive.
        targets: Observed values, same shape.
        quantile_levels: Nominal levels in ``(0, 1)`` with shape ``(n_levels,)``.

    Returns:
        Scalar mean absolute miscalibration as a JAX array.

    Raises:
        ValueError: If the shapes differ.
    """
    mean, target = _prepare_arrays(means, targets)
    variance, _ = _prepare_arrays(variances, targets)
    levels = jnp.asarray(quantile_levels)
    cdf_values = _norm_cdf(target, loc=mean, scale=jnp.sqrt(variance)).reshape(-1)

    def empirical_at(level: Any) -> Any:
        return jnp.mean((cdf_values <= level).astype(jnp.float32))

    return jnp.mean(jnp.abs(jax.vmap(empirical_at)(levels) - levels))


def predictive_entropy(ensemble_probabilities: Any) -> Any:
    """Entropy of the ensemble-averaged categorical distribution, per sample.

    ``H(mean_m p_m)`` with the ensemble on the leading axis and the classes on the
    last axis; the total predictive uncertainty of Gal & Ghahramani 2016.

    Note:
        Direction: INFO; higher means more uncertain.
        Range: [0, log(n_classes)].
        Returns one value per sample, so it is not a registered metric.

    Args:
        ensemble_probabilities: Probabilities with shape
            ``(n_members, *batch, n_classes)``, summing to one on the last axis.

    Returns:
        Entropies with shape ``(*batch,)``.
    """
    mean_probs = jnp.mean(jnp.asarray(ensemble_probabilities), axis=0)
    return -jnp.sum(mean_probs * jnp.log(mean_probs + _EPSILON), axis=-1)


def ensemble_mutual_information(ensemble_probabilities: Any) -> Any:
    """Epistemic uncertainty of an ensemble, per sample (BALD, Houlsby et al. 2011).

    ``H(mean_m p_m) - mean_m H(p_m)``: the predictive entropy minus the expected
    member entropy. Zero when every member agrees. Not the joint-table mutual
    information of the information domain.

    Note:
        Direction: INFO; higher means more disagreement between members.
        Range: [0, log(n_classes)].
        Returns one value per sample, so it is not a registered metric.

    Args:
        ensemble_probabilities: Probabilities with shape
            ``(n_members, *batch, n_classes)``, summing to one on the last axis.

    Returns:
        Mutual information with shape ``(*batch,)``.
    """
    probs = jnp.asarray(ensemble_probabilities)
    member_entropies = -jnp.sum(probs * jnp.log(probs + _EPSILON), axis=-1)
    return predictive_entropy(probs) - jnp.mean(member_entropies, axis=0)


def _mahalanobis(errors: Any, covariances: Any) -> Any:
    """Per-step squared Mahalanobis distance ``e_t^T C_t^{-1} e_t``."""
    solved = jax.vmap(jnp.linalg.solve)(covariances, errors)
    return jnp.einsum("ti,ti->t", errors, solved)


def anees(predicted_means: Any, predicted_covariances: Any, references: Any) -> Any:  # noqa: DOC502  # raised by _prepare_arrays
    """Average normalised estimation error squared (Bar-Shalom et al. 2002).

    ``(1 / (N d)) sum_t (r_t - m_t)^T P_t^{-1} (r_t - m_t)`` for predictions
    ``(m_t, P_t)`` and references ``r_t`` of dimension ``d``.

    Note:
        Direction: INFO; 1.0 is calibrated, above 1 over-confident, below 1
        under-confident.
        Range: [0, inf).

    Args:
        predicted_means: Predicted means with shape ``(n_steps, d)``.
        predicted_covariances: Predicted covariances with shape ``(n_steps, d, d)``.
        references: Reference values with shape ``(n_steps, d)``.

    Returns:
        Scalar ANEES as a JAX array.

    Raises:
        ValueError: If the means and references differ in shape.
    """
    mean, reference = _prepare_arrays(predicted_means, references)
    errors = reference - mean
    return jnp.mean(_mahalanobis(errors, jnp.asarray(predicted_covariances))) / errors.shape[-1]


def non_credibility_index(  # noqa: DOC502  # raised by _prepare_arrays
    predicted_means: Any,
    predicted_covariances: Any,
    references: Any,
    reference_covariances: Any,
) -> Any:
    """Non-credibility index of Li & Zhao 2006, in decibels.

    ``(10 / N) sum_t log10( e_t^T P_t^{-1} e_t / e_t^T S_t^{-1} e_t )`` with the
    predicted covariance ``P_t`` and the reference error covariance ``S_t``.

    Note:
        Direction: INFO; 0.0 is credible, positive means the predicted covariance
        underestimates the true error covariance.
        Range: (-inf, inf).

    Args:
        predicted_means: Predicted means with shape ``(n_steps, d)``.
        predicted_covariances: Predicted covariances with shape ``(n_steps, d, d)``.
        references: Reference values with shape ``(n_steps, d)``.
        reference_covariances: Reference error covariances, shape ``(n_steps, d, d)``.

    Returns:
        Scalar NCI as a JAX array.

    Raises:
        ValueError: If the means and references differ in shape.
    """
    mean, reference = _prepare_arrays(predicted_means, references)
    errors = reference - mean
    ratio = _mahalanobis(errors, jnp.asarray(predicted_covariances)) / _mahalanobis(
        errors, jnp.asarray(reference_covariances)
    )
    return 10.0 * jnp.mean(jnp.log10(ratio))


def chi2_confidence_interval(dim: int, *, percentile: float = 0.99) -> tuple[Any, Any]:
    """Symmetric confidence interval of the chi-squared distribution with ``dim`` degrees.

    The ``((1 - percentile) / 2, 1 - (1 - percentile) / 2)`` quantiles: the band an
    ANEES-style statistic of a calibrated estimator falls in. Needs the ``stats``
    extra (SciPy).

    Args:
        dim: Degrees of freedom.
        percentile: Interval coverage in ``(0, 1)``.

    Returns:
        ``(lower, upper)`` quantiles as JAX scalars.

    Raises:
        ValueError: If ``percentile`` is not strictly between 0 and 1.
    """
    if not 0.0 < percentile < 1.0:
        msg = f"percentile must be in (0, 1); got {percentile!r}"
        raise ValueError(msg)
    from scipy import stats

    tail = (1.0 - percentile) / 2.0
    distribution = stats.chi2(df=dim)
    return jnp.asarray(distribution.ppf(tail)), jnp.asarray(distribution.ppf(1.0 - tail))
