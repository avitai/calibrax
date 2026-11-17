"""Censored concordance and observable coverage metrics for survival and potency data.

Pure functions for evaluating predictive rankings and one-sided lower prediction
bounds when outcomes are right-, left-, or interval-censored (for example clinical
time-to-event endpoints or assay detection-limit bounds such as ``IC50 > 10 uM``).

Includes 3 functions: harrell_c_index, uno_c_index, censored_coverage_bounds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike


def _concordance_score_matrix(predictions: jax.Array) -> jax.Array:
    """Pairwise concordance matrix ``I(p_i > p_j) + 0.5 * I(p_i == p_j)``."""
    diff = predictions[:, None] - predictions[None, :]
    return jnp.where(diff > 0.0, 1.0, jnp.where(diff == 0.0, 0.5, 0.0))


def harrell_c_index(
    predictions: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike | None = None,
    *,
    event_indicator: ArrayLike | None = None,
) -> jax.Array:
    """Generalized Harrell concordance index for exact or censored outcomes.

    Evaluates the fraction of strictly ordered outcome pairs ``(i, j)`` with
    ``lower[i] > upper[j]`` for which ``predictions[i] > predictions[j]``, counting
    tied predictions as ``0.5`` (Harrell et al., JAMA 1982). Supports exact targets
    (``upper=None, event_indicator=None``), right-censored survival or potency bounds
    via ``event_indicator`` (``1`` = exact at ``lower``, ``0`` = right-censored
    ``[lower, +inf)``), and general left-/right-/interval-censored bounds
    ``[lower, upper]``.

    Note:
        Direction: HIGHER (1.0 = perfect concordance, 0.5 = random ordering).
        Range: [0, 1]. ``NaN`` when zero comparable pairs exist.

    Args:
        predictions: Predicted outcome values (higher predicts larger outcome).
        lower: Lower outcome bound (or observed time when ``upper`` is ``None``).
        upper: Optional upper outcome bound for interval/left/right censoring.
        event_indicator: Optional right-censoring indicator (nonzero = exact event,
            zero = right-censored ``[lower, +inf)``). Mutually exclusive with ``upper``.

    Returns:
        Concordance index as a scalar JAX array, or ``NaN`` when no comparable pairs exist.

    Raises:
        ValueError: If shapes do not match or both ``upper`` and ``event_indicator``
            are provided.
    """
    if upper is not None and event_indicator is not None:
        raise ValueError("Cannot specify both upper and event_indicator")

    pred = jnp.asarray(predictions, dtype=jnp.float32).ravel()
    lo = jnp.asarray(lower, dtype=jnp.float32).ravel()
    if pred.size != lo.size:
        raise ValueError(
            f"Shape mismatch: predictions has {pred.size} elements, lower has {lo.size}"
        )

    if upper is not None:
        hi = jnp.asarray(upper, dtype=jnp.float32).ravel()
        if hi.size != lo.size:
            raise ValueError(f"Shape mismatch: lower has {lo.size} elements, upper has {hi.size}")
    elif event_indicator is not None:
        ev = jnp.asarray(event_indicator).ravel()
        if ev.size != lo.size:
            raise ValueError(
                f"Shape mismatch: lower has {lo.size} elements, event_indicator has {ev.size}"
            )
        hi = jnp.where(ev > 0, lo, jnp.inf)
    else:
        hi = lo

    if pred.size < 2:  # noqa: PLR2004
        return jnp.array(jnp.nan, dtype=jnp.float32)

    invalid_bounds = jnp.any(lo > hi)
    comparable = (lo[:, None] > hi[None, :]).astype(jnp.float32)
    denom = jnp.sum(comparable)
    has_pairs = (~invalid_bounds) & (denom > 0.0)
    safe_denom = jnp.where(has_pairs, denom, 1.0)
    numer = jnp.sum(comparable * _concordance_score_matrix(pred))
    return jnp.where(has_pairs, numer / safe_denom, jnp.nan)


def _kaplan_meier_censoring_survival(
    observed_time: jax.Array,
    event_indicator: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Simultaneous risk-set Kaplan-Meier censoring survival ``G_hat(T_k)``."""
    size = observed_time.size
    order = jnp.argsort(observed_time)
    t_ord = observed_time[order]
    cens_ord = (~event_indicator[order]).astype(jnp.float32)

    idx = jnp.arange(size)
    starts = jnp.concatenate([jnp.ones(1, dtype=bool), t_ord[1:] != t_ord[:-1]])
    group = jnp.cumsum(starts) - 1
    group_start = lax.cummax(jnp.where(starts, idx, 0))
    at_risk_item = (size - group_start).astype(jnp.float32)

    cens_seg = jax.ops.segment_sum(cens_ord, group, num_segments=size)
    at_risk_seg = jax.ops.segment_max(at_risk_item, group, num_segments=size)
    has_cens = cens_seg > 0.0
    safe_risk = jnp.where(has_cens, jnp.maximum(at_risk_seg, 1.0), 1.0)
    factors = jnp.clip(1.0 - jnp.where(has_cens, cens_seg, 0.0) / safe_risk, 0.0, 1.0)
    g_ord = jnp.cumprod(factors)[group]
    return order, g_ord


def uno_c_index(
    scores: ArrayLike,
    observed_time: ArrayLike,
    event_indicator: ArrayLike,
    *,
    horizon: ArrayLike | None = None,
) -> jax.Array:
    """Inverse-probability-of-censoring weighted concordance (Uno et al., Stat. Med. 2011).

    Estimates the censoring-free concordance probability ``P(score_i > score_j | T_i > T_j)``
    (optionally restricted to ``T_j < horizon``) by weighting each comparable pair with
    ``G_hat(T_j)^(-2)``, where ``G_hat`` is the Kaplan-Meier survival estimator of the
    censoring distribution (Uno et al. 2011, Eqs. 2-5). Scores follow the same orientation
    as :func:`harrell_c_index` (larger ``scores`` predict longer survival or larger outcome).

    Note:
        Direction: HIGHER (1.0 = perfect concordance, 0.5 = random ordering).
        Range: [0, 1]. ``NaN`` when zero comparable pairs carry positive weight.

    Args:
        scores: Predicted outcome scores (larger predicts larger true time/value).
        observed_time: Observed follow-up times ``min(T_latent, C)``.
        event_indicator: Event indicator (nonzero = observed event, zero = right-censored).
        horizon: Optional positive truncation horizon ``tau`` restricting earlier events
            to ``observed_time < horizon``.

    Returns:
        IPCW concordance index as a scalar JAX array, or ``NaN`` when undefined.

    Raises:
        ValueError: If input lengths differ or eager ``horizon`` is not positive.
    """
    if (
        horizon is not None
        and isinstance(horizon, int | float)
        and not isinstance(horizon, jax.Array)
        and horizon <= 0.0
    ):
        raise ValueError(f"horizon must be positive, got {horizon}")

    pred = jnp.asarray(scores, dtype=jnp.float32).ravel()
    time = jnp.asarray(observed_time, dtype=jnp.float32).ravel()
    ev_raw = jnp.asarray(event_indicator).ravel()
    if not (pred.size == time.size == ev_raw.size):
        raise ValueError(
            f"Shape mismatch: scores={pred.size}, observed_time={time.size}, "
            f"event_indicator={ev_raw.size}"
        )
    if pred.size < 2:  # noqa: PLR2004
        return jnp.array(jnp.nan, dtype=jnp.float32)

    ev = ev_raw > 0
    order, g_ord = _kaplan_meier_censoring_survival(time, ev)
    eps = jnp.finfo(jnp.float32).eps
    safe_g_ord = jnp.maximum(g_ord, eps)
    ipcw_ord = 1.0 / (safe_g_ord * safe_g_ord)
    ipcw = jnp.zeros_like(time).at[order].set(ipcw_ord)

    if horizon is None:
        within_tau = jnp.ones_like(ev)
        valid_tau = jnp.array(True)
    else:
        tau = jnp.asarray(horizon, dtype=jnp.float32)
        within_tau = time < tau
        valid_tau = tau > 0.0

    # Pair (i, j) has later time[i] > earlier uncensored time[j] (with time[j] < tau).
    earlier_valid = ev & within_tau
    comparable = (time[:, None] > time[None, :]) & earlier_valid[None, :]
    pair_weights = jnp.where(comparable, ipcw[None, :], 0.0)
    denom = jnp.sum(pair_weights)
    has_weight = valid_tau & (denom > 0.0)
    safe_denom = jnp.where(has_weight, denom, 1.0)
    numer = jnp.sum(pair_weights * _concordance_score_matrix(pred))
    return jnp.where(has_weight, numer / safe_denom, jnp.nan)


def censored_coverage_bounds(
    lower_pred: ArrayLike,
    observed_time: ArrayLike,
    event_indicator: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Observable lower and upper bounds on one-sided coverage under right-censoring.

    For right-censored observations ``observed_time = min(T_latent, C)`` with
    ``event_indicator = I(T_latent <= C)``, latent one-sided coverage ``P(L <= T_latent)``
    is partially identified and sharply bracketed by the observable probabilities:

    - ``lower_bound = P_hat(lower_pred <= observed_time)``
    - ``upper_bound = 1 - P_hat(lower_pred > observed_time and event_indicator == 1)``

    Both bounds coincide with empirical coverage when all observations are uncensored.

    Args:
        lower_pred: Predicted one-sided lower bounds ``L``.
        observed_time: Observed right-censored values ``min(T_latent, C)``.
        event_indicator: Event indicator (nonzero = exact observation, zero = right-censored).

    Returns:
        Tuple ``(lower_bound, upper_bound)`` of scalar JAX arrays in ``[0, 1]``, or
        ``(NaN, NaN)`` when inputs are empty.

    Raises:
        ValueError: If input lengths do not match.
    """
    lo = jnp.asarray(lower_pred, dtype=jnp.float32).ravel()
    obs = jnp.asarray(observed_time, dtype=jnp.float32).ravel()
    ev_raw = jnp.asarray(event_indicator).ravel()
    if not (lo.size == obs.size == ev_raw.size):
        raise ValueError(
            f"Shape mismatch: lower_pred={lo.size}, observed_time={obs.size}, "
            f"event_indicator={ev_raw.size}"
        )
    if lo.size == 0:
        nan_val = jnp.array(jnp.nan, dtype=jnp.float32)
        return nan_val, nan_val

    ev = ev_raw > 0
    lower_bound = jnp.mean((lo <= obs).astype(jnp.float32))
    known_miscovered = jnp.mean(((lo > obs) & ev).astype(jnp.float32))
    upper_bound = 1.0 - known_miscovered
    return lower_bound, upper_bound
