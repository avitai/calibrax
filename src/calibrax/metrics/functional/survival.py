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

from calibrax.core._validation import eager_scalar_float


def _concordance_score_matrix(predictions: jax.Array) -> jax.Array:
    """Pairwise concordance matrix ``I(p_i > p_j) + 0.5 * I(p_i == p_j)``."""
    dt = predictions.dtype
    p_i = predictions[:, None]
    p_j = predictions[None, :]
    return jnp.where(
        p_i > p_j,
        jnp.asarray(1.0, dtype=dt),
        jnp.where(p_i == p_j, jnp.asarray(0.5, dtype=dt), jnp.asarray(0.0, dtype=dt)),
    )


def _validate_binary_events(event_indicator: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Return ``(events_bool, is_valid)`` requiring entries in ``{0, 1}`` without ``NaN``."""
    raw = jnp.asarray(event_indicator).ravel()
    if raw.dtype == jnp.bool_:
        return raw, jnp.array(True)
    if jnp.issubdtype(raw.dtype, jnp.integer):
        return raw == 1, jnp.all((raw == 0) | (raw == 1))
    ev_f = raw.astype(jnp.result_type(raw.dtype, jnp.float32))
    return ev_f == 1.0, jnp.all((ev_f == 0.0) | (ev_f == 1.0))


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
        Range: [0, 1]. ``NaN`` when zero comparable pairs exist, bounds are inverted,
        inputs contain ``NaN``, or ``event_indicator`` contains values outside ``{0, 1}``.

    Args:
        predictions: Predicted outcome values (higher predicts larger outcome).
        lower: Lower outcome bound (or observed time when ``upper`` is ``None``).
        upper: Optional upper outcome bound for interval/left/right censoring.
        event_indicator: Optional binary right-censoring indicator (``1``/``True`` = exact
            event, ``0``/``False`` = right-censored ``[lower, +inf)``). Mutually exclusive
            with ``upper``.

    Returns:
        Concordance index as a scalar JAX array, or ``NaN`` when undefined.

    Raises:
        ValueError: If shapes do not match or both ``upper`` and ``event_indicator``
            are provided.
    """
    if upper is not None and event_indicator is not None:
        raise ValueError("Cannot specify both upper and event_indicator")

    pred_raw = jnp.asarray(predictions).ravel()
    lo_raw = jnp.asarray(lower).ravel()
    if pred_raw.size != lo_raw.size:
        raise ValueError(
            f"Shape mismatch: predictions has {pred_raw.size} elements, lower has {lo_raw.size}"
        )

    if upper is not None:
        hi_raw = jnp.asarray(upper).ravel()
        if hi_raw.size != lo_raw.size:
            raise ValueError(
                f"Shape mismatch: lower has {lo_raw.size} elements, upper has {hi_raw.size}"
            )
        out_dtype = jnp.result_type(pred_raw.dtype, lo_raw.dtype, hi_raw.dtype, jnp.float32)
        pred = pred_raw.astype(out_dtype)
        lo = lo_raw.astype(out_dtype)
        hi = hi_raw.astype(out_dtype)
        valid_ev = jnp.array(True)
    elif event_indicator is not None:
        out_dtype = jnp.result_type(pred_raw.dtype, lo_raw.dtype, jnp.float32)
        pred = pred_raw.astype(out_dtype)
        lo = lo_raw.astype(out_dtype)
        ev, valid_ev = _validate_binary_events(event_indicator)
        if ev.size != lo.size:
            raise ValueError(
                f"Shape mismatch: lower has {lo.size} elements, event_indicator has {ev.size}"
            )
        hi = jnp.where(ev, lo, jnp.asarray(jnp.inf, dtype=out_dtype))
    else:
        out_dtype = jnp.result_type(pred_raw.dtype, lo_raw.dtype, jnp.float32)
        pred = pred_raw.astype(out_dtype)
        lo = lo_raw.astype(out_dtype)
        hi = lo
        valid_ev = jnp.array(True)

    if pred.size < 2:  # noqa: PLR2004
        return jnp.array(jnp.nan, dtype=out_dtype)

    has_nan = jnp.any(jnp.isnan(pred) | jnp.isnan(lo) | jnp.isnan(hi))
    invalid_bounds = jnp.any(lo > hi)
    comparable = (lo[:, None] > hi[None, :]).astype(out_dtype)
    denom = jnp.sum(comparable)
    has_pairs = valid_ev & (~has_nan) & (~invalid_bounds) & (denom > 0.0)
    safe_denom = jnp.where(has_pairs, denom, jnp.asarray(1.0, dtype=out_dtype))
    numer = jnp.sum(comparable * _concordance_score_matrix(pred))
    return jnp.where(has_pairs, numer / safe_denom, jnp.asarray(jnp.nan, dtype=out_dtype))


def _kaplan_meier_censoring_survival(
    observed_time: jax.Array,
    event_indicator: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Left-continuous reverse Kaplan-Meier censoring survival ``G_hat(T_k-)``.

    Matches Uno et al. (Stat. Med. 2011, ``cran/survC1``) and
    ``survival::survfit(Surv(time, 1 - status))`` by forming censoring hazards over the
    full risk set ``sum(observed_time >= t_k)``, evaluating the strictly left-continuous
    limit ``G_hat(t_k-)`` in log-space via ``cumsum(log1p(-hazard))`` to avoid cumulative
    product drift.
    """
    dtype = observed_time.dtype
    size = observed_time.size
    order = jnp.argsort(observed_time)
    t_ord = observed_time[order]
    ev_ord = event_indicator[order].astype(dtype)
    cens_ord = 1.0 - ev_ord

    idx = jnp.arange(size)
    starts = jnp.concatenate([jnp.ones(1, dtype=bool), t_ord[1:] != t_ord[:-1]])
    group = jnp.cumsum(starts) - 1
    group_start = lax.cummax(jnp.where(starts, idx, 0))
    at_risk_item = (size - group_start).astype(dtype)

    cens_seg = jax.ops.segment_sum(cens_ord, group, num_segments=size)
    at_risk_seg = jax.ops.segment_max(at_risk_item, group, num_segments=size)
    has_cens = cens_seg > 0.0
    one = jnp.asarray(1.0, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    safe_risk = jnp.where(has_cens, jnp.maximum(at_risk_seg, one), one)
    hazard = jnp.clip(jnp.where(has_cens, cens_seg, zero) / safe_risk, zero, one)
    all_censored = hazard >= one
    safe_hazard = jnp.where(all_censored, zero, hazard)
    log_factor = jnp.where(
        all_censored, jnp.asarray(-jnp.inf, dtype=dtype), jnp.log1p(-safe_hazard)
    )
    cum_log_prev = jnp.concatenate([jnp.zeros(1, dtype=dtype), jnp.cumsum(log_factor)[:-1]])
    g_ord = jnp.exp(cum_log_prev)[group]
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
    ``G_hat(T_j-)^{-2}``, where ``G_hat(t-)`` is the left-continuous Kaplan-Meier survival
    estimator of the censoring distribution (Uno et al. 2011, Eqs. 2-5). Scores follow the
    same orientation as :func:`harrell_c_index` (larger ``scores`` predict longer survival
    or larger outcome).

    Note:
        Direction: HIGHER (1.0 = perfect concordance, 0.5 = random ordering).
        Range: [0, 1]. ``NaN`` when zero comparable pairs carry positive weight, inputs
        contain ``NaN``, or ``event_indicator`` contains values outside ``{0, 1}``.

    Args:
        scores: Predicted outcome scores (larger predicts larger true time/value).
        observed_time: Observed follow-up times ``min(T_latent, C)``.
        event_indicator: Binary event indicator (``1``/``True`` = observed event,
            ``0``/``False`` = right-censored).
        horizon: Optional positive truncation horizon ``tau`` restricting earlier events
            to ``observed_time < horizon``.

    Returns:
        IPCW concordance index as a scalar JAX array, or ``NaN`` when undefined.

    Raises:
        ValueError: If input lengths differ or eager ``horizon`` is not positive.
    """
    pred_raw = jnp.asarray(scores).ravel()
    time_raw = jnp.asarray(observed_time).ravel()
    if horizon is None:
        out_dtype = jnp.result_type(pred_raw.dtype, time_raw.dtype, jnp.float32)
        tau = None
    else:
        eager_horizon = eager_scalar_float(horizon)
        if eager_horizon is not None and eager_horizon <= 0.0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        tau_raw = jnp.asarray(horizon)
        out_dtype = jnp.result_type(pred_raw.dtype, time_raw.dtype, tau_raw, jnp.float32)
        tau = tau_raw.astype(out_dtype)

    pred = pred_raw.astype(out_dtype)
    time = time_raw.astype(out_dtype)
    ev, valid_ev = _validate_binary_events(event_indicator)
    if pred.size != time.size or time.size != ev.size:
        raise ValueError(
            f"Shape mismatch: scores={pred.size}, observed_time={time.size}, "
            f"event_indicator={ev.size}"
        )
    if pred.size < 2:  # noqa: PLR2004
        return jnp.array(jnp.nan, dtype=out_dtype)

    has_nan = jnp.any(jnp.isnan(pred) | jnp.isnan(time))
    safe_time = jnp.where(jnp.isnan(time), jnp.asarray(0.0, dtype=out_dtype), time)
    order, g_ord = _kaplan_meier_censoring_survival(safe_time, ev)
    eps = jnp.asarray(jnp.finfo(out_dtype).eps, dtype=out_dtype)
    safe_g_ord = jnp.maximum(g_ord, eps)
    ipcw_ord = 1.0 / (safe_g_ord * safe_g_ord)
    ipcw = jnp.zeros_like(time).at[order].set(ipcw_ord)

    if tau is None:
        within_tau = jnp.ones_like(ev)
        valid_tau = jnp.array(True)
    else:
        within_tau = time < tau
        valid_tau = (tau > 0.0) & (~jnp.any(jnp.isnan(tau)))

    # Pair (i, j) has later time[i] > earlier uncensored time[j] (with time[j] < tau).
    earlier_valid = ev & within_tau
    comparable = (time[:, None] > time[None, :]) & earlier_valid[None, :]
    pair_weights = jnp.where(comparable, ipcw[None, :], jnp.asarray(0.0, dtype=out_dtype))
    denom = jnp.sum(pair_weights)
    has_weight = valid_ev & (~has_nan) & valid_tau & (denom > 0.0)
    safe_denom = jnp.where(has_weight, denom, jnp.asarray(1.0, dtype=out_dtype))
    numer = jnp.sum(pair_weights * _concordance_score_matrix(pred))
    return jnp.where(has_weight, numer / safe_denom, jnp.asarray(jnp.nan, dtype=out_dtype))


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
        event_indicator: Binary event indicator (``1``/``True`` = exact observation,
            ``0``/``False`` = right-censored).

    Returns:
        Tuple ``(lower_bound, upper_bound)`` of scalar JAX arrays in ``[0, 1]``, or
        ``(NaN, NaN)`` when inputs are empty, contain ``NaN``, or have non-binary
        ``event_indicator`` values.

    Raises:
        ValueError: If input lengths do not match.
    """
    lo_raw = jnp.asarray(lower_pred).ravel()
    obs_raw = jnp.asarray(observed_time).ravel()
    out_dtype = jnp.result_type(lo_raw.dtype, obs_raw.dtype, jnp.float32)
    lo = lo_raw.astype(out_dtype)
    obs = obs_raw.astype(out_dtype)
    ev, valid_ev = _validate_binary_events(event_indicator)
    if not (lo.size == obs.size == ev.size):
        raise ValueError(
            f"Shape mismatch: lower_pred={lo.size}, observed_time={obs.size}, "
            f"event_indicator={ev.size}"
        )
    nan_val = jnp.array(jnp.nan, dtype=out_dtype)
    if lo.size == 0:
        return nan_val, nan_val

    valid = valid_ev & (~jnp.any(jnp.isnan(lo) | jnp.isnan(obs)))
    lower_bound = jnp.mean((lo <= obs).astype(out_dtype))
    known_miscovered = jnp.mean(((lo > obs) & ev).astype(out_dtype))
    upper_bound = jnp.asarray(1.0, dtype=out_dtype) - known_miscovered
    return jnp.where(valid, lower_bound, nan_val), jnp.where(valid, upper_bound, nan_val)
