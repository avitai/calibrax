"""``calculate_all``: every requested registered metric in one call.

The same-shape regression metrics are computed in one fused traversal of their shared
subexpressions.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax.typing import ArrayLike

from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricSignature, MetricTier, MetricValues
from calibrax.metrics._utils import (
    _EPSILON,
    _prepare_arrays,
    coefficient_of_determination,
    safe_divide,
)
from calibrax.metrics.functional.regression import (
    _huber_elementwise,
    _quantile_elementwise,
    CHARBONNIER_ALPHA,
    CHARBONNIER_EPSILON,
    HUBER_DELTA,
    QUANTILE_LEVEL,
    relative_l2_error,
)


_FUSED_REGRESSION_NAMES = frozenset(
    {
        "mse",
        "mae",
        "rmse",
        "mean_error",
        "r_squared",
        "mape",
        "relative_error",
        "explained_variance",
        "max_error",
        "huber_loss",
        "quantile_loss",
        "log_cosh_loss",
        "smape",
        "relative_l2_error",
        "charbonnier_loss",
    }
)


def _calculate_regression_fused(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> MetricValues:
    """Compute all 15 same-shape regression metrics with shared subexpressions.

    Avoids redundant computation by reusing intermediate values (diff, abs_diff,
    sq_diff) across metrics instead of recomputing them independently. R-squared is the
    exception: it delegates to the shared core so the formula lives in one place. That
    costs nothing because the core's unweighted path spells its residual the same way
    this function binds ``diff``, which is what lets XLA reuse the binding instead of
    emitting a second one.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Dictionary mapping the 15 same-shape regression metric names to computed values.
    """
    p, t = _prepare_arrays(predictions, targets)
    diff = p - t
    abs_diff = jnp.abs(diff)
    sq_diff = diff**2

    # Core statistics (computed once)
    mse_val = jnp.mean(sq_diff)
    mae_val = jnp.mean(abs_diff)

    # Derived metrics using shared subexpressions
    return {
        "mse": mse_val,
        "mae": mae_val,
        "rmse": jnp.sqrt(mse_val),
        "mean_error": jnp.mean(diff),
        # Delegates rather than inlining a third copy of the formula. The core's
        # unweighted path is the same expression down to the spelling of the
        # residual, so the compiled suite is unchanged: 118 fusions against 118 at
        # both 4,096 and 262,144 points, and the value is bit-identical. Interleaved
        # medians agree to within run-to-run noise, so the fusion count is the claim.
        "r_squared": coefficient_of_determination(p, t),
        "mape": jnp.mean(abs_diff / (jnp.abs(t) + _EPSILON)),
        "relative_error": jnp.sqrt(jnp.sum(sq_diff)) / (jnp.sqrt(jnp.sum(t**2)) + _EPSILON),
        "explained_variance": 1.0 - jnp.var(diff) / (jnp.var(t) + _EPSILON),
        "max_error": jnp.max(abs_diff),
        "huber_loss": jnp.mean(_huber_elementwise(sq_diff, abs_diff, HUBER_DELTA)),
        "quantile_loss": jnp.mean(_quantile_elementwise(diff, QUANTILE_LEVEL)),
        "log_cosh_loss": jnp.mean(jnp.logaddexp(diff, -diff) - jnp.log(2.0)),
        "smape": jnp.mean(safe_divide(abs_diff, (jnp.abs(p) + jnp.abs(t)) / 2.0)),
        "relative_l2_error": relative_l2_error(p, t),
        # Reuses sq_diff, so this adds one sqrt and one mean rather than a second pass over
        # the inputs. Evaluated at the registered defaults, which is how calculate_all
        # invokes every entry here; the constants are imported from the function itself so
        # the fused copy cannot drift away from the default it stands in for.
        "charbonnier_loss": jnp.mean(
            jnp.sqrt(sq_diff + CHARBONNIER_EPSILON**2) ** CHARBONNIER_ALPHA
        ),
    }


def calculate_all(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    metrics: Sequence[str] | None = None,
) -> MetricValues:
    """Calculate multiple evaluation metrics at once.

    Delegates to MetricRegistry for function lookup, except for the metrics the fused
    single-pass regression suite covers. Those are taken from one traversal whenever the
    request includes all of them, and anything else requested is dispatched individually
    beside it, so adding a metric to the general domain costs one dispatch rather than the
    whole fused pass.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        metrics: Subset of metric names to compute. Defaults to same-shape
            Tier 0 general metrics.

    Returns:
        Dictionary mapping metric names to computed scalar values.

    Raises:
        ValueError: If an unknown metric is requested or metric is not Tier 0.
    """
    registry = MetricRegistry()
    if metrics is None:
        names = sorted(
            entry.name
            for entry in registry.list_by_tier(MetricTier.PURE_FUNCTION)
            if entry.domain == "general" and entry.signature == MetricSignature.PREDICTIONS_TARGETS
        )
    else:
        names = list(metrics)

    # The fused traversal computes `_FUSED_REGRESSION_NAMES` from shared subexpressions in
    # one pass, so it pays exactly when nothing it computes is wasted: when every metric it
    # produces was asked for. Whether anything else was also asked for has no bearing on
    # that, which is why this is a subset test rather than an equality one. Equality is
    # strictly stronger than the pass needs, and the surplus strength costs: under it one
    # extra metric in the request discards all fourteen. Since
    # `register_metric` defaults to `domain="general"` and the default request is exactly
    # the general same-shape metrics, registering any custom metric abandoned the pass for
    # the default call -- with no error and no wrong answer, just every metric dispatched
    # one at a time. Measured at 4,096 points that cost about 1.6x (1.65 ms against 1.02,
    # and 1.76 against 1.08 on another run); the claim is that the pass is taken at all,
    # since the timing moves between runs while the selection does not.
    # A request narrower than the pass still dispatches one by one, which is cheaper there.
    fused: MetricValues = (
        _calculate_regression_fused(predictions, targets)
        if set(names) >= _FUSED_REGRESSION_NAMES
        else {}
    )

    results: MetricValues = {}
    for name in names:
        if name in fused:
            results[name] = fused[name]
            continue
        if not registry.has(name):
            available = sorted(registry.list_names())
            msg = f"Unknown metric: {name!r}. Available: {available}"
            raise ValueError(msg)
        try:
            fn = registry.get_function(name)
        except TypeError as e:
            msg = f"Metric '{name}' is not a pure function and cannot be used with calculate_all"
            raise ValueError(msg) from e
        results[name] = fn(predictions, targets)
    return results
