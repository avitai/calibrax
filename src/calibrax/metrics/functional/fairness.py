"""Fairness metrics for algorithmic bias evaluation.

Pure functions for assessing disparities between demographic groups.
All require a ``protected_attribute`` array in addition to predictions/targets.

The impossibility theorem (Chouldechova 2017, Kleinberg et al. 2016) states that
demographic parity, equalized odds, and predictive value parity cannot all hold
simultaneously for an imperfect classifier with different base rates across groups.
This module provides multiple fairness criteria so users can understand the
inherent trade-offs.

Group labels are names: renaming a group changes nothing, and groups with no member take no
part. Pass ``num_groups`` to trace a metric under ``jax.jit``, labels then being ids below it;
without it the distinct labels are counted from the data, eagerly.

Registered with ``domain="fairness"``, ``signature=MetricSignature.CUSTOM``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from calibrax.metrics._types import MetricFn
from calibrax.metrics._utils import label_masks, safe_divide


# Scores and labels above this are the positive class; a group needs two members to test.
_POSITIVE_THRESHOLD = 0.5
_MIN_GROUP_SIZE = 2


def _group_rates(
    masks: jax.Array, predictions: ArrayLike, targets: ArrayLike, *, positive: bool
) -> jax.Array:
    """Each group's rate of positive predictions among its positive (or negative) targets.

    The true positive rate with ``positive``, the false positive rate without; a group with no
    such targets has rate 0.
    """
    predicted = (jnp.asarray(predictions, dtype=jnp.float32) > _POSITIVE_THRESHOLD).astype(
        jnp.float32
    )
    actual = jnp.asarray(targets, dtype=jnp.float32) > _POSITIVE_THRESHOLD
    among = (actual if positive else ~actual).astype(jnp.float32)
    return safe_divide(masks @ (predicted * among), masks @ among)


def _spread(rates: jax.Array, present: jax.Array) -> jax.Array:
    """The largest difference between two present groups' rates."""
    return jnp.max(jnp.where(present, rates, -jnp.inf)) - jnp.min(
        jnp.where(present, rates, jnp.inf)
    )


def demographic_parity_ratio(
    predictions: ArrayLike,
    protected_attribute: ArrayLike,
    *,
    num_groups: int | None = None,
) -> jax.Array:
    """Ratio of positive prediction rates across demographic groups.

    Computes min(rate_a/rate_b, rate_b/rate_a) for all group pairs,
    returning the minimum pairwise ratio. Does NOT use targets.

    Args:
        predictions: Binary predictions or probabilities, shape (n,).
        protected_attribute: Group membership labels, shape (n,).
        num_groups: Number of group ids, to trace; None counts the distinct labels.

    Returns:
        DPR in [0, 1]. 1.0 = perfect demographic parity.

    Examples:
        >>> import jax.numpy as jnp
        >>> preds = jnp.array([1, 1, 0, 0, 1, 1])
        >>> groups = jnp.array([0, 0, 0, 1, 1, 1])
        >>> demographic_parity_ratio(preds, groups)  # 2/3 / (2/3) = 1.0
        ...
    """
    masks, sizes = label_masks(protected_attribute, num_groups)
    present = sizes > 0
    selected = (jnp.asarray(predictions, dtype=jnp.float32) > _POSITIVE_THRESHOLD).astype(
        jnp.float32
    )
    rates = masks @ selected / jnp.where(present, sizes, 1.0)
    highest = jnp.max(jnp.where(present, rates, -jnp.inf))
    lowest = jnp.min(jnp.where(present, rates, jnp.inf))
    return jnp.where(highest > 0, lowest / jnp.where(highest > 0, highest, 1.0), 1.0)


def equalized_odds_difference(
    predictions: ArrayLike,
    targets: ArrayLike,
    protected_attribute: ArrayLike,
    *,
    num_groups: int | None = None,
) -> jax.Array:
    """Maximum absolute difference in TPR or FPR across groups.

    max(|TPR_a - TPR_b|, |FPR_a - FPR_b|) over all group pairs.

    Args:
        predictions: Binary predictions, shape (n,).
        targets: Binary ground truth, shape (n,).
        protected_attribute: Group membership labels, shape (n,).
        num_groups: Number of group ids, to trace; None counts the distinct labels.

    Returns:
        EOD in [0, 1]. 0.0 = perfect equalized odds.

    Examples:
        >>> import jax.numpy as jnp
        >>> preds = jnp.array([1, 1, 0, 1, 1, 0])
        >>> targets = jnp.array([1, 1, 0, 1, 1, 0])
        >>> groups = jnp.array([0, 0, 0, 1, 1, 1])
        >>> equalized_odds_difference(preds, targets, groups)
        0.0
    """
    masks, sizes = label_masks(protected_attribute, num_groups)
    tprs = _group_rates(masks, predictions, targets, positive=True)
    fprs = _group_rates(masks, predictions, targets, positive=False)
    present = sizes > 0
    return jnp.maximum(_spread(tprs, present), _spread(fprs, present))


def equal_opportunity_difference(
    predictions: ArrayLike,
    targets: ArrayLike,
    protected_attribute: ArrayLike,
    *,
    num_groups: int | None = None,
) -> jax.Array:
    """Absolute difference in TPR across demographic groups.

    Simpler than equalized odds — only examines positive outcomes.

    Args:
        predictions: Binary predictions, shape (n,).
        targets: Binary ground truth, shape (n,).
        protected_attribute: Group membership labels, shape (n,).
        num_groups: Number of group ids, to trace; None counts the distinct labels.

    Returns:
        EOD in [0, 1]. 0.0 = perfect equal opportunity.

    Examples:
        >>> import jax.numpy as jnp
        >>> preds = jnp.array([1, 1, 0, 1, 1, 0])
        >>> targets = jnp.array([1, 1, 0, 1, 1, 0])
        >>> groups = jnp.array([0, 0, 0, 1, 1, 1])
        >>> equal_opportunity_difference(preds, targets, groups)
        0.0
    """
    masks, sizes = label_masks(protected_attribute, num_groups)
    return _spread(_group_rates(masks, predictions, targets, positive=True), sizes > 0)


def disparate_impact_ratio(
    predictions: ArrayLike,
    protected_attribute: ArrayLike,
    *,
    num_groups: int | None = None,
) -> jax.Array:
    """Disparate impact ratio (same as demographic parity ratio).

    Named following US legal terminology (80% rule). Values < 0.8
    typically indicate disparate impact under US legal standards.

    Args:
        predictions: Binary predictions or probabilities, shape (n,).
        protected_attribute: Group membership labels, shape (n,).
        num_groups: Number of group ids, to trace; None counts the distinct labels.

    Returns:
        DIR in [0, 1]. Values >= 0.8 generally pass the 80% rule.

    Examples:
        >>> import jax.numpy as jnp
        >>> preds = jnp.array([1, 1, 0, 0, 1, 1])
        >>> groups = jnp.array([0, 0, 0, 1, 1, 1])
        >>> disparate_impact_ratio(preds, groups)
        ...
    """
    return demographic_parity_ratio(predictions, protected_attribute, num_groups=num_groups)


def group_metric_breakdown(
    metric_fn: MetricFn,
    predictions: ArrayLike,
    targets: ArrayLike,
    protected_attribute: ArrayLike,
) -> dict[str, float]:
    """Apply any metric function separately to each demographic group.

    Turns any (predictions, targets) -> float metric into a per-group
    breakdown. Groups with fewer than 2 samples are skipped.

    Args:
        metric_fn: Callable with signature (predictions, targets) -> float.
        predictions: Predicted values, shape (n,).
        targets: Ground truth values, shape (n,).
        protected_attribute: Group membership labels, shape (n,).

    Returns:
        Dictionary mapping group names (as strings) to metric values.

    Examples:
        >>> import jax.numpy as jnp
        >>> from calibrax.metrics.functional.regression import mse
        >>> preds = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> targets = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> groups = jnp.array([0, 0, 1, 1])
        >>> group_metric_breakdown(mse, preds, targets, groups)
        {'0': 0.0, '1': 0.0}
    """
    predictions = jnp.asarray(predictions, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    pa = jnp.asarray(protected_attribute)

    unique_vals = jnp.unique(pa)
    results: dict[str, float] = {}

    for val in unique_vals:
        mask = pa == val
        n_group = int(jnp.sum(mask))
        if n_group < _MIN_GROUP_SIZE:
            continue
        group_preds = predictions[mask]
        group_targets = targets[mask]
        results[str(int(val))] = float(metric_fn(group_preds, group_targets))

    return results
