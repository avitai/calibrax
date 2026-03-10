"""Fairness metrics for algorithmic bias evaluation.

Pure functions for assessing disparities between demographic groups.
All require a ``protected_attribute`` array in addition to predictions/targets.

The impossibility theorem (Chouldechova 2017, Kleinberg et al. 2016) states that
demographic parity, equalized odds, and predictive value parity cannot all hold
simultaneously for an imperfect classifier with different base rates across groups.
This module provides multiple fairness criteria so users can understand the
inherent trade-offs.

Registered with ``domain="fairness"``, ``signature=MetricSignature.CUSTOM``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON, safe_divide


def demographic_parity_ratio(
    predictions: Any,
    protected_attribute: Any,
) -> Any:
    """Ratio of positive prediction rates across demographic groups.

    Computes min(rate_a/rate_b, rate_b/rate_a) for all group pairs,
    returning the minimum pairwise ratio. Does NOT use targets.

    Args:
        predictions: Binary predictions or probabilities, shape (n,).
        protected_attribute: Group membership labels, shape (n,).

    Returns:
        DPR in [0, 1]. 1.0 = perfect demographic parity.

    Examples:
        >>> import jax.numpy as jnp
        >>> preds = jnp.array([1, 1, 0, 0, 1, 1])
        >>> groups = jnp.array([0, 0, 0, 1, 1, 1])
        >>> demographic_parity_ratio(preds, groups)  # 2/3 / (2/3) = 1.0
        ...
    """
    predictions = jnp.asarray(predictions, dtype=jnp.float32)
    pa = jnp.asarray(protected_attribute)
    unique_vals = jnp.unique(pa, size=int(jnp.max(pa)) + 1)
    n_groups = unique_vals.shape[0]

    # Build group masks: (n_groups, n)
    group_masks = pa[None, :] == unique_vals[:, None]
    group_sizes = jnp.sum(group_masks, axis=1)  # (n_groups,)
    group_positives = jnp.sum(group_masks * (predictions > 0.5)[None, :], axis=1)  # (n_groups,)
    rates = safe_divide(group_positives, group_sizes)  # (n_groups,)

    # Pairwise symmetric min ratio via broadcasting
    rate_matrix = jnp.minimum(
        safe_divide(rates[:, None], rates[None, :]),
        safe_divide(rates[None, :], rates[:, None]),
    )
    # Pairs where both rates are non-zero are valid
    valid = (rates[:, None] > _EPSILON) & (rates[None, :] > _EPSILON)
    off_diag = ~jnp.eye(n_groups, dtype=bool)
    # Replace diagonal and invalid pairs with 1.0 (neutral for min)
    rate_matrix = jnp.where(valid & off_diag, rate_matrix, 1.0)

    # Any off-diagonal pair where exactly one rate is zero → ratio is 0
    has_zero_pair = jnp.any(((rates[:, None] < _EPSILON) ^ (rates[None, :] < _EPSILON)) & off_diag)
    return jnp.where(has_zero_pair, 0.0, jnp.min(rate_matrix))


def equalized_odds_difference(
    predictions: Any,
    targets: Any,
    protected_attribute: Any,
) -> Any:
    """Maximum absolute difference in TPR or FPR across groups.

    max(|TPR_a - TPR_b|, |FPR_a - FPR_b|) over all group pairs.

    Args:
        predictions: Binary predictions, shape (n,).
        targets: Binary ground truth, shape (n,).
        protected_attribute: Group membership labels, shape (n,).

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
    predictions = jnp.asarray(predictions, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    pa = jnp.asarray(protected_attribute)
    unique_vals = jnp.unique(pa, size=int(jnp.max(pa)) + 1)

    # Build group masks: (n_groups, n)
    group_masks = pa[None, :] == unique_vals[:, None]  # (n_groups, n)

    pos_mask = targets > 0.5  # (n,)
    neg_mask = ~pos_mask  # (n,)

    # Per-group counts
    group_positives = jnp.sum(group_masks * pos_mask[None, :], axis=1)  # (n_groups,)
    group_negatives = jnp.sum(group_masks * neg_mask[None, :], axis=1)  # (n_groups,)

    pred_pos = predictions > 0.5  # (n,)

    # TP and FP per group
    group_tp = jnp.sum(group_masks * (pred_pos & pos_mask)[None, :], axis=1)
    group_fp = jnp.sum(group_masks * (pred_pos & neg_mask)[None, :], axis=1)

    tprs = safe_divide(group_tp, group_positives)  # (n_groups,)
    fprs = safe_divide(group_fp, group_negatives)  # (n_groups,)

    # Pairwise max absolute difference
    tpr_diff = jnp.max(jnp.abs(tprs[:, None] - tprs[None, :]))
    fpr_diff = jnp.max(jnp.abs(fprs[:, None] - fprs[None, :]))

    return jnp.maximum(tpr_diff, fpr_diff)


def equal_opportunity_difference(
    predictions: Any,
    targets: Any,
    protected_attribute: Any,
) -> Any:
    """Absolute difference in TPR across demographic groups.

    Simpler than equalized odds — only examines positive outcomes.

    Args:
        predictions: Binary predictions, shape (n,).
        targets: Binary ground truth, shape (n,).
        protected_attribute: Group membership labels, shape (n,).

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
    predictions = jnp.asarray(predictions, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    pa = jnp.asarray(protected_attribute)
    unique_vals = jnp.unique(pa, size=int(jnp.max(pa)) + 1)

    # Build group masks: (n_groups, n)
    group_masks = pa[None, :] == unique_vals[:, None]  # (n_groups, n)

    pos_mask = targets > 0.5  # (n,)

    # Per-group positives
    group_positives = jnp.sum(group_masks * pos_mask[None, :], axis=1)  # (n_groups,)

    # TP per group
    pred_pos = predictions > 0.5  # (n,)
    group_tp = jnp.sum(group_masks * (pred_pos & pos_mask)[None, :], axis=1)

    tprs = safe_divide(group_tp, group_positives)  # (n_groups,)

    # Max absolute pairwise difference
    return jnp.max(jnp.abs(tprs[:, None] - tprs[None, :]))


def disparate_impact_ratio(
    predictions: Any,
    protected_attribute: Any,
) -> Any:
    """Disparate impact ratio (same as demographic parity ratio).

    Named following US legal terminology (80% rule). Values < 0.8
    typically indicate disparate impact under US legal standards.

    Args:
        predictions: Binary predictions or probabilities, shape (n,).
        protected_attribute: Group membership labels, shape (n,).

    Returns:
        DIR in [0, 1]. Values >= 0.8 generally pass the 80% rule.

    Examples:
        >>> import jax.numpy as jnp
        >>> preds = jnp.array([1, 1, 0, 0, 1, 1])
        >>> groups = jnp.array([0, 0, 0, 1, 1, 1])
        >>> disparate_impact_ratio(preds, groups)
        ...
    """
    return demographic_parity_ratio(predictions, protected_attribute)


def group_metric_breakdown(
    metric_fn: Callable[..., float],
    predictions: Any,
    targets: Any,
    protected_attribute: Any,
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

    unique_vals = jnp.unique(pa, size=int(jnp.max(pa)) + 1)
    results: dict[str, float] = {}

    for val in unique_vals:
        mask = pa == val
        n_group = int(jnp.sum(mask))
        if n_group < 2:
            continue
        group_preds = predictions[mask]
        group_targets = targets[mask]
        results[str(int(val))] = float(metric_fn(group_preds, group_targets))

    return results
