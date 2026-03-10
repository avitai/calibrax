"""Segmentation metrics for pixel/voxel-level evaluation.

Pure functions for evaluating segmentation quality by comparing
predicted masks against ground truth masks. Supports binary and
multiclass segmentation with multiple averaging modes.

Includes 3 functions: iou, dice_coefficient, pixel_accuracy.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON, _prepare_class_arrays


def iou(
    predictions: Any,
    targets: Any,
    *,
    num_classes: int | None = None,
    average: str = "binary",
) -> Any:
    """Intersection over Union (Jaccard index) for segmentation.

    Measures overlap between predicted and ground truth masks.
    For binary: ``|P ∩ T| / |P ∪ T|``.
    For multiclass: per-class IoU, then averaged.

    Note:
        Direction: HIGHER (1.0 = perfect overlap).
        Range: [0, 1].
        Equivalent to 1 - Jaccard distance. Related to Dice via
        ``dice = 2 * iou / (1 + iou)``.

    Args:
        predictions: Predicted integer mask (0/1 for binary,
            0..num_classes-1 for multiclass).
        targets: Ground truth integer mask.
        num_classes: Number of classes. Required for macro/weighted.
            Inferred from data for binary.
        average: Averaging mode. ``"binary"`` for single-class,
            ``"macro"`` for unweighted mean across classes,
            ``"weighted"`` for frequency-weighted mean.

    Returns:
        IoU score as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> preds = jnp.array([1, 1, 0, 0])
        >>> truth = jnp.array([1, 0, 0, 0])
        >>> iou(preds, truth)  # binary IoU
        0.5
    """
    p, t = _prepare_class_arrays(predictions, targets)
    p, t = p.ravel(), t.ravel()

    if average == "binary":
        intersection = jnp.sum((p == 1) & (t == 1))
        union = jnp.sum((p == 1) | (t == 1))
        return intersection / (union + _EPSILON)

    if num_classes is None:
        num_classes = int(jnp.maximum(jnp.max(p), jnp.max(t))) + 1

    classes = jnp.arange(num_classes)
    p_eq = p[None, :] == classes[:, None]  # (num_classes, n)
    t_eq = t[None, :] == classes[:, None]  # (num_classes, n)
    intersections = jnp.sum(p_eq & t_eq, axis=1)  # (num_classes,)
    unions = jnp.sum(p_eq | t_eq, axis=1)  # (num_classes,)
    class_ious = jnp.where(unions > 0, intersections / (unions + _EPSILON), 0.0)
    class_counts = jnp.sum(t_eq, axis=1)

    if average == "macro":
        # Mean across classes with non-zero union
        valid = class_counts > 0
        n_valid = jnp.sum(valid)
        return jnp.where(n_valid > 0, jnp.sum(class_ious * valid) / n_valid, 0.0)

    # weighted: frequency-weighted mean
    total = jnp.sum(class_counts)
    weights = class_counts / (total + _EPSILON)
    return jnp.sum(class_ious * weights)


def dice_coefficient(
    predictions: Any,
    targets: Any,
    *,
    num_classes: int | None = None,
    average: str = "binary",
) -> Any:
    """Dice coefficient (F1 for segmentation).

    Measures overlap: ``2|P ∩ T| / (|P| + |T|)``. Equivalent to
    F1 score applied to pixel-level classification.

    Note:
        Direction: HIGHER (1.0 = perfect overlap).
        Range: [0, 1].
        Related to IoU via ``dice = 2 * iou / (1 + iou)``
        and ``iou = dice / (2 - dice)``.

    Args:
        predictions: Predicted integer mask (0/1 for binary,
            0..num_classes-1 for multiclass).
        targets: Ground truth integer mask.
        num_classes: Number of classes. Required for macro/weighted.
            Inferred from data for binary.
        average: Averaging mode. ``"binary"`` for single-class,
            ``"macro"`` for unweighted mean across classes,
            ``"weighted"`` for frequency-weighted mean.

    Returns:
        Dice coefficient as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> preds = jnp.array([1, 1, 0, 0])
        >>> truth = jnp.array([1, 0, 0, 0])
        >>> dice_coefficient(preds, truth)  # 2*1 / (2+1)
        0.666...
    """
    p, t = _prepare_class_arrays(predictions, targets)
    p, t = p.ravel(), t.ravel()

    if average == "binary":
        intersection = jnp.sum((p == 1) & (t == 1))
        sum_pred = jnp.sum(p == 1)
        sum_target = jnp.sum(t == 1)
        return 2.0 * intersection / (sum_pred + sum_target + _EPSILON)

    if num_classes is None:
        num_classes = int(jnp.maximum(jnp.max(p), jnp.max(t))) + 1

    classes = jnp.arange(num_classes)
    p_eq = p[None, :] == classes[:, None]
    t_eq = t[None, :] == classes[:, None]
    intersections = jnp.sum(p_eq & t_eq, axis=1)
    sum_preds = jnp.sum(p_eq, axis=1)
    sum_targets = jnp.sum(t_eq, axis=1)
    denoms = sum_preds + sum_targets
    class_dice = jnp.where(denoms > 0, 2.0 * intersections / (denoms + _EPSILON), 0.0)
    class_counts = sum_targets

    if average == "macro":
        valid = class_counts > 0
        n_valid = jnp.sum(valid)
        return jnp.where(n_valid > 0, jnp.sum(class_dice * valid) / n_valid, 0.0)

    # weighted
    total = jnp.sum(class_counts)
    weights = class_counts / (total + _EPSILON)
    return jnp.sum(class_dice * weights)


def pixel_accuracy(predictions: Any, targets: Any) -> Any:
    """Fraction of correctly classified pixels.

    Simple accuracy metric for segmentation tasks. Counts the
    proportion of pixels where prediction matches ground truth.

    Note:
        Direction: HIGHER (1.0 = all pixels correct).
        Range: [0, 1].
        Can be misleadingly high for imbalanced classes. Prefer
        IoU or Dice for class-imbalanced segmentation.

    Args:
        predictions: Predicted integer mask.
        targets: Ground truth integer mask.

    Returns:
        Pixel accuracy as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> preds = jnp.array([0, 1, 1, 0])
        >>> truth = jnp.array([0, 1, 0, 0])
        >>> pixel_accuracy(preds, truth)
        0.75
    """
    p, t = _prepare_class_arrays(predictions, targets)
    p, t = p.ravel(), t.ravel()
    return jnp.mean(p == t)
