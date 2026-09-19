"""Classification metrics for binary and multiclass evaluation.

Pure functions for computing standard classification metrics. All functions
accept JAX arrays and return scalar values (except confusion_matrix which
returns a JAX array).

Includes 14 functions: accuracy, precision, recall, f1_score, fbeta_score,
confusion_matrix, roc_auc, average_precision, log_loss, matthews_corrcoef,
cohen_kappa, balanced_accuracy, specificity, sensitivity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from calibrax.metrics._utils import _EPSILON, _prepare_class_arrays, reduce_values, safe_divide


# Predictions are class indices (1D) or per-class scores (2D).
_CLASS_SCORES_NDIM = 2


def _to_class_indices(predictions: ArrayLike) -> jax.Array:
    """Convert predictions to class indices.

    If 2D (probabilities), takes argmax along last axis.
    If 1D, assumes already class indices.

    Args:
        predictions: Class indices or probability array.

    Returns:
        1D array of integer class indices.
    """
    p = jnp.asarray(predictions)
    if p.ndim == _CLASS_SCORES_NDIM:
        return jnp.argmax(p, axis=-1)
    return p.astype(jnp.int32)


def _binary_confusion_counts(
    predictions: ArrayLike, targets: ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute binary confusion matrix counts (TP, FP, FN, TN).

    Args:
        predictions: Binary predictions (0 or 1).
        targets: Binary ground truth (0 or 1).

    Returns:
        Tuple of (tp, fp, fn, tn) as JAX scalars.
    """
    p, t = _prepare_class_arrays(predictions, targets)
    tp = jnp.sum((p == 1) & (t == 1))
    fp = jnp.sum((p == 1) & (t == 0))
    fn = jnp.sum((p == 0) & (t == 1))
    tn = jnp.sum((p == 0) & (t == 0))
    return tp, fp, fn, tn


def accuracy(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:
    """Fraction of correct predictions.

    Note:
        Direction: HIGHER (1.0 = perfect).
        Range: [0, 1].
        Not a proper scoring rule.

    Args:
        predictions: Predicted class indices or probability array.
        targets: Ground truth class indices.

    Returns:
        Accuracy as a scalar value.
    """
    p = _to_class_indices(predictions)
    p, t = _prepare_class_arrays(p, targets)
    return jnp.mean(p == t)


def confusion_matrix(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    num_classes: int | None = None,
) -> jax.Array:
    """Compute confusion matrix.

    Note:
        Helper function, not registered in MetricRegistry (returns array).
        Rows are true classes, columns are predicted classes.

    Args:
        predictions: Predicted class indices or probability array.
        targets: Ground truth class indices.
        num_classes: Number of classes. Inferred from data if None.

    Returns:
        Confusion matrix of shape (num_classes, num_classes) as JAX array.
    """
    p = _to_class_indices(predictions)
    p, t = _prepare_class_arrays(p, targets)
    if num_classes is None:
        num_classes = int(jnp.maximum(jnp.max(p), jnp.max(t))) + 1
    return jnp.zeros((num_classes, num_classes), dtype=jnp.int32).at[t, p].add(1)


def precision(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    average: str = "binary",
    num_classes: int | None = None,
) -> jax.Array:
    """Precision: TP / (TP + FP).

    Note:
        Direction: HIGHER (1.0 = no false positives).
        Range: [0, 1].

    Args:
        predictions: Predicted class indices or probability array.
        targets: Ground truth class indices.
        average: Averaging method. "binary" for binary classification,
            "micro" sums globally, "macro" averages per-class,
            "weighted" weights by class frequency.
        num_classes: Number of classes for the per-class averages; passed statically
            it makes the metric jit-compatible, ``None`` reads it from the data.

    Returns:
        Precision as a scalar value.
    """
    return _precision_recall_fbeta(
        predictions, targets, beta=0.0, average=average, num_classes=num_classes
    )[0]


def recall(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    average: str = "binary",
    num_classes: int | None = None,
) -> jax.Array:
    """Recall (sensitivity): TP / (TP + FN).

    Note:
        Direction: HIGHER (1.0 = no false negatives).
        Range: [0, 1].

    Args:
        predictions: Predicted class indices or probability array.
        targets: Ground truth class indices.
        average: Averaging method. "binary", "micro", "macro", "weighted".

        num_classes: Number of classes for the per-class averages; passed statically
            it makes the metric jit-compatible, ``None`` reads it from the data.

    Returns:
        Recall as a scalar value.
    """
    return _precision_recall_fbeta(
        predictions, targets, beta=0.0, average=average, num_classes=num_classes
    )[1]


def fbeta_score(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    beta: float = 1.0,
    average: str = "binary",
    num_classes: int | None = None,
) -> jax.Array:
    """Generalized F-measure with configurable beta.

    ``F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)``

    Note:
        Direction: HIGHER (1.0 = perfect).
        Range: [0, 1].
        beta < 1 weights precision more; beta > 1 weights recall more.

    Args:
        predictions: Predicted class indices or probability array.
        targets: Ground truth class indices.
        beta: Weight of recall vs precision. 1.0 = F1, 2.0 = F2.
        average: Averaging method. "binary", "micro", "macro", "weighted".

        num_classes: Number of classes for the per-class averages; passed statically
            it makes the metric jit-compatible, ``None`` reads it from the data.

    Returns:
        F-beta score as a scalar value.
    """
    return _precision_recall_fbeta(
        predictions, targets, beta=beta, average=average, num_classes=num_classes
    )[2]


def f1_score(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    average: str = "binary",
    num_classes: int | None = None,
) -> jax.Array:
    """F1 score: harmonic mean of precision and recall.

    Equivalent to ``fbeta_score(predictions, targets, beta=1.0)``.

    Note:
        Direction: HIGHER (1.0 = perfect).
        Range: [0, 1].

    Args:
        predictions: Predicted class indices or probability array.
        targets: Ground truth class indices.
        average: Averaging method. "binary", "micro", "macro", "weighted".
        num_classes: Number of classes for the per-class averages; passed statically
            it makes the metric jit-compatible, ``None`` reads it from the data.

    Returns:
        F1 score as a scalar value.
    """
    return fbeta_score(predictions, targets, beta=1.0, average=average, num_classes=num_classes)


def _fbeta_of(precision_value: jax.Array, recall_value: jax.Array, beta: float) -> jax.Array:
    """F-beta from precision and recall; ``precision`` itself when ``beta`` is 0."""
    beta_sq = beta**2
    if beta_sq == 0:
        return precision_value
    return (
        (1 + beta_sq)
        * precision_value
        * recall_value
        / (beta_sq * precision_value + recall_value + _EPSILON)
    )


def _precision_recall_fbeta(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    beta: float,
    average: str,
    num_classes: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute precision, recall, and F-beta together.

    ``"macro"`` and ``"weighted"`` average the per-class values, F-beta included:
    macro F-beta is the mean of the per-class F-betas and weighted F-beta their
    support-weighted mean, as scikit-learn and torchmetrics define them, not the
    F-beta of the averaged precision and recall.

    Args:
        predictions: Predicted class indices or probability array.
        targets: Ground truth class indices.
        beta: F-beta weight parameter.
        average: Averaging method.
        num_classes: Number of classes for the per-class averages; ``None`` reads
            it from the data, which is not jit-compatible.

    Returns:
        Tuple of (precision, recall, fbeta) as scalar values.

    Raises:
        ValueError: If ``average`` is not one of the supported modes.
    """
    p = _to_class_indices(predictions)
    p, t = _prepare_class_arrays(p, targets)

    if average == "binary":
        tp, fp, fn, _tn = _binary_confusion_counts(p, t)
        prec = safe_divide(tp, tp + fp)
        rec = safe_divide(tp, tp + fn)
        return prec, rec, _fbeta_of(prec, rec, beta)
    if average == "micro":
        cm = confusion_matrix(p, t, num_classes=num_classes)
        tp = jnp.trace(cm)
        total_pred = jnp.sum(cm)
        prec = safe_divide(tp, total_pred)
        rec = safe_divide(tp, total_pred)
        return prec, rec, _fbeta_of(prec, rec, beta)
    if average not in ("macro", "weighted"):
        msg = f"Unknown average mode: {average!r}. Use 'binary', 'micro', 'macro', or 'weighted'."
        raise ValueError(msg)

    if num_classes is None:
        num_classes = int(jnp.maximum(jnp.max(p), jnp.max(t))) + 1
    cm = confusion_matrix(p, t, num_classes=num_classes)
    per_class_tp = jnp.diag(cm)
    per_class_pred = jnp.sum(cm, axis=0)
    per_class_true = jnp.sum(cm, axis=1)
    per_class_prec = safe_divide(per_class_tp, per_class_pred)
    per_class_rec = safe_divide(per_class_tp, per_class_true)
    per_class_fb = _fbeta_of(per_class_prec, per_class_rec, beta)

    if average == "macro":
        return jnp.mean(per_class_prec), jnp.mean(per_class_rec), jnp.mean(per_class_fb)
    weights = per_class_true / (jnp.sum(per_class_true) + _EPSILON)
    return (
        jnp.sum(per_class_prec * weights),
        jnp.sum(per_class_rec * weights),
        jnp.sum(per_class_fb * weights),
    )


def softmax_cross_entropy(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    reduction: str = "mean",
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Cross-entropy of integer labels under the softmax of ``logits``.

    The per-element loss is ``-log_softmax(logits)[label]``, with the class axis last;
    the class count is the size of that axis. Reduced through the shared
    ``reduce_values``, so ``mask``, ``weights``, ``reduction`` and ``axis`` mean what they
    mean in every other loss.

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Not a true metric.

    Args:
        logits: Unnormalised scores with shape ``(..., num_classes)``.
        labels: Integer class labels with shape ``(...)``.
        mask: Elements to keep (boolean, broadcastable to ``labels``), or ``None``.
        weights: Element weights (broadcastable to ``labels``), or ``None``.
        reduction: ``"none"``, ``"mean"``, ``"sum"`` or ``"batch_sum"``.
        axis: Axis or axes for ``"mean"`` and ``"sum"``.

    Returns:
        Mean cross-entropy as a scalar value, or the reduced values.

    Raises:
        ValueError: If ``labels`` does not have the shape of ``logits`` without its class axis.
    """
    scores = jnp.asarray(logits)
    targets = jnp.asarray(labels).astype(jnp.int32)
    if targets.shape != scores.shape[:-1]:
        msg = (
            f"labels must have the shape of logits without the class axis: "
            f"logits {scores.shape}, labels {targets.shape}"
        )
        raise ValueError(msg)
    log_probs = jax.nn.log_softmax(scores, axis=-1)
    picked = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    return reduce_values(-picked, mask=mask, weights=weights, reduction=reduction, axis=axis)


def roc_auc(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:
    """Area under the ROC curve (binary classification only).

    Note:
        Direction: HIGHER (1.0 = perfect separation, 0.5 = random).
        Range: [0, 1].

    Args:
        predictions: Probability scores for the positive class.
        targets: Binary labels (0 or 1).

    Returns:
        AUC-ROC as a scalar value.
    """
    p = jnp.asarray(predictions).ravel()
    t = jnp.asarray(targets).astype(jnp.int32).ravel()

    # Sort by descending score
    order = jnp.argsort(-p)
    sorted_targets = t[order]

    # Compute TPR and FPR at each threshold
    tps = jnp.cumsum(sorted_targets)
    fps = jnp.cumsum(1 - sorted_targets)
    total_pos = jnp.sum(t)
    total_neg = jnp.sum(1 - t)

    tpr = safe_divide(tps, total_pos)
    fpr = safe_divide(fps, total_neg)

    # Prepend origin
    tpr = jnp.concatenate([jnp.array([0.0]), tpr])
    fpr = jnp.concatenate([jnp.array([0.0]), fpr])

    # Trapezoidal rule
    return jnp.trapezoid(tpr, fpr)


def average_precision(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:
    """Area under precision-recall curve.

    Note:
        Direction: HIGHER (1.0 = perfect).
        Range: [0, 1].

    Args:
        predictions: Probability scores for the positive class.
        targets: Binary labels (0 or 1).

    Returns:
        Average precision as a scalar value.
    """
    p = jnp.asarray(predictions).ravel()
    t = jnp.asarray(targets).astype(jnp.int32).ravel()

    order = jnp.argsort(-p)
    sorted_targets = t[order]

    tps = jnp.cumsum(sorted_targets)
    total_pred = jnp.arange(1, len(sorted_targets) + 1)
    precisions = safe_divide(tps, total_pred)

    # AP = sum of precision at each recall change point
    return jnp.sum(precisions * sorted_targets) / (jnp.sum(t) + _EPSILON)


def log_loss(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    eps: float = 1e-7,
) -> jax.Array:
    """Logarithmic loss (cross-entropy).

    Note:
        Direction: LOWER (0.0 = perfect).
        Range: [0, inf).
        Is a proper scoring rule — minimized by the true distribution.

    Args:
        predictions: Predicted probabilities. For binary: 1D array of
            P(class=1). For multiclass: 2D array of shape
            (n_samples, n_classes).
        targets: Ground truth class indices.
        eps: Clipping value to avoid log(0).

    Returns:
        Log loss as a scalar value.
    """
    p = jnp.asarray(predictions)
    t = jnp.asarray(targets).astype(jnp.int32)

    if p.ndim == 1:
        # Binary: clip and compute binary cross-entropy
        p_clipped = jnp.clip(p, eps, 1.0 - eps)
        return -jnp.mean(t * jnp.log(p_clipped) + (1 - t) * jnp.log(1 - p_clipped))
    # Multiclass: pick predicted probability for true class
    p_clipped = jnp.clip(p, eps, 1.0 - eps)
    log_probs = jnp.log(p_clipped[jnp.arange(len(t)), t])
    return -jnp.mean(log_probs)


def matthews_corrcoef(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:
    """Matthews correlation coefficient.

    Note:
        Direction: HIGHER (1.0 = perfect, 0.0 = random, -1.0 = inverse).
        Range: [-1, 1].
        Considered the most informative single score for binary classification.

    Args:
        predictions: Predicted class indices or probability array.
        targets: Ground truth class indices.

    Returns:
        MCC as a scalar value.
    """
    p = _to_class_indices(predictions)
    p, t = _prepare_class_arrays(p, targets)
    tp, fp, fn, tn = _binary_confusion_counts(p, t)

    numerator = tp * tn - fp * fn
    denominator = jnp.sqrt(
        (tp + fp).astype(jnp.float32)
        * (tp + fn).astype(jnp.float32)
        * (tn + fp).astype(jnp.float32)
        * (tn + fn).astype(jnp.float32)
    )
    return safe_divide(numerator, denominator)


def cohen_kappa(
    predictions: ArrayLike, targets: ArrayLike, *, num_classes: int | None = None
) -> jax.Array:
    """Cohen's kappa coefficient for inter-rater agreement.

    ``kappa = (accuracy - expected_accuracy) / (1 - expected_accuracy)``

    Note:
        Direction: HIGHER (1.0 = perfect agreement, 0.0 = chance agreement).
        Range: [-1, 1] (negative = worse than chance).

    Args:
        predictions: Predicted class indices or probability array.
        targets: Ground truth class indices.
        num_classes: Number of classes, to trace under ``jax.jit``; None reads it from the
            largest index, eagerly.

    Returns:
        Cohen's kappa as a scalar value.
    """
    p = _to_class_indices(predictions)
    p, t = _prepare_class_arrays(p, targets)
    n = len(t)

    cm = confusion_matrix(p, t, num_classes=num_classes)
    observed_accuracy = jnp.trace(cm) / n

    row_sums = jnp.sum(cm, axis=1).astype(jnp.float32)
    col_sums = jnp.sum(cm, axis=0).astype(jnp.float32)
    expected_accuracy = jnp.sum(row_sums * col_sums) / (n * n)

    return safe_divide(
        jnp.array(observed_accuracy - expected_accuracy),
        jnp.array(1.0 - expected_accuracy),
    )


def balanced_accuracy(
    predictions: ArrayLike, targets: ArrayLike, *, num_classes: int | None = None
) -> jax.Array:
    """Balanced accuracy: the mean recall over the classes the targets hold.

    A class no target has has no recall and takes no part, as in scikit-learn.

    Note:
        Direction: HIGHER (1.0 = perfect).
        Range: [0, 1].
        Equal to standard accuracy on balanced datasets. More informative
        than accuracy on imbalanced datasets.

    Args:
        predictions: Predicted class indices or probability array.
        targets: Ground truth class indices.
        num_classes: Number of classes, to trace under ``jax.jit``; None reads it from the
            largest index, eagerly.

    Returns:
        Balanced accuracy as a scalar value.
    """
    p = _to_class_indices(predictions)
    p, t = _prepare_class_arrays(p, targets)

    cm = confusion_matrix(p, t, num_classes=num_classes)
    support = jnp.sum(cm, axis=1)
    held = support > 0
    recall = jnp.diag(cm) / jnp.where(held, support, 1)
    return jnp.sum(jnp.where(held, recall, 0.0)) / jnp.sum(held)


def specificity(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:
    """Specificity (true negative rate): TN / (TN + FP).

    Note:
        Direction: HIGHER (1.0 = no false positives on negatives).
        Range: [0, 1].

    Args:
        predictions: Binary predictions (0 or 1).
        targets: Binary ground truth (0 or 1).

    Returns:
        Specificity as a scalar value.
    """
    p = _to_class_indices(predictions)
    p, t = _prepare_class_arrays(p, targets)
    _tp, fp, _fn, tn = _binary_confusion_counts(p, t)
    return safe_divide(tn, tn + fp)


def sensitivity(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:
    """Sensitivity (true positive rate): TP / (TP + FN).

    Equivalent to recall for binary classification.

    Note:
        Direction: HIGHER (1.0 = no false negatives).
        Range: [0, 1].

    Args:
        predictions: Binary predictions (0 or 1).
        targets: Binary ground truth (0 or 1).

    Returns:
        Sensitivity as a scalar value.
    """
    p = _to_class_indices(predictions)
    p, t = _prepare_class_arrays(p, targets)
    tp, _fp, fn, _tn = _binary_confusion_counts(p, t)
    return safe_divide(tp, tp + fn)
