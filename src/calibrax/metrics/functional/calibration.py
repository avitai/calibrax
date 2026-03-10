"""Calibration metrics for probability calibration assessment.

Pure functions for measuring how well predicted probabilities match
observed frequencies. Calibration metrics differ from classification
metrics: classification asks "which class?" while calibration asks
"how reliable is the stated confidence?"

Includes 7 functions: brier_score, expected_calibration_error,
maximum_calibration_error, reliability_diagram_bins,
brier_decomposition, adaptive_calibration_error, classwise_ece.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON, _prepare_arrays, safe_divide


def _bin_predictions(
    predictions: Any,
    targets: Any,
    num_bins: int,
) -> tuple[Any, Any, Any, Any]:
    """Bin predictions into equal-width confidence bins.

    Args:
        predictions: Predicted probabilities in [0, 1].
        targets: Binary ground truth (0 or 1).
        num_bins: Number of equal-width bins.

    Returns:
        Tuple of (bin_accuracies, bin_confidences, bin_counts, bin_edges)
        as JAX arrays.
    """
    p, t = _prepare_arrays(predictions, targets)
    p, t = p.ravel(), t.ravel()
    bin_edges = jnp.linspace(0.0, 1.0, num_bins + 1)

    # Assign each prediction to a bin (0-indexed)
    bin_indices = jnp.digitize(p, bin_edges[1:-1])
    bin_indices = jnp.clip(bin_indices, 0, num_bins - 1)

    # One-hot encode bin membership: shape (n, num_bins)
    one_hot = jnp.eye(num_bins)[bin_indices]

    bin_counts = jnp.sum(one_hot, axis=0)
    bin_accuracies = jnp.where(
        bin_counts > 0,
        jnp.sum(one_hot * t[:, None], axis=0) / (bin_counts + _EPSILON),
        0.0,
    )
    bin_confidences = jnp.where(
        bin_counts > 0,
        jnp.sum(one_hot * p[:, None], axis=0) / (bin_counts + _EPSILON),
        0.0,
    )

    return bin_accuracies, bin_confidences, bin_counts, bin_edges


def brier_score(predictions: Any, targets: Any) -> Any:
    """Brier score: mean squared error between probabilities and outcomes.

    Note:
        Direction: LOWER (0.0 = perfect calibration).
        Range: [0, 1].
        Strictly proper scoring rule — minimized by the true predictive
        distribution.

    Args:
        predictions: Predicted probabilities in [0, 1].
        targets: Binary ground truth (0 or 1).

    Returns:
        Brier score as a scalar value.
    """
    p, t = _prepare_arrays(predictions, targets)
    return jnp.mean((p.ravel() - t.ravel()) ** 2)


def expected_calibration_error(
    predictions: Any,
    targets: Any,
    *,
    num_bins: int = 10,
) -> Any:
    """Expected calibration error (ECE).

    Weighted average of |accuracy - confidence| across equal-width bins.

    Note:
        Direction: LOWER (0.0 = perfectly calibrated).
        Range: [0, 1].
        NOT a proper scoring rule. Sensitive to bin count — prefer
        adaptive_calibration_error for robustness.

    Args:
        predictions: Predicted probabilities in [0, 1].
        targets: Binary ground truth (0 or 1).
        num_bins: Number of equal-width bins.

    Returns:
        ECE as a scalar value.
    """
    bin_acc, bin_conf, bin_counts, _ = _bin_predictions(predictions, targets, num_bins)
    n = jnp.sum(bin_counts)
    weights = safe_divide(bin_counts, n)
    return jnp.sum(weights * jnp.abs(bin_acc - bin_conf))


def maximum_calibration_error(
    predictions: Any,
    targets: Any,
    *,
    num_bins: int = 10,
) -> Any:
    """Maximum calibration error (MCE).

    Maximum |accuracy - confidence| across all non-empty bins.

    Note:
        Direction: LOWER (0.0 = perfectly calibrated).
        Range: [0, 1].
        NOT a proper scoring rule. Reports worst-case bin calibration.

    Args:
        predictions: Predicted probabilities in [0, 1].
        targets: Binary ground truth (0 or 1).
        num_bins: Number of equal-width bins.

    Returns:
        MCE as a scalar value.
    """
    bin_acc, bin_conf, bin_counts, _ = _bin_predictions(predictions, targets, num_bins)
    errors = jnp.abs(bin_acc - bin_conf)
    # Only consider non-empty bins
    errors = jnp.where(bin_counts > 0, errors, 0.0)
    return jnp.max(errors)


def reliability_diagram_bins(
    predictions: Any,
    targets: Any,
    *,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute binned statistics for reliability diagram plotting.

    Note:
        Not a scalar metric — returns a dict for visualization.
        NOT registered in MetricRegistry.

    Args:
        predictions: Predicted probabilities in [0, 1].
        targets: Binary ground truth (0 or 1).
        num_bins: Number of equal-width bins.

    Returns:
        Dictionary with keys: bin_edges, bin_accuracies,
        bin_confidences, bin_counts.
    """
    bin_acc, bin_conf, bin_counts, bin_edges = _bin_predictions(predictions, targets, num_bins)
    return {
        "bin_edges": bin_edges,
        "bin_accuracies": bin_acc,
        "bin_confidences": bin_conf,
        "bin_counts": bin_counts,
    }


def brier_decomposition(
    predictions: Any,
    targets: Any,
    *,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Decompose Brier score into calibration, resolution, uncertainty.

    Property: ``brier_score = calibration - resolution + uncertainty``.

    Note:
        Not a scalar metric — returns a dict with decomposition components.
        NOT registered in MetricRegistry.

    Args:
        predictions: Predicted probabilities in [0, 1].
        targets: Binary ground truth (0 or 1).
        num_bins: Number of equal-width bins.

    Returns:
        Dictionary with keys: calibration, resolution, uncertainty, brier.
    """
    bin_acc, bin_conf, bin_counts, _ = _bin_predictions(predictions, targets, num_bins)
    t = jnp.asarray(targets).ravel()
    n = jnp.sum(bin_counts)
    base_rate = jnp.mean(t)

    # Calibration (reliability): how well probabilities match frequencies
    calibration = jnp.sum(bin_counts * (bin_acc - bin_conf) ** 2) / (n + _EPSILON)

    # Resolution (sharpness): how much forecasts differ from base rate
    resolution = jnp.sum(bin_counts * (bin_acc - base_rate) ** 2) / (n + _EPSILON)

    # Uncertainty: inherent unpredictability
    uncertainty = base_rate * (1.0 - base_rate)

    return {
        "calibration": calibration,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": brier_score(predictions, targets),
    }


def adaptive_calibration_error(
    predictions: Any,
    targets: Any,
    *,
    num_bins: int = 10,
) -> Any:
    """Adaptive calibration error (ACE) with equal-mass binning.

    Uses equal-mass bins (equal number of samples per bin) instead of
    ECE's equal-width bins. More robust to imbalanced confidence
    distributions.

    Note:
        Direction: LOWER (0.0 = perfectly calibrated).
        Range: [0, 1].
        NOT a proper scoring rule, but more robust than ECE.

    Args:
        predictions: Predicted probabilities in [0, 1].
        targets: Binary ground truth (0 or 1).
        num_bins: Number of equal-mass bins.

    Returns:
        ACE as a scalar value.
    """
    p, t = _prepare_arrays(predictions, targets)
    p, t = p.ravel(), t.ravel()

    order = jnp.argsort(p)
    sorted_p = p[order]
    sorted_t = t[order]

    n = len(sorted_p)
    bin_size = n // num_bins
    remainder = n % num_bins

    total_error = 0.0
    start = 0
    for i in range(num_bins):
        # Distribute remainder across first bins
        end = start + bin_size + (1 if i < remainder else 0)
        if end > start:
            bin_p = sorted_p[start:end]
            bin_t = sorted_t[start:end]
            bin_acc = float(jnp.mean(bin_t))
            bin_conf = float(jnp.mean(bin_p))
            total_error += abs(bin_acc - bin_conf) * (end - start)
        start = end

    return total_error / n


def classwise_ece(
    predictions: Any,
    targets: Any,
    *,
    num_bins: int = 10,
    num_classes: int | None = None,
) -> Any:
    """Classwise expected calibration error for multiclass problems.

    Computes one-vs-rest ECE for each class, then averages. More
    informative than single ECE for multiclass calibration.

    Note:
        Direction: LOWER (0.0 = perfectly calibrated).
        Range: [0, 1].
        NOT a proper scoring rule.

    Args:
        predictions: Predicted probability matrix of shape
            (n_samples, n_classes).
        targets: Ground truth class indices.
        num_bins: Number of bins for per-class ECE.
        num_classes: Number of classes. Inferred from predictions if None.

    Returns:
        Mean classwise ECE as a scalar value.
    """
    p = jnp.asarray(predictions)
    t = jnp.asarray(targets).astype(jnp.int32)

    if p.ndim == 1:
        msg = "classwise_ece requires 2D predictions (n_samples, n_classes)"
        raise ValueError(msg)

    if num_classes is None:
        num_classes = p.shape[1]

    def _ece_for_class(class_idx: Any) -> Any:
        class_probs = p[:, class_idx]
        class_targets = (t == class_idx).astype(jnp.float32)
        return expected_calibration_error(class_probs, class_targets, num_bins=num_bins)

    class_indices = jnp.arange(num_classes)
    per_class_ece = jax.vmap(_ece_for_class)(class_indices)
    return jnp.mean(per_class_ece)
