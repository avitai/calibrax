"""Calibration metrics for probability calibration assessment.

Pure functions for measuring how well predicted probabilities match
observed frequencies. Calibration metrics differ from classification
metrics: classification asks "which class?" while calibration asks
"how reliable is the stated confidence?"

Includes 10 functions: brier_score, expected_calibration_error,
maximum_calibration_error, reliability_diagram_bins,
brier_decomposition, adaptive_calibration_error, classwise_ece,
conformal_fdr_selection, false_discovery_proportion, empirical_power.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from calibrax.metrics._utils import _EPSILON, _prepare_arrays, safe_divide
from calibrax.statistics.significance import benjamini_hochberg_adjust


def _bin_predictions(
    predictions: ArrayLike,
    targets: ArrayLike,
    num_bins: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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


def brier_score(predictions: ArrayLike, targets: ArrayLike) -> jax.Array:
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
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    num_bins: int = 10,
) -> jax.Array:
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
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    num_bins: int = 10,
) -> jax.Array:
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
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    num_bins: int = 10,
) -> dict[str, jax.Array]:
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
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    num_bins: int = 10,
) -> dict[str, jax.Array]:
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
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    num_bins: int = 10,
) -> jax.Array:
    """Adaptive calibration error (ACE) with equal-mass binning (Nixon et al. 2019).

    Uses equal-mass bins (equal number of samples per bin) instead of
    ECE's equal-width bins. More robust to imbalanced confidence
    distributions. The sorted predictions fill the bins in order, the first ``n % num_bins``
    bins taking one extra; bin membership depends only on the sample count, so it is fixed at
    trace time and the metric runs under ``jax.jit`` and ``jax.grad``.

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
    p, t = p.ravel(), t.ravel().astype(p.dtype)
    order = jnp.argsort(p)
    n = p.shape[0]

    # Each sorted position's bin: the first n % num_bins bins hold one sample more.
    size, extra = divmod(n, num_bins)
    bins = np.repeat(np.arange(num_bins), [size + 1] * extra + [size] * (num_bins - extra))
    counts = jnp.asarray(np.bincount(bins, minlength=num_bins), p.dtype)
    confidence = jax.ops.segment_sum(p[order], bins, num_segments=num_bins)
    accuracy = jax.ops.segment_sum(t[order], bins, num_segments=num_bins)
    # A bin's |mean accuracy - mean confidence| times its size is |sum gap|.
    return jnp.sum(jnp.where(counts > 0, jnp.abs(accuracy - confidence), 0.0)) / n


def classwise_ece(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    num_bins: int = 10,
    num_classes: int | None = None,
) -> jax.Array:
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

    Raises:
        ValueError: If ``predictions`` is not two-dimensional.
    """
    p = jnp.asarray(predictions)
    t = jnp.asarray(targets).astype(jnp.int32)

    if p.ndim == 1:
        msg = "classwise_ece requires 2D predictions (n_samples, n_classes)"
        raise ValueError(msg)

    if num_classes is None:
        num_classes = p.shape[1]

    def _ece_for_class(class_idx: jax.Array) -> jax.Array:
        class_probs = p[:, class_idx]
        class_targets = (t == class_idx).astype(jnp.float32)
        return expected_calibration_error(class_probs, class_targets, num_bins=num_bins)

    class_indices = jnp.arange(num_classes)
    per_class_ece = jax.vmap(_ece_for_class)(class_indices)
    return jnp.mean(per_class_ece)


def conformal_fdr_selection(
    cal_scores: ArrayLike,
    test_scores: ArrayLike,
    *,
    q: ArrayLike = 0.1,
    cal_null_mask: ArrayLike | None = None,
) -> jax.Array:
    """Selective-conformal test-set discovery mask with finite-sample FDR control.

    Computes conformal p-values ``p_j = (1 + #{i in cal_null : S_i >= S_test_j}) / (1 + n_cal)``
    via ``O((n + m) log n)`` binary search and applies Benjamini-Hochberg step-up selection
    at level ``q`` (Jin and Candes, Biometrika 2023; Bates et al., Ann. Stat. 2023). Larger
    scores indicate stronger evidence against the null hypothesis.

    Note:
        Returns a boolean selection array rather than a scalar; not registered in
        ``MetricRegistry``.

    Args:
        cal_scores: Calibration scores of shape ``(n_cal,)``.
        test_scores: Candidate test scores of arbitrary shape.
        q: Target false discovery rate level in ``(0, 1)`` (defaults to ``0.1``).
        cal_null_mask: Optional boolean mask of shape ``(n_cal,)`` marking calibration
            observations that satisfy the null hypothesis. When ``None``, all entries
            of ``cal_scores`` are treated as null calibration scores.

    Returns:
        Boolean JAX array with the same shape as ``test_scores`` marking selected discoveries.

    Raises:
        ValueError: If eager ``q`` lies outside ``(0, 1)`` or ``cal_null_mask`` differs
            in length from ``cal_scores``.
    """
    if isinstance(q, int | float) and not isinstance(q, jax.Array) and not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q}")

    cal = jnp.asarray(cal_scores, dtype=jnp.float32).ravel()
    test = jnp.asarray(test_scores, dtype=jnp.float32)
    test_flat = test.ravel()

    if cal_null_mask is not None:
        null_mask = jnp.asarray(cal_null_mask).ravel().astype(bool)
        if null_mask.size != cal.size:
            raise ValueError(
                f"Shape mismatch: cal_scores has {cal.size} elements, "
                f"cal_null_mask has {null_mask.size}"
            )
    else:
        null_mask = None

    if cal.size == 0 or test_flat.size == 0:
        return jnp.zeros(test.shape, dtype=bool)

    order = jnp.argsort(cal)
    cal_sorted = cal[order]
    pos = jnp.searchsorted(cal_sorted, test_flat, side="left")
    if null_mask is None:
        ge_count = cal.size - pos
    else:
        null_sorted = null_mask[order].astype(jnp.int32)
        suffix_null = jnp.concatenate(
            [jnp.cumsum(null_sorted[::-1])[::-1], jnp.zeros(1, dtype=jnp.int32)]
        )
        ge_count = suffix_null[pos]

    p_vals = (1.0 + ge_count.astype(jnp.float32)) / (1.0 + float(cal.size))
    q_vals = benjamini_hochberg_adjust(p_vals)
    q_arr = jnp.asarray(q, dtype=jnp.float32)
    valid_q = (q_arr > 0.0) & (q_arr < 1.0)
    return (valid_q & (q_vals <= q_arr)).reshape(test.shape)


def false_discovery_proportion(selected: ArrayLike, is_null: ArrayLike) -> jax.Array:
    """Realized false discovery proportion ``V / max(R, 1)`` (Benjamini and Hochberg 1995).

    Evaluates the fraction of selected discoveries that belong to the null hypothesis.
    Making zero selections (``R == 0``) on a non-empty test set incurs zero false
    discoveries and returns ``0.0``; an empty test set returns ``NaN``.

    Note:
        Direction: LOWER (0.0 = zero false discoveries).
        Range: [0, 1]. ``NaN`` when ``selected`` is empty.

    Args:
        selected: Boolean or binary selection indicator (nonzero = selected).
        is_null: Boolean or binary null-hypothesis indicator (nonzero = true null).

    Returns:
        False discovery proportion as a scalar JAX array, or ``NaN`` on empty inputs.

    Raises:
        ValueError: If ``selected`` and ``is_null`` differ in size.
    """
    sel_raw = jnp.asarray(selected).ravel()
    null_raw = jnp.asarray(is_null).ravel()
    if sel_raw.size != null_raw.size:
        raise ValueError(
            f"Shape mismatch: selected has {sel_raw.size} elements, is_null has {null_raw.size}"
        )
    if sel_raw.size == 0:
        return jnp.array(jnp.nan, dtype=jnp.float32)

    sel = sel_raw > 0
    null = null_raw > 0
    n_sel = jnp.sum(sel.astype(jnp.float32))
    n_false = jnp.sum((sel & null).astype(jnp.float32))
    return jnp.where(n_sel > 0.0, n_false / jnp.maximum(n_sel, 1.0), 0.0)


def empirical_power(selected: ArrayLike, is_alternative: ArrayLike) -> jax.Array:
    """Realized discovery power (true positive selection rate among true alternatives).

    Note:
        Direction: HIGHER (1.0 = all true alternatives selected).
        Range: [0, 1]. ``NaN`` when ``selected`` is empty or zero true alternatives exist.

    Args:
        selected: Boolean or binary selection indicator (nonzero = selected).
        is_alternative: Boolean or binary alternative indicator (nonzero = true non-null).

    Returns:
        Empirical power as a scalar JAX array, or ``NaN`` when undefined.

    Raises:
        ValueError: If ``selected`` and ``is_alternative`` differ in size.
    """
    sel_raw = jnp.asarray(selected).ravel()
    alt_raw = jnp.asarray(is_alternative).ravel()
    if sel_raw.size != alt_raw.size:
        raise ValueError(
            f"Shape mismatch: selected has {sel_raw.size} elements, "
            f"is_alternative has {alt_raw.size}"
        )
    if sel_raw.size == 0:
        return jnp.array(jnp.nan, dtype=jnp.float32)

    sel = sel_raw > 0
    alt = alt_raw > 0
    n_alt = jnp.sum(alt.astype(jnp.float32))
    n_true_disc = jnp.sum((sel & alt).astype(jnp.float32))
    has_alt = n_alt > 0.0
    safe_alt = jnp.where(has_alt, n_alt, 1.0)
    return jnp.where(has_alt, n_true_disc / safe_alt, jnp.nan)
