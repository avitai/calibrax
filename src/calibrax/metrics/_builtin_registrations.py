"""Built-in metric registrations for the MetricRegistry.

Contains all _register_*_metrics() functions and the _register_all_builtins()
orchestrator that calls them. Separated from _registry.py to keep the registry
class and decorator focused on infrastructure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from calibrax.core.models import MetricDirection
from calibrax.metrics._types import MetricEntry, MetricProperties, MetricSignature, MetricTier


@dataclass(frozen=True)
class _BuiltinMetricSpec:
    """Named built-in metric registration metadata."""

    name: str
    fn: Any
    description: str
    direction: MetricDirection
    properties: MetricProperties
    signature: MetricSignature = MetricSignature.PREDICTIONS_TARGETS
    domain: str = "general"


_FUSED_REGRESSION_NAMES = frozenset(
    {
        "mse",
        "mae",
        "rmse",
        "r_squared",
        "mape",
        "relative_error",
        "explained_variance",
        "max_error",
        "huber_loss",
        "quantile_loss",
        "log_cosh_loss",
        "smape",
    }
)


def _calculate_regression_fused(
    predictions: Any,
    targets: Any,
) -> dict[str, Any]:
    """Compute all 12 same-shape regression metrics with shared subexpressions.

    Avoids redundant computation by reusing intermediate values (diff, abs_diff,
    sq_diff, mean_t) across metrics instead of recomputing them independently.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.

    Returns:
        Dictionary mapping the 12 same-shape regression metric names to computed values.
    """
    from calibrax.metrics._utils import _EPSILON, _prepare_arrays, safe_divide

    p, t = _prepare_arrays(predictions, targets)
    diff = p - t
    abs_diff = jnp.abs(diff)
    sq_diff = diff**2
    mean_t = jnp.mean(t)

    # Core statistics (computed once)
    mse_val = jnp.mean(sq_diff)
    mae_val = jnp.mean(abs_diff)

    # Derived metrics using shared subexpressions
    return {
        "mse": mse_val,
        "mae": mae_val,
        "rmse": jnp.sqrt(mse_val),
        "r_squared": 1.0 - jnp.sum(sq_diff) / (jnp.sum((t - mean_t) ** 2) + _EPSILON),
        "mape": jnp.mean(abs_diff / (jnp.abs(t) + _EPSILON)),
        "relative_error": jnp.sqrt(jnp.sum(sq_diff)) / (jnp.sqrt(jnp.sum(t**2)) + _EPSILON),
        "explained_variance": 1.0 - jnp.var(diff) / (jnp.var(t) + _EPSILON),
        "max_error": jnp.max(abs_diff),
        "huber_loss": jnp.mean(jnp.where(abs_diff <= 1.0, 0.5 * sq_diff, abs_diff - 0.5)),
        "quantile_loss": jnp.mean(jnp.where(diff <= 0, 0.5 * (-diff), 0.5 * diff)),
        "log_cosh_loss": jnp.mean(jnp.logaddexp(diff, -diff) - jnp.log(2.0)),
        "smape": jnp.mean(safe_divide(abs_diff, (jnp.abs(p) + jnp.abs(t)) / 2.0)),
    }


def _register_all_builtins() -> None:
    """Register all built-in metrics across all domains."""
    _register_regression_metrics()
    _register_classification_metrics()
    _register_calibration_metrics()
    _register_segmentation_metrics()
    _register_distance_metrics()
    _register_divergence_metrics()
    _register_information_metrics()
    _register_ranking_metrics()
    _register_statistical_metrics()
    _register_text_metrics()
    _register_audio_metrics()
    _register_geometric_metrics()
    _register_manifold_metrics()
    _register_graph_metrics()
    _register_image_metrics()
    _register_fairness_metrics()
    _register_clustering_metrics()


def _register_regression_metrics() -> None:
    """Register all 13 regression metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.regression import (
        crps,
        explained_variance,
        huber_loss,
        log_cosh_loss,
        mae,
        mape,
        max_error,
        mse,
        quantile_loss,
        r_squared,
        relative_error,
        rmse,
        smape,
    )

    registry = MetricRegistry()
    builtins = [
        _BuiltinMetricSpec(
            "mse",
            mse,
            "Mean squared error",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "mae",
            mae,
            "Mean absolute error",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_true_metric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "rmse",
            rmse,
            "Root mean squared error",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_true_metric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "r_squared",
            r_squared,
            "Coefficient of determination",
            MetricDirection.HIGHER,
            MetricProperties(is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "mape",
            mape,
            "Mean absolute percentage error",
            MetricDirection.LOWER,
            MetricProperties(is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "relative_error",
            relative_error,
            "Mean relative error (L2 norm ratio)",
            MetricDirection.LOWER,
            MetricProperties(is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "explained_variance",
            explained_variance,
            "Explained variance score",
            MetricDirection.HIGHER,
            MetricProperties(is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "max_error",
            max_error,
            "Maximum absolute error",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "huber_loss",
            huber_loss,
            "Huber loss (robust regression)",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "quantile_loss",
            quantile_loss,
            "Quantile (pinball) loss",
            MetricDirection.LOWER,
            MetricProperties(is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "log_cosh_loss",
            log_cosh_loss,
            "Log-cosh loss",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "smape",
            smape,
            "Symmetric mean absolute percentage error",
            MetricDirection.LOWER,
            MetricProperties(is_symmetric=True, is_differentiable=True),
        ),
        _BuiltinMetricSpec(
            "crps",
            crps,
            "Continuous ranked probability score for ensemble forecasts",
            MetricDirection.LOWER,
            MetricProperties(is_proper=True, is_differentiable=True, is_jit_compatible=True),
            signature=MetricSignature.ENSEMBLE_PREDICTIONS_TARGETS,
        ),
    ]
    for spec in builtins:
        if not registry.has(spec.name):
            entry = MetricEntry(
                name=spec.name,
                fn=spec.fn,
                tier=MetricTier.PURE_FUNCTION,
                domain=spec.domain,
                direction=spec.direction,
                description=spec.description,
                signature=spec.signature,
                properties=spec.properties,
            )
            registry.register(spec.name, entry)


def _register_classification_metrics() -> None:
    """Register 13 classification metrics at import time.

    Note: confusion_matrix is NOT registered (returns array, not float).
    """
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.classification import (
        accuracy,
        average_precision,
        balanced_accuracy,
        cohen_kappa,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision,
        recall,
        roc_auc,
        sensitivity,
        specificity,
    )

    registry = MetricRegistry()
    # (name, fn, description, direction, is_symmetric, is_proper, is_differentiable)
    builtins: list[tuple[str, Any, str, MetricDirection, bool, bool, bool]] = [
        (
            "accuracy",
            accuracy,
            "Fraction of correct predictions",
            MetricDirection.HIGHER,
            True,
            False,
            False,
        ),
        (
            "precision",
            precision,
            "Precision (TP / (TP + FP))",
            MetricDirection.HIGHER,
            False,
            False,
            False,
        ),
        ("recall", recall, "Recall (TP / (TP + FN))", MetricDirection.HIGHER, False, False, False),
        (
            "f1_score",
            f1_score,
            "Harmonic mean of precision and recall",
            MetricDirection.HIGHER,
            False,
            False,
            False,
        ),
        (
            "roc_auc",
            roc_auc,
            "Area under the ROC curve",
            MetricDirection.HIGHER,
            False,
            False,
            False,
        ),
        (
            "average_precision",
            average_precision,
            "Area under precision-recall curve",
            MetricDirection.HIGHER,
            False,
            False,
            False,
        ),
        (
            "log_loss",
            log_loss,
            "Logarithmic loss (cross-entropy)",
            MetricDirection.LOWER,
            False,
            True,
            True,
        ),
        (
            "matthews_corrcoef",
            matthews_corrcoef,
            "Matthews correlation coefficient",
            MetricDirection.HIGHER,
            True,
            False,
            False,
        ),
        (
            "cohen_kappa",
            cohen_kappa,
            "Cohen's kappa coefficient",
            MetricDirection.HIGHER,
            True,
            False,
            False,
        ),
        (
            "balanced_accuracy",
            balanced_accuracy,
            "Average recall per class",
            MetricDirection.HIGHER,
            True,
            False,
            False,
        ),
        (
            "specificity",
            specificity,
            "True negative rate",
            MetricDirection.HIGHER,
            False,
            False,
            False,
        ),
        (
            "sensitivity",
            sensitivity,
            "True positive rate",
            MetricDirection.HIGHER,
            False,
            False,
            False,
        ),
    ]
    for name, fn, desc, direction, is_sym, is_proper_flag, is_diff in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="classification",
                direction=direction,
                description=desc,
                signature=MetricSignature.PREDICTIONS_TARGETS,
                properties=MetricProperties(
                    is_symmetric=is_sym,
                    is_proper=is_proper_flag,
                    is_differentiable=is_diff,
                ),
            )
            registry.register(name, entry)


def _register_calibration_metrics() -> None:
    """Register 5 calibration metrics at import time.

    Note: reliability_diagram_bins and brier_decomposition are NOT
    registered (they return dicts, not floats).
    """
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.calibration import (
        adaptive_calibration_error,
        brier_score,
        classwise_ece,
        expected_calibration_error,
        maximum_calibration_error,
    )

    registry = MetricRegistry()
    # (name, fn, description, direction, is_proper)
    builtins: list[tuple[str, Any, str, MetricDirection, bool]] = [
        ("brier_score", brier_score, "Brier score (probability MSE)", MetricDirection.LOWER, True),
        (
            "expected_calibration_error",
            expected_calibration_error,
            "Expected calibration error",
            MetricDirection.LOWER,
            False,
        ),
        (
            "maximum_calibration_error",
            maximum_calibration_error,
            "Maximum calibration error",
            MetricDirection.LOWER,
            False,
        ),
        (
            "adaptive_calibration_error",
            adaptive_calibration_error,
            "Adaptive calibration error",
            MetricDirection.LOWER,
            False,
        ),
        (
            "classwise_ece",
            classwise_ece,
            "Classwise expected calibration error",
            MetricDirection.LOWER,
            False,
        ),
    ]
    for name, fn, desc, direction, is_proper_flag in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="calibration",
                direction=direction,
                description=desc,
                signature=MetricSignature.PREDICTIONS_TARGETS,
                properties=MetricProperties(
                    is_proper=is_proper_flag,
                    is_differentiable=False,
                ),
            )
            registry.register(name, entry)


def _register_segmentation_metrics() -> None:
    """Register 3 segmentation metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.segmentation import (
        dice_coefficient,
        iou,
        pixel_accuracy,
    )

    registry = MetricRegistry()
    # (name, fn, description)
    builtins: list[tuple[str, Any, str]] = [
        ("iou", iou, "Intersection over Union (Jaccard index)"),
        ("dice_coefficient", dice_coefficient, "Dice coefficient (F1 for segmentation)"),
        ("pixel_accuracy", pixel_accuracy, "Fraction of correctly classified pixels"),
    ]
    for name, fn, desc in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="segmentation",
                direction=MetricDirection.HIGHER,
                description=desc,
                signature=MetricSignature.PREDICTIONS_TARGETS,
                properties=MetricProperties(is_symmetric=True),
            )
            registry.register(name, entry)


def _register_distance_metrics() -> None:
    """Register 11 distance metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.distance import (
        chebyshev_distance,
        cosine_distance,
        euclidean_distance,
        hamming_distance,
        jaccard_distance,
        lorentz_distance,
        mahalanobis_distance,
        manhattan_distance,
        minkowski_distance,
        poincare_distance,
        randers_distance,
    )

    registry = MetricRegistry()
    # (name, fn, description, is_true_metric, is_symmetric, is_differentiable, invariances)
    builtins: list[tuple[str, Any, str, bool, bool, bool, tuple[str, ...]]] = [
        (
            "cosine_distance",
            cosine_distance,
            "Cosine distance (1 - similarity)",
            False,
            True,
            True,
            ("scale",),
        ),
        (
            "euclidean_distance",
            euclidean_distance,
            "Euclidean (L2) distance",
            True,
            True,
            True,
            ("rotation", "translation"),
        ),
        (
            "manhattan_distance",
            manhattan_distance,
            "Manhattan (L1) distance",
            True,
            True,
            True,
            ("translation",),
        ),
        (
            "chebyshev_distance",
            chebyshev_distance,
            "Chebyshev (L-infinity) distance",
            True,
            True,
            True,
            ("translation",),
        ),
        (
            "mahalanobis_distance",
            mahalanobis_distance,
            "Mahalanobis distance",
            True,
            True,
            True,
            (),
        ),
        (
            "hamming_distance",
            hamming_distance,
            "Hamming distance (fraction differing)",
            True,
            True,
            False,
            (),
        ),
        (
            "minkowski_distance",
            minkowski_distance,
            "Minkowski (Lp) distance",
            True,
            True,
            True,
            ("translation",),
        ),
        (
            "jaccard_distance",
            jaccard_distance,
            "Jaccard distance (set-based)",
            True,
            True,
            False,
            (),
        ),
        (
            "poincare_distance",
            poincare_distance,
            "Poincare ball geodesic distance",
            True,
            True,
            True,
            (),
        ),
        (
            "lorentz_distance",
            lorentz_distance,
            "Lorentz hyperboloid geodesic distance",
            True,
            True,
            True,
            ("lorentz",),
        ),
        (
            "randers_distance",
            randers_distance,
            "Randers asymmetric Finsler distance",
            False,
            False,
            True,
            (),
        ),
    ]
    for name, fn, desc, is_true, is_sym, is_diff, invs in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="distance",
                direction=MetricDirection.LOWER,
                description=desc,
                signature=MetricSignature.PREDICTIONS_TARGETS,
                properties=MetricProperties(
                    is_true_metric=is_true,
                    is_symmetric=is_sym,
                    is_differentiable=is_diff,
                    invariances=invs,
                ),
            )
            registry.register(name, entry)


def _register_divergence_metrics() -> None:
    """Register 13 divergence metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.divergence import (
        bregman_divergence,
        chi_squared_divergence,
        f_divergence,
        hellinger_distance,
        js_divergence,
        kl_divergence,
        mmd,
        renyi_divergence,
        reverse_kl_divergence,
        sinkhorn_divergence,
        sliced_wasserstein,
        total_variation,
        wasserstein_1d,
    )

    registry = MetricRegistry()
    # (name, fn, description, is_true_metric, is_symmetric, is_differentiable, signature)
    builtins: list[tuple[str, Any, str, bool, bool, bool, MetricSignature]] = [
        (
            "kl_divergence",
            kl_divergence,
            "Kullback-Leibler divergence",
            False,
            False,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "reverse_kl_divergence",
            reverse_kl_divergence,
            "Reverse KL divergence (mode-seeking)",
            False,
            False,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "js_divergence",
            js_divergence,
            "Jensen-Shannon divergence",
            True,
            True,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "total_variation",
            total_variation,
            "Total variation distance",
            True,
            True,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "hellinger_distance",
            hellinger_distance,
            "Hellinger distance",
            True,
            True,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "chi_squared_divergence",
            chi_squared_divergence,
            "Pearson chi-squared divergence",
            False,
            False,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "renyi_divergence",
            renyi_divergence,
            "Renyi alpha-divergence",
            False,
            False,
            True,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "f_divergence",
            f_divergence,
            "Unified f-divergence",
            False,
            False,
            True,
            MetricSignature.CUSTOM,
        ),
        (
            "wasserstein_1d",
            wasserstein_1d,
            "1D Wasserstein-1 distance",
            True,
            True,
            True,
            MetricSignature.SAMPLES,
        ),
        (
            "mmd",
            mmd,
            "Maximum Mean Discrepancy",
            True,
            True,
            True,
            MetricSignature.SAMPLES,
        ),
        (
            "sinkhorn_divergence",
            sinkhorn_divergence,
            "Debiased Sinkhorn divergence",
            False,
            True,
            True,
            MetricSignature.SAMPLES,
        ),
        (
            "sliced_wasserstein",
            sliced_wasserstein,
            "Sliced Wasserstein distance",
            True,
            True,
            True,
            MetricSignature.SAMPLES,
        ),
        (
            "bregman_divergence",
            bregman_divergence,
            "Bregman divergence with convex generator",
            False,
            False,
            True,
            MetricSignature.CUSTOM,
        ),
    ]
    for name, fn, desc, is_true, is_sym, is_diff, sig in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="divergence",
                direction=MetricDirection.LOWER,
                description=desc,
                signature=sig,
                properties=MetricProperties(
                    is_true_metric=is_true,
                    is_symmetric=is_sym,
                    is_differentiable=is_diff,
                ),
            )
            registry.register(name, entry)


def _register_information_metrics() -> None:
    """Register 5 information-theoretic metrics at import time.

    Note: fisher_information_matrix is NOT registered (returns matrix, not float).
    """
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.information import (
        conditional_entropy,
        cross_entropy,
        entropy,
        mutual_information,
        normalized_mutual_information,
    )

    registry = MetricRegistry()
    # (name, fn, description, direction, signature)
    builtins: list[tuple[str, Any, str, MetricDirection, MetricSignature]] = [
        (
            "entropy",
            entropy,
            "Shannon entropy",
            MetricDirection.INFO,
            MetricSignature.SINGLE_INPUT,
        ),
        (
            "cross_entropy",
            cross_entropy,
            "Cross-entropy",
            MetricDirection.LOWER,
            MetricSignature.PREDICTIONS_TARGETS,
        ),
        (
            "mutual_information",
            mutual_information,
            "Mutual information",
            MetricDirection.HIGHER,
            MetricSignature.SINGLE_INPUT,
        ),
        (
            "conditional_entropy",
            conditional_entropy,
            "Conditional entropy H(Y|X)",
            MetricDirection.LOWER,
            MetricSignature.SINGLE_INPUT,
        ),
        (
            "normalized_mutual_information",
            normalized_mutual_information,
            "Normalized mutual information",
            MetricDirection.HIGHER,
            MetricSignature.SINGLE_INPUT,
        ),
    ]
    for name, fn, desc, direction, sig in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="information",
                direction=direction,
                description=desc,
                signature=sig,
            )
            registry.register(name, entry)


def _register_ranking_metrics() -> None:
    """Register 8 ranking/retrieval metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.ranking import (
        coverage,
        hit_rate,
        mean_average_precision,
        mean_reciprocal_rank,
        ndcg,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )

    registry = MetricRegistry()
    builtins: list[tuple[str, Any, str]] = [
        ("ndcg", ndcg, "Normalized Discounted Cumulative Gain"),
        ("ndcg_at_k", ndcg_at_k, "NDCG truncated to top-k"),
        ("mean_average_precision", mean_average_precision, "Mean Average Precision"),
        ("precision_at_k", precision_at_k, "Precision at k"),
        ("recall_at_k", recall_at_k, "Recall at k"),
        ("mean_reciprocal_rank", mean_reciprocal_rank, "Mean Reciprocal Rank"),
        ("hit_rate", hit_rate, "Hit rate at k"),
        ("coverage", coverage, "Catalog coverage"),
    ]
    for name, fn, desc in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="ranking",
                direction=MetricDirection.HIGHER,
                description=desc,
                signature=MetricSignature.PREDICTIONS_TARGETS,
            )
            registry.register(name, entry)


def _register_statistical_metrics() -> None:
    """Register 5 statistical correlation metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.statistical import (
        concordance_correlation,
        kendall_tau,
        pearson_correlation,
        r_squared_adjusted,
        spearman_rank_correlation,
    )

    registry = MetricRegistry()
    builtins: list[tuple[str, Any, str]] = [
        ("pearson_correlation", pearson_correlation, "Pearson correlation coefficient"),
        ("spearman_rank_correlation", spearman_rank_correlation, "Spearman rank correlation"),
        ("kendall_tau", kendall_tau, "Kendall rank correlation coefficient"),
        ("concordance_correlation", concordance_correlation, "Lin's concordance correlation"),
        ("r_squared_adjusted", r_squared_adjusted, "Adjusted R-squared"),
    ]
    for name, fn, desc in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="statistical",
                direction=MetricDirection.HIGHER,
                description=desc,
                signature=MetricSignature.PREDICTIONS_TARGETS,
                properties=MetricProperties(is_symmetric=True),
            )
            registry.register(name, entry)


def _register_text_metrics() -> None:
    """Register 5 text metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.text import (
        bleu,
        distinct_n,
        perplexity,
        rouge_l,
        rouge_n,
    )

    registry = MetricRegistry()
    # (name, fn, description, direction, is_jit_compatible)
    builtins: list[tuple[str, Any, str, MetricDirection, bool]] = [
        ("bleu", bleu, "BLEU score for translation evaluation", MetricDirection.HIGHER, False),
        ("rouge_n", rouge_n, "ROUGE-N recall for summarization", MetricDirection.HIGHER, False),
        ("rouge_l", rouge_l, "ROUGE-L LCS-based F-measure", MetricDirection.HIGHER, False),
        (
            "perplexity",
            perplexity,
            "Perplexity from log-probabilities",
            MetricDirection.LOWER,
            True,
        ),
        ("distinct_n", distinct_n, "Distinct-N lexical diversity", MetricDirection.HIGHER, False),
    ]
    for name, fn, desc, direction, jit_compat in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="text",
                direction=direction,
                description=desc,
                signature=MetricSignature.CUSTOM,
                properties=MetricProperties(
                    is_jit_compatible=jit_compat,
                    is_differentiable=jit_compat,
                ),
            )
            registry.register(name, entry)


def _register_audio_metrics() -> None:
    """Register 3 audio metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.audio import (
        mel_cepstral_distortion,
        signal_to_noise_ratio,
        spectral_convergence,
    )

    registry = MetricRegistry()
    builtins: list[tuple[str, Any, str, MetricDirection]] = [
        (
            "spectral_convergence",
            spectral_convergence,
            "Spectral convergence ratio",
            MetricDirection.LOWER,
        ),
        (
            "mel_cepstral_distortion",
            mel_cepstral_distortion,
            "Mel cepstral distortion (dB)",
            MetricDirection.LOWER,
        ),
        (
            "signal_to_noise_ratio",
            signal_to_noise_ratio,
            "Signal-to-noise ratio (dB)",
            MetricDirection.HIGHER,
        ),
    ]
    for name, fn, desc, direction in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="audio",
                direction=direction,
                description=desc,
                signature=MetricSignature.PREDICTIONS_TARGETS,
                properties=MetricProperties(is_symmetric=True),
            )
            registry.register(name, entry)


def _register_geometric_metrics() -> None:
    """Register 4 geometric metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.geometric import (
        chamfer_distance,
        directed_hausdorff,
        earth_movers_distance_1d,
        hausdorff_distance,
    )

    registry = MetricRegistry()
    # (name, fn, description, is_true_metric, is_symmetric)
    builtins: list[tuple[str, Any, str, bool, bool]] = [
        ("chamfer_distance", chamfer_distance, "Chamfer distance between point sets", False, True),
        (
            "earth_movers_distance_1d",
            earth_movers_distance_1d,
            "1D Earth Mover's Distance (Wasserstein-1)",
            True,
            True,
        ),
        (
            "directed_hausdorff",
            directed_hausdorff,
            "Directed Hausdorff distance (asymmetric)",
            False,
            False,
        ),
        (
            "hausdorff_distance",
            hausdorff_distance,
            "Hausdorff distance (symmetric)",
            True,
            True,
        ),
    ]
    for name, fn, desc, is_true, is_sym in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="geometric",
                direction=MetricDirection.LOWER,
                description=desc,
                signature=MetricSignature.SAMPLES,
                properties=MetricProperties(
                    is_true_metric=is_true,
                    is_symmetric=is_sym,
                ),
            )
            registry.register(name, entry)


def _register_manifold_metrics() -> None:
    """Register 5 manifold distance metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.manifold import (
        grassmann_distance,
        spd_affine_invariant_distance,
        spd_log_euclidean_distance,
        stiefel_distance,
        ultrahyperbolic_distance,
    )

    registry = MetricRegistry()

    # (name, fn, description, is_true_metric, invariances)
    builtins: list[tuple[str, Any, str, bool, tuple[str, ...]]] = [
        (
            "spd_affine_invariant_distance",
            spd_affine_invariant_distance,
            "Affine-invariant Riemannian distance on SPD manifold",
            True,
            ("affine", "congruence"),
        ),
        (
            "spd_log_euclidean_distance",
            spd_log_euclidean_distance,
            "Log-Euclidean distance between SPD matrices",
            True,
            ("orthogonal",),
        ),
        (
            "grassmann_distance",
            grassmann_distance,
            "Geodesic distance on the Grassmann manifold",
            True,
            ("orthogonal",),
        ),
        (
            "stiefel_distance",
            stiefel_distance,
            "Frobenius distance on the Stiefel manifold",
            True,
            (),
        ),
        (
            "ultrahyperbolic_distance",
            ultrahyperbolic_distance,
            "Geodesic distance on pseudo-hyperboloid with mixed signature",
            False,
            (),
        ),
    ]
    for name, fn, desc, is_true, invs in builtins:
        if not registry.has(name):
            registry.register(
                name,
                MetricEntry(
                    name=name,
                    fn=fn,
                    tier=MetricTier.PURE_FUNCTION,
                    domain="manifold",
                    direction=MetricDirection.LOWER,
                    description=desc,
                    signature=MetricSignature.SAMPLES,
                    properties=MetricProperties(
                        is_true_metric=is_true,
                        is_symmetric=True,
                        invariances=invs,
                    ),
                ),
            )


def _register_graph_metrics() -> None:
    """Register 4 graph distance metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.graph import (
        graph_edit_distance_approx,
        resistance_distance,
        shortest_path_distance,
        spectral_distance,
    )

    registry = MetricRegistry()

    # Between-graph metrics (two adjacency matrices)
    between_graph: list[tuple[str, Any, str, bool, bool, bool]] = [
        (
            "spectral_distance",
            spectral_distance,
            "Laplacian eigenvalue spectrum distance between graphs",
            True,
            True,
            True,
        ),
        (
            "graph_edit_distance_approx",
            graph_edit_distance_approx,
            "Approximate graph edit distance via spectral relaxation",
            False,
            True,
            True,
        ),
    ]
    for name, fn, desc, is_true, is_sym, is_diff in between_graph:
        if not registry.has(name):
            registry.register(
                name,
                MetricEntry(
                    name=name,
                    fn=fn,
                    tier=MetricTier.PURE_FUNCTION,
                    domain="graph",
                    direction=MetricDirection.LOWER,
                    description=desc,
                    signature=MetricSignature.SAMPLES,
                    properties=MetricProperties(
                        is_true_metric=is_true,
                        is_symmetric=is_sym,
                        is_differentiable=is_diff,
                        invariances=("permutation",),
                    ),
                ),
            )

    # Within-graph metrics (single adjacency matrix)
    within_graph: list[tuple[str, Any, str, bool]] = [
        (
            "resistance_distance",
            resistance_distance,
            "Effective electrical resistance distance matrix",
            True,
        ),
        (
            "shortest_path_distance",
            shortest_path_distance,
            "Floyd-Warshall all-pairs shortest paths",
            False,
        ),
    ]
    for name, fn, desc, is_diff in within_graph:
        if not registry.has(name):
            registry.register(
                name,
                MetricEntry(
                    name=name,
                    fn=fn,
                    tier=MetricTier.PURE_FUNCTION,
                    domain="graph",
                    direction=MetricDirection.LOWER,
                    description=desc,
                    signature=MetricSignature.SINGLE_INPUT,
                    properties=MetricProperties(
                        is_true_metric=True,
                        is_symmetric=True,
                        is_differentiable=is_diff,
                    ),
                ),
            )


def _register_image_metrics() -> None:
    """Register 4 image quality metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.image import (
        ms_ssim,
        psnr,
        ssim,
        vendi_score,
    )

    registry = MetricRegistry()
    builtins: list[tuple[str, Any, str]] = [
        ("psnr", psnr, "Peak Signal-to-Noise Ratio (dB)"),
        ("ssim", ssim, "Structural Similarity Index Measure"),
        ("ms_ssim", ms_ssim, "Multi-Scale Structural Similarity"),
        ("vendi_score", vendi_score, "Vendi Score (diversity via eigenvalue entropy)"),
    ]
    for name, fn, desc in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="image",
                direction=MetricDirection.HIGHER,
                description=desc,
                signature=MetricSignature.PREDICTIONS_TARGETS,
                properties=MetricProperties(is_symmetric=True),
            )
            registry.register(name, entry)


def _register_fairness_metrics() -> None:
    """Register 4 fairness metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.fairness import (
        demographic_parity_ratio,
        disparate_impact_ratio,
        equal_opportunity_difference,
        equalized_odds_difference,
    )

    registry = MetricRegistry()
    # (name, fn, description, direction)
    builtins: list[tuple[str, Any, str, MetricDirection]] = [
        (
            "demographic_parity_ratio",
            demographic_parity_ratio,
            "Ratio of positive prediction rates across demographic groups",
            MetricDirection.HIGHER,
        ),
        (
            "equalized_odds_difference",
            equalized_odds_difference,
            "Max absolute difference in TPR or FPR across groups",
            MetricDirection.LOWER,
        ),
        (
            "equal_opportunity_difference",
            equal_opportunity_difference,
            "Absolute difference in TPR across demographic groups",
            MetricDirection.LOWER,
        ),
        (
            "disparate_impact_ratio",
            disparate_impact_ratio,
            "Disparate impact ratio (80% rule)",
            MetricDirection.HIGHER,
        ),
    ]
    for name, fn, desc, direction in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="fairness",
                direction=direction,
                description=desc,
                signature=MetricSignature.CUSTOM,
            )
            registry.register(name, entry)


def _register_clustering_metrics() -> None:
    """Register 7 clustering metrics at import time."""
    from calibrax.metrics._registry import MetricRegistry
    from calibrax.metrics.functional.clustering import (
        adjusted_mutual_information,
        adjusted_rand_index,
        calinski_harabasz_score,
        davies_bouldin_score,
        normalized_mutual_information_clustering,
        silhouette_score,
        v_measure,
    )

    registry = MetricRegistry()
    # (name, fn, description, direction)
    builtins: list[tuple[str, Any, str, MetricDirection]] = [
        (
            "adjusted_rand_index",
            adjusted_rand_index,
            "Chance-adjusted Rand index for clustering agreement",
            MetricDirection.HIGHER,
        ),
        (
            "normalized_mutual_information_clustering",
            normalized_mutual_information_clustering,
            "Normalized mutual information for clustering",
            MetricDirection.HIGHER,
        ),
        (
            "adjusted_mutual_information",
            adjusted_mutual_information,
            "Chance-adjusted mutual information for clustering",
            MetricDirection.HIGHER,
        ),
        (
            "v_measure",
            v_measure,
            "Harmonic mean of homogeneity and completeness",
            MetricDirection.HIGHER,
        ),
        (
            "silhouette_score",
            silhouette_score,
            "Mean silhouette coefficient for cluster separation",
            MetricDirection.HIGHER,
        ),
        (
            "calinski_harabasz_score",
            calinski_harabasz_score,
            "Calinski-Harabasz variance ratio criterion",
            MetricDirection.HIGHER,
        ),
        (
            "davies_bouldin_score",
            davies_bouldin_score,
            "Davies-Bouldin index for cluster separation",
            MetricDirection.LOWER,
        ),
    ]
    for name, fn, desc, direction in builtins:
        if not registry.has(name):
            entry = MetricEntry(
                name=name,
                fn=fn,
                tier=MetricTier.PURE_FUNCTION,
                domain="clustering",
                direction=direction,
                description=desc,
                signature=MetricSignature.FEATURES_LABELS,
                properties=MetricProperties(is_symmetric=True),
            )
            registry.register(name, entry)


def calculate_all(
    predictions: Any,
    targets: Any,
    *,
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Calculate multiple evaluation metrics at once.

    Delegates to MetricRegistry for function lookup.

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
    from calibrax.metrics._registry import MetricRegistry

    registry = MetricRegistry()
    if metrics is None:
        entries = registry.list_by_tier(MetricTier.PURE_FUNCTION)
        names_set = {
            e.name
            for e in entries
            if e.domain == "general" and e.signature == MetricSignature.PREDICTIONS_TARGETS
        }
        # Use fused single-pass path when the default set matches exactly
        if names_set == _FUSED_REGRESSION_NAMES:
            return _calculate_regression_fused(predictions, targets)
        # Fall through to dynamic dispatch if registry has changed
        names = sorted(names_set)
    else:
        names = list(metrics)
        # Use fused path when explicitly requesting exactly the regression set
        if set(names) == _FUSED_REGRESSION_NAMES:
            return _calculate_regression_fused(predictions, targets)

    results: dict[str, Any] = {}
    for name in names:
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
