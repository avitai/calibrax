"""Registrations of the built-in ``classification`` metrics."""

from __future__ import annotations

from calibrax.core.models import MetricDirection
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import (
    MetricEntry,
    MetricFn,
    MetricProperties,
    MetricSignature,
    MetricTier,
)
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
    softmax_cross_entropy,
    specificity,
)


def register() -> None:
    """Register 14 classification metrics at import time.

    Note: confusion_matrix is NOT registered (returns array, not float).
    """
    registry = MetricRegistry()
    # (name, fn, description, direction, is_symmetric, is_proper, is_differentiable)
    builtins: list[tuple[str, MetricFn, str, MetricDirection, bool, bool, bool]] = [
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

    # Logits and labels, not (predictions, targets): registered with its own signature so a
    # registry-driven collection computed on class indices leaves it out.
    if not registry.has("softmax_cross_entropy"):
        registry.register(
            "softmax_cross_entropy",
            MetricEntry(
                name="softmax_cross_entropy",
                fn=softmax_cross_entropy,
                tier=MetricTier.PURE_FUNCTION,
                domain="classification",
                direction=MetricDirection.LOWER,
                description="Cross-entropy of integer labels under softmax logits",
                signature=MetricSignature.CUSTOM,
                properties=MetricProperties(
                    is_proper=True, is_differentiable=True, is_jit_compatible=True
                ),
            ),
        )
