"""Registrations of the built-in ``clustering`` metrics."""

from __future__ import annotations

from typing import Protocol

import jax
from jax.typing import ArrayLike

from calibrax.core.models import MetricDirection
from calibrax.metrics._registrations._spec import _BuiltinMetricSpec, _register_specs
from calibrax.metrics._registry import MetricRegistry
from calibrax.metrics._types import MetricFn, MetricProperties, MetricSignature
from calibrax.metrics.functional.clustering import (
    adjusted_mutual_information,
    adjusted_rand_index,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_information_clustering,
    silhouette_score,
    v_measure,
)


class _ContingencyMetric(Protocol):
    """A metric of two labelings, ground truth first, with static label counts."""

    def __call__(
        self,
        labels_true: ArrayLike,
        labels_pred: ArrayLike,
        *,
        num_classes: int | None = None,
        num_clusters: int | None = None,
    ) -> jax.Array: ...


def _predictions_first(metric: _ContingencyMetric) -> MetricFn:
    """``metric`` as the registry calls a metric, ``fn(predictions, targets)``.

    The clustering metrics take the ground truth first, as scikit-learn does; a suite passes
    the predicted labels first. The counts stay keywords, static under ``jax.jit``.
    """

    def by_predictions(
        predictions: ArrayLike,
        targets: ArrayLike,
        *,
        num_classes: int | None = None,
        num_clusters: int | None = None,
    ) -> jax.Array:
        return metric(targets, predictions, num_classes=num_classes, num_clusters=num_clusters)

    by_predictions.__name__ = by_predictions.__qualname__ = getattr(metric, "__name__", "metric")
    by_predictions.__doc__ = metric.__doc__
    return by_predictions


def register() -> None:
    """Register 7 clustering metrics at import time.

    The four agreement scores compare predicted labels with the ground truth, so the registry
    calls them ``fn(predictions, targets)``; labels carry no gradient. Silhouette,
    Calinski-Harabasz and Davies-Bouldin score features by their cluster labels and are
    differentiable in the features.
    """
    registry = MetricRegistry()
    agreement = MetricProperties(is_symmetric=True, is_differentiable=False)
    agreements: list[tuple[str, _ContingencyMetric, str]] = [
        (
            "adjusted_rand_index",
            adjusted_rand_index,
            "Chance-adjusted Rand index for clustering agreement",
        ),
        (
            "normalized_mutual_information_clustering",
            normalized_mutual_information_clustering,
            "Normalized mutual information for clustering",
        ),
        (
            "adjusted_mutual_information",
            adjusted_mutual_information,
            "Chance-adjusted mutual information for clustering",
        ),
        ("v_measure", v_measure, "Harmonic mean of homogeneity and completeness"),
    ]
    internal: list[tuple[str, MetricFn, str, MetricDirection]] = [
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
    _register_specs(
        registry,
        [
            *(
                _BuiltinMetricSpec(
                    name,
                    _predictions_first(metric),
                    desc,
                    MetricDirection.HIGHER,
                    agreement,
                    domain="clustering",
                )
                for name, metric, desc in agreements
            ),
            *(
                _BuiltinMetricSpec(
                    name,
                    fn,
                    desc,
                    direction,
                    MetricProperties(),
                    signature=MetricSignature.FEATURES_LABELS,
                    domain="clustering",
                )
                for name, fn, desc, direction in internal
            ),
        ],
    )
