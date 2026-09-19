"""Label-based metrics read labels as names: renaming classes, clusters or groups changes nothing.

References are scikit-learn's implementations. A static count (``num_classes``,
``num_clusters``, ``num_groups``) makes each metric traceable; without one the distinct
labels are counted from the data, which only runs eagerly.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn import metrics as sk

from calibrax.metrics._utils import dense_labels
from calibrax.metrics.functional.classification import balanced_accuracy, cohen_kappa
from calibrax.metrics.functional.clustering import (
    adjusted_mutual_information,
    adjusted_rand_index,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_information_clustering,
    silhouette_score,
    v_measure,
)
from calibrax.metrics.functional.fairness import (
    demographic_parity_ratio,
    disparate_impact_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    group_metric_breakdown,
)
from calibrax.metrics.functional.regression import mse


_RNG = np.random.default_rng(0)
# Two random labelings of 60 points with 4 and 3 labels, and the same labelings renamed.
_TRUE = _RNG.integers(0, 4, 60)
_PRED = _RNG.integers(0, 3, 60)
_RENAMES = {
    "contiguous": (lambda y: y),
    "gaps": (lambda y: y * 3 + 1),
    "shifted": (lambda y: y + 5),
}
_FEATURES = (_RNG.normal(size=(60, 2)) + _TRUE[:, None] * 3.0).astype(np.float32)


class TestDenseLabels:
    def test_distinct_values_are_numbered_in_sorted_order(self) -> None:
        ids, count = dense_labels(jnp.array([7, 3, 7, 11, 3]), None)

        assert ids.tolist() == [1, 0, 1, 2, 0]
        assert count == 3

    def test_a_given_count_takes_labels_as_ids(self) -> None:
        ids, count = dense_labels(jnp.array([0, 2, 2]), 4)

        assert ids.tolist() == [0, 2, 2]
        assert count == 4


_CONTINGENCY: list[tuple[str, Callable[..., jax.Array], Callable[..., float]]] = [
    ("ari", adjusted_rand_index, sk.adjusted_rand_score),
    ("nmi", normalized_mutual_information_clustering, sk.normalized_mutual_info_score),
    ("ami", adjusted_mutual_information, sk.adjusted_mutual_info_score),
    ("v_measure", v_measure, sk.v_measure_score),
]


class TestContingencyMetrics:
    @pytest.mark.parametrize(("name", "ours", "reference"), _CONTINGENCY)
    @pytest.mark.parametrize("rename", sorted(_RENAMES))
    def test_matches_sklearn_under_any_naming(
        self,
        name: str,
        ours: Callable[..., jax.Array],
        reference: Callable[..., float],
        rename: str,
    ) -> None:
        relabel = _RENAMES[rename]
        value = ours(relabel(_TRUE), relabel(_PRED))

        assert float(value) == pytest.approx(reference(_TRUE, _PRED), abs=1e-5), name

    @pytest.mark.parametrize(("name", "ours", "_reference"), _CONTINGENCY)
    def test_traces_with_static_counts(
        self, name: str, ours: Callable[..., jax.Array], _reference: Callable[..., float]
    ) -> None:
        eager = ours(_TRUE, _PRED)
        traced = jax.jit(ours, static_argnames=("num_classes", "num_clusters"))(
            _TRUE, _PRED, num_classes=4, num_clusters=3
        )

        assert float(traced) == pytest.approx(float(eager), abs=1e-5), name

    @pytest.mark.parametrize("average", ["arithmetic", "geometric", "min", "max"])
    def test_ami_averages_as_sklearn_does(self, average: str) -> None:
        value = adjusted_mutual_information(_TRUE, _PRED, average=average)

        reference = sk.adjusted_mutual_info_score(_TRUE, _PRED, average_method=average)
        assert float(value) == pytest.approx(reference, abs=1e-5)

    @pytest.mark.parametrize("beta", [0.5, 2.0])
    def test_v_measure_weights_as_rosenberg_and_hirschberg(self, beta: float) -> None:
        value = v_measure(_TRUE, _PRED, beta=beta)

        assert float(value) == pytest.approx(sk.v_measure_score(_TRUE, _PRED, beta=beta), abs=1e-5)

    @pytest.mark.parametrize(("name", "ours", "reference"), _CONTINGENCY)
    def test_identical_labelings_score_one(
        self, name: str, ours: Callable[..., jax.Array], reference: Callable[..., float]
    ) -> None:
        assert float(ours(_TRUE, _TRUE)) == pytest.approx(reference(_TRUE, _TRUE), abs=1e-5), name


_INTERNAL: list[tuple[str, Callable[..., jax.Array], Callable[..., float]]] = [
    ("silhouette", silhouette_score, sk.silhouette_score),
    ("calinski_harabasz", calinski_harabasz_score, sk.calinski_harabasz_score),
    ("davies_bouldin", davies_bouldin_score, sk.davies_bouldin_score),
]


class TestInternalClusterScores:
    @pytest.mark.parametrize(("name", "ours", "reference"), _INTERNAL)
    @pytest.mark.parametrize("rename", sorted(_RENAMES))
    def test_matches_sklearn_under_any_naming(
        self,
        name: str,
        ours: Callable[..., jax.Array],
        reference: Callable[..., float],
        rename: str,
    ) -> None:
        value = ours(_FEATURES, _RENAMES[rename](_TRUE))

        assert float(value) == pytest.approx(reference(_FEATURES, _TRUE), rel=1e-4), name

    @pytest.mark.parametrize(("name", "ours", "reference"), _INTERNAL)
    def test_traces_with_a_static_count_and_empty_clusters(
        self, name: str, ours: Callable[..., jax.Array], reference: Callable[..., float]
    ) -> None:
        # Six ids of which four are used: the two empty clusters take no part.
        traced = jax.jit(ours, static_argnames=("num_clusters",))(_FEATURES, _TRUE, num_clusters=6)

        assert float(traced) == pytest.approx(reference(_FEATURES, _TRUE), rel=1e-4), name

    def test_a_singleton_cluster_has_silhouette_zero(self) -> None:
        features = np.array(
            [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [9.0, 9.0], [9.1, 9.0]], np.float32
        )
        labels = np.array([0, 0, 1, 2, 2])

        value = silhouette_score(features, labels)

        assert float(value) == pytest.approx(sk.silhouette_score(features, labels), rel=1e-4)


_PREDICTED = (_RNG.random(60) > 0.4).astype(np.float32)
_OUTCOME = (_RNG.random(60) > 0.5).astype(np.float32)
_GROUPS = _RNG.integers(0, 3, 60)


def _fairness_calls() -> list[tuple[str, Callable[..., jax.Array], tuple[np.ndarray, ...]]]:
    return [
        ("demographic_parity_ratio", demographic_parity_ratio, (_PREDICTED,)),
        ("disparate_impact_ratio", disparate_impact_ratio, (_PREDICTED,)),
        ("equalized_odds_difference", equalized_odds_difference, (_PREDICTED, _OUTCOME)),
        ("equal_opportunity_difference", equal_opportunity_difference, (_PREDICTED, _OUTCOME)),
    ]


class TestFairnessGroups:
    @pytest.mark.parametrize(("name", "fn", "leading"), _fairness_calls())
    @pytest.mark.parametrize("rename", sorted(_RENAMES))
    def test_group_names_change_nothing(
        self, name: str, fn: Callable[..., jax.Array], leading: tuple[np.ndarray, ...], rename: str
    ) -> None:
        renamed = fn(*leading, _RENAMES[rename](_GROUPS))

        assert float(renamed) == pytest.approx(float(fn(*leading, _GROUPS)), abs=1e-6), name

    @pytest.mark.parametrize(("name", "fn", "leading"), _fairness_calls())
    def test_traces_with_a_static_count_and_empty_groups(
        self, name: str, fn: Callable[..., jax.Array], leading: tuple[np.ndarray, ...]
    ) -> None:
        traced = jax.jit(fn, static_argnames=("num_groups",))(*leading, _GROUPS, num_groups=5)

        assert float(traced) == pytest.approx(float(fn(*leading, _GROUPS)), abs=1e-6), name

    def test_a_breakdown_is_keyed_by_the_group_names(self) -> None:
        breakdown = group_metric_breakdown(mse, _OUTCOME, _PREDICTED, _GROUPS * 3 + 1)

        assert sorted(breakdown) == ["1", "4", "7"]


_CLASS_TARGETS = _RNG.integers(0, 4, 60)
_CLASS_PREDICTIONS = np.where(_RNG.random(60) < 0.7, _CLASS_TARGETS, _RNG.integers(0, 5, 60))


class TestClassIndexMetrics:
    """Class ids are indices ``0..C-1``; a static ``num_classes`` traces."""

    @pytest.mark.parametrize(
        ("name", "ours", "reference"),
        [
            ("cohen_kappa", cohen_kappa, sk.cohen_kappa_score),
            ("balanced_accuracy", balanced_accuracy, sk.balanced_accuracy_score),
        ],
    )
    def test_matches_sklearn(
        self, name: str, ours: Callable[..., jax.Array], reference: Callable[..., float]
    ) -> None:
        # Predictions include class 4, which no target has.
        value = ours(_CLASS_PREDICTIONS, _CLASS_TARGETS)

        assert float(value) == pytest.approx(
            reference(_CLASS_TARGETS, _CLASS_PREDICTIONS), abs=1e-5
        ), name

    @pytest.mark.parametrize("ours", [cohen_kappa, balanced_accuracy])
    def test_traces_with_a_static_class_count(self, ours: Callable[..., jax.Array]) -> None:
        eager = ours(_CLASS_PREDICTIONS, _CLASS_TARGETS)
        traced = jax.jit(ours, static_argnames=("num_classes",))(
            _CLASS_PREDICTIONS, _CLASS_TARGETS, num_classes=6
        )

        assert float(traced) == pytest.approx(float(eager), abs=1e-6)
