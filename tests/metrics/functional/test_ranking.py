"""Tests for ranking and retrieval metrics."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.metrics import ndcg_score
from substrax.testing import TraceCounter

from calibrax.core.models import MetricDirection
from calibrax.metrics import MetricRegistry
from calibrax.metrics._types import MetricSignature
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


class TestNDCG:
    """Tests for ndcg."""

    def test_perfect_ranking(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([3, 2, 1])
        assert ndcg(scores, relevance) == pytest.approx(1.0, abs=1e-5)

    def test_worst_ranking(self) -> None:
        scores = jnp.array([1.0, 2.0, 3.0])
        relevance = jnp.array([3, 2, 1])
        result = ndcg(scores, relevance)
        assert result < 1.0

    def test_known_value(self) -> None:
        # 3 items: relevance [3, 2, 1], perfect ranking → NDCG=1.0
        scores = jnp.array([10.0, 5.0, 1.0])
        relevance = jnp.array([3, 2, 1])
        assert ndcg(scores, relevance) == pytest.approx(1.0, abs=1e-5)

    def test_all_irrelevant(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([0, 0, 0])
        assert ndcg(scores, relevance) == pytest.approx(0.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        scores = jnp.array([1.0, 2.0])
        relevance = jnp.array([1, 0])
        result = ndcg(scores, relevance)
        assert isinstance(result, jax.Array)


class TestNDCGAtK:
    """Tests for ndcg_at_k."""

    def test_perfect_at_k(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([3, 2, 1])
        assert ndcg_at_k(scores, relevance, k=2) == pytest.approx(1.0, abs=1e-5)

    def test_k_larger_than_list(self) -> None:
        scores = jnp.array([2.0, 1.0])
        relevance = jnp.array([1, 0])
        result = ndcg_at_k(scores, relevance, k=10)
        assert isinstance(result, jax.Array)

    def test_returns_jax_scalar(self) -> None:
        scores = jnp.array([1.0, 2.0])
        relevance = jnp.array([1, 0])
        result = ndcg_at_k(scores, relevance, k=1)
        assert isinstance(result, jax.Array)


class TestMeanAveragePrecision:
    """Tests for mean_average_precision."""

    def test_perfect_ranking(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([1, 1, 0])
        assert mean_average_precision(scores, relevance) == pytest.approx(1.0, abs=1e-5)

    def test_known_value(self) -> None:
        # Ranked: [1, 0, 1] → AP = (1/1 + 2/3) / 2 = 5/6
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([1, 0, 1])
        expected = (1.0 / 1.0 + 2.0 / 3.0) / 2.0
        assert mean_average_precision(scores, relevance) == pytest.approx(expected, abs=1e-5)

    def test_no_relevant(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([0, 0, 0])
        assert mean_average_precision(scores, relevance) == pytest.approx(0.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        scores = jnp.array([1.0, 2.0])
        relevance = jnp.array([1, 0])
        result = mean_average_precision(scores, relevance)
        assert isinstance(result, jax.Array)


class TestPrecisionAtK:
    """Tests for precision_at_k."""

    def test_all_relevant(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([1, 1, 1])
        assert precision_at_k(scores, relevance, k=3) == pytest.approx(1.0, abs=1e-5)

    def test_none_relevant(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([0, 0, 0])
        assert precision_at_k(scores, relevance, k=2) == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self) -> None:
        # Top 3 by score: all 3 items, 2 relevant → 2/3
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([1, 1, 0])
        assert precision_at_k(scores, relevance, k=3) == pytest.approx(2.0 / 3.0, abs=1e-5)


class TestRecallAtK:
    """Tests for recall_at_k."""

    def test_all_found(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([1, 1, 0])
        assert recall_at_k(scores, relevance, k=3) == pytest.approx(1.0, abs=1e-5)

    def test_partial(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([1, 0, 1])
        # Top 1: score=3 → rel=1. Total relevant=2. Recall@1 = 1/2
        assert recall_at_k(scores, relevance, k=1) == pytest.approx(0.5, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        scores = jnp.array([1.0, 2.0])
        relevance = jnp.array([1, 0])
        result = recall_at_k(scores, relevance, k=1)
        assert isinstance(result, jax.Array)


class TestMeanReciprocalRank:
    """Tests for mean_reciprocal_rank."""

    def test_first_is_relevant(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([1, 0, 0])
        assert mean_reciprocal_rank(scores, relevance) == pytest.approx(1.0, abs=1e-5)

    def test_second_is_relevant(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([0, 1, 0])
        assert mean_reciprocal_rank(scores, relevance) == pytest.approx(0.5, abs=1e-5)

    def test_none_relevant(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([0, 0, 0])
        assert mean_reciprocal_rank(scores, relevance) == pytest.approx(0.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        scores = jnp.array([1.0, 2.0])
        relevance = jnp.array([1, 0])
        result = mean_reciprocal_rank(scores, relevance)
        assert isinstance(result, jax.Array)


class TestHitRate:
    """Tests for hit_rate."""

    def test_hit(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([1, 0, 0])
        assert hit_rate(scores, relevance, k=1) == pytest.approx(1.0, abs=1e-5)

    def test_miss(self) -> None:
        scores = jnp.array([3.0, 2.0, 1.0])
        relevance = jnp.array([0, 0, 1])
        assert hit_rate(scores, relevance, k=1) == pytest.approx(0.0, abs=1e-5)


class TestCoverage:
    """Catalog coverage: distinct recommended items over the catalog size (Ge et al. 2010)."""

    def test_full_coverage(self) -> None:
        assert float(coverage(jnp.array([0, 1, 2, 3, 4]), catalog_size=5)) == pytest.approx(1.0)

    def test_duplicates_count_once(self) -> None:
        assert float(coverage(jnp.array([0, 0, 1, 1]), catalog_size=5)) == pytest.approx(0.4)

    def test_a_batch_of_lists_is_covered_together(self) -> None:
        lists = jnp.array([[0, 1, 2], [2, 3, 3]])
        assert float(coverage(lists, catalog_size=8)) == pytest.approx(0.5)

    def test_ids_outside_the_catalog_are_not_counted(self) -> None:
        """jnp.bincount clips a negative id to item 0; it must not be counted as item 0."""
        items = jnp.array([-1, 5, 7, 1])
        assert float(coverage(items, catalog_size=5)) == pytest.approx(0.2)

    def test_jit_traces_once_with_a_static_catalog_size(self) -> None:
        counter = TraceCounter()
        compiled = jax.jit(counter.wrap(coverage), static_argnames=("catalog_size",))
        with counter.expect(new_traces=1):
            compiled(jnp.array([0, 1, 1]), catalog_size=4)
        with counter.expect(new_traces=0):
            result = compiled(jnp.array([3, 2, 2]), catalog_size=4)
        assert float(result) == pytest.approx(0.5)

    def test_vmap_gives_per_user_coverage(self) -> None:
        per_user = jax.vmap(lambda items: coverage(items, catalog_size=4))(
            jnp.array([[0, 0, 0], [0, 1, 2]])
        )
        assert per_user.tolist() == pytest.approx([0.25, 0.75])

    def test_returns_jax_scalar(self) -> None:
        result = coverage(jnp.array([0, 1]), catalog_size=10)
        assert isinstance(result, jax.Array)
        assert result.shape == ()

    def test_the_registry_marks_it_custom(self) -> None:
        assert MetricRegistry().get("coverage").signature == MetricSignature.CUSTOM


class TestRankingMetricRegistration:
    """Tests for ranking metric registration."""

    def test_all_registered(self) -> None:
        registry = MetricRegistry()
        expected = [
            "ndcg",
            "ndcg_at_k",
            "mean_average_precision",
            "precision_at_k",
            "recall_at_k",
            "mean_reciprocal_rank",
            "hit_rate",
            "coverage",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_ranking_domain(self) -> None:
        registry = MetricRegistry()
        ranking_metrics = registry.list_by_domain("ranking")
        assert len(ranking_metrics) == 8

    def test_all_direction_higher(self) -> None:
        registry = MetricRegistry()
        for m in registry.list_by_domain("ranking"):
            assert m.direction == MetricDirection.HIGHER


# --- Tied scores -------------------------------------------------------------------------------
#
# A tie in the scores orders the tied items neither way, so each metric is its mean over every
# ordering of each tied group: McSherry and Najork, "Computing Information Retrieval Performance
# Measures Efficiently in the Presence of Tied Scores", ECIR 2008 (doi:10.1007/978-3-540-78646-7_38),
# which scikit-learn's ``ndcg_score`` implements (``_tie_averaged_dcg``). The oracle below is that
# definition itself: it computes each metric on every ordering consistent with the scores, in
# float64, and averages.


def _one_ordering(metric: str, relevance: np.ndarray, k: int) -> float:
    """The metric on relevance already in ranked order, by calibrax's existing definitions."""
    n = relevance.size
    kk = min(k, n)
    if metric in ("ndcg", "ndcg_at_k"):
        cut = n if metric == "ndcg" else kk
        discounts = np.log2(np.arange(cut) + 2.0)
        dcg = np.sum((2.0 ** relevance[:cut] - 1.0) / discounts)
        idcg = np.sum((2.0 ** np.sort(relevance)[::-1][:cut] - 1.0) / discounts)
        return float(dcg / idcg) if idcg > 0 else 0.0
    if metric == "precision_at_k":
        return float(np.sum(relevance[:kk]) / kk)
    if metric == "recall_at_k":
        total = np.sum(relevance)
        return float(np.sum(relevance[:kk]) / total) if total > 0 else 0.0
    if metric == "mean_average_precision":
        total = np.sum(relevance)
        precision = np.cumsum(relevance) / np.arange(1, n + 1)
        return float(np.sum(precision * relevance) / total) if total > 0 else 0.0
    if metric == "mean_reciprocal_rank":
        hits = np.flatnonzero(relevance > 0)
        return float(1.0 / (hits[0] + 1)) if hits.size else 0.0
    if metric == "hit_rate":
        return float(np.sum(relevance[:kk]) > 0)
    raise ValueError(metric)


def _tie_averaged(metric: str, scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """The metric's mean over every ordering of each group of tied scores (brute force)."""
    groups = [relevance[scores == value] for value in sorted(set(scores.tolist()), reverse=True)]
    per_group = [list(itertools.permutations(group)) for group in groups]
    values = [
        _one_ordering(metric, np.concatenate([np.asarray(order, float) for order in combo]), k)
        for combo in itertools.product(*per_group)
    ]
    return float(np.mean(values))


def _tied_ranking_fixtures(count: int, size: int = 6) -> list[tuple[np.ndarray, np.ndarray]]:
    """Scores from three values, so most fixtures tie relevant and irrelevant items together."""
    rng = np.random.default_rng(0)
    fixtures = []
    while len(fixtures) < count:
        scores = rng.integers(0, 3, size).astype(np.float32)
        relevance = rng.integers(0, 2, size).astype(np.float32)
        if relevance.any():
            fixtures.append((scores, relevance))
    return fixtures


def _graded_fixtures(count: int, size: int = 6) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(1)
    return [
        (rng.integers(0, 3, size).astype(np.float32), rng.integers(0, 4, size).astype(np.float32))
        for _ in range(count)
    ]


_UNIT_ROUNDOFF = float(np.finfo(np.float32).eps) / 2


def _ranking_bound(size: int) -> float:
    """The float32 error bound on any of these metrics over ``size`` items, derived, not measured.

    Each metric is a sum or a product of at most ``size`` terms. A term is formed with at most four
    rounded operations: a gain or count, a group mean or ratio, a ``log2`` discount (within two
    ulp), and one division. A sum or product of n terms, each relatively within ``gamma_4``, is
    within ``gamma_(n+4)`` (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., secs.
    3.1, 4.2), and NDCG's ratio of two such sums is within twice that plus one rounding. Every
    metric is at most 1, so ``gamma_(2 size + 9)`` bounds the absolute error of each.
    """
    k = 2 * size + 9
    return k * _UNIT_ROUNDOFF / (1 - k * _UNIT_ROUNDOFF)


_BINARY = {
    "mean_average_precision": lambda s, r, _k: mean_average_precision(s, r),
    "precision_at_k": lambda s, r, k: precision_at_k(s, r, k=k),
    "recall_at_k": lambda s, r, k: recall_at_k(s, r, k=k),
    "mean_reciprocal_rank": lambda s, r, _k: mean_reciprocal_rank(s, r),
    "hit_rate": lambda s, r, k: hit_rate(s, r, k=k),
}
_GRADED = {
    "ndcg": lambda s, r, _k: ndcg(s, r),
    "ndcg_at_k": lambda s, r, k: ndcg_at_k(s, r, k=k),
}
TIED_BINARY = _tied_ranking_fixtures(60)
TIED_GRADED = _graded_fixtures(60)


class TestTiedScores:
    """Each ranking metric is its mean over the orderings of tied items, so their order is moot."""

    @pytest.mark.parametrize("name", sorted(_BINARY))
    @pytest.mark.parametrize("k", [1, 3, 6])
    def test_binary_metrics_are_tie_averaged(self, name: str, k: int) -> None:
        for scores, relevance in TIED_BINARY:
            expected = _tie_averaged(name, scores, relevance, k)
            assert float(_BINARY[name](scores, relevance, k)) == pytest.approx(
                expected, abs=_ranking_bound(scores.size)
            ), (name, k, scores.tolist(), relevance.tolist())

    @pytest.mark.parametrize("name", sorted(_GRADED))
    @pytest.mark.parametrize("k", [1, 3, 6])
    def test_ndcg_is_tie_averaged(self, name: str, k: int) -> None:
        for scores, relevance in TIED_GRADED:
            expected = _tie_averaged(name, scores, relevance, k)
            assert float(_GRADED[name](scores, relevance, k)) == pytest.approx(
                expected, abs=_ranking_bound(scores.size)
            ), (name, k, scores.tolist(), relevance.tolist())

    def test_ndcg_matches_scikit_learn_with_the_exponential_gain(self) -> None:
        """scikit-learn's ndcg_score is the tie-averaged NDCG; given 2^rel - 1 as its gain it is
        calibrax's."""
        for scores, relevance in TIED_GRADED:
            expected = ndcg_score([2.0**relevance - 1.0], [scores], k=3)
            assert float(ndcg_at_k(scores, relevance, k=3)) == pytest.approx(
                expected, abs=_ranking_bound(scores.size)
            )

    @pytest.mark.parametrize("name", sorted({**_BINARY, **_GRADED}))
    def test_reordering_tied_items_changes_nothing(self, name: str) -> None:
        metric = {**_BINARY, **_GRADED}[name]
        rng = np.random.default_rng(2)
        for scores, relevance in TIED_GRADED if name in _GRADED else TIED_BINARY:
            order = rng.permutation(scores.size)
            assert float(metric(scores[order], relevance[order], 3)) == pytest.approx(
                float(metric(scores, relevance, 3)), abs=_ranking_bound(scores.size)
            )

    def test_four_tied_items_with_two_relevant(self) -> None:
        scores = jnp.full(4, 0.5)
        for relevance in ([1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]):
            rel = jnp.array(relevance)
            assert float(precision_at_k(scores, rel, k=1)) == pytest.approx(0.5)
            assert float(hit_rate(scores, rel, k=1)) == pytest.approx(0.5)
            # The first relevant item is at rank 1 with probability 1/2, rank 2 with 1/3, rank 3
            # with 1/6: 1/2 + 1/6 + 1/18.
            assert float(mean_reciprocal_rank(scores, rel)) == pytest.approx(13 / 18)

    @pytest.mark.parametrize("name", sorted({**_BINARY, **_GRADED}))
    def test_jit_and_vmap_give_the_tie_averaged_value(self, name: str) -> None:
        metric = {**_BINARY, **_GRADED}[name]
        fixtures = (TIED_GRADED if name in _GRADED else TIED_BINARY)[:8]
        scores = np.stack([s for s, _ in fixtures])
        relevance = np.stack([r for _, r in fixtures])
        compiled = np.asarray(jax.jit(jax.vmap(lambda s, r: metric(s, r, 3)))(scores, relevance))
        expected = np.array([_tie_averaged(name, s, r, 3) for s, r in fixtures])
        np.testing.assert_allclose(compiled, expected, rtol=0, atol=_ranking_bound(scores.shape[1]))
