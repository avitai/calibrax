"""Tests for ranking and retrieval metrics."""

from __future__ import annotations

import fractions
import itertools
import math

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
    bedroc,
    coverage,
    enrichment_factor,
    hit_rate,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    sequential_discovery_nauc,
    sequential_shortfall,
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
            "bedroc",
            "enrichment_factor",
            "sequential_discovery_nauc",
            "sequential_shortfall",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_ranking_domain(self) -> None:
        registry = MetricRegistry()
        ranking_metrics = registry.list_by_domain("ranking")
        assert len(ranking_metrics) == 12

    def test_ranking_directions(self) -> None:
        registry = MetricRegistry()
        for m in registry.list_by_domain("ranking"):
            expected = (
                MetricDirection.LOWER
                if m.name == "sequential_shortfall"
                else MetricDirection.HIGHER
            )
            assert m.direction == expected


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


def _truchon_bayly_bedroc_f64(ranks_1based: np.ndarray, n_items: int, alpha: float) -> float:
    """Truchon & Bayly (JCIM 2007, Eqs. 26 & 36) textbook RIE normalization in extended precision."""
    n_actives = ranks_1based.size
    a = np.longdouble(alpha)
    n = np.longdouble(n_items)
    ra = np.longdouble(n_actives) / n
    ranks = ranks_1based.astype(np.longdouble)
    rie_num = np.sum(np.exp(-a * ranks / n)) / np.longdouble(n_actives)
    rie_den = (1 - np.exp(-a)) / (n * (np.exp(a / n) - 1))
    rie = rie_num / rie_den
    rie_min = (1 - np.exp(a * ra)) / (ra * (1 - np.exp(a)))
    rie_max = (1 - np.exp(-a * ra)) / (ra * (1 - np.exp(-a)))
    return float((rie - rie_min) / (rie_max - rie_min))


class TestBEDROC:
    """Boltzmann-Enhanced Discrimination of ROC (Truchon & Bayly, JCIM 2007)."""

    def test_perfect_and_worst_rankings_hit_unit_bounds(self) -> None:
        scores_top = jnp.array([10.0, 9.0, 4.0, 3.0, 2.0, 1.0], dtype=jnp.float32)
        scores_bot = jnp.array([1.0, 2.0, 9.0, 8.0, 7.0, 6.0], dtype=jnp.float32)
        relevance = jnp.array([1, 1, 0, 0, 0, 0], dtype=jnp.float32)
        assert float(bedroc(scores_top, relevance, alpha=20.0)) == pytest.approx(1.0, abs=1e-6)
        assert float(bedroc(scores_bot, relevance, alpha=20.0)) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("alpha", [0.5, 5.0, 20.0, 80.5, 200.0])
    def test_matches_truchon_bayly_eq26_across_alphas_without_overflow(self, alpha: float) -> None:
        rng = np.random.default_rng(17)
        n_items = 64
        scores = rng.permutation(n_items).astype(np.float32)
        relevance = (rng.random(n_items) < 0.15).astype(np.float32)
        relevance[:2] = [1.0, 0.0]

        order = np.argsort(-scores)
        active_ranks = np.flatnonzero(relevance[order] > 0) + 1
        expected = _truchon_bayly_bedroc_f64(active_ranks, n_items, alpha)
        actual = float(bedroc(jnp.asarray(scores), jnp.asarray(relevance), alpha=alpha))
        assert np.isfinite(actual)
        assert actual == pytest.approx(expected, abs=_ranking_bound(n_items))

    def test_tied_scores_are_permutation_invariant_and_match_bruteforce_mean(self) -> None:
        rng = np.random.default_rng(23)
        for scores, relevance in TIED_BINARY:
            if relevance.sum() in (0, scores.size):
                continue
            groups = [relevance[scores == v] for v in sorted(set(scores.tolist()), reverse=True)]
            per_group = [list(itertools.permutations(g)) for g in groups]
            brute = [
                _truchon_bayly_bedroc_f64(
                    np.flatnonzero(np.concatenate([np.asarray(o, float) for o in combo]) > 0) + 1,
                    scores.size,
                    20.0,
                )
                for combo in itertools.product(*per_group)
            ]
            val = float(bedroc(scores, relevance, alpha=20.0))
            assert val == pytest.approx(float(np.mean(brute)), abs=_ranking_bound(scores.size))
            perm = rng.permutation(scores.size)
            assert float(bedroc(scores[perm], relevance[perm], alpha=20.0)) == pytest.approx(
                val, abs=_ranking_bound(scores.size)
            )

    def test_empty_or_single_class_returns_nan_and_nonpositive_alpha_raises(self) -> None:
        empty = jnp.zeros(0, dtype=jnp.float32)
        assert np.isnan(float(bedroc(empty, empty)))
        scores = jnp.array([3.0, 2.0, 1.0], dtype=jnp.float32)
        assert np.isnan(float(bedroc(scores, jnp.zeros(3, dtype=jnp.float32))))
        assert np.isnan(float(bedroc(scores, jnp.ones(3, dtype=jnp.float32))))
        nan_scores = jnp.array([jnp.nan, 3.0, 2.0, 1.0], dtype=jnp.float32)
        nan_rel = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
        assert np.isnan(float(bedroc(nan_scores, nan_rel)))
        assert np.isnan(float(jax.jit(bedroc)(nan_scores, nan_rel)))
        assert np.isnan(
            float(
                bedroc(
                    jnp.array([4.0, 3.0, 2.0, 1.0], dtype=jnp.float32),
                    jnp.array([jnp.nan, 1.0, 0.0, 0.0], dtype=jnp.float32),
                )
            )
        )
        assert np.isnan(float(bedroc(scores, jnp.array([1.0, 0.0, 0.0]), alpha=float("nan"))))
        with pytest.raises(ValueError, match="alpha"):
            bedroc(scores, jnp.array([1.0, 0.0, 0.0]), alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            bedroc(scores, jnp.array([1.0, 0.0, 0.0]), alpha=-5.0)
        with pytest.raises(ValueError, match="alpha"):
            bedroc(scores, jnp.array([1.0, 0.0, 0.0]), alpha=np.float32(-1.0))
        with pytest.raises(ValueError, match="alpha"):
            bedroc(scores, jnp.array([1.0, 0.0, 0.0]), alpha=np.array(-1.0, dtype=np.float32))

    def test_sequence_inputs_traced_alpha_and_dtype_preservation(self) -> None:
        val_list = bedroc([3.0, 2.0, 1.0], [1, 0, 0], alpha=20.0)  # type: ignore[arg-type]
        val_tuple = bedroc((3.0, 2.0, 1.0), (1, 0, 0), alpha=20.0)  # type: ignore[arg-type]
        assert float(val_list) == pytest.approx(1.0, abs=1e-6)
        assert float(val_tuple) == pytest.approx(1.0, abs=1e-6)

        scores = jnp.array([3.0, 2.0, 1.0], dtype=jnp.float32)
        rel = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
        traced = jax.jit(jax.vmap(lambda a: bedroc(scores, rel, alpha=a)))(
            jnp.array([10.0, 20.0], dtype=jnp.float32)
        )
        np.testing.assert_allclose(np.asarray(traced), np.array([1.0, 1.0]), atol=1e-6)

        with jax.enable_x64(True):
            out_f32 = bedroc(scores, rel, alpha=20.0)
            out_f64 = bedroc(scores.astype(jnp.float64), rel.astype(jnp.float64), alpha=20.0)
            assert out_f32.dtype == jnp.float32
            assert out_f64.dtype == jnp.float64

    def test_jit_and_vmap_trace_once(self) -> None:
        counter = TraceCounter()
        compiled = jax.jit(counter.wrap(lambda s, r: bedroc(s, r, alpha=20.0)))
        s = jnp.arange(24.0, dtype=jnp.float32).reshape(3, 8)
        r = jnp.array(
            [[0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0]],
            dtype=jnp.float32,
        )
        with counter.expect(new_traces=1):
            compiled(s[0], r[0])
        with counter.expect(new_traces=0):
            compiled(s[1], r[1])
        batched = jax.vmap(lambda x, y: bedroc(x, y, alpha=20.0))(s, r)
        assert batched.shape == (3,)
        assert float(batched[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(batched[1]) == pytest.approx(0.0, abs=1e-6)


class TestEnrichmentFactor:
    """Early enrichment factor EF_f at screening fraction ``fraction``."""

    def test_known_enrichment_and_tie_averaging(self) -> None:
        # N=10, n=2 (base rate 0.2). Top 20% -> k=2. Both actives in top 2 -> precision=1.0 -> EF=5.0.
        scores = jnp.arange(10.0, 0.0, -1.0, dtype=jnp.float32)
        relevance = jnp.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
        assert float(enrichment_factor(scores, relevance, fraction=0.2)) == pytest.approx(5.0)

        # All tied scores -> expected precision equals base rate -> EF == 1.0.
        tied = jnp.ones(10, dtype=jnp.float32)
        assert float(enrichment_factor(tied, relevance, fraction=0.2)) == pytest.approx(1.0)

    def test_empty_or_zero_actives_returns_nan_and_invalid_fraction_raises(self) -> None:
        empty = jnp.zeros(0, dtype=jnp.float32)
        assert np.isnan(float(enrichment_factor(empty, empty)))
        scores = jnp.array([3.0, 2.0, 1.0], dtype=jnp.float32)
        assert np.isnan(float(enrichment_factor(scores, jnp.zeros(3, dtype=jnp.float32))))
        with pytest.raises(ValueError, match="fraction"):
            enrichment_factor(scores, jnp.array([1.0, 0.0, 0.0]), fraction=0.0)
        with pytest.raises(ValueError, match="fraction"):
            enrichment_factor(scores, jnp.array([1.0, 0.0, 0.0]), fraction=1.5)

    def test_python_float_fraction_avoids_float32_downward_cutoff_rounding(self) -> None:
        # Exact RDKit CalcEnrichment cutoff convention: k = min(N, max(1, ceil(N * fraction))).
        # Specifically checks binary-float pitfalls: 0.07 * 100 -> 7 (not ceil(7.000000000000001)=8),
        # 0.29 * 100 -> 29 (not 28), 0.05 * 30 -> 2, 0.05 * 3 -> 1, and fraction=1.0 -> EF=1.0.
        rng = np.random.default_rng(101)
        for n_items in (3, 10, 30, 100, 300):
            scores_np = np.arange(n_items, 0, -1, dtype=np.float32)
            rel_np = (rng.random(n_items) < 0.25).astype(np.float32)
            rel_np[0] = 1.0
            n_actives = float(np.sum(rel_np))
            scores_j = jnp.asarray(scores_np)
            rel_j = jnp.asarray(rel_np)

            for frac in (0.01, 0.05, 0.07, 0.29, 1.0):
                k_exact = min(n_items, max(1, math.ceil(fractions.Fraction(str(frac)) * n_items)))
                hits_exact = float(np.sum(rel_np[:k_exact]))
                expected_ef = (hits_exact / k_exact) / (n_actives / n_items)
                actual_ef = float(enrichment_factor(scores_j, rel_j, fraction=frac))
                assert actual_ef == pytest.approx(
                    expected_ef, rel=4 * _UNIT_ROUNDOFF, abs=4 * _UNIT_ROUNDOFF
                ), (n_items, frac, k_exact)

        # Explicitly verify boundary element k=7 at (N=100, fraction=0.07): active at rank 8 must
        # NOT be counted inside top 7%.
        scores_100 = jnp.arange(100.0, 0.0, -1.0, dtype=jnp.float32)
        rel_rank8 = jnp.zeros(100, dtype=jnp.float32).at[7].set(1.0)
        assert float(enrichment_factor(scores_100, rel_rank8, fraction=0.07)) == 0.0
        assert float(enrichment_factor(scores_100, rel_rank8, fraction=np.float32(0.07))) == 0.0
        assert float(enrichment_factor(scores_100, rel_rank8, fraction=jnp.float32(0.07))) == 0.0
        jitted_static = jax.jit(enrichment_factor, static_argnames=("fraction",))
        assert float(jitted_static(scores_100, rel_rank8, fraction=0.07)) == 0.0
        with jax.enable_x64(True):
            scores_100_f64 = scores_100.astype(jnp.float64)
            rel_rank8_f64 = rel_rank8.astype(jnp.float64)
            assert (
                float(enrichment_factor(scores_100_f64, rel_rank8_f64, fraction=jnp.float32(0.07)))
                == 0.0
            )
        with pytest.raises(ValueError, match="fraction"):
            enrichment_factor(scores_100, rel_rank8, fraction=np.array([0.07, 0.2]))
        with pytest.raises(ValueError, match="fraction"):
            enrichment_factor(scores_100, rel_rank8, fraction=jnp.asarray(1.5))
        with pytest.raises(TypeError, match="fraction"):
            enrichment_factor(scores_100, rel_rank8, fraction=True)
        with pytest.raises(TypeError, match="Python float"):
            enrichment_factor(
                jnp.zeros(0), jnp.zeros(0), fraction=jnp.asarray(0.3, dtype=jnp.bfloat16)
            )
        with pytest.raises(TypeError, match="static_argnames"):
            jax.jit(lambda f: enrichment_factor(scores_100, rel_rank8, fraction=f))(
                jnp.float32(0.07)
            )
        rel_rank7 = jnp.zeros(100, dtype=jnp.float32).at[6].set(1.0)
        assert float(enrichment_factor(scores_100, rel_rank7, fraction=0.07)) == pytest.approx(
            100.0 / 7.0, rel=4 * _UNIT_ROUNDOFF
        )

        # Single-trace compilation under jax.jit and jax.vmap with static fraction.
        counter = TraceCounter()
        compiled = jax.jit(counter.wrap(lambda s, r: enrichment_factor(s, r, fraction=0.07)))
        with counter.expect(new_traces=1):
            compiled(scores_100, rel_rank7)
        with counter.expect(new_traces=0):
            compiled(scores_100 * 2.0, rel_rank8)

    def test_eager_jit_cutoff_identity_scalar_bounds_and_nan_propagation(self) -> None:
        jitted_ef = jax.jit(enrichment_factor, static_argnames=("fraction",))
        for n_items, frac, expected_k in (
            (100, 0.07, 7),
            (1000, 0.007, 7),
            (50, 0.14, 7),
            (200, 0.035, 7),
            (29, 0.05, 2),
            (10, 0.3, 3),
            (10, 0.6, 6),
            (3, 1.0 / 3.0, 1),
            (3, 2.0 / 3.0, 2),
        ):
            scores = jnp.arange(float(n_items), 0.0, -1.0, dtype=jnp.float32)
            rel_out = jnp.zeros(n_items, dtype=jnp.int32).at[expected_k].set(1)
            ef_eager = float(enrichment_factor(scores, rel_out, fraction=frac))
            ef_jit = float(jitted_ef(scores, rel_out, fraction=frac))
            assert ef_eager == 0.0
            assert ef_jit == 0.0

            rel_in = jnp.zeros(n_items, dtype=jnp.int32).at[expected_k - 1].set(1)
            ef_in_eager = float(enrichment_factor(scores, rel_in, fraction=frac))
            ef_in_jit = float(jitted_ef(scores, rel_in, fraction=frac))
            assert ef_in_eager == pytest.approx(float(n_items) / float(expected_k), rel=1e-6)
            assert ef_in_jit == ef_in_eager

        scores_100 = jnp.arange(100.0, 0.0, -1.0, dtype=jnp.float32)
        rel_100 = jnp.zeros(100, dtype=jnp.int32).at[7].set(1)
        frac_above_f32 = jnp.nextafter(jnp.float32(0.07), jnp.float32(1.0))
        assert float(
            enrichment_factor(scores_100, rel_100, fraction=float(frac_above_f32))
        ) == pytest.approx(12.5)
        assert float(
            enrichment_factor(scores_100, rel_100, fraction=frac_above_f32)
        ) == pytest.approx(12.5)

        for bad_frac in (
            0.0,
            -0.1,
            1.1,
            float("nan"),
            np.float64(-0.1),
            np.float32(-0.1),
            np.array(-0.1, dtype=np.float32),
            jnp.array(-0.1, dtype=jnp.float32),
        ):
            with pytest.raises(ValueError, match="fraction"):
                enrichment_factor(jnp.array([2.0, 1.0]), jnp.array([1, 0]), fraction=bad_frac)

        assert float(
            enrichment_factor([3.0, 2.0, 1.0], (1, 0, 0), fraction=0.5)  # type: ignore[arg-type]
        ) == pytest.approx(1.5, abs=1e-6)
        with jax.enable_x64(True):
            assert (
                enrichment_factor(
                    jnp.array([3.0, 2.0, 1.0], dtype=jnp.float32),
                    jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
                    fraction=0.5,
                ).dtype
                == jnp.float32
            )
            assert (
                enrichment_factor(
                    jnp.array([3.0, 2.0, 1.0], dtype=jnp.float64),
                    jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
                    fraction=0.5,
                ).dtype
                == jnp.float64
            )

    def test_large_library_concrete_fraction_cutoff_in_float64(self) -> None:
        n_items = 10_000_001
        scores = jnp.arange(n_items, 0, -1, dtype=jnp.int32)
        zeros = jnp.zeros(n_items, dtype=jnp.int32)
        jitted_1pct = jax.jit(lambda s, r: enrichment_factor(s, r, fraction=0.01))

        # ceil(10_000_001 * 0.01) = ceil(100_000.01) = 100_001 (0-based index 100_000).
        expected = float(n_items) / 100_001.0
        assert float(
            enrichment_factor(scores, zeros.at[100_000].set(1), fraction=0.01)
        ) == pytest.approx(expected, rel=1e-6)
        assert float(jitted_1pct(scores, zeros.at[100_000].set(1))) == pytest.approx(
            expected, rel=1e-6
        )
        assert float(jitted_1pct(scores, zeros.at[99_999].set(1))) == pytest.approx(
            expected, rel=1e-6
        )
        assert float(jitted_1pct(scores, zeros.at[100_001].set(1))) == 0.0

        nan_scores = jnp.array([jnp.nan, 3.0, 2.0, 1.0], dtype=jnp.float32)
        rel_valid = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
        assert np.isnan(float(enrichment_factor(nan_scores, rel_valid, fraction=0.5)))
        assert np.isnan(
            float(
                jax.jit(lambda s, r: enrichment_factor(s, r, fraction=0.5))(nan_scores, rel_valid)
            )
        )
        assert np.isnan(
            float(
                enrichment_factor(
                    jnp.array([4.0, 3.0, 2.0, 1.0], dtype=jnp.float32),
                    jnp.array([jnp.nan, 1.0, 0.0, 0.0], dtype=jnp.float32),
                    fraction=0.5,
                )
            )
        )


class TestSequentialDiscoveryMetrics:
    """Oracle-normalized sequential discovery AUC and shortfall (AssayBench protocol)."""

    def test_oracle_and_zero_trajectories_and_complementarity(self) -> None:
        # Total actives = 3 over 5 steps: oracle cumulative hits are [1, 2, 3, 3, 3].
        oracle = jnp.array([1.0, 2.0, 3.0, 3.0, 3.0], dtype=jnp.float32)
        policy = jnp.array([0.0, 1.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
        zeros = jnp.zeros(5, dtype=jnp.float32)

        assert float(sequential_discovery_nauc(oracle, total_actives=3)) == pytest.approx(1.0)
        assert float(sequential_shortfall(oracle, oracle_cumulative_hits=oracle)) == pytest.approx(
            0.0
        )
        assert float(sequential_discovery_nauc(zeros, total_actives=3)) == pytest.approx(0.0)
        assert float(sequential_shortfall(zeros, oracle_cumulative_hits=oracle)) == pytest.approx(
            1.0
        )

        nauc_val = float(sequential_discovery_nauc(policy, total_actives=3))
        sf_val = float(sequential_shortfall(policy, oracle_cumulative_hits=oracle))
        # Oracle trapezoid on [0, 1, 2, 3, 3, 3] = 0.5 + 1.5 + 2.5 + 3.0 + 3.0 = 10.5.
        # Policy trapezoid on [0, 0, 1, 1, 2, 3] = 0.0 + 0.5 + 1.0 + 1.5 + 2.5 = 5.5.
        assert nauc_val == pytest.approx(5.5 / 10.5, abs=1e-6)
        assert sf_val == pytest.approx(1.0 - 5.5 / 10.5, abs=1e-6)

    def test_budget_truncation_empty_and_differentiability(self) -> None:
        policy = jnp.array([1.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
        # Truncated to budget=2: oracle [0, 1, 2] area = 0.5 + 1.5 = 2.0; policy [0, 1, 1] area = 1.5.
        assert float(sequential_discovery_nauc(policy, total_actives=3, budget=2)) == pytest.approx(
            0.75
        )
        with pytest.raises(ValueError, match="budget"):
            sequential_discovery_nauc(policy, total_actives=3, budget=0)

        empty = jnp.zeros(0, dtype=jnp.float32)
        assert np.isnan(float(sequential_discovery_nauc(empty, total_actives=3)))
        assert np.isnan(float(sequential_discovery_nauc(policy, total_actives=0)))
        assert np.isnan(float(sequential_shortfall(policy, oracle_cumulative_hits=jnp.zeros(4))))

        grad = jax.grad(lambda h: sequential_discovery_nauc(h, total_actives=3))(policy)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert bool(jnp.all(grad > 0))
