"""Tests for ranking and retrieval metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
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
