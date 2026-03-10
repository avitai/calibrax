"""Ranking and retrieval metrics.

Pure functions for evaluating the quality of ranked lists and
information retrieval systems. Functions accept predicted scores
and ground truth relevance labels.

Includes 8 functions: ndcg, ndcg_at_k, mean_average_precision,
precision_at_k, recall_at_k, mean_reciprocal_rank, hit_rate, coverage.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON


def _rank_by_scores(scores: Any, relevance: Any) -> Any:
    """Sort relevance by descending predicted scores.

    Args:
        scores: Predicted relevance scores.
        relevance: Ground truth relevance labels.

    Returns:
        Relevance labels sorted by descending score.
    """
    order = jnp.argsort(-jnp.asarray(scores).ravel())
    return jnp.asarray(relevance).ravel()[order]


def ndcg(scores: Any, relevance: Any) -> Any:
    """Normalized Discounted Cumulative Gain (full list).

    ``DCG / IDCG`` where ``DCG = sum((2^rel_i - 1) / log2(i+2))``.

    Note:
        Direction: HIGHER (1.0 = perfect ranking).
        Range: [0, 1].

    Args:
        scores: Predicted relevance scores.
        relevance: Ground truth relevance labels (non-negative).

    Returns:
        NDCG as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> ndcg(jnp.array([3.0, 2.0, 1.0]), jnp.array([3, 2, 1]))
        1.0
    """
    ranked_rel = _rank_by_scores(scores, relevance)
    rel_arr = jnp.asarray(relevance).ravel()

    positions = jnp.arange(len(ranked_rel)) + 2  # i+2 for 0-indexed
    discounts = jnp.log2(positions.astype(jnp.float32))
    gains = (2.0**ranked_rel - 1.0) / discounts
    dcg = jnp.sum(gains)

    # Ideal DCG
    ideal_rel = jnp.sort(rel_arr)[::-1]
    ideal_gains = (2.0**ideal_rel - 1.0) / discounts
    idcg = jnp.sum(ideal_gains)

    return jnp.where(idcg > _EPSILON, dcg / idcg, 0.0)


def ndcg_at_k(scores: Any, relevance: Any, *, k: int) -> Any:
    """NDCG truncated to top-k results.

    Note:
        Direction: HIGHER (1.0 = perfect ranking in top-k).
        Range: [0, 1].

    Args:
        scores: Predicted relevance scores.
        relevance: Ground truth relevance labels.
        k: Number of top results to consider.

    Returns:
        NDCG@k as a scalar value.
    """
    ranked_rel = _rank_by_scores(scores, relevance)
    rel_arr = jnp.asarray(relevance).ravel()
    k = min(k, len(ranked_rel))

    positions = jnp.arange(k) + 2
    discounts = jnp.log2(positions.astype(jnp.float32))
    dcg = jnp.sum((2.0 ** ranked_rel[:k] - 1.0) / discounts)

    ideal_rel = jnp.sort(rel_arr)[::-1][:k]
    idcg = jnp.sum((2.0**ideal_rel - 1.0) / discounts)

    return jnp.where(idcg > _EPSILON, dcg / idcg, 0.0)


def mean_average_precision(scores: Any, relevance: Any) -> Any:
    """Mean Average Precision for a single query.

    Average of precision at each relevant position.

    Note:
        Direction: HIGHER (1.0 = all relevant items ranked first).
        Range: [0, 1].
        Relevance must be binary (0/1).

    Args:
        scores: Predicted relevance scores.
        relevance: Binary ground truth (0 or 1).

    Returns:
        Average precision as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> mean_average_precision(jnp.array([3.0, 1.0, 2.0]), jnp.array([1, 0, 1]))
        1.0
    """
    ranked_rel = _rank_by_scores(scores, relevance).astype(jnp.float32)
    n = len(ranked_rel)
    n_relevant = jnp.sum(ranked_rel)

    cumsum = jnp.cumsum(ranked_rel)
    positions = jnp.arange(1, n + 1, dtype=jnp.float32)
    precisions = cumsum / positions
    ap_sum = jnp.sum(precisions * ranked_rel)
    return jnp.where(n_relevant == 0, 0.0, ap_sum / n_relevant)


def precision_at_k(scores: Any, relevance: Any, *, k: int) -> Any:
    """Fraction of relevant items in top-k.

    Note:
        Direction: HIGHER (1.0 = all top-k are relevant).
        Range: [0, 1].

    Args:
        scores: Predicted relevance scores.
        relevance: Binary ground truth (0 or 1).
        k: Number of top results to consider.

    Returns:
        Precision@k as a scalar value.
    """
    ranked_rel = _rank_by_scores(scores, relevance).astype(jnp.float32)
    k = min(k, len(ranked_rel))
    return jnp.sum(ranked_rel[:k]) / k


def recall_at_k(scores: Any, relevance: Any, *, k: int) -> Any:
    """Fraction of relevant items found in top-k.

    Note:
        Direction: HIGHER (1.0 = all relevant items in top-k).
        Range: [0, 1].

    Args:
        scores: Predicted relevance scores.
        relevance: Binary ground truth (0 or 1).
        k: Number of top results to consider.

    Returns:
        Recall@k as a scalar value.
    """
    ranked_rel = _rank_by_scores(scores, relevance).astype(jnp.float32)
    k = min(k, len(ranked_rel))
    n_relevant = jnp.sum(jnp.asarray(relevance).ravel().astype(jnp.float32))
    hits = jnp.sum(ranked_rel[:k])
    return jnp.where(n_relevant == 0, 0.0, hits / n_relevant)


def mean_reciprocal_rank(scores: Any, relevance: Any) -> Any:
    """Reciprocal of the rank of the first relevant item.

    Note:
        Direction: HIGHER (1.0 = first item is relevant).
        Range: [0, 1].
        Returns 0.0 if no relevant items.

    Args:
        scores: Predicted relevance scores.
        relevance: Binary ground truth (0 or 1).

    Returns:
        MRR as a scalar value.

    Examples:
        >>> import jax.numpy as jnp
        >>> mean_reciprocal_rank(jnp.array([1.0, 3.0, 2.0]), jnp.array([0, 1, 0]))
        1.0
    """
    ranked_rel = _rank_by_scores(scores, relevance).astype(jnp.float32)
    # Find first relevant position (1-indexed)
    positions = jnp.arange(1, len(ranked_rel) + 1, dtype=jnp.float32)
    # Set non-relevant to inf so argmin finds first relevant
    reciprocals = jnp.where(ranked_rel > 0, 1.0 / positions, 0.0)
    rank = jnp.max(reciprocals)
    return jnp.where(rank == 0, 0.0, rank)


def hit_rate(scores: Any, relevance: Any, *, k: int) -> Any:
    """Whether any relevant item appears in top-k.

    Note:
        Direction: HIGHER.
        Range: {0.0, 1.0} (binary).

    Args:
        scores: Predicted relevance scores.
        relevance: Binary ground truth (0 or 1).
        k: Number of top results to consider.

    Returns:
        1.0 if any relevant item in top-k, 0.0 otherwise.
    """
    ranked_rel = _rank_by_scores(scores, relevance).astype(jnp.float32)
    k = min(k, len(ranked_rel))
    return jnp.where(jnp.sum(ranked_rel[:k]) > 0, 1.0, 0.0)


def coverage(scores: Any, relevance: Any, *, catalog_size: int) -> Any:
    """Fraction of catalog covered by recommendations.

    Note:
        Direction: HIGHER (1.0 = full catalog coverage).
        Range: [0, 1].

    Args:
        scores: Recommended item IDs (1D integer array).
        relevance: Unused (present for API consistency). Pass any array.
        catalog_size: Total number of unique items in catalog.

    Returns:
        Coverage as a scalar value.
    """
    items = jnp.asarray(scores).ravel()
    unique_count = jnp.float32(len(jnp.unique(items)))
    return unique_count / catalog_size
