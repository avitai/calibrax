"""Ranking and retrieval metrics.

Pure functions for evaluating the quality of ranked lists and
information retrieval systems. Functions accept predicted scores
and ground truth relevance labels.

Includes 8 functions: ndcg, ndcg_at_k, mean_average_precision,
precision_at_k, recall_at_k, mean_reciprocal_rank, hit_rate, coverage.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

from calibrax.metrics._utils import _EPSILON


class _TiedRanking(NamedTuple):
    """Items in descending score order, each with its group of tied scores.

    Attributes:
        relevance: Relevance in descending score order.
        position: 0-based ranked position.
        size: Size of the item's tied group.
        offset: 0-based offset of the position within its group.
        group_relevance: Relevance summed over the item's tied group.
        before: Relevance summed over the groups ranked above the item's group.
    """

    relevance: jax.Array
    position: jax.Array
    size: jax.Array
    offset: jax.Array
    group_relevance: jax.Array
    before: jax.Array


def _tied_ranking(scores: ArrayLike, relevance: ArrayLike) -> _TiedRanking:
    """Rank by descending score and describe each item's group of tied scores.

    A tie orders the tied items neither way, so every metric below is its mean over the orderings
    of each tied group (McSherry and Najork, "Computing Information Retrieval Performance Measures
    Efficiently in the Presence of Tied Scores", ECIR 2008), which scikit-learn's ``ndcg_score``
    implements. Each metric reads only group totals, so the input order of tied items cannot move
    it. Static shapes (segment sums over the groups) keep every metric jittable and vmappable.

    Args:
        scores: Predicted relevance scores.
        relevance: Ground truth relevance labels.

    Returns:
        The ranking with each item's group size, offset, relevance total and the relevance above it.
    """
    keys, ranked = lax.sort_key_val(-jnp.asarray(scores).ravel(), jnp.asarray(relevance).ravel())
    ranked = ranked.astype(jnp.float32)
    size = keys.size
    position = jnp.arange(size)
    starts = jnp.concatenate([jnp.ones(min(size, 1), dtype=bool), keys[1:] != keys[:-1]])
    group = jnp.cumsum(starts) - 1
    start = lax.cummax(jnp.where(starts, position, 0))
    group_total = functools.partial(jax.ops.segment_sum, segment_ids=group, num_segments=size)
    above = (jnp.cumsum(ranked) - ranked)[start]
    return _TiedRanking(
        relevance=ranked,
        position=position,
        size=group_total(jnp.ones(size, jnp.float32))[group],
        offset=position - start,
        group_relevance=group_total(ranked)[group],
        before=above,
    )


def _expected_relevance(ranking: _TiedRanking) -> jax.Array:
    """Each position's expected relevance: its group's relevance spread over the group."""
    return ranking.group_relevance / ranking.size


def _irrelevant_given_earlier(ranking: _TiedRanking) -> jax.Array:
    """The chance a position is irrelevant, given the earlier positions of its group were.

    Within a group of ``n`` items holding ``r`` relevant ones, the item at offset ``o`` is
    relevant with probability ``r / (n - o)`` once the ``o`` before it were not (McSherry and
    Najork's reciprocal-rank case, sec. 2.5); the chain of these is the chance that no relevant
    item has appeared by that position.
    """
    remaining = ranking.size - ranking.offset
    return 1.0 - jnp.minimum(ranking.group_relevance / remaining, 1.0)


def _discounted_gain(ranking: _TiedRanking, gains: jax.Array, cut: int) -> jax.Array:
    """DCG of the first ``cut`` slots, each slot holding its group's mean gain."""
    size = ranking.relevance.size
    group = jnp.cumsum(ranking.offset == 0) - 1
    mean_gain = jax.ops.segment_sum(gains, group, num_segments=size)[group] / ranking.size
    discounts = jnp.log2(ranking.position.astype(jnp.float32) + 2.0)
    return jnp.sum(jnp.where(ranking.position < cut, mean_gain / discounts, 0.0))


def _ideal_gain(relevance: ArrayLike, cut: int) -> jax.Array:
    """DCG of the ideal ranking's first ``cut`` slots; ties in the relevance do not move it."""
    ideal = jnp.sort(jnp.asarray(relevance).ravel().astype(jnp.float32))[::-1][:cut]
    discounts = jnp.log2(jnp.arange(ideal.size, dtype=jnp.float32) + 2.0)
    return jnp.sum((2.0**ideal - 1.0) / discounts)


def _normalized_gain(scores: ArrayLike, relevance: ArrayLike, cut: int) -> jax.Array:
    ranking = _tied_ranking(scores, relevance)
    dcg = _discounted_gain(ranking, 2.0**ranking.relevance - 1.0, cut)
    idcg = _ideal_gain(relevance, cut)
    return jnp.where(idcg > _EPSILON, dcg / jnp.maximum(idcg, _EPSILON), 0.0)


def ndcg(scores: ArrayLike, relevance: ArrayLike) -> jax.Array:
    """Normalized Discounted Cumulative Gain (full list).

    ``DCG / IDCG`` where ``DCG = sum((2^rel_i - 1) / log2(i+2))``. Tied scores share their group's
    mean gain, slot by slot: the mean over the orderings of the tied items (McSherry and Najork
    2008; scikit-learn's ``ndcg_score`` with ``2^rel - 1`` as its gain).

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
    return _normalized_gain(scores, relevance, jnp.asarray(scores).size)


def ndcg_at_k(scores: ArrayLike, relevance: ArrayLike, *, k: int) -> jax.Array:
    """NDCG truncated to top-k results, tied scores averaged as in :func:`ndcg`.

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
    return _normalized_gain(scores, relevance, min(k, jnp.asarray(scores).size))


def mean_average_precision(scores: ArrayLike, relevance: ArrayLike) -> jax.Array:
    """Mean Average Precision for a single query.

    Average of precision at each relevant position, its mean over the orderings of tied items
    (McSherry and Najork 2008, sec. 2.4). scikit-learn's ``average_precision_score`` instead takes
    one operating point per tied group, so on tied scores the two differ; on untied scores they
    agree.

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
    ranking = _tied_ranking(scores, relevance)
    n_relevant = jnp.sum(ranking.relevance)
    others = jnp.maximum(ranking.size - 1.0, 1.0)
    relevant_before_in_group = ranking.offset * (ranking.group_relevance - 1.0) / others
    precision_if_relevant = (ranking.before + relevant_before_in_group + 1.0) / (
        ranking.position + 1.0
    )
    ap_sum = jnp.sum(_expected_relevance(ranking) * precision_if_relevant)
    return jnp.where(n_relevant == 0, 0.0, ap_sum / jnp.maximum(n_relevant, 1.0))


def _expected_hits(scores: ArrayLike, relevance: ArrayLike, k: int) -> tuple[jax.Array, int]:
    """The expected relevant items among the top ``k``, and that ``k`` capped at the list length."""
    ranking = _tied_ranking(scores, relevance)
    k = min(k, ranking.relevance.size)
    return jnp.sum(jnp.where(ranking.position < k, _expected_relevance(ranking), 0.0)), k


def precision_at_k(scores: ArrayLike, relevance: ArrayLike, *, k: int) -> jax.Array:
    """Fraction of relevant items in top-k, its mean over the orderings of tied items.

    A tied group straddling the cutoff contributes its relevant share of the slots inside it
    (McSherry and Najork 2008, sec. 2.1).

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
    hits, k = _expected_hits(scores, relevance, k)
    return hits / k


def recall_at_k(scores: ArrayLike, relevance: ArrayLike, *, k: int) -> jax.Array:
    """Fraction of relevant items found in top-k, tied items averaged (McSherry and Najork, 2.2).

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
    hits, _ = _expected_hits(scores, relevance, k)
    n_relevant = jnp.sum(jnp.asarray(relevance).ravel().astype(jnp.float32))
    return jnp.where(n_relevant == 0, 0.0, hits / jnp.maximum(n_relevant, 1.0))


def _first_relevant_at(ranking: _TiedRanking) -> jax.Array:
    """The chance that each position holds the first relevant item, over the tied orderings."""
    irrelevant = _irrelevant_given_earlier(ranking)
    none_yet = jnp.cumprod(jnp.concatenate([jnp.ones(1, jnp.float32), irrelevant[:-1]]))
    return none_yet * (1.0 - irrelevant)


def mean_reciprocal_rank(scores: ArrayLike, relevance: ArrayLike) -> jax.Array:
    """Reciprocal of the rank of the first relevant item, its mean over tied orderings.

    McSherry and Najork 2008, sec. 2.5: a relevant item is first at a position with the chance that
    none appeared before it and one appears there.

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
    ranking = _tied_ranking(scores, relevance)
    return jnp.sum(_first_relevant_at(ranking) / (ranking.position + 1.0))


def hit_rate(scores: ArrayLike, relevance: ArrayLike, *, k: int) -> jax.Array:
    """Whether any relevant item appears in top-k: its mean over the orderings of tied items.

    Over the orderings it is the chance that a relevant item reaches the top ``k``, so a tied group
    straddling the cutoff gives a value between 0 and 1.

    Note:
        Direction: HIGHER.
        Range: [0, 1]; 0.0 or 1.0 when no tied group straddles the cutoff.

    Args:
        scores: Predicted relevance scores.
        relevance: Binary ground truth (0 or 1).
        k: Number of top results to consider.

    Returns:
        The chance that a relevant item is in the top-k.
    """
    ranking = _tied_ranking(scores, relevance)
    k = min(k, ranking.relevance.size)
    return jnp.sum(jnp.where(ranking.position < k, _first_relevant_at(ranking), 0.0))


def coverage(items: ArrayLike, *, catalog_size: int) -> jax.Array:
    """Catalog coverage: the fraction of the catalog's items that appear in the recommendations.

    ``|distinct recommended items| / catalog_size`` (Ge, Delgado-Battenfeld and Jannach 2010).
    Items are integer ids in ``[0, catalog_size)``; an id outside that range is not a catalog item
    and is not counted. The count uses ``jnp.bincount`` with a static ``length``, so the function
    works under ``jax.jit`` (``catalog_size`` static) and ``jax.vmap``.

    Note:
        Direction: HIGHER (1.0 = full catalog coverage).
        Range: [0, 1].

    Args:
        items: Recommended item ids, any shape; every list in a batch is covered together.
        catalog_size: Number of items in the catalog.

    Returns:
        Coverage as a scalar value.
    """
    ids = jnp.asarray(items).ravel()
    in_catalog = (ids >= 0) & (ids < catalog_size)
    # jnp.bincount clips negative ids to 0, so ids outside the catalog carry zero weight.
    counts = jnp.bincount(
        jnp.where(in_catalog, ids, 0), weights=in_catalog.astype(jnp.int32), length=catalog_size
    )
    return jnp.count_nonzero(counts) / catalog_size
