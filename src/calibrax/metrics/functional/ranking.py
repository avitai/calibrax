"""Ranking and retrieval metrics.

Pure functions for evaluating the quality of ranked lists and
information retrieval systems. Functions accept predicted scores
and ground truth relevance labels.

Includes 12 functions: ndcg, ndcg_at_k, mean_average_precision,
precision_at_k, recall_at_k, mean_reciprocal_rank, hit_rate, coverage,
bedroc, enrichment_factor, sequential_discovery_nauc, sequential_shortfall.
"""

from __future__ import annotations

import fractions
import functools
import math
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
    s_arr = jnp.asarray(scores).ravel()
    r_arr = jnp.asarray(relevance).ravel()
    out_dtype = jnp.result_type(s_arr.dtype, r_arr.dtype, jnp.float32)
    keys, ranked = lax.sort_key_val(-s_arr, r_arr)
    ranked = ranked.astype(out_dtype)
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
        size=group_total(jnp.ones(size, out_dtype))[group],
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


def bedroc(
    scores: ArrayLike,
    relevance: ArrayLike,
    *,
    alpha: ArrayLike = 20.0,
) -> jax.Array:
    """Boltzmann-Enhanced Discrimination of ROC (Truchon and Bayly, JCIM 2007).

    Evaluates early recognition of active items using exponential rank weights
    ``exp(-alpha * rank / N)`` normalized to ``[0, 1]`` via the exact discrete geometric
    extrema (Truchon and Bayly 2007, Eqs. 26 and 36). Tied score groups receive the exact
    group-mean Boltzmann weight across all within-group permutations (McSherry and Najork
    2008), and factoring out ``exp(-alpha / N)`` avoids positive-exponent overflow in
    ``float32`` at large ``alpha``.

    Note:
        Direction: HIGHER (1.0 = all active items ranked ahead of all inactive items).
        Range: [0, 1]. ``NaN`` when the list is empty or contains only actives or only
        inactives (where ``RIE_max == RIE_min``).

    Args:
        scores: Predicted ranking scores (higher ranks earlier).
        relevance: Binary ground-truth activity indicators (0 or 1).
        alpha: Exponential early-recognition decay parameter (must be positive).

    Returns:
        BEDROC score as a scalar JAX array, or ``NaN`` when undefined.

    Raises:
        ValueError: If ``scores`` and ``relevance`` differ in size or eager ``alpha <= 0``.
    """
    if isinstance(alpha, int | float) and not isinstance(alpha, jax.Array) and alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    s = jnp.asarray(scores).ravel()
    r = jnp.asarray(relevance).ravel()
    if s.size != r.size:
        raise ValueError(f"Shape mismatch: scores has {s.size} elements, relevance has {r.size}")
    if s.size == 0:
        return jnp.array(jnp.nan, dtype=jnp.float32)

    ranking = _tied_ranking(s, r)
    n_items = float(s.size)
    n_actives = jnp.sum(ranking.relevance)
    alpha_arr = jnp.asarray(alpha, dtype=jnp.float32)
    valid = (n_actives > 0.0) & (n_actives < n_items) & (alpha_arr > 0.0)

    safe_alpha = jnp.where(valid, alpha_arr, 1.0)
    safe_actives = jnp.where(valid, n_actives, 1.0)
    step = safe_alpha / n_items

    weights = jnp.exp(-step * ranking.position.astype(jnp.float32))
    weighted_sum = jnp.sum(_expected_relevance(ranking) * weights)

    denom_step = -jnp.expm1(-step)
    safe_denom = jnp.where(denom_step > 0.0, denom_step, 1.0)
    s_max = -jnp.expm1(-safe_actives * step) / safe_denom
    s_min = jnp.exp(-step * (n_items - safe_actives)) * s_max
    span = s_max - s_min
    safe_span = jnp.where(span > 0.0, span, 1.0)
    return jnp.where(valid & (span > 0.0), (weighted_sum - s_min) / safe_span, jnp.nan)


def enrichment_factor(
    scores: ArrayLike,
    relevance: ArrayLike,
    *,
    fraction: ArrayLike = 0.01,
) -> jax.Array:
    """Early enrichment factor at screening cutoff ``fraction``.

    Ratio of tie-averaged precision among the top ``k = max(1, ceil(fraction * N))``
    ranked items (Riniker and Landrum, J. Cheminf. 2013; McSherry and Najork, ECIR 2008)
    to the library base rate ``n_actives / N``.

    Note:
        Direction: HIGHER (values above 1.0 outperform random selection).
        Range: ``[0, N / n_actives]``. ``NaN`` when the list is empty or has zero actives.

    Args:
        scores: Predicted ranking scores (higher ranks earlier).
        relevance: Binary ground-truth activity indicators (0 or 1).
        fraction: Screening fraction in ``(0, 1]`` (defaults to ``0.01``).

    Returns:
        Enrichment factor as a scalar JAX array, or ``NaN`` when undefined.

    Raises:
        ValueError: If ``scores`` and ``relevance`` differ in size or eager ``fraction``
            lies outside ``(0, 1]``.
    """
    if (
        isinstance(fraction, int | float)
        and not isinstance(fraction, jax.Array)
        and not (0.0 < fraction <= 1.0)
    ):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    s = jnp.asarray(scores).ravel()
    r = jnp.asarray(relevance).ravel()
    if s.size != r.size:
        raise ValueError(f"Shape mismatch: scores has {s.size} elements, relevance has {r.size}")
    is_python_frac = isinstance(fraction, int | float) and not isinstance(fraction, jax.Array)
    out_dtype = (
        jnp.result_type(s.dtype, r.dtype, jnp.float32)
        if is_python_frac
        else jnp.result_type(s.dtype, r.dtype, fraction, jnp.float32)
    )
    if s.size == 0:
        return jnp.array(jnp.nan, dtype=out_dtype)

    ranking = _tied_ranking(s, r)
    n_items = jnp.asarray(float(s.size), dtype=out_dtype)
    if is_python_frac:
        k_int = min(s.size, max(1, math.ceil(fractions.Fraction(str(fraction)) * s.size)))
        k = jnp.asarray(float(k_int), dtype=out_dtype)
        valid_frac = jnp.array(True)
    else:
        frac = jnp.asarray(fraction, dtype=out_dtype)
        valid_frac = (frac > 0.0) & (frac <= 1.0)
        prod = jnp.where(valid_frac, frac, jnp.ones_like(frac)) * n_items
        k = jnp.clip(
            jnp.ceil(prod - 4.0 * jnp.finfo(out_dtype).eps * prod),
            jnp.asarray(1.0, dtype=out_dtype),
            n_items,
        )
    hits = jnp.sum(
        jnp.where(ranking.position.astype(out_dtype) < k, _expected_relevance(ranking), 0.0)
    )
    precision = hits / k
    n_actives = jnp.sum(ranking.relevance).astype(out_dtype)
    valid = valid_frac & (n_actives > 0.0)
    safe_base_rate = jnp.where(valid, n_actives / n_items, jnp.asarray(1.0, dtype=out_dtype))
    return jnp.where(valid, precision / safe_base_rate, jnp.asarray(jnp.nan, dtype=out_dtype))


def _normalized_discovery_areas(
    cumulative_hits: jax.Array,
    oracle_cumulative_hits: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Trapezoidal areas under policy and oracle cumulative-hit curves anchored at ``(0, 0)``."""
    zero = jnp.zeros(1, dtype=cumulative_hits.dtype)
    policy_curve = jnp.concatenate([zero, cumulative_hits])
    oracle_curve = jnp.concatenate([zero, oracle_cumulative_hits])
    policy_area = jnp.trapezoid(policy_curve)
    oracle_area = jnp.trapezoid(oracle_curve)
    valid = oracle_area > 0.0
    safe_oracle = jnp.where(valid, oracle_area, jnp.ones_like(oracle_area))
    return policy_area, safe_oracle, valid


def sequential_discovery_nauc(
    cumulative_hits: ArrayLike,
    *,
    total_actives: ArrayLike,
    budget: int | None = None,
) -> jax.Array:
    """Oracle-normalized area under the sequential hit-recovery curve (AssayBench protocol).

    Integrates ``cumulative_hits`` anchored at ``(0, 0)`` with the trapezoidal rule and
    divides by the ideal oracle trajectory ``min(step, total_actives)`` evaluated on the
    same discrete step grid.

    Note:
        Direction: HIGHER (1.0 = ideal oracle recovery trajectory).
        Range: [0, 1]. ``NaN`` when ``cumulative_hits`` is empty or ``total_actives <= 0``.

    Args:
        cumulative_hits: Cumulative active discoveries after each acquisition step.
        total_actives: Total number of discoverable active items in the pool.
        budget: Optional positive step horizon truncating ``cumulative_hits``.

    Returns:
        Normalized discovery AUC as a scalar JAX array, or ``NaN`` when undefined.

    Raises:
        ValueError: If ``budget`` is provided and is not positive.
    """
    hits = jnp.asarray(cumulative_hits, dtype=jnp.float32).ravel()
    if budget is not None:
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        hits = hits[:budget]
    if hits.size == 0:
        return jnp.array(jnp.nan, dtype=jnp.float32)

    actives = jnp.asarray(total_actives, dtype=hits.dtype)
    steps = jnp.arange(1, hits.size + 1, dtype=hits.dtype)
    oracle = jnp.minimum(steps, jnp.maximum(actives, 0.0))
    policy_area, safe_oracle, valid = _normalized_discovery_areas(hits, oracle)
    return jnp.where(valid & (actives > 0.0), policy_area / safe_oracle, jnp.nan)


def sequential_shortfall(
    cumulative_hits: ArrayLike,
    *,
    oracle_cumulative_hits: ArrayLike,
) -> jax.Array:
    """Relative sequential discovery shortfall ``1 - AUC(policy) / AUC(oracle)``.

    Measures the normalized area gap between an ideal ``oracle_cumulative_hits`` curve
    and the realized ``cumulative_hits`` trajectory anchored at ``(0, 0)``.

    Note:
        Direction: LOWER (0.0 = matches the oracle trajectory).
        Range: [0, 1] when ``0 <= cumulative_hits <= oracle_cumulative_hits``.
        ``NaN`` when ``cumulative_hits`` is empty or the oracle area is non-positive.

    Args:
        cumulative_hits: Realized cumulative discoveries at each step.
        oracle_cumulative_hits: Oracle cumulative discoveries on the same step grid.

    Returns:
        Normalized discovery shortfall as a scalar JAX array, or ``NaN`` when undefined.

    Raises:
        ValueError: If ``cumulative_hits`` and ``oracle_cumulative_hits`` differ in length.
    """
    hits = jnp.asarray(cumulative_hits, dtype=jnp.float32).ravel()
    oracle = jnp.asarray(oracle_cumulative_hits, dtype=hits.dtype).ravel()
    if hits.size != oracle.size:
        raise ValueError(
            f"Shape mismatch: cumulative_hits has {hits.size} steps, "
            f"oracle_cumulative_hits has {oracle.size}"
        )
    if hits.size == 0:
        return jnp.array(jnp.nan, dtype=jnp.float32)

    policy_area, safe_oracle, valid = _normalized_discovery_areas(hits, oracle)
    return jnp.where(valid, (safe_oracle - policy_area) / safe_oracle, jnp.nan)
