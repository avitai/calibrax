"""Clustering evaluation metrics for unsupervised learning.

Pure functions for evaluating clustering quality. Divided into two categories:

**External evaluation** (requires ground truth labels):
- adjusted_rand_index, normalized_mutual_information_clustering,
  adjusted_mutual_information, v_measure

**Internal evaluation** (no ground truth, uses feature distances):
- silhouette_score, calinski_harabasz_score, davies_bouldin_score

All accept integer label arrays, read as names: renaming a label changes nothing. Pass the
number of labels (``num_classes`` for the ground truth, ``num_clusters`` for the clustering)
to trace a metric under ``jax.jit``, labels then being ids below it, empty ids allowed;
without it the distinct labels are counted from the data, eagerly. Edge cases follow
scikit-learn's conventions. Internal metrics additionally require a feature matrix.
Registered with ``domain="clustering"`` and ``signature=MetricSignature.FEATURES_LABELS``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import betaln
from jax.typing import ArrayLike

from calibrax.metrics._utils import _EPSILON, dense_labels, label_masks, safe_norm


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_AVERAGES = ("arithmetic", "geometric", "min", "max")


def _contingency(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    num_classes: int | None,
    num_clusters: int | None,
) -> jax.Array:
    """Counts of samples per (true label, predicted label), shape ``(classes, clusters)``."""
    true_ids, classes = dense_labels(labels_true, num_classes)
    pred_ids, clusters = dense_labels(labels_pred, num_clusters)
    return jnp.zeros((classes, clusters), jnp.float32).at[true_ids, pred_ids].add(1.0)


def _entropy(counts: jax.Array) -> jax.Array:
    """Shannon entropy in nats of the distribution ``counts`` describe; empty counts add 0."""
    probs = counts / jnp.sum(counts)
    present = probs > 0
    return -jnp.sum(jnp.where(present, probs * jnp.log(jnp.where(present, probs, 1.0)), 0.0))


def _mutual_information(contingency: jax.Array) -> jax.Array:
    """Mutual information in nats between the two labelings a contingency table counts."""
    n = jnp.sum(contingency)
    outer = jnp.sum(contingency, axis=1)[:, None] * jnp.sum(contingency, axis=0)[None, :]
    present = contingency > 0
    ratio = jnp.where(present, contingency * n / jnp.where(present, outer, 1.0), 1.0)
    return jnp.sum(jnp.where(present, contingency / n * jnp.log(ratio), 0.0))


def _generalized_mean(h_true: jax.Array, h_pred: jax.Array, average: str) -> jax.Array:
    """The two entropies' mean named ``average``, as scikit-learn normalises by."""
    if average == "arithmetic":
        return (h_true + h_pred) / 2.0
    if average == "geometric":
        return jnp.sqrt(h_true * h_pred)
    if average == "min":
        return jnp.minimum(h_true, h_pred)
    return jnp.maximum(h_true, h_pred)


def _check_average(average: str) -> None:
    """Refuse an unknown normaliser name.

    Args:
        average: The normaliser's name.

    Raises:
        ValueError: If ``average`` is not one of the supported names.
    """
    if average not in _AVERAGES:
        msg = f"average must be one of {_AVERAGES}, got '{average}'"
        raise ValueError(msg)


def _log_binomial(n: jax.Array, k: jax.Array) -> jax.Array:
    """``log C(n, k)`` through ``betaln``, which keeps precision where ``lgamma`` terms cancel."""
    return -jnp.log1p(n) - betaln(n - k + 1.0, k + 1.0)


def _expected_mutual_information(contingency: jax.Array, num_samples: int) -> jax.Array:
    """Expected mutual information under the hypergeometric model of random labelings.

    Vinh, Epps and Bailey (2010), JMLR 11, eq. 24: the sum over cells and their possible
    counts ``n_ij`` of ``n_ij / N * log(N n_ij / (a_i b_j))`` weighted by the hypergeometric
    probability of ``n_ij``. The count runs over ``1..num_samples`` in a ``fori_loop`` (the
    memory of one table), terms outside ``[max(1, a_i + b_j - N), min(a_i, b_j)]`` masked.
    """
    n = jnp.sum(contingency)
    a = jnp.sum(contingency, axis=1)[:, None]
    b = jnp.sum(contingency, axis=0)[None, :]
    lower = jnp.maximum(1.0, a + b - n)
    upper = jnp.minimum(a, b)
    log_total = _log_binomial(n, b)

    def add_count(count: int, emi: jax.Array) -> jax.Array:
        nij = jnp.asarray(count, jnp.float32)
        valid = (nij >= lower) & (nij <= upper)
        # Arguments that keep every log finite where the term is masked out.
        a_safe = jnp.where(valid, a, nij)
        b_safe = jnp.where(valid, b, nij)
        log_pmf = (
            _log_binomial(a_safe, nij)
            + _log_binomial(n - a_safe, b_safe - nij)
            - jnp.where(valid, log_total, 0.0)
        )
        term = nij / n * jnp.log(n * nij / (a_safe * b_safe)) * jnp.exp(log_pmf)
        return emi + jnp.sum(jnp.where(valid, term, 0.0))

    return jax.lax.fori_loop(1, num_samples + 1, add_count, jnp.float32(0.0))


def _pairwise_distances(features: jax.Array) -> jax.Array:
    """Pairwise Euclidean distances, exactly 0 on the diagonal, shape ``(n, n)``."""
    return safe_norm(features[:, None, :] - features[None, :, :], axis=-1)


# ---------------------------------------------------------------------------
# External evaluation metrics
# ---------------------------------------------------------------------------


def adjusted_rand_index(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    *,
    num_classes: int | None = None,
    num_clusters: int | None = None,
) -> jax.Array:
    """Adjusted Rand Index for clustering agreement (Hubert and Arabie 1985).

    ARI = (RI - E[RI]) / (max(RI) - E[RI]) over the contingency table's pair counts. Two
    labelings that are both one cluster, or both all singletons, agree perfectly: 1.0.

    Args:
        labels_true: Ground truth integer labels, shape (n,).
        labels_pred: Predicted integer labels, shape (n,).
        num_classes: Number of ground-truth label ids, to trace; None counts them.
        num_clusters: Number of predicted label ids, to trace; None counts them.

    Returns:
        ARI value in [-1, 1]. 1.0 = perfect agreement, 0.0 = random,
        negative = worse than random.

    Examples:
        >>> import jax.numpy as jnp
        >>> adjusted_rand_index(jnp.array([0, 0, 1, 1]), jnp.array([0, 0, 1, 1]))
        1.0
    """
    contingency = _contingency(labels_true, labels_pred, num_classes, num_clusters)
    n = jnp.sum(contingency)
    sum_comb_c = jnp.sum(contingency * (contingency - 1) / 2.0)
    row_sums = jnp.sum(contingency, axis=1)
    col_sums = jnp.sum(contingency, axis=0)
    sum_comb_a = jnp.sum(row_sums * (row_sums - 1) / 2.0)
    sum_comb_b = jnp.sum(col_sums * (col_sums - 1) / 2.0)

    expected = sum_comb_a * sum_comb_b / (n * (n - 1) / 2.0)
    denominator = (sum_comb_a + sum_comb_b) / 2.0 - expected
    trivial = jnp.abs(denominator) < _EPSILON
    return jnp.where(trivial, 1.0, (sum_comb_c - expected) / jnp.where(trivial, 1.0, denominator))


def normalized_mutual_information_clustering(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    *,
    average: str = "arithmetic",
    num_classes: int | None = None,
    num_clusters: int | None = None,
) -> jax.Array:
    """Normalized Mutual Information for clustering: MI(true, pred) over a mean of entropies.

    Two single-cluster labelings agree perfectly (1.0); otherwise a zero normaliser gives 0.

    Args:
        labels_true: Ground truth integer labels, shape (n,).
        labels_pred: Predicted integer labels, shape (n,).
        average: Normalizer type. One of "arithmetic" (default), "geometric",
            "min", "max".
        num_classes: Number of ground-truth label ids, to trace; None counts them.
        num_clusters: Number of predicted label ids, to trace; None counts them.

    Returns:
        NMI value in [0, 1]. 1.0 = perfect agreement.

    Examples:
        >>> import jax.numpy as jnp
        >>> normalized_mutual_information_clustering(
        ...     jnp.array([0, 0, 1, 1]), jnp.array([0, 0, 1, 1])
        ... )
        1.0
    """
    _check_average(average)
    contingency = _contingency(labels_true, labels_pred, num_classes, num_clusters)
    h_true = _entropy(jnp.sum(contingency, axis=1))
    h_pred = _entropy(jnp.sum(contingency, axis=0))
    normalizer = jnp.maximum(_generalized_mean(h_true, h_pred, average), _EPSILON)
    both_trivial = (h_true < _EPSILON) & (h_pred < _EPSILON)
    return jnp.where(both_trivial, 1.0, _mutual_information(contingency) / normalizer)


def adjusted_mutual_information(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    *,
    average: str = "arithmetic",
    num_classes: int | None = None,
    num_clusters: int | None = None,
) -> jax.Array:
    """Adjusted Mutual Information (Vinh, Epps and Bailey 2010).

    AMI = (MI - E[MI]) / (mean(H_true, H_pred) - E[MI]), with the exact expected mutual
    information under the hypergeometric model of random labelings (their eq. 24).

    Args:
        labels_true: Ground truth integer labels, shape (n,).
        labels_pred: Predicted integer labels, shape (n,).
        average: The entropies' mean in the normaliser: "arithmetic" (default),
            "geometric", "min" or "max".
        num_classes: Number of ground-truth label ids, to trace; None counts them.
        num_clusters: Number of predicted label ids, to trace; None counts them.

    Returns:
        AMI value, at most 1.0 (perfect agreement); 0.0 is the expected value by chance.

    Examples:
        >>> import jax.numpy as jnp
        >>> adjusted_mutual_information(
        ...     jnp.array([0, 0, 1, 1]), jnp.array([0, 0, 1, 1])
        ... )
        1.0
    """
    _check_average(average)
    contingency = _contingency(labels_true, labels_pred, num_classes, num_clusters)
    h_true = _entropy(jnp.sum(contingency, axis=1))
    h_pred = _entropy(jnp.sum(contingency, axis=0))
    emi = _expected_mutual_information(contingency, jnp.shape(labels_true)[0])
    denominator = _generalized_mean(h_true, h_pred, average) - emi
    denominator = jnp.where(
        denominator < 0, jnp.minimum(denominator, -_EPSILON), jnp.maximum(denominator, _EPSILON)
    )
    ami = (_mutual_information(contingency) - emi) / denominator
    both_trivial = (h_true < _EPSILON) & (h_pred < _EPSILON)
    return jnp.where(both_trivial, 1.0, ami)


def v_measure(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    *,
    beta: float = 1.0,
    num_classes: int | None = None,
    num_clusters: int | None = None,
) -> jax.Array:
    """V-measure: the weighted harmonic mean of homogeneity and completeness.

    Rosenberg and Hirschberg (2007): ``V = (1 + beta) h c / (beta h + c)``, homogeneity
    ``h = MI / H(true)`` and completeness ``c = MI / H(pred)``, each 1.0 where its entropy
    is zero. With beta = 1 it equals NMI with the arithmetic normaliser.

    Args:
        labels_true: Ground truth integer labels, shape (n,).
        labels_pred: Predicted integer labels, shape (n,).
        beta: Weight parameter. 1.0 = equal weight. >1 = favor completeness.
        num_classes: Number of ground-truth label ids, to trace; None counts them.
        num_clusters: Number of predicted label ids, to trace; None counts them.

    Returns:
        V-measure in [0, 1]. 1.0 = perfect clustering.

    Examples:
        >>> import jax.numpy as jnp
        >>> v_measure(jnp.array([0, 0, 1, 1]), jnp.array([0, 0, 1, 1]))
        1.0
    """
    contingency = _contingency(labels_true, labels_pred, num_classes, num_clusters)
    mi = _mutual_information(contingency)
    h_true = _entropy(jnp.sum(contingency, axis=1))
    h_pred = _entropy(jnp.sum(contingency, axis=0))
    homogeneity = jnp.where(h_true > 0, mi / jnp.where(h_true > 0, h_true, 1.0), 1.0)
    completeness = jnp.where(h_pred > 0, mi / jnp.where(h_pred > 0, h_pred, 1.0), 1.0)
    total = beta * homogeneity + completeness
    return jnp.where(
        total > 0,
        (1 + beta) * homogeneity * completeness / jnp.where(total > 0, total, 1.0),
        0.0,
    )


# ---------------------------------------------------------------------------
# Internal evaluation metrics
# ---------------------------------------------------------------------------


def silhouette_score(
    features: ArrayLike, labels: ArrayLike, *, num_clusters: int | None = None
) -> jax.Array:
    """Mean silhouette coefficient across all samples (Rousseeuw 1987).

    For each sample: s = (b - a) / max(a, b) where a = mean intra-cluster distance and
    b = mean distance to the nearest other cluster; a sample alone in its cluster scores 0.
    O(n^2) complexity.

    Args:
        features: Feature matrix of shape (n, d).
        labels: Cluster assignment integer labels, shape (n,).
        num_clusters: Number of cluster ids, to trace; None counts the distinct labels.

    Returns:
        Mean silhouette in [-1, 1]. Higher = better separated clusters.

    Examples:
        >>> import jax.numpy as jnp
        >>> features = jnp.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
        >>> labels = jnp.array([0, 0, 1, 1])
        >>> silhouette_score(features, labels)  # Close to 1.0
        ...
    """
    features = jnp.asarray(features, dtype=jnp.float32)
    masks, sizes = label_masks(labels, num_clusters)
    own = masks.T  # (n, k)
    to_cluster = _pairwise_distances(features) @ own  # summed distance to each cluster, (n, k)

    own_size = own @ sizes
    alone = own_size <= 1
    a = jnp.sum(to_cluster * own, axis=1) / jnp.where(alone, 1.0, own_size - 1.0)
    other = (own == 0) & (sizes[None, :] > 0)
    mean_to_other = to_cluster / jnp.where(sizes > 0, sizes, 1.0)[None, :]
    b = jnp.min(jnp.where(other, mean_to_other, jnp.inf), axis=1)
    spread = jnp.maximum(a, b)
    silhouettes = jnp.where(
        alone | (spread <= 0), 0.0, (b - a) / jnp.where(spread > 0, spread, 1.0)
    )
    return jnp.mean(silhouettes)


def _centroids(features: jax.Array, masks: jax.Array, sizes: jax.Array) -> jax.Array:
    """Each cluster's mean feature vector, zero for an empty cluster, shape ``(k, d)``."""
    return (masks @ features) / jnp.where(sizes > 0, sizes, 1.0)[:, None]


def calinski_harabasz_score(
    features: ArrayLike, labels: ArrayLike, *, num_clusters: int | None = None
) -> jax.Array:
    """Calinski-Harabasz Index (Variance Ratio Criterion; Calinski and Harabasz 1974).

    Ratio of between-cluster to within-cluster dispersion, adjusted for the counts of
    non-empty clusters ``k`` and samples ``n``: ``B (n - k) / (W (k - 1))``; 1.0 when every
    cluster is a single point (W = 0).

    Args:
        features: Feature matrix of shape (n, d).
        labels: Cluster assignment integer labels, shape (n,).
        num_clusters: Number of cluster ids, to trace; None counts the distinct labels.

    Returns:
        Calinski-Harabasz score (>= 0). Higher is better.

    Examples:
        >>> import jax.numpy as jnp
        >>> features = jnp.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
        >>> labels = jnp.array([0, 0, 1, 1])
        >>> calinski_harabasz_score(features, labels)  # Large value
        ...
    """
    features = jnp.asarray(features, dtype=jnp.float32)
    masks, sizes = label_masks(labels, num_clusters)
    n = features.shape[0]
    k = jnp.sum(sizes > 0)
    centroids = _centroids(features, masks, sizes)
    between = jnp.sum(sizes * jnp.sum((centroids - jnp.mean(features, axis=0)) ** 2, axis=1))
    within = jnp.sum((features - masks.T @ centroids) ** 2)
    return jnp.where(
        within > 0, between * (n - k) / (jnp.where(within > 0, within, 1.0) * (k - 1)), 1.0
    )


def davies_bouldin_score(
    features: ArrayLike, labels: ArrayLike, *, num_clusters: int | None = None
) -> jax.Array:
    """Davies-Bouldin Index for cluster separation (Davies and Bouldin 1979).

    The mean over non-empty clusters of the largest ``(s_i + s_j) / d(c_i, c_j)`` against
    another cluster, ``s`` a cluster's mean distance to its centroid. 0 when every cluster
    is a single point or all centroids coincide. Lower = better separated clusters.

    Args:
        features: Feature matrix of shape (n, d).
        labels: Cluster assignment integer labels, shape (n,).
        num_clusters: Number of cluster ids, to trace; None counts the distinct labels.

    Returns:
        Davies-Bouldin score (>= 0). Lower is better.

    Examples:
        >>> import jax.numpy as jnp
        >>> features = jnp.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
        >>> labels = jnp.array([0, 0, 1, 1])
        >>> davies_bouldin_score(features, labels)  # Close to 0
        ...
    """
    features = jnp.asarray(features, dtype=jnp.float32)
    masks, sizes = label_masks(labels, num_clusters)
    present = sizes > 0
    centroids = _centroids(features, masks, sizes)
    to_own = safe_norm(features - masks.T @ centroids, axis=1)  # each sample to its centroid
    spread = (masks @ to_own) / jnp.where(present, sizes, 1.0)
    separation = _pairwise_distances(centroids)

    # Pairs of distinct non-empty clusters, the diagonal excluded by index: under jit a
    # centroid's distance to itself need not come out exactly 0.
    pair = present[:, None] & present[None, :] & ~jnp.eye(present.shape[0], dtype=bool)
    apart = pair & (separation > 0)  # coinciding centroids add no ratio, as in scikit-learn
    ratios = jnp.where(
        apart, (spread[:, None] + spread[None, :]) / jnp.where(apart, separation, 1.0), 0.0
    )
    worst = jnp.max(ratios, axis=1)
    score = jnp.sum(jnp.where(present, worst, 0.0)) / jnp.sum(present)
    degenerate = jnp.all(jnp.where(present, spread, 0.0) <= 0) | ~jnp.any(apart)
    return jnp.where(degenerate, 0.0, score)
