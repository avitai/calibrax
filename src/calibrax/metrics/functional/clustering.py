"""Clustering evaluation metrics for unsupervised learning.

Pure functions for evaluating clustering quality. Divided into two categories:

**External evaluation** (requires ground truth labels):
- adjusted_rand_index, normalized_mutual_information_clustering,
  adjusted_mutual_information, v_measure

**Internal evaluation** (no ground truth, uses feature distances):
- silhouette_score, calinski_harabasz_score, davies_bouldin_score

All accept integer label arrays. Internal metrics additionally require
a feature matrix. Registered with ``domain="clustering"`` and
``signature=MetricSignature.FEATURES_LABELS``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON, safe_divide


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _contingency_table(
    labels_true: Any,
    labels_pred: Any,
) -> tuple[Any, Any, Any]:
    """Build contingency table from two label arrays.

    Args:
        labels_true: Ground truth integer labels, shape (n,).
        labels_pred: Predicted integer labels, shape (n,).

    Returns:
        Tuple of (contingency_matrix, classes_true, classes_pred).
    """
    lt = jnp.asarray(labels_true, dtype=jnp.int32)
    lp = jnp.asarray(labels_pred, dtype=jnp.int32)
    classes_true = jnp.unique(lt, size=lt.shape[0])
    classes_pred = jnp.unique(lp, size=lp.shape[0])
    # Remove padding from jnp.unique (size= pads with max value)
    classes_true = jnp.sort(jnp.unique(lt, size=int(jnp.max(lt)) + 1))
    classes_pred = jnp.sort(jnp.unique(lp, size=int(jnp.max(lp)) + 1))

    # Broadcasting: compare each sample against each class
    lt_match = lt[None, :] == classes_true[:, None]  # (n_true, n_samples)
    lp_match = lp[None, :] == classes_pred[:, None]  # (n_pred, n_samples)
    contingency = jnp.sum(lt_match[:, None, :] & lp_match[None, :, :], axis=2)  # (n_true, n_pred)
    contingency = contingency.astype(jnp.float32)
    return contingency, classes_true, classes_pred


def _entropy_from_counts(counts: Any) -> Any:
    """Compute Shannon entropy from a count array.

    Args:
        counts: Array of counts.

    Returns:
        Entropy in nats.
    """
    n = jnp.sum(counts)
    probs = counts / (n + _EPSILON)
    probs = jnp.asarray(jnp.where(probs > 0, probs, 1.0))  # avoid log(0)
    return -jnp.sum(counts / (n + _EPSILON) * jnp.log(probs))


def _pairwise_distances(features: Any) -> Any:
    """Compute pairwise Euclidean distance matrix.

    Args:
        features: Feature matrix of shape (n, d).

    Returns:
        Distance matrix of shape (n, n).
    """
    diff = features[:, None, :] - features[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1) + _EPSILON)


# ---------------------------------------------------------------------------
# External evaluation metrics
# ---------------------------------------------------------------------------


def adjusted_rand_index(labels_true: Any, labels_pred: Any) -> Any:
    """Adjusted Rand Index for clustering agreement.

    Measures similarity between two clusterings, corrected for chance.
    ARI = (RI - E[RI]) / (max(RI) - E[RI]).

    Args:
        labels_true: Ground truth integer labels, shape (n,).
        labels_pred: Predicted integer labels, shape (n,).

    Returns:
        ARI value in [-1, 1]. 1.0 = perfect agreement, 0.0 = random,
        negative = worse than random.

    Examples:
        >>> import jax.numpy as jnp
        >>> adjusted_rand_index(jnp.array([0, 0, 1, 1]), jnp.array([0, 0, 1, 1]))
        1.0
    """
    contingency, _, _ = _contingency_table(labels_true, labels_pred)
    n = jnp.sum(contingency)

    # Sum of C(n_ij, 2) over all cells
    sum_comb_c = jnp.sum(contingency * (contingency - 1) / 2.0)

    # Row and column sums
    row_sums = jnp.sum(contingency, axis=1)
    col_sums = jnp.sum(contingency, axis=0)

    sum_comb_a = jnp.sum(row_sums * (row_sums - 1) / 2.0)
    sum_comb_b = jnp.sum(col_sums * (col_sums - 1) / 2.0)

    total_comb = n * (n - 1) / 2.0
    expected = safe_divide(sum_comb_a * sum_comb_b, total_comb)
    max_index = (sum_comb_a + sum_comb_b) / 2.0

    numerator = sum_comb_c - expected
    denominator = max_index - expected

    return safe_divide(numerator, denominator)


def normalized_mutual_information_clustering(
    labels_true: Any,
    labels_pred: Any,
    *,
    average: str = "arithmetic",
) -> Any:
    """Normalized Mutual Information for clustering.

    MI(true, pred) / normalizer. Range [0, 1].

    Args:
        labels_true: Ground truth integer labels, shape (n,).
        labels_pred: Predicted integer labels, shape (n,).
        average: Normalizer type. One of "arithmetic" (default), "geometric",
            "min", "max".

    Returns:
        NMI value in [0, 1]. 1.0 = perfect agreement.

    Raises:
        ValueError: If average is not one of the supported options.

    Examples:
        >>> import jax.numpy as jnp
        >>> normalized_mutual_information_clustering(
        ...     jnp.array([0, 0, 1, 1]), jnp.array([0, 0, 1, 1])
        ... )
        1.0
    """
    valid = {"arithmetic", "geometric", "min", "max"}
    if average not in valid:
        msg = f"average must be one of {valid}, got '{average}'"
        raise ValueError(msg)

    contingency, _, _ = _contingency_table(labels_true, labels_pred)
    n = jnp.sum(contingency)

    row_sums = jnp.sum(contingency, axis=1)
    col_sums = jnp.sum(contingency, axis=0)

    # Mutual information from contingency
    outer = row_sums[:, None] * col_sums[None, :]
    log_term = jnp.where(
        contingency > 0,
        jnp.log(contingency * n / (outer + _EPSILON) + _EPSILON),
        0.0,
    )
    mi = jnp.sum((contingency / n) * log_term)

    h_true = _entropy_from_counts(row_sums)
    h_pred = _entropy_from_counts(col_sums)

    if h_true < _EPSILON and h_pred < _EPSILON:
        return jnp.float32(1.0)

    if average == "arithmetic":
        normalizer = (h_true + h_pred) / 2.0
    elif average == "geometric":
        normalizer = jnp.sqrt(h_true * h_pred + _EPSILON)
    elif average == "min":
        normalizer = jnp.minimum(h_true, h_pred)
    else:  # max
        normalizer = jnp.maximum(h_true, h_pred)

    return safe_divide(mi, normalizer)


def adjusted_mutual_information(
    labels_true: Any,
    labels_pred: Any,
) -> Any:
    """Adjusted Mutual Information for clustering.

    Chance-adjusted version of NMI: AMI = (MI - E[MI]) / (max(H_true, H_pred) - E[MI]).
    More robust than NMI for comparing clusterings of different sizes.

    Args:
        labels_true: Ground truth integer labels, shape (n,).
        labels_pred: Predicted integer labels, shape (n,).

    Returns:
        AMI value in [-1, 1]. 1.0 = perfect agreement,
        0.0 = random labeling.

    Examples:
        >>> import jax.numpy as jnp
        >>> adjusted_mutual_information(
        ...     jnp.array([0, 0, 1, 1]), jnp.array([0, 0, 1, 1])
        ... )
        1.0
    """
    contingency, _, _ = _contingency_table(labels_true, labels_pred)
    n = jnp.sum(contingency)

    row_sums = jnp.sum(contingency, axis=1)
    col_sums = jnp.sum(contingency, axis=0)

    # MI from contingency
    outer = row_sums[:, None] * col_sums[None, :]
    log_term = jnp.where(
        contingency > 0,
        jnp.log(contingency * n / (outer + _EPSILON) + _EPSILON),
        0.0,
    )
    mi = jnp.sum((contingency / n) * log_term)

    h_true = _entropy_from_counts(row_sums)
    h_pred = _entropy_from_counts(col_sums)

    # Expected MI under random model (simplified approximation)
    # E[MI] ≈ (R-1)(C-1)/(2N)
    r = jnp.sum(row_sums > 0)
    c = jnp.sum(col_sums > 0)
    emi = (r - 1) * (c - 1) / (2.0 * n + _EPSILON)

    normalizer = jnp.maximum(h_true, h_pred) - emi
    ami = (mi - emi) / (normalizer + _EPSILON)

    # Edge cases: trivial clusterings or degenerate normalizer
    trivial = (h_true < _EPSILON) & (h_pred < _EPSILON)
    degenerate = jnp.abs(normalizer) < _EPSILON
    return jnp.where(trivial, 1.0, jnp.where(degenerate, 0.0, ami))


def v_measure(
    labels_true: Any,
    labels_pred: Any,
    *,
    beta: float = 1.0,
) -> Any:
    """V-measure: harmonic mean of homogeneity and completeness.

    Equivalent to NMI with arithmetic normalizer when beta=1.0.
    beta > 1 weights completeness more, beta < 1 weights homogeneity more.

    Args:
        labels_true: Ground truth integer labels, shape (n,).
        labels_pred: Predicted integer labels, shape (n,).
        beta: Weight parameter. 1.0 = equal weight. >1 = favor completeness.

    Returns:
        V-measure in [0, 1]. 1.0 = perfect clustering.

    Examples:
        >>> import jax.numpy as jnp
        >>> v_measure(jnp.array([0, 0, 1, 1]), jnp.array([0, 0, 1, 1]))
        1.0
    """
    contingency, _, _ = _contingency_table(labels_true, labels_pred)
    n = jnp.sum(contingency)

    row_sums = jnp.sum(contingency, axis=1)
    col_sums = jnp.sum(contingency, axis=0)

    h_true = _entropy_from_counts(row_sums)
    h_pred = _entropy_from_counts(col_sums)

    # H(C|K) — conditional entropy of classes given clusters (matrix operation)
    col_probs = contingency / (col_sums[None, :] + _EPSILON)
    col_probs_safe = jnp.asarray(jnp.where(col_probs > 0, col_probs, 1.0))
    h_c_given_k = -jnp.sum((contingency / (n + _EPSILON)) * jnp.log(col_probs_safe))

    # H(K|C) — conditional entropy of clusters given classes (matrix operation)
    row_probs = contingency / (row_sums[:, None] + _EPSILON)
    row_probs_safe = jnp.asarray(jnp.where(row_probs > 0, row_probs, 1.0))
    h_k_given_c = -jnp.sum((contingency / (n + _EPSILON)) * jnp.log(row_probs_safe))

    # Homogeneity: 1 - H(C|K) / H(C)
    homogeneity = jnp.asarray(
        jnp.where(h_true > _EPSILON, 1.0 - h_c_given_k / (h_true + _EPSILON), 1.0)
    )
    # Completeness: 1 - H(K|C) / H(K)
    completeness = jnp.asarray(
        jnp.where(h_pred > _EPSILON, 1.0 - h_k_given_c / (h_pred + _EPSILON), 1.0)
    )

    v = (
        (1 + beta**2)
        * homogeneity
        * completeness
        / (beta**2 * homogeneity + completeness + _EPSILON)
    )
    return jnp.where(homogeneity + completeness < _EPSILON, 0.0, v)


# ---------------------------------------------------------------------------
# Internal evaluation metrics
# ---------------------------------------------------------------------------


def silhouette_score(features: Any, labels: Any) -> Any:
    """Mean silhouette coefficient across all samples.

    For each sample: s = (b - a) / max(a, b) where a = mean intra-cluster
    distance and b = mean nearest-cluster distance. O(n^2) complexity.

    Args:
        features: Feature matrix of shape (n, d).
        labels: Cluster assignment integer labels, shape (n,).

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
    labels = jnp.asarray(labels, dtype=jnp.int32)
    n = features.shape[0]

    distances = _pairwise_distances(features)
    unique_labels = jnp.unique(labels, size=int(jnp.max(labels)) + 1)
    n_clusters = unique_labels.shape[0]

    # Build per-cluster masks: (n_clusters, n)
    cluster_masks = labels[None, :] == unique_labels[:, None]
    cluster_sizes = jnp.sum(cluster_masks, axis=1)  # (n_clusters,)

    # Mean distance from each sample to each cluster: (n, n_clusters)
    dist_to_clusters = jnp.sum(distances[:, None, :] * cluster_masks[None, :, :], axis=2) / (
        cluster_sizes[None, :] + _EPSILON
    )

    # For own cluster: a_i (mean intra-cluster distance, excluding self)
    sample_cluster = jnp.argmax(cluster_masks[:, jnp.arange(n)].T, axis=1)  # (n,)
    own_sizes = cluster_sizes[sample_cluster]
    a_i = jnp.sum(distances * cluster_masks[sample_cluster], axis=1) / (own_sizes - 1 + _EPSILON)

    # b_i: min mean distance to any OTHER cluster
    own_mask = jnp.eye(n_clusters, dtype=bool)[sample_cluster]  # (n, n_clusters)
    other_dists = jnp.where(own_mask, jnp.inf, dist_to_clusters)
    b_i = jnp.min(other_dists, axis=1)
    b_i = jnp.where(jnp.isinf(b_i), 0.0, b_i)

    denom = jnp.maximum(a_i, b_i) + _EPSILON
    silhouettes = (b_i - a_i) / denom
    return jnp.mean(silhouettes)


def calinski_harabasz_score(features: Any, labels: Any) -> Any:
    """Calinski-Harabasz Index (Variance Ratio Criterion).

    Ratio of between-cluster to within-cluster dispersion, adjusted
    for cluster and sample counts. Higher = better-separated clusters.

    Args:
        features: Feature matrix of shape (n, d).
        labels: Cluster assignment integer labels, shape (n,).

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
    labels = jnp.asarray(labels, dtype=jnp.int32)
    n = features.shape[0]

    global_centroid = jnp.mean(features, axis=0)
    unique_labels = jnp.unique(labels, size=int(jnp.max(labels)) + 1)
    k = unique_labels.shape[0]

    if k <= 1 or n <= k:
        return jnp.float32(0.0)

    # Vectorized cluster masks and centroids
    cluster_masks = labels[None, :] == unique_labels[:, None]  # (k, n)
    cluster_sizes = jnp.sum(cluster_masks, axis=1)  # (k,)

    masked_features = features[None, :, :] * cluster_masks[:, :, None]  # (k, n, d)
    centroids = jnp.sum(masked_features, axis=1) / (cluster_sizes[:, None] + _EPSILON)  # (k, d)

    # BGSS: between-group sum of squares
    bgss = jnp.sum(cluster_sizes * jnp.sum((centroids - global_centroid) ** 2, axis=1))

    # WGSS: within-group sum of squares
    sample_centroids = centroids[jnp.argmax(cluster_masks.T, axis=1)]  # (n, d)
    wgss = jnp.sum((features - sample_centroids) ** 2)

    return safe_divide(
        bgss * (n - k),
        wgss * (k - 1),
    )


def davies_bouldin_score(features: Any, labels: Any) -> Any:
    """Davies-Bouldin Index for cluster separation.

    For each cluster, finds the worst-case similarity ratio with another
    cluster. Lower = better separated clusters.

    Args:
        features: Feature matrix of shape (n, d).
        labels: Cluster assignment integer labels, shape (n,).

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
    labels = jnp.asarray(labels, dtype=jnp.int32)

    unique_labels = jnp.unique(labels, size=int(jnp.max(labels)) + 1)
    k = unique_labels.shape[0]

    if k <= 1:
        return jnp.float32(0.0)

    # Vectorized cluster masks and centroids
    cluster_masks = labels[None, :] == unique_labels[:, None]  # (k, n)
    cluster_sizes = jnp.sum(cluster_masks, axis=1)  # (k,)

    masked_features = features[None, :, :] * cluster_masks[:, :, None]  # (k, n, d)
    centroids = jnp.sum(masked_features, axis=1) / (cluster_sizes[:, None] + _EPSILON)  # (k, d)

    # Intra-cluster distances: mean distance from each sample to its centroid
    diffs = features[None, :, :] - centroids[:, None, :]  # (k, n, d)
    per_sample_dist = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + _EPSILON)  # (k, n)
    masked_dists = per_sample_dist * cluster_masks  # (k, n)
    s_values = jnp.sum(masked_dists, axis=1) / (cluster_sizes + _EPSILON)  # (k,)

    # Pairwise centroid distances
    centroid_diffs = centroids[:, None, :] - centroids[None, :, :]  # (k, k, d)
    centroid_dists = jnp.sqrt(jnp.sum(centroid_diffs**2, axis=-1) + _EPSILON)  # (k, k)

    # Similarity ratios: (s_i + s_j) / d_ij
    ratios = (s_values[:, None] + s_values[None, :]) / (centroid_dists + _EPSILON)
    ratios = jnp.where(jnp.eye(k, dtype=bool), -jnp.inf, ratios)
    max_ratios = jnp.max(ratios, axis=1)  # (k,)
    return jnp.mean(max_ratios)
