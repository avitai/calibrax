"""Sample-based evaluation of generative models: fidelity, diversity and privacy.

All functions take feature matrices ``(n_samples, n_features)`` of real and
generated samples (raw values or the output of a feature extractor) and are pure
``jax.numpy`` computations.

References:
    * Kynkaanniemi et al. 2019, "Improved precision and recall metric for assessing
      generative models": k-nearest-neighbour manifold precision and recall.
    * Distance to closest record and exact-match rates are the standard privacy
      checks for synthetic tabular data.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp

from calibrax.metrics._utils import _EPSILON


# A sample covariance needs at least two samples; with one, the estimate divides by zero.
_MIN_COVARIANCE_SAMPLES = 2


def _prepare_feature_arrays(real: Any, generated: Any) -> tuple[Any, Any]:
    """Validate two feature matrices share their feature dimension and convert them.

    Args:
        real: Real features with shape ``(n_real, n_features)``.
        generated: Generated features with shape ``(n_generated, n_features)``.

    Returns:
        Tuple of float32 JAX arrays ``(real, generated)``.

    Raises:
        ValueError: If either input is not two-dimensional or the feature
            dimensions differ.
    """
    real_features = jnp.asarray(real, dtype=jnp.float32)
    generated_features = jnp.asarray(generated, dtype=jnp.float32)
    if real_features.ndim != 2 or generated_features.ndim != 2:  # noqa: PLR2004
        msg = (
            "features must be 2-dimensional (samples, features), got "
            f"{real_features.shape} and {generated_features.shape}"
        )
        raise ValueError(msg)
    if real_features.shape[1] != generated_features.shape[1]:
        msg = (
            "real and generated features must share their feature dimension: "
            f"{real_features.shape[1]} != {generated_features.shape[1]}"
        )
        raise ValueError(msg)
    return real_features, generated_features


def _pairwise_squared_distances(query: Any, data: Any) -> Any:
    """Squared Euclidean distances with shape ``(n_query, n_data)``."""
    return jnp.sum((query[:, None, :] - data[None, :, :]) ** 2, axis=-1)


def _nearest_neighbors(query: Any, data: Any, k: int) -> tuple[Any, Any]:
    """Distances and indices of the ``k`` nearest data points of each query point.

    Args:
        query: Query points with shape ``(n_query, n_features)``.
        data: Data points with shape ``(n_data, n_features)``.
        k: Number of neighbours; capped at ``n_data``.

    Returns:
        Tuple ``(distances, indices)`` with shape ``(n_query, min(k, n_data))``,
        sorted by increasing distance.
    """
    squared = _pairwise_squared_distances(query, data)
    indices = jnp.argsort(squared, axis=1)[:, : min(k, data.shape[0])]
    return jnp.sqrt(jnp.take_along_axis(squared, indices, axis=1)), indices


def manifold_radii(features: Any, *, k: int) -> Any:
    """Distance from each point to its ``k``-th nearest other point.

    The radius of the ball that approximates the data manifold around each point in
    Kynkaanniemi et al. 2019; the point itself (distance zero) is excluded.

    Args:
        features: Points with shape ``(n_samples, n_features)``.
        k: Neighbour order, at least 1 and below ``n_samples``.

    Returns:
        Radii with shape ``(n_samples,)``.
    """
    points = jnp.asarray(features, dtype=jnp.float32)
    distances, _ = _nearest_neighbors(points, points, k + 1)
    return distances[:, k]


def _covered_fraction(queries: Any, manifold: Any, *, k: int) -> Any:
    """Fraction of ``queries`` inside the k-NN manifold of ``manifold`` points."""
    radii = manifold_radii(manifold, k=k)
    distances, indices = _nearest_neighbors(queries, manifold, 1)
    return jnp.mean((distances[:, 0] <= radii[indices[:, 0]]).astype(jnp.float32))


def manifold_precision(real: Any, generated: Any, *, k: int = 3) -> Any:  # noqa: DOC502  # raised by _prepare_feature_arrays
    """Fraction of generated samples inside the real k-NN manifold (fidelity).

    Note:
        Direction: HIGHER (1.0 = every generated sample looks real).
        Range: [0, 1].

    Args:
        real: Real features with shape ``(n_real, n_features)``.
        generated: Generated features with shape ``(n_generated, n_features)``.
        k: Neighbour order of the manifold estimate.

    Returns:
        Scalar precision as a JAX array.

    Raises:
        ValueError: If the feature matrices are not compatible.
    """
    real_features, generated_features = _prepare_feature_arrays(real, generated)
    return _covered_fraction(generated_features, real_features, k=k)


def manifold_recall(real: Any, generated: Any, *, k: int = 3) -> Any:  # noqa: DOC502  # raised by _prepare_feature_arrays
    """Fraction of real samples inside the generated k-NN manifold (diversity).

    Note:
        Direction: HIGHER (1.0 = every real sample is covered).
        Range: [0, 1].

    Args:
        real: Real features with shape ``(n_real, n_features)``.
        generated: Generated features with shape ``(n_generated, n_features)``.
        k: Neighbour order of the manifold estimate.

    Returns:
        Scalar recall as a JAX array.

    Raises:
        ValueError: If the feature matrices are not compatible.
    """
    real_features, generated_features = _prepare_feature_arrays(real, generated)
    return _covered_fraction(real_features, generated_features, k=k)


def _density_weighted_coverage(queries: Any, manifold: Any, *, k: int) -> Any:
    """Coverage of ``queries`` by the ``manifold`` balls, weighted by local density.

    Each query counts with the density ``1 / r_k`` of its nearest manifold point,
    normalised over the queries, so dense regions of the manifold dominate.
    """
    radii = manifold_radii(manifold, k=k)
    densities = 1.0 / (radii + _EPSILON)
    distances, indices = _nearest_neighbors(queries, manifold, 1)
    nearest = indices[:, 0]
    weights = densities[nearest] / jnp.sum(densities[nearest])
    inside = (distances[:, 0] <= radii[nearest]).astype(jnp.float32)
    return jnp.sum(inside * weights)


def density_weighted_precision(real: Any, generated: Any, *, k: int = 5) -> Any:  # noqa: DOC502  # raised by _prepare_feature_arrays
    """Manifold precision weighted by the density of the real manifold.

    Generated samples whose nearest real point sits in a dense region weigh more,
    which makes the score robust to real outliers with large radii.

    Note:
        Direction: HIGHER.
        Range: [0, 1].

    Args:
        real: Real features with shape ``(n_real, n_features)``.
        generated: Generated features with shape ``(n_generated, n_features)``.
        k: Neighbour order of the density estimate.

    Returns:
        Scalar weighted precision as a JAX array.

    Raises:
        ValueError: If the feature matrices are not compatible.
    """
    real_features, generated_features = _prepare_feature_arrays(real, generated)
    return _density_weighted_coverage(generated_features, real_features, k=k)


def density_weighted_recall(real: Any, generated: Any, *, k: int = 5) -> Any:  # noqa: DOC502  # raised by _prepare_feature_arrays
    """Manifold recall weighted by the density of the generated manifold.

    Note:
        Direction: HIGHER.
        Range: [0, 1].

    Args:
        real: Real features with shape ``(n_real, n_features)``.
        generated: Generated features with shape ``(n_generated, n_features)``.
        k: Neighbour order of the density estimate.

    Returns:
        Scalar weighted recall as a JAX array.

    Raises:
        ValueError: If the feature matrices are not compatible.
    """
    real_features, generated_features = _prepare_feature_arrays(real, generated)
    return _density_weighted_coverage(real_features, generated_features, k=k)


def distance_to_closest_record(real: Any, generated: Any) -> Any:  # noqa: DOC502  # raised by _prepare_feature_arrays
    """Mean squared distance from each generated record to its closest real record.

    Features are min-max normalised over the union of both sets first, so every
    feature contributes on the same scale. Larger values mean the generated records
    sit further from any real record, which is what a privacy check wants.

    Note:
        Direction: HIGHER (0.0 = every generated record copies a real one).
        Range: [0, n_features].

    Args:
        real: Real records with shape ``(n_real, n_features)``.
        generated: Generated records with shape ``(n_generated, n_features)``.

    Returns:
        Scalar mean closest-record distance as a JAX array.

    Raises:
        ValueError: If the feature matrices are not compatible.
    """
    real_features, generated_features = _prepare_feature_arrays(real, generated)
    combined = jnp.concatenate([real_features, generated_features], axis=0)
    low = jnp.min(combined, axis=0)
    span = jnp.max(combined, axis=0) - low + _EPSILON
    squared = _pairwise_squared_distances(
        (generated_features - low) / span, (real_features - low) / span
    )
    return jnp.mean(jnp.min(squared, axis=1))


def memorization_rate(real: Any, generated: Any) -> Any:  # noqa: DOC502  # raised by _prepare_feature_arrays
    """Fraction of generated records that exactly equal some real record.

    Note:
        Direction: LOWER (0.0 = nothing copied).
        Range: [0, 1].

    Args:
        real: Real records with shape ``(n_real, n_features)``.
        generated: Generated records with shape ``(n_generated, n_features)``.

    Returns:
        Scalar exact-match rate as a JAX array.

    Raises:
        ValueError: If the feature matrices are not compatible.
    """
    real_features, generated_features = _prepare_feature_arrays(real, generated)
    matches = jnp.all(generated_features[:, None, :] == real_features[None, :, :], axis=-1)
    return jnp.mean(jnp.any(matches, axis=1).astype(jnp.float32))


def frechet_distance(mean_a: Any, cov_a: Any, mean_b: Any, cov_b: Any) -> Any:
    """Fréchet distance between two Gaussians given their means and covariances.

    ``|mu_a - mu_b|^2 + Tr(S_a + S_b - 2 (S_a^{1/2} S_b S_a^{1/2})^{1/2})``. The
    cross term is computed through the symmetric sandwich and the
    eigendecomposition-based :func:`calibrax.metrics._utils.matrix_sqrtm`, which
    stays on backends where ``jax.scipy.linalg.sqrtm`` has no kernel. This is the
    core of the Fréchet Inception Distance once features have been extracted.
    In float32 this form loses digits when the covariances are ill-conditioned;
    from feature matrices, :func:`frechet_feature_distance` avoids forming them.

    Note:
        Direction: LOWER (0.0 = identical Gaussians).
        Range: [0, inf).

    Args:
        mean_a: Mean of the first Gaussian with shape ``(d,)``.
        cov_a: Covariance of the first Gaussian with shape ``(d, d)``.
        mean_b: Mean of the second Gaussian with shape ``(d,)``.
        cov_b: Covariance of the second Gaussian with shape ``(d, d)``.

    Returns:
        Scalar distance as a JAX array, floored at zero.
    """
    from calibrax.metrics._utils import matrix_sqrtm

    def _symmetrize(matrix: Any) -> Any:
        return 0.5 * (matrix + matrix.T)

    cov_a = _symmetrize(jnp.asarray(cov_a, dtype=jnp.float32))
    cov_b = _symmetrize(jnp.asarray(cov_b, dtype=jnp.float32))
    diff = jnp.asarray(mean_a, dtype=jnp.float32) - jnp.asarray(mean_b, dtype=jnp.float32)
    sqrt_cov_a = matrix_sqrtm(cov_a)
    cross = matrix_sqrtm(_symmetrize(sqrt_cov_a @ cov_b @ sqrt_cov_a))
    distance = jnp.sum(diff**2) + jnp.trace(cov_a) + jnp.trace(cov_b) - 2.0 * jnp.trace(cross)
    return jnp.maximum(distance, 0.0)


def frechet_feature_distance(real: Any, generated: Any) -> Any:
    """Fréchet distance between the Gaussians fitted to two feature matrices.

    With Inception features this is the Fréchet Inception Distance; with any
    other extractor it is the same fidelity measure in that feature space.

    The covariances are never formed. With thin QR factors ``X - mean = Q R`` of
    each feature matrix, ``Tr(S) = ||R||_F^2 / (n - 1)``, and the cross term
    ``Tr((S_r S_g)^{1/2})`` is the sum of the singular values of ``R_r R_g^T``
    divided by ``sqrt((n_r - 1)(n_g - 1))``. Correlated features give covariances
    with condition numbers near 1e7, where float32 square roots of covariance
    matrices lose digits and differ between LAPACK builds; the factored form
    stays within about 1e-6 of a float64 reference.

    Note:
        Direction: LOWER (0.0 = identical feature distributions).
        Range: [0, inf).

    Args:
        real: Real features with shape ``(n_real, n_features)``.
        generated: Generated features with shape ``(n_generated, n_features)``.

    Returns:
        Scalar distance as a JAX array.

    Raises:
        ValueError: If the feature matrices are not compatible, or either has fewer
            than two samples.
    """
    real_features, generated_features = _prepare_feature_arrays(real, generated)
    n_real, n_generated = real_features.shape[0], generated_features.shape[0]
    if min(n_real, n_generated) < _MIN_COVARIANCE_SAMPLES:
        msg = (
            "frechet_feature_distance needs at least two samples per side to estimate a "
            f"covariance, got {n_real} real and {n_generated} generated"
        )
        raise ValueError(msg)
    real_mean = jnp.mean(real_features, axis=0)
    generated_mean = jnp.mean(generated_features, axis=0)
    real_factor = jnp.linalg.qr(real_features - real_mean, mode="r")
    generated_factor = jnp.linalg.qr(generated_features - generated_mean, mode="r")
    real_dof, generated_dof = n_real - 1, n_generated - 1
    cross = jnp.sum(jnp.linalg.svd(real_factor @ generated_factor.T, compute_uv=False))
    distance = (
        jnp.sum((real_mean - generated_mean) ** 2)
        + jnp.sum(real_factor**2) / real_dof
        + jnp.sum(generated_factor**2) / generated_dof
        - 2.0 * cross / math.sqrt(real_dof * generated_dof)
    )
    return jnp.maximum(distance, 0.0)


def inception_score_per_split(probabilities: Any, *, splits: int = 10) -> Any:
    """Inception score of each of ``splits`` equal chunks of class probabilities.

    Each chunk's score is ``exp(E_x KL(p(y|x) || p(y)))`` with ``p(y)`` the chunk's
    marginal; the spread across chunks is the score's usual error bar.

    Args:
        probabilities: Class probabilities with shape ``(n_samples, n_classes)``.
        splits: Number of equal chunks; rows beyond ``splits * (n // splits)`` are dropped.

    Returns:
        Per-chunk scores with shape ``(splits,)``.

    Raises:
        ValueError: If ``splits`` is not between 1 and the number of samples.
    """
    p = jnp.asarray(probabilities, dtype=jnp.float32)
    n_samples = p.shape[0]
    if splits < 1 or splits > n_samples:
        msg = f"splits must be between 1 and the number of samples ({n_samples}), got {splits}"
        raise ValueError(msg)
    split_size = n_samples // splits
    chunks = p[: splits * split_size].reshape(splits, split_size, -1)
    safe = jnp.maximum(chunks, _EPSILON)
    marginal = jnp.maximum(jnp.mean(chunks, axis=1, keepdims=True), _EPSILON)
    kl = jnp.sum(safe * (jnp.log(safe) - jnp.log(marginal)), axis=-1)
    return jnp.exp(jnp.mean(kl, axis=1))


def inception_score(probabilities: Any, *, splits: int = 10) -> Any:  # noqa: DOC502  # raised by inception_score_per_split
    """Inception score: mean over ``splits`` chunks of ``exp(E KL(p(y|x) || p(y)))``.

    Note:
        Direction: HIGHER (up to the number of classes, reached by confident and
        evenly spread predictions).
        Range: [1, n_classes].

    Args:
        probabilities: Class probabilities with shape ``(n_samples, n_classes)``.
        splits: Number of equal chunks the score is averaged over.

    Returns:
        Scalar score as a JAX array.

    Raises:
        ValueError: If ``splits`` is not between 1 and the number of samples.
    """
    return jnp.mean(inception_score_per_split(probabilities, splits=splits))
