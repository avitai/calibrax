"""Scientific metrics for molecular and protein modeling.

Mostly pure JAX math -- no external chemistry dependencies required.
Designed for evaluating molecular geometry generation, binding affinity
prediction, and conformational sampling.

Typical use cases:
- Molecular dynamics simulations
- Drug discovery pipelines
- Protein structure prediction
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from calibrax.metrics.functional.regression import mae, mse, r_squared
from calibrax.metrics.functional.statistical import pearson_correlation


def chemical_validity(
    bond_lengths: jnp.ndarray,
    bond_angles: jnp.ndarray,
    *,
    length_thresholds: tuple[float, float] = (0.8, 2.0),
    angle_thresholds: tuple[float, float] = (1.5, 3.5),
) -> Any:
    """Fraction of bonds and angles within acceptable physical ranges.

    Checks if generated molecular geometries are physically plausible
    by validating bond lengths (in Angstroms) and bond angles (in radians)
    against threshold ranges.

    Args:
        bond_lengths: 1D array of bond lengths in Angstroms.
        bond_angles: 1D array of bond angles in radians.
        length_thresholds: (min, max) acceptable bond length range.
            Defaults to (0.8, 2.0) Angstroms.
        angle_thresholds: (min, max) acceptable bond angle range.
            Defaults to (1.5, 3.5) radians (~86-200 degrees).

    Returns:
        Fraction of valid bonds and angles in [0, 1]. Higher is better.

    Examples:
        >>> lengths = jnp.array([1.0, 1.2, 1.5])
        >>> angles = jnp.array([2.0, 2.5, 3.0])
        >>> chemical_validity(lengths, angles)
        1.0
    """
    valid_lengths = (bond_lengths >= length_thresholds[0]) & (bond_lengths <= length_thresholds[1])
    valid_angles = (bond_angles >= angle_thresholds[0]) & (bond_angles <= angle_thresholds[1])

    all_valid = jnp.concatenate([valid_lengths, valid_angles])
    return jnp.mean(all_valid)


def binding_affinity_metrics(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
) -> dict[str, Any]:
    """Compute regression metrics for binding affinity predictions.

    Evaluates pKd/pKi predictions using standard regression metrics.
    Delegates to existing regression and statistical functions (DRY).

    Args:
        predictions: Predicted binding affinity values (e.g., pKd).
        targets: Ground truth binding affinity values.

    Returns:
        Dictionary with binding_mse, binding_mae, binding_r_squared,
        and binding_pearson values.

    Examples:
        >>> preds = jnp.array([6.5, 7.2, 8.1])
        >>> targets = jnp.array([6.8, 7.0, 8.3])
        >>> result = binding_affinity_metrics(preds, targets)
        >>> result["binding_mse"]  # MSE of predictions
    """
    return {
        "binding_mse": mse(predictions, targets),
        "binding_mae": mae(predictions, targets),
        "binding_r_squared": r_squared(predictions, targets),
        "binding_pearson": pearson_correlation(predictions, targets),
    }


def conformational_diversity(
    coordinates: jnp.ndarray,
) -> Any:
    """Mean pairwise RMSD across molecular conformations.

    Measures diversity of molecular conformations by computing the
    root mean square deviation (RMSD) between all pairs of structures.

    Args:
        coordinates: Array of shape (num_conformations, num_atoms, 3).
            Each conformation is a set of 3D atomic coordinates.

    Returns:
        Mean pairwise RMSD in the same units as input coordinates
        (typically Angstroms). Higher means more diverse conformations.

    Examples:
        >>> coords = jnp.array([
        ...     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ...     [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
        ... ])
        >>> conformational_diversity(coords)  # > 0.0
    """
    n = coordinates.shape[0]
    if n < 2:
        return 0.0

    # Broadcasting: (n, 1, atoms, 3) - (1, n, atoms, 3) -> (n, n, atoms, 3)
    diffs = coordinates[:, None] - coordinates[None, :]
    per_pair_rmsd = jnp.sqrt(jnp.mean(jnp.sum(diffs**2, axis=-1), axis=-1))  # (n, n)
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    total_rmsd = jnp.sum(per_pair_rmsd * mask)
    count = n * (n - 1) // 2
    return total_rmsd / count
