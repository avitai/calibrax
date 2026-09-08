"""Tests for the generative domain (Kynkaanniemi et al. 2019 precision/recall,
density-weighted variants, and the tabular privacy checks)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from calibrax.metrics.functional.generative import (
    density_weighted_precision,
    density_weighted_recall,
    distance_to_closest_record,
    manifold_precision,
    manifold_radii,
    manifold_recall,
    memorization_rate,
)


def _blobs(seed: int, center: tuple[float, float], scale: float, n: int = 100) -> jax.Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.array(center) + scale * rng.standard_normal((n, 2)))


@pytest.fixture
def real() -> jax.Array:
    return jnp.concatenate([_blobs(42, (0.0, 0.0), 0.5, 50), _blobs(43, (4.0, 4.0), 0.5, 50)])


@pytest.fixture
def far() -> jax.Array:
    return _blobs(44, (10.0, 10.0), 0.5)


class TestManifoldRadii:
    def test_unit_grid(self) -> None:
        grid = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert bool(jnp.allclose(manifold_radii(grid, k=1), 1.0, rtol=1e-5))
        assert bool(jnp.allclose(manifold_radii(grid, k=2), 1.0, rtol=1e-5))
        assert bool(jnp.allclose(manifold_radii(grid, k=3), jnp.sqrt(2.0), rtol=1e-5))


class TestManifoldPrecisionRecall:
    def test_identical_samples_score_one(self, real: jax.Array) -> None:
        assert manifold_precision(real, real, k=3) == pytest.approx(1.0, abs=1e-6)
        assert manifold_recall(real, real, k=3) == pytest.approx(1.0, abs=1e-6)

    def test_far_samples_score_near_zero(self, real: jax.Array, far: jax.Array) -> None:
        assert float(manifold_precision(real, far, k=3)) < 0.1
        assert float(manifold_recall(real, far, k=3)) < 0.1

    def test_one_covered_mode_has_high_precision_and_partial_recall(self, real: jax.Array) -> None:
        generated = _blobs(45, (0.0, 0.0), 0.7)
        assert float(manifold_precision(real, generated, k=3)) > 0.5
        assert 0.1 < float(manifold_recall(real, generated, k=3)) < 0.9

    def test_rejects_mismatched_feature_dimensions(self) -> None:
        with pytest.raises(ValueError, match="feature dimension"):
            manifold_precision(jnp.ones((5, 2)), jnp.ones((5, 3)))

    @given(seed=st.integers(min_value=0, max_value=200), k=st.integers(min_value=1, max_value=5))
    @settings(max_examples=25, deadline=None)
    def test_values_are_fractions(self, seed: int, k: int) -> None:
        rng = np.random.default_rng(seed)
        real_features = jnp.asarray(rng.standard_normal((12, 3)))
        generated = jnp.asarray(rng.standard_normal((9, 3)))
        for value in (
            manifold_precision(real_features, generated, k=k),
            manifold_recall(real_features, generated, k=k),
            density_weighted_precision(real_features, generated, k=k),
            density_weighted_recall(real_features, generated, k=k),
        ):
            assert 0.0 <= float(value) <= 1.0 + 1e-6


class TestDensityWeighted:
    def test_identical_samples_score_one(self, real: jax.Array) -> None:
        assert density_weighted_precision(real, real, k=5) == pytest.approx(1.0, abs=1e-5)
        assert density_weighted_recall(real, real, k=5) == pytest.approx(1.0, abs=1e-5)

    def test_far_samples_score_near_zero(self, real: jax.Array, far: jax.Array) -> None:
        assert float(density_weighted_precision(real, far, k=5)) < 0.1
        assert float(density_weighted_recall(real, far, k=5)) < 0.1


class TestPrivacy:
    def test_copies_have_zero_distance_and_full_memorization(self, real: jax.Array) -> None:
        assert distance_to_closest_record(real, real) == pytest.approx(0.0, abs=1e-6)
        assert memorization_rate(real, real) == pytest.approx(1.0, abs=1e-6)

    def test_perturbed_records_are_not_memorised(self, real: jax.Array) -> None:
        generated = real + 0.01
        assert memorization_rate(real, generated) == pytest.approx(0.0, abs=1e-6)
        assert float(distance_to_closest_record(real, generated)) > 0.0

    def test_closest_record_distance_is_normalised(self) -> None:
        real_records = jnp.array([[0.0, 0.0], [10.0, 100.0]])
        generated = jnp.array([[5.0, 50.0]])
        # both features span [0, 1] after normalisation; the point sits at (0.5, 0.5)
        assert distance_to_closest_record(real_records, generated) == pytest.approx(0.5, abs=1e-5)

    def test_partial_memorization(self) -> None:
        real_records = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        generated = jnp.array([[1.0, 2.0], [9.0, 9.0], [3.0, 4.0], [0.0, 0.0]])
        assert memorization_rate(real_records, generated) == pytest.approx(0.5, abs=1e-6)
