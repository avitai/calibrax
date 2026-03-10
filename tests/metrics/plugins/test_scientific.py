"""Tests for scientific domain metrics plugin."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.plugins.scientific import (
    binding_affinity_metrics,
    chemical_validity,
    conformational_diversity,
)


class TestChemicalValidity:
    """Tests for chemical_validity."""

    def test_all_valid(self) -> None:
        bond_lengths = jnp.array([1.0, 1.2, 1.5])
        bond_angles = jnp.array([2.0, 2.5, 3.0])
        result = chemical_validity(bond_lengths, bond_angles)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_all_invalid(self) -> None:
        bond_lengths = jnp.array([0.1, 5.0, 10.0])
        bond_angles = jnp.array([0.1, 5.0, 6.0])
        result = chemical_validity(bond_lengths, bond_angles)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_mixed(self) -> None:
        bond_lengths = jnp.array([1.0, 5.0])  # 1 valid, 1 invalid
        bond_angles = jnp.array([2.0, 5.0])  # 1 valid, 1 invalid
        result = chemical_validity(bond_lengths, bond_angles)
        assert result == pytest.approx(0.5, abs=1e-5)

    def test_custom_thresholds(self) -> None:
        bond_lengths = jnp.array([3.0])  # Invalid with default, valid with custom
        bond_angles = jnp.array([2.0])  # Valid with default
        result = chemical_validity(
            bond_lengths,
            bond_angles,
            length_thresholds=(0.5, 4.0),
        )
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        result = chemical_validity(jnp.array([1.0]), jnp.array([2.0]))
        assert isinstance(result, jax.Array)


class TestBindingAffinityMetrics:
    """Tests for binding_affinity_metrics."""

    def test_perfect_predictions(self) -> None:
        values = jnp.array([1.0, 2.0, 3.0])
        result = binding_affinity_metrics(values, values)
        assert result["binding_mse"] == pytest.approx(0.0, abs=1e-5)
        assert result["binding_r_squared"] == pytest.approx(1.0, abs=1e-4)
        assert result["binding_pearson"] == pytest.approx(1.0, abs=1e-4)

    def test_known_values(self) -> None:
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.5, 2.5, 3.5])
        result = binding_affinity_metrics(predictions, targets)
        assert result["binding_mse"] > 0.0
        assert result["binding_mae"] > 0.0

    def test_returns_all_keys(self) -> None:
        predictions = jnp.array([1.0, 2.0])
        targets = jnp.array([1.1, 2.1])
        result = binding_affinity_metrics(predictions, targets)
        expected_keys = {"binding_mse", "binding_mae", "binding_r_squared", "binding_pearson"}
        assert set(result.keys()) == expected_keys

    def test_delegates_to_regression(self) -> None:
        """Results should match calling regression metrics directly."""
        from calibrax.metrics.functional.regression import mae, mse

        predictions = jnp.array([1.0, 3.0, 5.0])
        targets = jnp.array([1.5, 2.5, 4.5])
        result = binding_affinity_metrics(predictions, targets)
        assert result["binding_mse"] == pytest.approx(mse(predictions, targets), abs=1e-5)
        assert result["binding_mae"] == pytest.approx(mae(predictions, targets), abs=1e-5)


class TestConformationalDiversity:
    """Tests for conformational_diversity."""

    def test_identical_conformations(self) -> None:
        coords = jnp.ones((5, 10, 3))
        result = conformational_diversity(coords)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_diverse_conformations(self) -> None:
        coords = jnp.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
            ]
        )
        result = conformational_diversity(coords)
        assert result > 0.0

    def test_single_conformation(self) -> None:
        coords = jnp.ones((1, 10, 3))
        result = conformational_diversity(coords)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_known_rmsd(self) -> None:
        """Two conformations with known RMSD."""
        # 2 conformations, 1 atom, 3D
        # RMSD = sqrt(mean(sum((c1 - c2)^2, axis=-1)))
        # = sqrt((1^2 + 1^2 + 1^2)) = sqrt(3)
        coords = jnp.array(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0]],
            ]
        )
        result = conformational_diversity(coords)
        import math

        assert result == pytest.approx(math.sqrt(3.0), abs=1e-4)
