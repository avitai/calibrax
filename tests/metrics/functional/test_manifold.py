"""Tests for manifold and non-Euclidean distance metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.manifold import (
    grassmann_distance,
    spd_affine_invariant_distance,
    spd_log_euclidean_distance,
    stiefel_distance,
    ultrahyperbolic_distance,
)


# --- SPD matrix helpers ---


def _make_spd(n: int, seed: int = 0) -> jnp.ndarray:
    """Create a random SPD matrix."""
    # Use deterministic values for reproducibility
    a = jnp.eye(n) + 0.1 * seed * jnp.ones((n, n))
    return a @ a.T + 0.1 * jnp.eye(n)


class TestSPDAffineInvariantDistance:
    """Tests for spd_affine_invariant_distance."""

    def test_identical_matrices(self) -> None:
        """Distance between identical SPD matrices is 0."""
        a = jnp.eye(3) * 2.0
        result = spd_affine_invariant_distance(a, a)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self) -> None:
        """2x2 diagonal SPD matrices with known distance."""
        # For diagonal A=diag(a1,a2), B=diag(b1,b2):
        # d = sqrt(log(b1/a1)^2 + log(b2/a2)^2)
        a = jnp.diag(jnp.array([1.0, 4.0]))
        b = jnp.diag(jnp.array([4.0, 1.0]))
        expected = jnp.sqrt(jnp.log(4.0) ** 2 + jnp.log(0.25) ** 2)
        result = spd_affine_invariant_distance(a, b)
        assert result == pytest.approx(float(expected), abs=1e-4)

    def test_symmetric(self) -> None:
        """d(A, B) = d(B, A)."""
        a = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        b = jnp.diag(jnp.array([2.0, 1.0, 4.0]))
        assert spd_affine_invariant_distance(a, b) == pytest.approx(
            spd_affine_invariant_distance(b, a), abs=1e-4
        )

    def test_affine_invariant(self) -> None:
        """d(A, B) = d(MAM^T, MBM^T) for invertible M."""
        a = jnp.diag(jnp.array([1.0, 3.0]))
        b = jnp.diag(jnp.array([2.0, 5.0]))
        m = jnp.array([[2.0, 1.0], [0.5, 1.5]])
        a_t = m @ a @ m.T
        b_t = m @ b @ m.T
        d_orig = spd_affine_invariant_distance(a, b)
        d_transformed = spd_affine_invariant_distance(a_t, b_t)
        assert d_orig == pytest.approx(d_transformed, abs=1e-3)

    def test_triangle_inequality(self) -> None:
        """d(A, C) <= d(A, B) + d(B, C)."""
        a = jnp.diag(jnp.array([1.0, 2.0]))
        b = jnp.diag(jnp.array([2.0, 3.0]))
        c = jnp.diag(jnp.array([4.0, 1.0]))
        d_ac = spd_affine_invariant_distance(a, c)
        d_ab = spd_affine_invariant_distance(a, b)
        d_bc = spd_affine_invariant_distance(b, c)
        assert d_ac <= d_ab + d_bc + 1e-5

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        a = jnp.eye(2)
        result = spd_affine_invariant_distance(a, a)
        assert isinstance(result, jax.Array)


class TestSPDLogEuclideanDistance:
    """Tests for spd_log_euclidean_distance."""

    def test_identical_matrices(self) -> None:
        """Distance between identical SPD matrices is 0."""
        a = jnp.eye(3) * 2.0
        result = spd_log_euclidean_distance(a, a)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self) -> None:
        """Diagonal SPD: d = ||log(A) - log(B)||_F."""
        a = jnp.diag(jnp.array([1.0, 4.0]))
        b = jnp.diag(jnp.array([2.0, 2.0]))
        # log(A) = diag(0, log4), log(B) = diag(log2, log2)
        # diff = diag(-log2, log4-log2) = diag(-log2, log2)
        # ||diff||_F = sqrt(2*log2^2) = log2 * sqrt(2)
        expected = float(jnp.log(2.0) * jnp.sqrt(2.0))
        result = spd_log_euclidean_distance(a, b)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_symmetric(self) -> None:
        """d(A, B) = d(B, A)."""
        a = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        b = jnp.diag(jnp.array([2.0, 1.0, 4.0]))
        assert spd_log_euclidean_distance(a, b) == pytest.approx(
            spd_log_euclidean_distance(b, a), abs=1e-4
        )

    def test_not_affine_invariant(self) -> None:
        """Log-Euclidean is NOT affine-invariant in general."""
        a = jnp.diag(jnp.array([1.0, 10.0, 0.5]))
        b = jnp.diag(jnp.array([5.0, 2.0, 3.0]))
        m = jnp.array(
            [
                [3.0, 1.0, 0.5],
                [0.5, 2.0, 0.3],
                [0.2, 0.1, 1.5],
            ]
        )
        a_t = m @ a @ m.T
        b_t = m @ b @ m.T
        d_orig = spd_log_euclidean_distance(a, b)
        d_transformed = spd_log_euclidean_distance(a_t, b_t)
        # These should differ (not affine-invariant)
        assert abs(d_orig - d_transformed) > 0.1

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        a = jnp.eye(2)
        result = spd_log_euclidean_distance(a, a)
        assert isinstance(result, jax.Array)


class TestGrassmannDistance:
    """Tests for grassmann_distance."""

    def test_identical_subspaces(self) -> None:
        """Distance between identical subspaces is 0."""
        u = jnp.eye(3, 2)  # First 2 columns of identity
        result = grassmann_distance(u, u)
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_orthogonal_subspaces(self) -> None:
        """Orthogonal 1D subspaces have distance pi/2."""
        u = jnp.array([[1.0], [0.0]])
        v = jnp.array([[0.0], [1.0]])
        result = grassmann_distance(u, v)
        assert result == pytest.approx(jnp.pi / 2, abs=1e-4)

    def test_known_principal_angles(self) -> None:
        """Known angle between 1D subspaces."""
        angle = jnp.pi / 4  # 45 degrees
        u = jnp.array([[1.0], [0.0]])
        v = jnp.array([[jnp.cos(angle)], [jnp.sin(angle)]])
        result = grassmann_distance(u, v)
        assert result == pytest.approx(float(angle), abs=1e-4)

    def test_symmetric(self) -> None:
        """d(U, V) = d(V, U)."""
        u = jnp.eye(3, 2)
        v = jnp.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ]
        )
        assert grassmann_distance(u, v) == pytest.approx(grassmann_distance(v, u), abs=1e-4)

    def test_basis_independent(self) -> None:
        """Same subspace with different basis gives ~0."""
        u = jnp.eye(3, 2)  # span{e1, e2}
        # Rotate basis within the subspace
        c, s = jnp.cos(0.3), jnp.sin(0.3)
        rot = jnp.array([[c, -s], [s, c], [0.0, 0.0]])
        result = grassmann_distance(u, rot)
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        u = jnp.eye(3, 2)
        result = grassmann_distance(u, u)
        assert isinstance(result, jax.Array)


class TestStiefelDistance:
    """Tests for stiefel_distance."""

    def test_identical_frames(self) -> None:
        """Distance between identical frames is 0."""
        u = jnp.eye(3, 2)
        result = stiefel_distance(u, u)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self) -> None:
        """Known Frobenius distance between simple orthonormal frames."""
        u = jnp.eye(3, 2)
        v = jnp.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ]
        )
        # ||U - V||_F for swapped first two columns
        expected = float(jnp.linalg.norm(u - v))
        result = stiefel_distance(u, v)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_symmetric(self) -> None:
        """d(U, V) = d(V, U)."""
        u = jnp.eye(3, 2)
        v = jnp.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ]
        )
        assert stiefel_distance(u, v) == pytest.approx(stiefel_distance(v, u), abs=1e-4)

    def test_basis_dependent(self) -> None:
        """Same subspace, different basis gives nonzero (unlike Grassmann)."""
        u = jnp.eye(3, 2)  # span{e1, e2}
        c, s = jnp.cos(0.3), jnp.sin(0.3)
        rot = jnp.array([[c, -s], [s, c], [0.0, 0.0]])
        # Same subspace but different orthonormal basis
        result = stiefel_distance(u, rot)
        assert result > 0.01  # Should be nonzero

    def test_full_frame_frobenius(self) -> None:
        """p=n: Frobenius distance between orthogonal matrices."""
        u = jnp.eye(3)
        # Rotation around z-axis by 30 degrees
        angle = jnp.pi / 6
        c, s = jnp.cos(angle), jnp.sin(angle)
        v = jnp.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        result = stiefel_distance(u, v)
        expected = float(jnp.linalg.norm(u - v))
        assert result == pytest.approx(expected, abs=1e-4)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        u = jnp.eye(3, 2)
        result = stiefel_distance(u, u)
        assert isinstance(result, jax.Array)


class TestUltrahyperbolicDistance:
    """Tests for ultrahyperbolic_distance."""

    def test_lorentz_special_case(self) -> None:
        """signature=(1, n) matches lorentz_distance."""
        from calibrax.metrics.functional.distance import lorentz_distance

        # Points on the hyperboloid: -x0^2 + x1^2 + x2^2 = -1
        a = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])
        b = jnp.array([jnp.sqrt(5.0), 2.0, 0.0])
        d_ultra = ultrahyperbolic_distance(a, b, signature=(1, 2))
        d_lorentz = lorentz_distance(a, b)
        assert d_ultra == pytest.approx(d_lorentz, abs=1e-3)

    def test_identical_points(self) -> None:
        """Distance between identical points is ~0."""
        a = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])
        result = ultrahyperbolic_distance(a, a, signature=(1, 2))
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_symmetric(self) -> None:
        """d(a, b) = d(b, a)."""
        a = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])
        b = jnp.array([jnp.sqrt(5.0), 2.0, 0.0])
        assert ultrahyperbolic_distance(a, b, signature=(1, 2)) == pytest.approx(
            ultrahyperbolic_distance(b, a, signature=(1, 2)), abs=1e-4
        )

    def test_different_signatures(self) -> None:
        """Different (p, q) gives different distances."""
        # Create valid points for (1,3) and (2,2) signatures
        a_13 = jnp.array([jnp.sqrt(3.0), 1.0, 1.0, 0.0])
        b_13 = jnp.array([jnp.sqrt(6.0), 2.0, 1.0, 0.0])

        a_22 = jnp.array([jnp.sqrt(2.0), jnp.sqrt(2.0), 1.0, 1.0])
        b_22 = jnp.array([jnp.sqrt(3.0), jnp.sqrt(3.0), 2.0, 1.0])

        d1 = ultrahyperbolic_distance(a_13, b_13, signature=(1, 3))
        d2 = ultrahyperbolic_distance(a_22, b_22, signature=(2, 2))
        # Different signatures, different geometry — distances should differ
        assert d1 != pytest.approx(d2, abs=0.01)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        a = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])
        result = ultrahyperbolic_distance(a, a, signature=(1, 2))
        assert isinstance(result, jax.Array)
