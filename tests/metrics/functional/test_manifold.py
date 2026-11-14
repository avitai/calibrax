"""Tests for manifold and non-Euclidean distance metrics."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import subspace_angles

from calibrax.metrics.functional.distance import lorentz_distance
from calibrax.metrics.functional.manifold import (
    grassmann_distance,
    RANK_DEFICIENT_CHOICES,
    RankDeficient,
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
        assert float(grassmann_distance(u, u)) == 0.0

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
        assert result == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("tilt", [1e-4, 0.3, 1.2])
    def test_matches_scipy_principal_angles(self, tilt: float) -> None:
        """Knyazev and Argentati's angles, as scipy.linalg.subspace_angles computes them."""
        rng = np.random.default_rng(3)
        u = np.linalg.qr(rng.normal(size=(6, 3)))[0]
        step = np.linalg.qr(rng.normal(size=(6, 3)))[0]
        v = np.linalg.qr(u + tilt * step)[0]

        expected = np.linalg.norm(subspace_angles(u, v))
        assert float(grassmann_distance(u, v)) == pytest.approx(expected, rel=1e-3, abs=1e-6)

    @pytest.mark.parametrize("same", [True, False])
    def test_gradient_is_finite_under_jit(self, same: bool) -> None:
        rng = np.random.default_rng(4)
        u = jnp.asarray(np.linalg.qr(rng.normal(size=(5, 2)))[0], jnp.float32)
        v = u if same else jnp.asarray(np.linalg.qr(rng.normal(size=(5, 2)))[0], jnp.float32)

        gradient = jax.jit(jax.grad(grassmann_distance))(u, v)

        assert bool(jnp.all(jnp.isfinite(gradient)))

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        u = jnp.eye(3, 2)
        result = grassmann_distance(u, u)
        assert isinstance(result, jax.Array)

    @pytest.mark.parametrize("scale", [3.0, 0.5])
    def test_any_basis_of_the_subspace_gives_its_distance(self, scale: float) -> None:
        """Each input is orthonormalised first (Bjorck and Golub; scipy's ``subspace_angles``).

        A scaled or non-orthonormal basis spans the same subspace, so it gives the same distance
        in either argument order. The tolerance is the one the scipy comparison above uses.
        """
        rng = np.random.default_rng(0)
        u = np.linalg.qr(rng.normal(size=(5, 2)))[0]
        w = np.linalg.qr(rng.normal(size=(5, 2)))[0]
        skewed = w @ np.array([[1.0, 0.7], [0.0, 2.0]])
        expected = float(np.linalg.norm(subspace_angles(u, w)))
        for basis in (scale * w, skewed):
            assert float(grassmann_distance(u, basis)) == pytest.approx(
                expected, rel=1e-3, abs=1e-6
            )
            assert float(grassmann_distance(basis, u)) == pytest.approx(
                expected, rel=1e-3, abs=1e-6
            )

    def test_the_two_choices_agree_on_full_rank_bases(self) -> None:
        rng = np.random.default_rng(1)
        u, v = (jnp.asarray(rng.normal(size=(6, 3)), dtype=jnp.float32) for _ in range(2))
        default = float(grassmann_distance(u, v))
        assert float(grassmann_distance(u, v, rank_deficient="ye_lim")) == default

    @pytest.mark.parametrize(
        "deficient",
        [np.zeros((4, 2)), np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])],
        ids=["zero", "repeated column"],
    )
    def test_a_rank_deficient_basis_is_nan_by_default(self, deficient: np.ndarray) -> None:
        """A basis spanning fewer than p dimensions is not a point of Gr(p, n)."""
        u = jnp.eye(4, 2)
        assert np.isnan(float(grassmann_distance(u, jnp.asarray(deficient, dtype=jnp.float32))))

    def test_ye_lim_counts_each_missing_angle_as_a_right_angle(self) -> None:
        """Ye & Lim 2016, Table 3: sqrt(abs(k - l) pi^2 / 4 + sum of the min(k, l) angles squared).

        The zero matrix spans {0} (k = 0) against a plane (l = 2): sqrt(2) pi / 2. A repeated column
        spans the line e1 (k = 1) inside the plane span(e1, e2) (l = 2): its one angle is 0, so pi / 2.
        """
        u = jnp.eye(4, 2)
        zero = jnp.zeros((4, 2))
        line = jnp.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        eps = float(jnp.finfo(jnp.float32).eps)
        zero_distance = float(grassmann_distance(u, zero, rank_deficient="ye_lim"))
        assert zero_distance == pytest.approx(np.sqrt(2.0) * np.pi / 2, rel=10 * eps)
        assert float(grassmann_distance(u, line, rank_deficient="ye_lim")) == pytest.approx(
            np.pi / 2, rel=10 * eps
        )
        assert float(grassmann_distance(zero, zero, rank_deficient="ye_lim")) == 0.0

    def test_different_dimensions_are_refused_by_default_and_measured_by_ye_lim(self) -> None:
        line, plane = jnp.eye(4, 1), jnp.eye(4, 2)
        with pytest.raises(ValueError, match="columns"):
            grassmann_distance(line, plane)
        assert float(grassmann_distance(line, plane, rank_deficient="ye_lim")) == pytest.approx(
            np.pi / 2, rel=10 * float(jnp.finfo(jnp.float32).eps)
        )

    def test_gradients_are_finite_for_a_scaled_basis(self) -> None:
        """QR keeps the gradient finite where an SVD's, with repeated singular values, is NaN."""
        rng = np.random.default_rng(2)
        u = jnp.asarray(np.linalg.qr(rng.normal(size=(5, 2)))[0], jnp.float32)
        w = jnp.asarray(np.linalg.qr(rng.normal(size=(5, 2)))[0], jnp.float32)
        gradient = jax.jit(jax.grad(grassmann_distance))(3.0 * u, w)
        assert bool(jnp.all(jnp.isfinite(gradient)))

    def test_refuses_an_unknown_choice(self) -> None:
        u = jnp.eye(4, 2)
        with pytest.raises(ValueError, match="rank_deficient"):
            grassmann_distance(u, u, rank_deficient="clamp")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            ("nan", [np.sqrt(2.0) * np.pi / 2, 0.0, np.nan, np.nan]),
            ("ye_lim", [np.sqrt(2.0) * np.pi / 2, 0.0, np.pi / 2, np.sqrt(2.0) * np.pi / 2]),
        ],
    )
    def test_a_jitted_vmap_over_mixed_bases_gives_each_ones_distance(
        self, choice: RankDeficient, expected: list[float]
    ) -> None:
        """One batch: the orthogonal plane, a scaled basis of ``u``, a line, the zero matrix.

        The rank test and both branches are traced per lane, so each lane takes its own branch.
        """
        u = jnp.eye(4)[:, :2]
        line = jnp.stack([u[:, 0], 2.0 * u[:, 0]], axis=1)
        batch = jnp.stack([jnp.eye(4)[:, 2:], 2.0 * u, line, jnp.zeros((4, 2))])
        distance = functools.partial(grassmann_distance, rank_deficient=choice)
        out = np.asarray(jax.jit(jax.vmap(distance, in_axes=(None, 0)))(u, batch))
        eps = float(jnp.finfo(jnp.float32).eps)
        np.testing.assert_allclose(out, expected, rtol=10 * eps, atol=10 * eps)

    @pytest.mark.parametrize("choice", RANK_DEFICIENT_CHOICES)
    def test_the_gradient_at_a_rank_deficient_basis_is_zero_not_nan(
        self, choice: RankDeficient
    ) -> None:
        """QR reads a full-rank stand-in there (inner where), as its own gradient divides by zero.

        The rank-deficient value carries no gradient (the column space is not differentiable in
        the matrix where the rank drops), so it leaves a batch's other gradients finite.
        """
        u = jnp.eye(4)[:, :2]
        line = jnp.stack([u[:, 0], 2.0 * u[:, 0]], axis=1)
        distance = functools.partial(grassmann_distance, rank_deficient=choice)
        for deficient in (line, jnp.zeros((4, 2))):
            gradient = np.asarray(jax.jit(jax.grad(distance, argnums=1))(u, deficient))
            assert np.all(gradient == 0.0)


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
        """signature=(1, n) is lorentz_distance, the same haversine computation."""
        # Points on the hyperboloid: -x0^2 + x1^2 + x2^2 = -1
        a = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])
        b = jnp.array([jnp.sqrt(5.0), 2.0, 0.0])
        assert float(ultrahyperbolic_distance(a, b, signature=(1, 2))) == float(
            lorentz_distance(a, b)
        )

    @pytest.mark.parametrize(
        ("point", "signature"),
        [
            ([2.0**0.5, 1.0, 0.0], (1, 2)),
            ([1.0, 0.0, 0.0, 0.0], (2, 2)),
            ([2.0**0.5, 2.0**0.5, 1.0, 1.0, 0.5], (3, 2)),
        ],
    )
    def test_identical_points_are_exactly_zero_with_a_zero_gradient(
        self, point: list[float], signature: tuple[int, int]
    ) -> None:
        """One point given as both arguments: distance 0 and gradient 0, with no floor."""
        a = jnp.asarray(point, dtype=jnp.float32)
        assert float(ultrahyperbolic_distance(a, a, signature=signature)) == 0.0
        gradient = jax.grad(lambda x: ultrahyperbolic_distance(x, x, signature=signature))(a)
        assert np.all(np.asarray(gradient) == 0.0)

    @pytest.mark.parametrize("angle", [0.3, 1.0, 2.5])
    def test_a_timelike_pair_is_at_the_angle_between_them(self, angle: float) -> None:
        """On (2, 2), (1, 0, 0, 0) and (cos t, sin t, 0, 0) have u = cos t, so d = t (Bengtsson eq. 20).

        The tolerance is a few float32 roundings of the half-angle ``atan2`` form, 10 eps relative.
        """
        a = jnp.array([1.0, 0.0, 0.0, 0.0])
        b = jnp.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=jnp.float32)
        result = float(ultrahyperbolic_distance(a, b, signature=(2, 2)))
        assert result == pytest.approx(angle, rel=10 * float(jnp.finfo(jnp.float32).eps))

    @pytest.mark.parametrize("rapidity", [0.5, 1.5])
    def test_a_spacelike_pair_is_at_the_hyperbolic_angle_between_them(
        self, rapidity: float
    ) -> None:
        """On (2, 2), (1, 0, 0, 0) and (cosh s, 0, sinh s, 0) have u = cosh s, so d = s (eq. 19)."""
        a = jnp.array([1.0, 0.0, 0.0, 0.0])
        b = jnp.array([np.cosh(rapidity), 0.0, np.sinh(rapidity), 0.0], dtype=jnp.float32)
        result = float(ultrahyperbolic_distance(a, b, signature=(2, 2)))
        assert result == pytest.approx(rapidity, rel=10 * float(jnp.finfo(jnp.float32).eps))

    @pytest.mark.parametrize("rapidity", [0.5, 1.5])
    def test_a_disconnected_pair_is_at_the_broken_geodesic_length(self, rapidity: float) -> None:
        """(1, 0, 0, 0) and (-cosh s, 0, sinh s, 0) have u = -cosh s < -1: no geodesic joins them.

        The default is the broken geodesic through -x (Law & Stam 2020, p. 5; QGCN): its first leg,
        x to -x, has length pi, and -x and y are spacelike at distance s. With
        ``disconnected="nan"`` the undefined distance is NaN.
        """
        a = jnp.array([1.0, 0.0, 0.0, 0.0])
        b = jnp.array([-np.cosh(rapidity), 0.0, np.sinh(rapidity), 0.0], dtype=jnp.float32)
        broken = float(ultrahyperbolic_distance(a, b, signature=(2, 2)))
        assert broken == pytest.approx(np.pi + rapidity, rel=10 * float(jnp.finfo(jnp.float32).eps))
        assert np.isnan(float(ultrahyperbolic_distance(a, b, signature=(2, 2), disconnected="nan")))

    @pytest.mark.parametrize("rapidity", [0.5, 1.5])
    def test_the_two_sheets_of_the_hyperboloid(self, rapidity: float) -> None:
        """At (1, 2), points on opposite sheets have no geodesic; a pair on the lower sheet does.

        (1, 0, 0) and (-cosh s, sinh s, 0) sit on opposite sheets: the broken geodesic is pi + s,
        as -a and b share the lower sheet at distance s. Two lower-sheet points are the mirror
        image of an upper-sheet pair, at that pair's distance.
        """
        eps = float(jnp.finfo(jnp.float32).eps)
        a = jnp.array([1.0, 0.0, 0.0])
        lower = jnp.array([-np.cosh(rapidity), np.sinh(rapidity), 0.0], dtype=jnp.float32)
        broken = float(ultrahyperbolic_distance(a, lower, signature=(1, 2)))
        assert broken == pytest.approx(np.pi + rapidity, rel=10 * eps)
        assert np.isnan(
            float(ultrahyperbolic_distance(a, lower, signature=(1, 2), disconnected="nan"))
        )
        mirrored = float(ultrahyperbolic_distance(-a, lower, signature=(1, 2)))
        assert mirrored == pytest.approx(rapidity, rel=10 * eps)

    def test_the_broken_geodesic_meets_the_timelike_branch_at_pi(self) -> None:
        """At u = -1 both branches reach pi, so the default is continuous across the boundary."""
        a = jnp.array([1.0, 0.0, 0.0, 0.0])
        near = np.pi - 1e-3
        just_connected = jnp.array([np.cos(near), np.sin(near), 0.0, 0.0], dtype=jnp.float32)
        just_disconnected = jnp.array([-np.cosh(1e-3), 0.0, np.sinh(1e-3), 0.0], dtype=jnp.float32)
        before = float(ultrahyperbolic_distance(a, just_connected, signature=(2, 2)))
        after = float(ultrahyperbolic_distance(a, just_disconnected, signature=(2, 2)))
        assert before == pytest.approx(np.pi - 1e-3, abs=1e-5)
        assert after == pytest.approx(np.pi + 1e-3, abs=1e-5)

    @pytest.mark.parametrize(
        "pair",
        [
            ([1.0, 0.0, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0]),
            ([1.0, 0.0, 0.0, 0.0], [1.25, 0.0, 0.75, 0.0]),
            ([1.0, 0.0, 0.0, 0.0], [-1.25, 0.0, 0.75, 0.0]),
        ],
        ids=["timelike", "spacelike", "disconnected"],
    )
    def test_gradients_are_finite_in_every_regime(
        self, pair: tuple[list[float], list[float]]
    ) -> None:
        a, b = (jnp.asarray(p, dtype=jnp.float32) for p in pair)
        gradient = jax.grad(lambda x: ultrahyperbolic_distance(x, b, signature=(2, 2)))(a)
        assert np.all(np.isfinite(np.asarray(gradient)))

    def test_jit_with_the_static_choice(self) -> None:
        a = jnp.array([1.0, 0.0, 0.0, 0.0])
        b = jnp.array([-1.25, 0.0, 0.75, 0.0])
        jitted = jax.jit(ultrahyperbolic_distance, static_argnames=("signature", "disconnected"))
        eager = float(ultrahyperbolic_distance(a, b, signature=(2, 2)))
        assert float(jitted(a, b, signature=(2, 2))) == eager
        assert np.isnan(float(jitted(a, b, signature=(2, 2), disconnected="nan")))

    @pytest.mark.parametrize(
        ("signature", "disconnected", "message"),
        [
            ((0, 4), "broken_geodesic", "timelike"),
            ((1, 2), "broken_geodesic", "4 coordinates"),
            ((2, 2), "clamp", "disconnected"),
        ],
    )
    def test_refuses_a_signature_or_choice_it_cannot_honour(
        self, signature: tuple[int, int], disconnected: str, message: str
    ) -> None:
        """(0, q) has no points (<x, x> = -1 needs a timelike coordinate); p + q is the size."""
        a = jnp.array([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match=message):
            ultrahyperbolic_distance(a, a, signature=signature, disconnected=disconnected)  # type: ignore[arg-type]

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
