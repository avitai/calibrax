"""Tests for divergence metrics."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing import TraceCounter

from calibrax.core.models import MetricDirection
from calibrax.metrics import MetricRegistry
from calibrax.metrics.functional.divergence import (
    bregman_divergence,
    chi_squared_divergence,
    f_divergence,
    hellinger_distance,
    js_divergence,
    kl_divergence,
    kolmogorov_smirnov_distance,
    mmd,
    mmd_squared_unbiased,
    renyi_divergence,
    reverse_kl_divergence,
    sinkhorn_divergence,
    sliced_wasserstein,
    SLICED_WASSERSTEIN_REGISTRY_SEED,
    total_variation,
    wasserstein_1d,
)


class TestKLDivergence:
    """Tests for kl_divergence."""

    def test_identical_distributions(self) -> None:
        p = jnp.array([0.25, 0.25, 0.25, 0.25])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self) -> None:
        p = jnp.array([0.5, 0.5])
        q = jnp.array([0.25, 0.75])
        # KL = 0.5*log(0.5/0.25) + 0.5*log(0.5/0.75)
        expected = 0.5 * jnp.log(2.0) + 0.5 * jnp.log(2.0 / 3.0)
        assert kl_divergence(p, q) == pytest.approx(float(expected), abs=1e-4)

    def test_non_negative(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        assert kl_divergence(p, q) >= -1e-6

    def test_handles_zero_probabilities(self) -> None:
        p = jnp.array([0.0, 1.0])
        q = jnp.array([0.5, 0.5])
        result = kl_divergence(p, q)
        assert jnp.isfinite(result)

    def test_asymmetric(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), abs=1e-3)

    def test_returns_jax_scalar(self) -> None:
        p = jnp.array([0.5, 0.5])
        result = kl_divergence(p, p)
        assert isinstance(result, jax.Array)


class TestReverseKLDivergence:
    """Tests for reverse_kl_divergence."""

    def test_identical(self) -> None:
        p = jnp.array([0.5, 0.5])
        assert reverse_kl_divergence(p, p) == pytest.approx(0.0, abs=1e-5)

    def test_delegates_to_kl(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        assert reverse_kl_divergence(p, q) == pytest.approx(kl_divergence(q, p), abs=1e-6)

    def test_asymmetric(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        assert reverse_kl_divergence(p, q) != pytest.approx(kl_divergence(p, q), abs=1e-3)


class TestJSDivergence:
    """Tests for js_divergence."""

    def test_identical_distributions(self) -> None:
        p = jnp.array([0.5, 0.5])
        assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-5)

    def test_symmetric(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=1e-6)

    def test_bounded(self) -> None:
        p = jnp.array([1.0, 0.0])
        q = jnp.array([0.0, 1.0])
        result = js_divergence(p, q)
        assert 0.0 <= result <= float(jnp.log(2.0)) + 1e-5

    def test_known_value(self) -> None:
        p = jnp.array([0.5, 0.5])
        q = jnp.array([0.5, 0.5])
        assert js_divergence(p, q) == pytest.approx(0.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        p = jnp.array([0.5, 0.5])
        result = js_divergence(p, p)
        assert isinstance(result, jax.Array)


class TestTotalVariation:
    """Tests for total_variation."""

    def test_identical(self) -> None:
        p = jnp.array([0.5, 0.5])
        assert total_variation(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_disjoint(self) -> None:
        p = jnp.array([1.0, 0.0])
        q = jnp.array([0.0, 1.0])
        assert total_variation(p, q) == pytest.approx(1.0, abs=1e-6)

    def test_symmetric(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        assert total_variation(p, q) == pytest.approx(total_variation(q, p), abs=1e-6)

    def test_bounded(self) -> None:
        p = jnp.array([0.2, 0.3, 0.5])
        q = jnp.array([0.4, 0.1, 0.5])
        result = total_variation(p, q)
        assert 0.0 <= result <= 1.0 + 1e-6

    def test_returns_jax_scalar(self) -> None:
        p = jnp.array([0.5, 0.5])
        result = total_variation(p, p)
        assert isinstance(result, jax.Array)


class TestHellingerDistance:
    """Tests for hellinger_distance."""

    def test_identical(self) -> None:
        p = jnp.array([0.5, 0.5])
        assert hellinger_distance(p, p) == pytest.approx(0.0, abs=1e-5)

    def test_disjoint(self) -> None:
        p = jnp.array([1.0, 0.0])
        q = jnp.array([0.0, 1.0])
        assert hellinger_distance(p, q) == pytest.approx(1.0, abs=1e-5)

    def test_symmetric(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        assert hellinger_distance(p, q) == pytest.approx(hellinger_distance(q, p), abs=1e-6)

    def test_bounded(self) -> None:
        p = jnp.array([0.2, 0.3, 0.5])
        q = jnp.array([0.4, 0.1, 0.5])
        result = hellinger_distance(p, q)
        assert 0.0 <= result <= 1.0 + 1e-6

    def test_pinsker_inequality(self) -> None:
        # TV <= sqrt(2) * H
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        tv = total_variation(p, q)
        h = hellinger_distance(p, q)
        assert tv <= jnp.sqrt(2.0) * h + 1e-5


class TestChiSquaredDivergence:
    """Tests for chi_squared_divergence."""

    def test_identical(self) -> None:
        p = jnp.array([0.5, 0.5])
        assert chi_squared_divergence(p, p) == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self) -> None:
        p = jnp.array([0.6, 0.4])
        q = jnp.array([0.5, 0.5])
        # (0.1^2)/0.5 + (0.1^2)/0.5 = 0.02 + 0.02 = 0.04
        assert chi_squared_divergence(p, q) == pytest.approx(0.04, abs=1e-4)

    def test_non_negative(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        assert chi_squared_divergence(p, q) >= -1e-6


class TestRenyiDivergence:
    """Tests for renyi_divergence."""

    def test_identical(self) -> None:
        p = jnp.array([0.5, 0.5])
        assert renyi_divergence(p, p) == pytest.approx(0.0, abs=1e-4)

    def test_alpha_near_1_approaches_kl(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        kl = kl_divergence(p, q)
        renyi = renyi_divergence(p, q, alpha=0.999)
        assert renyi == pytest.approx(kl, abs=0.05)

    def test_non_negative(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])
        assert renyi_divergence(p, q, alpha=2.0) >= -1e-6

    def test_alpha_1_raises(self) -> None:
        p = jnp.array([0.5, 0.5])
        with pytest.raises(ValueError, match=r"alpha=1\.0"):
            renyi_divergence(p, p, alpha=1.0)


class TestFDivergence:
    """Tests for f_divergence."""

    def test_kl_via_generator(self) -> None:
        # f(u) = u * log(u) reproduces KL
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])

        def kl_generator(u: jax.Array) -> jax.Array:
            return u * jnp.log(jnp.maximum(u, 1e-8))

        f_div = f_divergence(p, q, generator=kl_generator)
        kl = kl_divergence(p, q)
        assert f_div == pytest.approx(kl, abs=0.05)

    def test_tv_via_generator(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])

        def tv_generator(u: jax.Array) -> jax.Array:
            return 0.5 * jnp.abs(u - 1.0)

        f_div = f_divergence(p, q, generator=tv_generator)
        tv = total_variation(p, q)
        assert f_div == pytest.approx(tv, abs=1e-4)

    def test_custom_generator(self) -> None:
        p = jnp.array([0.5, 0.5])

        def custom_f(u: jax.Array) -> jax.Array:
            return (u - 1.0) ** 2

        result = f_divergence(p, p, generator=custom_f)
        assert result == pytest.approx(0.0, abs=1e-5)


class TestWasserstein1D:
    """Tests for wasserstein_1d."""

    def test_identical_samples(self) -> None:
        p = jnp.array([1.0, 2.0, 3.0])
        assert wasserstein_1d(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_known_shift(self) -> None:
        p = jnp.array([1.0, 2.0, 3.0])
        q = jnp.array([2.0, 3.0, 4.0])
        assert wasserstein_1d(p, q) == pytest.approx(1.0, abs=1e-5)

    def test_non_negative(self) -> None:
        p = jnp.array([1.0, 3.0, 5.0])
        q = jnp.array([2.0, 4.0, 6.0])
        assert wasserstein_1d(p, q) >= -1e-6

    def test_returns_jax_scalar(self) -> None:
        p = jnp.array([1.0, 2.0])
        result = wasserstein_1d(p, p)
        assert isinstance(result, jax.Array)


def _gram(a: np.ndarray, b: np.ndarray, kernel: str, bandwidth: float) -> np.ndarray:
    """The kernel matrix in float64, straight from its definition."""
    diff = a[:, None, :] - b[None, :, :]
    if kernel == "rbf":
        return np.exp(-np.sum(diff**2, axis=-1) / (2.0 * bandwidth**2))
    return np.exp(-np.sum(np.abs(diff), axis=-1) / bandwidth)


class TestMMD:
    """``mmd``: Gretton et al. (2012) eq. 5, the RKHS distance between empirical mean embeddings."""

    _X = np.random.default_rng(0).normal(size=(40, 3))
    _Y = np.random.default_rng(1).normal(size=(30, 3)) + 0.4

    @pytest.mark.parametrize("kernel", ["rbf", "laplace"])
    def test_is_the_biased_statistic(self, kernel: str) -> None:
        x, y = self._X, self._Y
        squared = (
            _gram(x, x, kernel, 1.5).mean()
            + _gram(y, y, kernel, 1.5).mean()
            - 2.0 * _gram(x, y, kernel, 1.5).mean()
        )

        value = mmd(x, y, kernel=kernel, bandwidth=1.5)

        assert float(value) == pytest.approx(np.sqrt(squared), rel=1e-4)

    def test_identical_samples(self) -> None:
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert float(mmd(x, x)) == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions(self) -> None:
        x = jnp.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        y = jnp.array([[5.0, 5.0], [5.1, 5.1], [5.2, 5.2]])
        assert float(mmd(x, y, bandwidth=1.0)) > 0.0

    def test_symmetric(self) -> None:
        x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        y = jnp.array([[2.0, 0.0], [0.0, 2.0]])
        assert mmd(x, y) == pytest.approx(mmd(y, x), abs=1e-6)

    def test_returns_jax_scalar(self) -> None:
        result = mmd(jnp.array([[1.0, 0.0]]), jnp.array([[0.0, 1.0]]))
        assert isinstance(result, jax.Array)

    def test_an_unknown_kernel_is_refused(self) -> None:
        with pytest.raises(ValueError, match="kernel"):
            mmd(self._X, self._Y, kernel="gaussian")


class TestMMDSquaredUnbiased:
    """``mmd_squared_unbiased``: Gretton et al. (2012) Lemma 6, the U-statistic MMD^2_u."""

    def test_is_the_unbiased_statistic(self) -> None:
        x, y = TestMMD._X, TestMMD._Y
        n, m = len(x), len(y)
        kxx, kyy, kxy = _gram(x, x, "rbf", 1.0), _gram(y, y, "rbf", 1.0), _gram(x, y, "rbf", 1.0)
        expected = (
            (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
            + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
            - 2.0 * kxy.mean()
        )

        assert float(mmd_squared_unbiased(x, y)) == pytest.approx(expected, rel=1e-4)

    def test_averages_zero_between_samples_of_one_distribution(self) -> None:
        # Unbiased: over many pairs of samples from one distribution, the mean is 0 within its
        # standard error, and individual values fall below 0 (they are not clamped).
        keys = jax.random.split(jax.random.key(0), 400)
        draws = jax.vmap(lambda k: jax.random.normal(k, (2, 20, 2)))(keys)
        values = jax.vmap(lambda d: mmd_squared_unbiased(d[0], d[1]))(draws)

        standard_error = float(jnp.std(values)) / np.sqrt(len(values))
        assert abs(float(jnp.mean(values))) < 3.0 * standard_error
        assert float(jnp.min(values)) < 0.0

    def test_needs_two_samples_on_each_side(self) -> None:
        with pytest.raises(ValueError, match="two samples"):
            mmd_squared_unbiased(jnp.ones((1, 2)), jnp.ones((3, 2)))


class TestSinkhornDivergence:
    """Tests for sinkhorn_divergence."""

    def test_identical_samples(self) -> None:
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert float(sinkhorn_divergence(x, x)) == pytest.approx(0.0, abs=1e-6)

    def test_well_separated_clouds_at_small_regularization(self) -> None:
        # eps = 0.001 against costs near 200: exp(-C / eps) underflows, the log domain does not.
        x = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = x + 10.0
        result = sinkhorn_divergence(x, y, regularization=0.001)
        # A translation by t moves every point |t|^2 = 200 in squared Euclidean cost.
        assert float(result) == pytest.approx(200.0, rel=1e-3)

    def test_symmetric(self) -> None:
        x = jnp.array([[0.0], [1.0]])
        y = jnp.array([[2.0], [3.0]])
        assert float(sinkhorn_divergence(x, y)) == pytest.approx(
            float(sinkhorn_divergence(y, x)), abs=1e-5
        )

    def test_matches_ott_jax(self) -> None:
        # OTT-JAX 0.6.0's sinkhorn_divergence on these clouds (threshold 1e-6): 1.5737164 at
        # eps = 0.1 and 1.1528621 at eps = 1.0.
        x = jax.random.normal(jax.random.key(0), (32, 3))
        y = jax.random.normal(jax.random.key(1), (32, 3)) + 0.5

        for eps, reference in ((0.1, 1.5737164), (1.0, 1.1528621)):
            value = sinkhorn_divergence(x, y, regularization=eps, threshold=1e-6)
            assert float(value) == pytest.approx(reference, abs=5e-6)

    def test_reverse_mode_gradient_under_jit(self) -> None:
        x = jax.random.normal(jax.random.key(0), (16, 2))
        y = jax.random.normal(jax.random.key(1), (16, 2)) + 1.0

        gradient = jax.jit(jax.grad(sinkhorn_divergence))(x, y)

        assert bool(jnp.all(jnp.isfinite(gradient)))
        # Moving x toward y lowers the divergence: the gradient points away from y.
        assert float(jnp.sum(gradient * (y.mean(0) - x.mean(0)))) < 0

    def test_returns_jax_scalar(self) -> None:
        x = jnp.array([[0.0], [1.0]])
        result = sinkhorn_divergence(x, x)
        assert isinstance(result, jax.Array)


class TestSlicedWasserstein:
    """``SW_p = (E_theta[W_p^p(theta)])^(1/p)`` over random unit directions (Bonneel et al. 2015)."""

    def test_identical_samples(self) -> None:
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert sliced_wasserstein(x, x, key=jax.random.key(0)) == pytest.approx(0.0, abs=1e-6)

    def test_one_dimension_is_the_exact_wasserstein_distance(self) -> None:
        x = jnp.array([0.0, 1.0, 2.0, 5.0])
        y = jnp.array([1.0, 3.0, 4.0, 6.0])
        # Sorted differences 1, 2, 2, 1: W_2 = sqrt(mean([1, 4, 4, 1])) = sqrt(2.5).
        result = sliced_wasserstein(x, y, key=jax.random.key(0), num_projections=8)
        assert float(result) == pytest.approx(np.sqrt(2.5), rel=1e-5)

    @pytest.mark.parametrize("dimension", [2, 10])
    def test_a_shift_gives_its_norm_over_root_dimension(self, dimension: int) -> None:
        """For y = x + v every direction's W_2 is |<v, theta>|, and E[<v, theta>^2] = |v|^2 / d.

        The mean of per-direction W_2, which is not SW_2, converges elsewhere: |v| E|theta_1|
        (2 / pi for d = 2, about 0.25 for d = 10 against 0.316).
        """
        x = jax.random.normal(jax.random.key(1), (64, dimension))
        v = jnp.arange(1.0, dimension + 1.0)
        result = sliced_wasserstein(x, x + v, key=jax.random.key(2), num_projections=20_000)
        expected = float(jnp.linalg.norm(v)) / np.sqrt(dimension)
        assert float(result) == pytest.approx(expected, rel=0.02)

    def test_order_one_is_the_mean_of_per_direction_distances(self) -> None:
        x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        y = x + jnp.array([3.0, 4.0])
        # W_1 along theta is |<(3, 4), theta>|; E|<v, theta>| = |v| * 2 / pi in two dimensions.
        result = sliced_wasserstein(x, y, p=1.0, key=jax.random.key(3), num_projections=20_000)
        assert float(result) == pytest.approx(5.0 * 2.0 / np.pi, rel=0.02)

    def test_symmetric(self) -> None:
        x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        y = jnp.array([[2.0, 2.0], [3.0, 3.0]])
        key = jax.random.key(0)
        assert float(sliced_wasserstein(x, y, key=key)) == pytest.approx(
            float(sliced_wasserstein(y, x, key=key)), rel=1e-6
        )

    def test_a_key_is_required(self) -> None:
        x = jnp.array([[0.0], [1.0]])
        with pytest.raises(TypeError, match="sliced_wasserstein"):
            sliced_wasserstein(x, x, key=None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            sliced_wasserstein(x, x)  # type: ignore[call-arg]

    def test_an_rngs_stream_supplies_the_key(self) -> None:
        x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        y = x + 1.0
        from_rngs = sliced_wasserstein(x, y, key=nnx.Rngs(sample=7))
        from_key = sliced_wasserstein(x, y, key=nnx.Rngs(sample=7).sample())
        assert float(from_rngs) == pytest.approx(float(from_key))

    def test_the_default_projection_count(self) -> None:
        assert inspect.signature(sliced_wasserstein).parameters["num_projections"].default == 256

    def test_unequal_sample_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same number of samples"):
            sliced_wasserstein(jnp.zeros((3, 2)), jnp.zeros((4, 2)), key=jax.random.key(0))

    def test_jit_traces_once_for_new_keys_and_data(self) -> None:
        counter = TraceCounter()
        compiled = jax.jit(counter.wrap(sliced_wasserstein), static_argnames=("num_projections",))
        x = jnp.ones((5, 3))
        with counter.expect(new_traces=1):
            compiled(x, x + 1.0, key=jax.random.key(0), num_projections=16)
        with counter.expect(new_traces=0):
            compiled(x * 2.0, x, key=jax.random.key(1), num_projections=16)

    def test_vmap_over_keys_matches_separate_calls(self) -> None:
        x = jax.random.normal(jax.random.key(4), (8, 3))
        y = x + 0.5
        keys = jax.random.split(jax.random.key(5), 3)
        batched = jax.vmap(lambda k: sliced_wasserstein(x, y, key=k, num_projections=32))(keys)
        separate = jnp.stack([sliced_wasserstein(x, y, key=k, num_projections=32) for k in keys])
        np.testing.assert_allclose(np.asarray(batched), np.asarray(separate), rtol=1e-5, atol=1e-6)

    def test_gradient_is_finite_including_at_identical_samples(self) -> None:
        x = jax.random.normal(jax.random.key(6), (6, 2))
        key = jax.random.key(7)
        at_zero = jax.grad(lambda z: sliced_wasserstein(z, x, key=key))(x)
        apart = jax.grad(lambda z: sliced_wasserstein(z, x + 1.0, key=key))(x)
        assert bool(jnp.all(jnp.isfinite(at_zero)))
        assert bool(jnp.all(jnp.isfinite(apart)))
        assert float(jnp.abs(apart).sum()) > 0.0

    def test_returns_jax_scalar(self) -> None:
        x = jnp.array([[0.0], [1.0]])
        result = sliced_wasserstein(x, x, key=jax.random.key(0))
        assert isinstance(result, jax.Array)
        assert result.shape == ()


class TestBregmanDivergence:
    """Tests for bregman_divergence."""

    def test_squared_euclidean(self) -> None:
        # psi = 0.5 * ||x||^2 → D = 0.5 * ||x - y||^2
        x = jnp.array([1.0, 0.0])
        y = jnp.array([0.0, 1.0])

        def psi(z: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(z**2)

        result = bregman_divergence(x, y, generator=psi)
        expected = 0.5 * jnp.sum((x - y) ** 2)
        assert result == pytest.approx(float(expected), abs=1e-4)

    def test_auto_grad(self) -> None:
        # generator_grad=None → uses jax.grad
        x = jnp.array([2.0, 1.0])
        y = jnp.array([1.0, 2.0])

        def psi(z: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(z**2)

        result = bregman_divergence(x, y, generator=psi)
        assert isinstance(result, jax.Array)
        assert result >= -1e-6

    def test_non_negative(self) -> None:
        x = jnp.array([0.3, 0.7])
        y = jnp.array([0.6, 0.4])

        def psi(z: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(z**2)

        result = bregman_divergence(x, y, generator=psi)
        assert result >= -1e-6

    def test_identical_is_zero(self) -> None:
        x = jnp.array([0.5, 0.5])

        def psi(z: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(z**2)

        result = bregman_divergence(x, x, generator=psi)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        x = jnp.array([1.0])

        def psi(z: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(z**2)

        result = bregman_divergence(x, x, generator=psi)
        assert isinstance(result, jax.Array)


class TestDivergenceMetricRegistration:
    """Tests for divergence metric registration in MetricRegistry."""

    def test_all_divergence_metrics_registered(self) -> None:
        registry = MetricRegistry()
        expected = [
            "kl_divergence",
            "reverse_kl_divergence",
            "js_divergence",
            "total_variation",
            "hellinger_distance",
            "chi_squared_divergence",
            "renyi_divergence",
            "f_divergence",
            "wasserstein_1d",
            "kolmogorov_smirnov_distance",
            "mmd",
            "mmd_squared_unbiased",
            "sinkhorn_divergence",
            "sliced_wasserstein",
            "bregman_divergence",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_divergence_domain(self) -> None:
        registry = MetricRegistry()
        div_metrics = registry.list_by_domain("divergence")
        assert len(div_metrics) == 15

    def test_the_registered_sliced_wasserstein_uses_a_fixed_projection_set(self) -> None:
        """The registry calls ``fn(predictions, targets)``, so its entry fixes the directions.

        A fixed set makes values from different suite runs comparable, at the price of being
        a pseudometric, which the entry's properties record.
        """

        entry = MetricRegistry().get("sliced_wasserstein")
        assert entry.fn is not None
        x = jax.random.normal(jax.random.key(8), (16, 3))
        y = x + 0.25

        first, second = entry.fn(x, y), entry.fn(x, y)

        assert float(first) == float(second)
        expected = sliced_wasserstein(x, y, key=jax.random.key(SLICED_WASSERSTEIN_REGISTRY_SEED))
        assert float(first) == pytest.approx(float(expected), rel=1e-6)
        assert entry.properties.is_true_metric is False
        assert entry.properties.is_symmetric is True

    def test_all_direction_lower(self) -> None:
        registry = MetricRegistry()
        div_metrics = registry.list_by_domain("divergence")
        for m in div_metrics:
            assert m.direction == MetricDirection.LOWER

    def test_symmetry_flags(self) -> None:
        registry = MetricRegistry()
        assert registry.get("js_divergence").properties.is_symmetric is True
        assert registry.get("kl_divergence").properties.is_symmetric is False
        assert registry.get("total_variation").properties.is_symmetric is True


class TestKolmogorovSmirnovDistance:
    def test_identical_samples_have_zero_distance(self) -> None:
        sample = jnp.array([0.1, 0.5, 0.9, 1.3])
        assert kolmogorov_smirnov_distance(sample, sample) == pytest.approx(0.0, abs=1e-6)

    def test_disjoint_samples_have_distance_one(self) -> None:
        assert kolmogorov_smirnov_distance(
            jnp.array([0.0, 1.0]), jnp.array([5.0, 6.0])
        ) == pytest.approx(1.0, abs=1e-6)

    def test_matches_scipy(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        a, b = rng.standard_normal(40), rng.standard_normal(55) + 0.3
        assert kolmogorov_smirnov_distance(jnp.asarray(a), jnp.asarray(b)) == pytest.approx(
            stats.ks_2samp(a, b).statistic, abs=1e-6
        )
