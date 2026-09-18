"""Tests for divergence metrics."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.metrics.functional.divergence import (
    bregman_divergence,
    chi_squared_divergence,
    f_divergence,
    hellinger_distance,
    js_divergence,
    kl_divergence,
    mmd,
    renyi_divergence,
    reverse_kl_divergence,
    sinkhorn_divergence,
    sliced_wasserstein,
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

        def kl_generator(u: Any) -> Any:
            return u * jnp.log(jnp.maximum(u, 1e-8))

        f_div = f_divergence(p, q, generator=kl_generator)
        kl = kl_divergence(p, q)
        assert f_div == pytest.approx(kl, abs=0.05)

    def test_tv_via_generator(self) -> None:
        p = jnp.array([0.3, 0.7])
        q = jnp.array([0.6, 0.4])

        def tv_generator(u: Any) -> Any:
            return 0.5 * jnp.abs(u - 1.0)

        f_div = f_divergence(p, q, generator=tv_generator)
        tv = total_variation(p, q)
        assert f_div == pytest.approx(tv, abs=1e-4)

    def test_custom_generator(self) -> None:
        p = jnp.array([0.5, 0.5])

        def custom_f(u: Any) -> Any:
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


class TestMMD:
    """Tests for mmd."""

    def test_identical_samples(self) -> None:
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert mmd(x, x) == pytest.approx(0.0, abs=1e-4)

    def test_different_distributions(self) -> None:
        x = jnp.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        y = jnp.array([[5.0, 5.0], [5.1, 5.1], [5.2, 5.2]])
        result = mmd(x, y, bandwidth=1.0)
        assert result > 0.0

    def test_symmetric(self) -> None:
        x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        y = jnp.array([[2.0, 0.0], [0.0, 2.0]])
        assert mmd(x, y) == pytest.approx(mmd(y, x), abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        x = jnp.array([[1.0, 0.0]])
        y = jnp.array([[0.0, 1.0]])
        result = mmd(x, y)
        assert isinstance(result, jax.Array)


class TestSinkhornDivergence:
    """Tests for sinkhorn_divergence."""

    def test_identical_samples(self) -> None:
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        result = sinkhorn_divergence(x, x)
        assert result == pytest.approx(0.0, abs=0.1)

    def test_different_distributions(self) -> None:
        # With small regularization and well-separated distributions, Sinkhorn > 0
        x = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = jnp.array([[10.0, 10.0], [10.5, 10.5], [11.0, 10.0], [10.0, 11.0], [11.0, 11.0]])
        result = sinkhorn_divergence(x, y, regularization=0.001, max_iter=200)
        assert result >= 0.0

    def test_symmetric(self) -> None:
        x = jnp.array([[0.0], [1.0]])
        y = jnp.array([[2.0], [3.0]])
        assert sinkhorn_divergence(x, y) == pytest.approx(sinkhorn_divergence(y, x), abs=0.1)

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
        from flax import nnx

        x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        y = x + 1.0
        from_rngs = sliced_wasserstein(x, y, key=nnx.Rngs(sample=7))
        from_key = sliced_wasserstein(x, y, key=nnx.Rngs(sample=7).sample())
        assert float(from_rngs) == pytest.approx(float(from_key))

    def test_the_default_projection_count(self) -> None:
        import inspect

        assert inspect.signature(sliced_wasserstein).parameters["num_projections"].default == 256

    def test_unequal_sample_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same number of samples"):
            sliced_wasserstein(jnp.zeros((3, 2)), jnp.zeros((4, 2)), key=jax.random.key(0))

    def test_jit_traces_once_for_new_keys_and_data(self) -> None:
        from substrax.testing import TraceCounter

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

        def psi(z: Any) -> Any:
            return 0.5 * jnp.sum(z**2)

        result = bregman_divergence(x, y, generator=psi)
        expected = 0.5 * jnp.sum((x - y) ** 2)
        assert result == pytest.approx(float(expected), abs=1e-4)

    def test_auto_grad(self) -> None:
        # generator_grad=None → uses jax.grad
        x = jnp.array([2.0, 1.0])
        y = jnp.array([1.0, 2.0])

        def psi(z: Any) -> Any:
            return 0.5 * jnp.sum(z**2)

        result = bregman_divergence(x, y, generator=psi)
        assert isinstance(result, jax.Array)
        assert result >= -1e-6

    def test_non_negative(self) -> None:
        x = jnp.array([0.3, 0.7])
        y = jnp.array([0.6, 0.4])

        def psi(z: Any) -> Any:
            return 0.5 * jnp.sum(z**2)

        result = bregman_divergence(x, y, generator=psi)
        assert result >= -1e-6

    def test_identical_is_zero(self) -> None:
        x = jnp.array([0.5, 0.5])

        def psi(z: Any) -> Any:
            return 0.5 * jnp.sum(z**2)

        result = bregman_divergence(x, x, generator=psi)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        x = jnp.array([1.0])

        def psi(z: Any) -> Any:
            return 0.5 * jnp.sum(z**2)

        result = bregman_divergence(x, x, generator=psi)
        assert isinstance(result, jax.Array)


class TestDivergenceMetricRegistration:
    """Tests for divergence metric registration in MetricRegistry."""

    def test_all_divergence_metrics_registered(self) -> None:
        from calibrax.metrics import MetricRegistry

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
            "sinkhorn_divergence",
            "sliced_wasserstein",
            "bregman_divergence",
        ]
        for name in expected:
            assert registry.has(name), f"Metric '{name}' not registered"

    def test_divergence_domain(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        div_metrics = registry.list_by_domain("divergence")
        assert len(div_metrics) == 14

    def test_the_registered_sliced_wasserstein_uses_a_fixed_projection_set(self) -> None:
        """The registry calls ``fn(predictions, targets)``, so its entry fixes the directions.

        A fixed set makes values from different suite runs comparable, at the price of being
        a pseudometric, which the entry's properties record.
        """
        from calibrax.metrics import MetricRegistry
        from calibrax.metrics.functional.divergence import SLICED_WASSERSTEIN_REGISTRY_SEED

        entry = MetricRegistry().get("sliced_wasserstein")
        x = jax.random.normal(jax.random.key(8), (16, 3))
        y = x + 0.25

        first, second = entry.fn(x, y), entry.fn(x, y)

        assert float(first) == float(second)
        expected = sliced_wasserstein(x, y, key=jax.random.key(SLICED_WASSERSTEIN_REGISTRY_SEED))
        assert float(first) == pytest.approx(float(expected), rel=1e-6)
        assert entry.properties.is_true_metric is False
        assert entry.properties.is_symmetric is True

    def test_all_direction_lower(self) -> None:
        from calibrax.core.models import MetricDirection
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        div_metrics = registry.list_by_domain("divergence")
        for m in div_metrics:
            assert m.direction == MetricDirection.LOWER

    def test_symmetry_flags(self) -> None:
        from calibrax.metrics import MetricRegistry

        registry = MetricRegistry()
        assert registry.get("js_divergence").properties.is_symmetric is True
        assert registry.get("kl_divergence").properties.is_symmetric is False
        assert registry.get("total_variation").properties.is_symmetric is True


class TestKolmogorovSmirnovDistance:
    def test_identical_samples_have_zero_distance(self) -> None:
        from calibrax.metrics.functional.divergence import kolmogorov_smirnov_distance

        sample = jnp.array([0.1, 0.5, 0.9, 1.3])
        assert kolmogorov_smirnov_distance(sample, sample) == pytest.approx(0.0, abs=1e-6)

    def test_disjoint_samples_have_distance_one(self) -> None:
        from calibrax.metrics.functional.divergence import kolmogorov_smirnov_distance

        assert kolmogorov_smirnov_distance(
            jnp.array([0.0, 1.0]), jnp.array([5.0, 6.0])
        ) == pytest.approx(1.0, abs=1e-6)

    def test_matches_scipy(self) -> None:
        from calibrax.metrics.functional.divergence import kolmogorov_smirnov_distance

        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        a, b = rng.standard_normal(40), rng.standard_normal(55) + 0.3
        assert kolmogorov_smirnov_distance(jnp.asarray(a), jnp.asarray(b)) == pytest.approx(
            stats.ks_2samp(a, b).statistic, abs=1e-6
        )
