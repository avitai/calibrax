"""Frequency weights and a mask on ``kendall_tau``, against physical replication.

Kendall's tau was the one ratio statistic left without either, on the stated grounds that
weighted Kendall had the same no-accepted-definition problem as weighted Spearman. That was
wrong, and these tests are the evidence.

Kendall's tau is a statistic over **pairs**, so a per-element weight lifts to the pair
weight ``w_i * w_j`` with nothing left to choose. The test that settles it is the defining
property of a frequency weight: integer weights must equal physically replicating each
element that many times. They do, ties included, against ``scipy.stats.kendalltau`` --
which shares no implementation with this one.

The reference is ``scipy``, kept out of the comparison's own machinery: these assertions
compare against an independent implementation rather than against a second expression of
the same formula.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import kendalltau

from calibrax.metrics.functional.statistical import kendall_tau


#: Float64 agreement with scipy, which computes in double precision. The weighted path
#: runs in float32 here, so this is the float32 budget rather than scipy's own accuracy.
TOLERANCE = 1e-5

#: Deliberately small. The statistic is O(n^2) in memory, and replication multiplies the
#: reference's length by the weights, so a large fixture buys nothing and costs quadratically.
SIZE = 12


def _fixture(seed: int, *, tied: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Two correlated series, optionally rounded so that ties occur.

    Args:
        seed: Seed for the generator, so each case is reproducible on its own.
        tied: Round both series to integers, which forces ties in both variables.

    Returns:
        The two series as float64 numpy arrays.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=SIZE)
    b = a + rng.normal(scale=0.6, size=SIZE)
    return (np.round(a), np.round(b)) if tied else (a, b)


class TestIntegerWeightsEqualReplication:
    """The property that makes these frequency weights rather than a convention."""

    @pytest.mark.parametrize("tied", [False, True])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 11])
    def test_matches_scipy_on_the_replicated_sample(self, seed: int, tied: bool) -> None:
        """Weighting by ``w`` equals repeating each element ``w`` times, ties included."""
        a, b = _fixture(seed, tied=tied)
        rng = np.random.default_rng(seed + 100)
        w = rng.integers(1, 5, size=SIZE)
        reference = kendalltau(np.repeat(a, w), np.repeat(b, w)).statistic
        assert np.isfinite(reference), "degenerate fixture: scipy returned no statistic"
        weighted = float(
            kendall_tau(jnp.asarray(a), jnp.asarray(b), weights=jnp.asarray(w, dtype=float))
        )
        assert weighted == pytest.approx(reference, abs=TOLERANCE)

    def test_the_wrong_lift_does_not_have_the_property(self) -> None:
        """Positive control: ``w_i + w_j`` is the plausible alternative, and it fails.

        Without this, the test above would pass for a weighting that merely correlates
        with the right one on easy fixtures.
        """
        a, b = _fixture(1)
        rng = np.random.default_rng(101)
        w = rng.integers(1, 5, size=SIZE).astype(float)
        reference = kendalltau(np.repeat(a, w.astype(int)), np.repeat(b, w.astype(int))).statistic

        upper = np.triu(np.ones((SIZE, SIZE), dtype=bool), k=1)
        additive = np.add.outer(w, w) * upper
        sign_a = np.sign(a[:, None] - a[None, :])
        sign_b = np.sign(b[:, None] - b[None, :])
        total = additive.sum()
        tied_a = (additive * (sign_a == 0)).sum()
        tied_b = (additive * (sign_b == 0)).sum()
        wrong = (additive * sign_a * sign_b).sum() / np.sqrt((total - tied_a) * (total - tied_b))

        assert abs(wrong - reference) > 1e-3, (
            "the additive lift now reproduces replication too, so this fixture no longer "
            "distinguishes the two and the multiplicative lift is unverified"
        )


class TestTheTieCorrectionCancels:
    """Why no within-element term appears, pinned so its absence does not read as a bug."""

    def test_adding_the_within_element_pairs_changes_nothing(self) -> None:
        """Replicating an element makes pairs tied in **both** variables.

        There are ``w_i * (w_i - 1) / 2`` of them, and they add equally to the pair total
        and to both tie totals, so they cancel from ``(n0 - n1)`` and ``(n0 - n2)``. The
        intuitive correction therefore produces the same number, which is exactly why the
        omission needs a test: both forms are right, and a reader who adds the term back
        will not see a failure telling them it was unnecessary.
        """
        a, b = _fixture(2)
        rng = np.random.default_rng(102)
        w = rng.integers(1, 5, size=SIZE).astype(float)

        upper = np.triu(np.ones((SIZE, SIZE), dtype=bool), k=1)
        pair = np.outer(w, w) * upper
        sign_a = np.sign(a[:, None] - a[None, :])
        sign_b = np.sign(b[:, None] - b[None, :])
        numerator = (pair * sign_a * sign_b).sum()
        within = float(np.sum(w * (w - 1.0) / 2.0))

        def tau_b(extra: float) -> float:
            total = pair.sum() + extra
            tied_a = (pair * (sign_a == 0)).sum() + extra
            tied_b = (pair * (sign_b == 0)).sum() + extra
            return float(numerator / np.sqrt((total - tied_a) * (total - tied_b)))

        assert within > 0.0, "the fixture has no replicated elements, so nothing is tested"
        assert tau_b(0.0) == pytest.approx(tau_b(within), abs=1e-12)
        assert float(
            kendall_tau(jnp.asarray(a), jnp.asarray(b), weights=jnp.asarray(w))
        ) == pytest.approx(tau_b(within), abs=TOLERANCE)


class TestTheMask:
    """A mask is the unambiguous half: zero weight removes every pair an element is in."""

    @pytest.mark.parametrize("seed", [0, 3, 11])
    def test_a_mask_equals_subsetting(self, seed: int) -> None:
        a, b = _fixture(seed)
        rng = np.random.default_rng(seed + 200)
        keep = rng.random(SIZE) > 0.35
        assert keep.sum() >= 3, "fixture kept too few elements to correlate"
        assert float(
            kendall_tau(jnp.asarray(a), jnp.asarray(b), mask=jnp.asarray(keep))
        ) == pytest.approx(
            float(kendall_tau(jnp.asarray(a[keep]), jnp.asarray(b[keep]))), abs=TOLERANCE
        )

    def test_a_mask_composes_with_weights(
        self,
    ) -> None:
        """Masking then weighting equals weighting the subset, as `sample_weights` means."""
        a, b = _fixture(7)
        rng = np.random.default_rng(207)
        keep = rng.random(SIZE) > 0.3
        w = rng.integers(1, 4, size=SIZE).astype(float)
        combined = float(
            kendall_tau(
                jnp.asarray(a), jnp.asarray(b), mask=jnp.asarray(keep), weights=jnp.asarray(w)
            )
        )
        subset = float(
            kendall_tau(jnp.asarray(a[keep]), jnp.asarray(b[keep]), weights=jnp.asarray(w[keep]))
        )
        assert combined == pytest.approx(subset, abs=TOLERANCE)


class TestTheUnweightedPathIsUnchanged:
    """The default path must be the previous computation, bit for bit."""

    @pytest.mark.parametrize("tied", [False, True])
    @pytest.mark.parametrize("seed", [0, 5, 11])
    def test_unit_weights_match_the_default_path(self, seed: int, tied: bool) -> None:
        """Equal weights reproduce the unweighted value.

        Not bit-identical, deliberately: the weighted branch sums a matrix of ones where
        the default branch uses exact pair counts, so this is an equality of value and the
        tolerance is the float32 budget. The default branch exists so that callers who pass
        nothing keep the old number exactly.
        """
        a, b = _fixture(seed, tied=tied)
        plain = float(kendall_tau(jnp.asarray(a), jnp.asarray(b)))
        ones = float(kendall_tau(jnp.asarray(a), jnp.asarray(b), weights=jnp.ones(SIZE)))
        assert ones == pytest.approx(plain, abs=TOLERANCE)

    def test_the_default_path_still_matches_scipy(self) -> None:
        a, b = _fixture(3, tied=True)
        assert float(kendall_tau(jnp.asarray(a), jnp.asarray(b))) == pytest.approx(
            kendalltau(a, b).statistic, abs=TOLERANCE
        )


class TestTransformations:
    """Both branches must survive the transformations the package promises."""

    def test_the_weighted_path_jits_vmaps_and_differentiates(self) -> None:
        a, b = _fixture(0)
        a_j, b_j = jnp.asarray(a), jnp.asarray(b)
        w = jnp.asarray(np.random.default_rng(0).integers(1, 4, size=SIZE), dtype=float)

        def metric(x: jnp.ndarray, y: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
            return kendall_tau(x, y, weights=v)

        assert jnp.isfinite(jax.jit(metric)(a_j, b_j, w))
        batched = jax.jit(jax.vmap(metric))(
            jnp.stack([a_j] * 3), jnp.stack([b_j] * 3), jnp.stack([w] * 3)
        )
        assert batched.shape == (3,)
        # `sign` has zero gradient almost everywhere, so the gradient with respect to the
        # weights is the one that carries signal; it must at least be finite.
        assert jnp.all(jnp.isfinite(jax.grad(lambda v: metric(a_j, b_j, v))(w)))

    def test_the_masked_path_traces(self) -> None:
        a, b = _fixture(1)
        keep = jnp.asarray(np.random.default_rng(1).random(SIZE) > 0.3)
        traced = jax.jit(lambda x, y, m: kendall_tau(x, y, mask=m))(
            jnp.asarray(a), jnp.asarray(b), keep
        )
        assert jnp.isfinite(traced)
