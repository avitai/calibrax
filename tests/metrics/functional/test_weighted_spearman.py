"""Frequency weights on ``spearman_rank_correlation``, against physical replication.

A frequency weight ``w_i`` says element ``i`` was observed ``w_i`` times, so the Spearman
coefficient of the weighted sample is the Spearman coefficient of the replicated sample, with
no convention to choose. An element's midrank in the replicated sample is the weight strictly
below its value, plus half the weight tied with it, plus one half; a correlation ignores the
constant half. Spearman's coefficient of a contingency table is computed the same way from its
counts (the R package ``contingencytables``, chapter 7, "The r x c table"), and it is what SAS's
``FREQ`` statement means: each observation "represents" that many observations.

Sampling or probability weights are a different question, which ``wCorr`` answers with its own
convention and Stata and SAS's ``WEIGHT`` statement do not answer at all. SPSS answers it with
replication too: ``expss``'s ``w_spearman`` rounds the weights and repeats each row that many
times, and documents that "it gives the same result as in SPSS Statistics software".

For weights that are not counts, replication has no meaning, and the ranks are R Hmisc's
``wtd.rank`` (``cumsum(freqs) - .5*(freqs-1)`` per distinct value), which takes any
non-negative weights and equals the replicated midranks on counts.

The reference is ``scipy.stats.spearmanr`` on ``np.repeat`` of the data, an implementation of
the unweighted statistic that shares nothing with this one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from scipy.stats import rankdata, spearmanr

from calibrax.metrics.functional.statistical import spearman_rank_correlation
from tests.metrics import float64_references as references


#: Float32 against float64 references. Measured over 360 tied fixtures (40 seeds by 4, 50 and
#: 10,000 distinct values by sizes 16, 256 and 4,096), counts eager and jitted against
#: replication and uniform weights against the Hmisc reference, the worst difference was
#: 2.89e-07, 2.4 float32 ULP at 1; this allows 1e-6.
TOLERANCE = 1e-6

SIZE = 16


def _fixture(seed: int, *, levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Two integer-valued series over ``levels`` values, so ties occur in both.

    Args:
        seed: Seed for the generator, so each case is reproducible on its own.
        levels: Number of distinct values ``a`` draws from; few levels means many ties.

    Returns:
        The two series as float64 numpy arrays.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, levels, SIZE).astype(np.float64)
    b = a + rng.integers(-2, 3, SIZE)
    return a, b


def _counts(seed: int) -> np.ndarray:
    """Integer frequencies from 0 to 4, zeros included, as a bootstrap resample produces."""
    return np.random.default_rng(seed + 100).integers(0, 5, SIZE)


def _replicated(a: np.ndarray, b: np.ndarray, counts: np.ndarray) -> float:
    """Scipy's coefficient of the sample with element ``i`` repeated ``counts[i]`` times."""
    return float(spearmanr(np.repeat(a, counts), np.repeat(b, counts)).statistic)


def _weighted(
    a: np.ndarray, b: np.ndarray, weights: np.ndarray, *, mask: ArrayLike | None = None
) -> float:
    """The function under test, in float32, as a Python float."""
    return float(
        spearman_rank_correlation(
            jnp.asarray(a, dtype=jnp.float32),
            jnp.asarray(b, dtype=jnp.float32),
            mask=mask,
            weights=jnp.asarray(weights, dtype=jnp.float32),
        )
    )


class TestIntegerWeightsEqualReplication:
    """The property that makes these frequency weights rather than a convention."""

    @pytest.mark.parametrize("levels", [4, 50])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 11])
    def test_matches_scipy_on_the_replicated_sample(self, seed: int, levels: int) -> None:
        """Weighting by ``w`` equals repeating each element ``w`` times, ties and zeros included."""
        a, b = _fixture(seed, levels=levels)
        counts = _counts(seed)
        reference = _replicated(a, b, counts)
        assert np.isfinite(reference), "degenerate fixture: scipy returned no statistic"
        assert _weighted(a, b, counts) == pytest.approx(reference, abs=TOLERANCE)

    def test_the_sample_weight_convention_does_not_have_the_property(self) -> None:
        """Positive control: ``wCorr``'s ranks, ``below + (n_tied + 1) / 2 * mean_tied_weight``,
        are the plausible alternative, and on this fixture they miss replication by 3.3e-02.
        Without this, the test above could pass for a rank that merely agrees with
        replication on easy fixtures; over 40 such fixtures the gap ranges from 1.1e-04 to
        3.3e-02, and this is the widest."""
        a, b = _fixture(26, levels=4)
        counts = _counts(26).astype(np.float64)

        def wcorr_ranks(x: np.ndarray) -> np.ndarray:
            below = np.array([counts[x < v].sum() for v in x])
            tied = [x == v for v in x]
            return below + np.array([(t.sum() + 1) / 2 * counts[t].mean() for t in tied])

        ra, rb = wcorr_ranks(a), wcorr_ranks(b)
        mean_a, mean_b = np.average(ra, weights=counts), np.average(rb, weights=counts)
        covariance = np.sum(counts * (ra - mean_a) * (rb - mean_b))
        alternative = covariance / np.sqrt(
            np.sum(counts * (ra - mean_a) ** 2) * np.sum(counts * (rb - mean_b) ** 2)
        )
        reference = _replicated(a, b, counts.astype(int))
        assert abs(alternative - reference) > 1_000 * TOLERANCE


class TestNonIntegerWeightsFollowHmisc:
    """Weights that are not counts take Hmisc's ranks, which replication cannot check."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_the_hmisc_port_ranks_like_the_replicated_sample(self, seed: int) -> None:
        """The reference is itself checked, against scipy, before it judges anything.

        On counts each kept element's rank must equal ``rankdata`` of the replicated sample at
        that element's copies, exactly: both are sums of integers and halves. An element with
        count zero is absent from the replicated sample and has no rank there to compare.
        """
        a, _ = _fixture(seed, levels=4)
        counts = _counts(seed)
        replicated_ranks = rankdata(np.repeat(a, counts))
        first_copy = np.concatenate([[0], np.cumsum(counts)[:-1]])
        kept = counts > 0
        np.testing.assert_array_equal(
            references.frequency_ranks(a, counts.astype(np.float64))[kept],
            replicated_ranks[first_copy[kept]],
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 11])
    def test_matches_the_hmisc_reference(self, seed: int) -> None:
        a, b = _fixture(seed, levels=4)
        weights = np.random.default_rng(seed + 200).uniform(0.0, 3.0, SIZE)
        assert _weighted(a, b, weights) == pytest.approx(
            references.spearman(a, b, weights), abs=TOLERANCE
        )


class TestMaskAndWeights:
    """A mask is a 0/1 frequency, and the two compose the way ``sample_weights`` folds them."""

    def test_unit_weights_equal_the_unweighted_statistic(self) -> None:
        a, b = _fixture(2, levels=4)
        unweighted = float(spearman_rank_correlation(jnp.asarray(a), jnp.asarray(b)))
        assert _weighted(a, b, np.ones(SIZE)) == pytest.approx(unweighted, abs=TOLERANCE)

    def test_a_mask_equals_zero_one_weights(self) -> None:
        a, b = _fixture(3, levels=4)
        keep = np.random.default_rng(3).random(SIZE) > 0.3
        masked = float(
            spearman_rank_correlation(
                jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32), mask=jnp.asarray(keep)
            )
        )
        assert masked == pytest.approx(_weighted(a, b, keep.astype(float)), abs=TOLERANCE)

    def test_mask_and_weights_combine(self) -> None:
        """An element the mask excludes contributes nothing, whatever its weight."""
        a, b = _fixture(7, levels=4)
        counts = _counts(7)
        keep = np.random.default_rng(7).random(SIZE) > 0.3
        combined = _weighted(a, b, counts, mask=jnp.asarray(keep))
        assert combined == pytest.approx(_replicated(a, b, counts * keep), abs=TOLERANCE)

    @pytest.mark.parametrize("scale", [0.5, 3.0])
    def test_scaling_every_weight_changes_nothing(self, scale: float) -> None:
        """Frequencies matter only in proportion, since every rank scales by the same factor."""
        a, b = _fixture(11, levels=4)
        counts = _counts(11).astype(np.float64)
        assert _weighted(a, b, scale * counts) == pytest.approx(
            _weighted(a, b, counts), abs=TOLERANCE
        )

    def test_a_kept_infinity_ranks_among_the_kept(self) -> None:
        """An infinity is a value like any other and sorts last among the kept elements."""
        a, b = _fixture(0, levels=50)
        a[[2, 9]] = np.inf
        counts = _counts(0)
        counts[2], counts[9] = 2, 0
        assert _weighted(a, b, counts) == pytest.approx(_replicated(a, b, counts), abs=TOLERANCE)


class TestEmptyAndNonFinite:
    """An empty selection reads NaN, and only a kept non-finite value reaches the result."""

    def test_all_zero_weights_read_nan(self) -> None:
        a, b = _fixture(0, levels=4)
        assert np.isnan(_weighted(a, b, np.zeros(SIZE)))

    def test_a_zero_weighted_nan_is_neutral(self) -> None:
        a, b = _fixture(1, levels=50)
        counts = _counts(1)
        counts[4] = 0
        dirty = a.copy()
        dirty[4] = np.nan
        assert _weighted(dirty, b, counts) == pytest.approx(
            _replicated(a, b, counts), abs=TOLERANCE
        )

    def test_a_weighted_nan_propagates(self) -> None:
        a, b = _fixture(1, levels=50)
        counts = _counts(1)
        counts[4] = 2
        dirty = a.copy()
        dirty[4] = np.nan
        assert np.isnan(_weighted(dirty, b, counts))


class TestTheBootstrapItServes:
    """A within-subset bootstrap at a fixed shape: multinomial counts over the subset's
    members, one count vector per resample, under ``jit(vmap)``."""

    def test_vmapped_resamples_equal_scipy_on_each_replicated_resample(self) -> None:
        a, b = _fixture(2, levels=50)
        members = np.flatnonzero(np.random.default_rng(2).random(SIZE) > 0.4)
        rng = np.random.default_rng(12)
        counts = np.zeros((5, SIZE), dtype=np.int64)
        for row in counts:
            np.add.at(row, rng.choice(members, size=members.size), 1)

        batched = jax.jit(
            jax.vmap(lambda w: spearman_rank_correlation(jnp.asarray(a), jnp.asarray(b), weights=w))
        )(jnp.asarray(counts, dtype=jnp.float32))

        expected = np.array([_replicated(a, b, row) for row in counts])
        np.testing.assert_allclose(np.asarray(batched), expected, atol=TOLERANCE, rtol=0.0)

    def test_the_weight_gradient_is_finite_under_jit(self) -> None:
        a, b = _fixture(3, levels=4)
        counts = jnp.asarray(_counts(3), dtype=jnp.float32)
        gradient = jax.jit(
            jax.grad(lambda w: spearman_rank_correlation(jnp.asarray(a), jnp.asarray(b), weights=w))
        )(counts)
        assert bool(jnp.all(jnp.isfinite(gradient)))
