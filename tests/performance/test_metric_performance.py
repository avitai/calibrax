"""Performance benchmarks for the metric paths an eager call pays for.

CodSpeed runs this directory under its simulation instrument, which counts instructions rather
than timing them, so a benchmark here moves when the Python-level work of an eager call moves,
even where the compiled program is identical. That is the cost a helper adds when it does work
the compiler would have absorbed, and it is invisible to anything measured under ``jax.jit``.

A metric that takes ``mask`` or ``weights`` has two paths. The unweighted path is the one every
existing caller uses, so it is benchmarked for each ratio statistic; the masked path is the
cost a caller opts into, so it is benchmarked for the losses whose substitution it carries and
for the Spearman correlation, whose ranking it changes. The Spearman correlation is also
benchmarked compiled, since its sort is the one compiled step here that grows faster than the
input.

Each benchmark asserts its value. A benchmark that asserts nothing is free to measure nothing.
Every value below is exact by construction, or, for the Spearman correlation, scipy's float64
answer on a fixed fixture, so none is a recorded measurement that could drift with the platform.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from scipy import stats

from calibrax.metrics.functional.regression import mse, r_squared, relative_l2_error
from calibrax.metrics.functional.statistical import (
    concordance_correlation,
    pearson_correlation,
    spearman_rank_correlation,
)


#: Small on purpose. CodSpeed's simulation instrument cannot simulate system calls
#: consistently and leaves them out, so a benchmark dominated by them is measured as less than
#: it is, and allocation is what generates them here. The size buys nothing for what these
#: benchmarks detect: every benchmarked path makes the same number of Python-level calls at
#: 100,000, 10,000 and 2,000 elements.
_SIZE = 10_000

#: A field-shaped fixture for ``relative_l2_error``, whose mask is per sample.
_SAMPLES, _FIELD = 400, 50


@pytest.mark.performance
def test_mse_large_vector_performance(benchmark: BenchmarkFixture) -> None:
    """MSE on a large vector, the benchmark CodSpeed holds a baseline for.

    It keeps its own 100,000-element fixture: a smaller workload would be compared against
    that baseline and read as an improvement.
    """
    predictions = jnp.ones((100_000,), dtype=jnp.float32)
    targets = jnp.zeros((100_000,), dtype=jnp.float32)

    result = benchmark(mse, predictions, targets)

    assert result == pytest.approx(1.0)


@pytest.mark.performance
def test_pearson_unweighted_performance(benchmark: BenchmarkFixture) -> None:
    """The unweighted path of the Pearson correlation.

    ``y`` is an exact affine function of ``x``, so the correlation is 1 by construction.
    """
    x = jnp.linspace(-1.0, 1.0, _SIZE, dtype=jnp.float32)
    y = 2.0 * x + 3.0

    result = benchmark(pearson_correlation, x, y)

    assert result == pytest.approx(1.0, abs=1e-6)


@pytest.mark.performance
def test_concordance_unweighted_performance(benchmark: BenchmarkFixture) -> None:
    """The unweighted path of the concordance correlation; identical inputs give 1."""
    x = jnp.linspace(-1.0, 1.0, _SIZE, dtype=jnp.float32)

    result = benchmark(concordance_correlation, x, x)

    assert result == pytest.approx(1.0, abs=1e-6)


def _spearman_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffled integer scores with heavy ties, a noisy monotone partner, and a mask.

    100 distinct values over 10,000 elements, so every rank is an average over a tie group
    and the sort has real work to do -- unlike a sorted, tie-free input, which ranks by
    position.
    """
    rng = np.random.default_rng(0)
    x = rng.integers(0, 100, _SIZE).astype(np.float32)
    y = (x + rng.normal(scale=25.0, size=_SIZE)).round().astype(np.float32)
    keep = rng.random(_SIZE) > 0.3
    return x, y, keep


_SPEARMAN_X, _SPEARMAN_Y, _SPEARMAN_KEEP = _spearman_fixture()

#: scipy's average-rank answers in float64, the reference every Spearman benchmark asserts.
_SPEARMAN_RHO = float(stats.spearmanr(_SPEARMAN_X, _SPEARMAN_Y).statistic)
_SPEARMAN_RHO_KEPT = float(
    stats.spearmanr(_SPEARMAN_X[_SPEARMAN_KEEP], _SPEARMAN_Y[_SPEARMAN_KEEP]).statistic
)

#: float32 against scipy's float64. Measured over 120 fixtures of this shape (40 seeds by 20,
#: 100 and 1,000 distinct values), eager, masked and jitted alike, the worst difference was
#: 1.73e-07, about 1.5 float32 ULP at 1.
_SPEARMAN_TOLERANCE = 1e-6


def test_the_spearman_fixture_tells_ranking_defects_apart() -> None:
    """The answers the benchmarks assert move by far more than the tolerance if ranks are
    taken by position (ties broken by order) or over the whole array instead of the kept
    subset, so a benchmark asserting them also detects either defect."""
    ordinal = stats.pearsonr(
        stats.rankdata(_SPEARMAN_X, method="ordinal"),
        stats.rankdata(_SPEARMAN_Y, method="ordinal"),
    ).statistic
    assert abs(ordinal - _SPEARMAN_RHO) > 10 * _SPEARMAN_TOLERANCE
    assert abs(_SPEARMAN_RHO_KEPT - _SPEARMAN_RHO) > 10 * _SPEARMAN_TOLERANCE


@pytest.mark.performance
def test_spearman_unweighted_performance(benchmark: BenchmarkFixture) -> None:
    """The unweighted path of the Spearman correlation, which runs through Pearson's.

    It takes no ``weights``, and is benchmarked anyway: the ranks it correlates go through
    ``pearson_correlation``, so a cost added there reaches it.
    """
    x, y = jnp.asarray(_SPEARMAN_X), jnp.asarray(_SPEARMAN_Y)

    result = benchmark(spearman_rank_correlation, x, y)

    assert result == pytest.approx(_SPEARMAN_RHO, abs=_SPEARMAN_TOLERANCE)


@pytest.mark.performance
def test_spearman_masked_performance(benchmark: BenchmarkFixture) -> None:
    """The masked path, which ranks the kept subset and correlates through Pearson's
    weighted branch, the cost a caller opts into."""
    x, y = jnp.asarray(_SPEARMAN_X), jnp.asarray(_SPEARMAN_Y)
    keep = jnp.asarray(_SPEARMAN_KEEP)

    result = benchmark(spearman_rank_correlation, x, y, mask=keep)

    assert result == pytest.approx(_SPEARMAN_RHO_KEPT, abs=_SPEARMAN_TOLERANCE)


@pytest.mark.performance
def test_spearman_jitted_performance(benchmark: BenchmarkFixture) -> None:
    """The compiled program, its sort and correlation, with dispatch out of the way.

    Compiled before timing, and each round waits for the result, so a round is one run of
    the program rather than an enqueue.
    """
    x, y = jnp.asarray(_SPEARMAN_X), jnp.asarray(_SPEARMAN_Y)
    compiled = jax.jit(spearman_rank_correlation)
    jax.block_until_ready(compiled(x, y))

    result = benchmark(lambda: jax.block_until_ready(compiled(x, y)))

    assert result == pytest.approx(_SPEARMAN_RHO, abs=_SPEARMAN_TOLERANCE)


@pytest.mark.performance
def test_r_squared_unweighted_performance(benchmark: BenchmarkFixture) -> None:
    """The unweighted path of R-squared, through the shared core; identical inputs give 1."""
    x = jnp.linspace(-1.0, 1.0, _SIZE, dtype=jnp.float32)

    result = benchmark(r_squared, x, x)

    assert result == pytest.approx(1.0, abs=1e-6)


@pytest.mark.performance
def test_mse_masked_performance(benchmark: BenchmarkFixture) -> None:
    """The masked path of MSE, the cost a caller opts into.

    Every kept element has a squared error of exactly 1, so the mean over them is 1
    whatever the mask keeps.
    """
    predictions = jnp.ones((_SIZE,), dtype=jnp.float32)
    targets = jnp.zeros((_SIZE,), dtype=jnp.float32)
    keep = jnp.arange(_SIZE) % 2 == 0

    result = benchmark(mse, predictions, targets, mask=keep)

    assert result == pytest.approx(1.0)


@pytest.mark.performance
def test_relative_l2_error_masked_performance(benchmark: BenchmarkFixture) -> None:
    """The masked path of the one metric whose mask is dimensioned per sample.

    ``||2f - f|| / ||f||`` is 1 for every sample, so the mean over the kept ones is 1.
    """
    fields = jnp.ones((_SAMPLES, _FIELD), dtype=jnp.float32)
    keep = jnp.arange(_SAMPLES) % 2 == 0

    result = benchmark(relative_l2_error, 2.0 * fields, fields, mask=keep)

    assert result == pytest.approx(1.0)
