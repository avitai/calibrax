"""Performance benchmarks for the metric paths an eager call pays for.

CodSpeed runs this directory under its simulation instrument, which counts instructions rather
than timing them, so a benchmark here moves when the Python-level work of an eager call moves,
even where the compiled program is identical. That is the cost a helper adds when it does work
the compiler would have absorbed, and it is invisible to anything measured under ``jax.jit``.

A metric that takes ``mask`` or ``weights`` has two paths. The unweighted path is the one every
existing caller uses, so it is benchmarked for each ratio statistic; the masked path is the
cost a caller opts into, so it is benchmarked for the losses whose substitution it carries.

Each benchmark calls the metric eagerly and asserts its value. A benchmark that asserts
nothing is free to measure nothing, and every value below is exact by construction rather than
a recorded measurement, so it cannot drift with the platform.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

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


@pytest.mark.performance
def test_spearman_unweighted_performance(benchmark: BenchmarkFixture) -> None:
    """The unweighted path of the Spearman correlation, which runs through Pearson's.

    It takes no ``weights``, and is benchmarked anyway: the ranks it correlates go through
    ``pearson_correlation``, so a cost added there reaches it.
    """
    x = jnp.linspace(-1.0, 1.0, _SIZE, dtype=jnp.float32)

    result = benchmark(spearman_rank_correlation, x, x)

    assert result == pytest.approx(1.0, abs=1e-6)


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
