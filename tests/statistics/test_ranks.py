"""Tests for exact host decimal rank, screening cutoff, and Farey BH threshold helpers."""

from __future__ import annotations

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.statistics import (
    conformal_plus_ranks,
    conformal_quantile_rank,
    decimal_fraction,
    discrete_bh_count_thresholds,
    farey_floor_rational,
    minimum_conformal_calibration_size,
    screening_cutoff_count,
)


_R_FUZZ_EPS64 = 4.0 * float(np.finfo(np.float64).eps)


def _r_quantile_floor(x: float) -> int:
    """R ``stats::quantile.default`` relative-fuzz floor (``src/library/stats/R/quantile.R:53``)."""
    return math.floor(x * (1.0 + _R_FUZZ_EPS64))


class TestDecimalRankAndCutoffHelpers:
    """Tests for exact decimal probability-to-rank conversion in calibrax.statistics."""

    def test_decimal_fraction_recovers_caller_decimal_across_scalar_types(self) -> None:
        for val in (
            0.1,
            np.float64(0.1),
            np.float32(0.1),
            np.float16(0.1),
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(0.1, dtype=jnp.float16),
        ):
            assert decimal_fraction(val) == Fraction(1, 10)
        for val in (
            0.3,
            np.float64(0.3),
            np.float32(0.3),
            np.float16(0.3),
            jnp.asarray(0.3, dtype=jnp.float32),
            jnp.asarray(0.3, dtype=jnp.float16),
        ):
            assert decimal_fraction(val) == Fraction(3, 10)
        with jax.enable_x64(True):
            assert decimal_fraction(jnp.asarray(0.1, dtype=jnp.float64)) == Fraction(1, 10)
            assert decimal_fraction(jnp.asarray(0.3, dtype=jnp.float64)) == Fraction(3, 10)
        for val in (0.9, np.float64(0.9), np.float32(0.9), jnp.asarray(0.9)):
            assert decimal_fraction(val) == Fraction(9, 10)
        for val in (0.07, np.float64(0.07), np.float32(0.07), jnp.float32(0.07)):
            assert decimal_fraction(val) == Fraction(7, 100)
        assert decimal_fraction(1) == Fraction(1, 1)
        assert decimal_fraction(jnp.array(1, dtype=jnp.int32)) == Fraction(1, 1)

        with pytest.raises(TypeError, match="Python float"):
            decimal_fraction(jnp.asarray(0.3, dtype=jnp.bfloat16))
        with pytest.raises(TypeError, match="Python float"):
            conformal_quantile_rank(10, jnp.asarray(0.3, dtype=jnp.bfloat16))

        q_below_f32 = np.nextafter(np.float32(0.05), np.float32(0.0))
        assert decimal_fraction(q_below_f32) < Fraction(1, 20)
        assert decimal_fraction(float(q_below_f32)) < Fraction(1, 20)

    def test_matches_r_stats_quantile_relative_fuzz_and_lei_2018_boundaries(self) -> None:
        # Explicit cases where IEEE-754 binary-double Fraction(alpha) was 1 rank too high:
        assert conformal_quantile_rank(19, 0.15) == 17
        assert conformal_quantile_rank(9, 0.3) == 7
        assert conformal_quantile_rank(19, 0.35) == 13
        assert minimum_conformal_calibration_size(0.1) == 9
        assert minimum_conformal_calibration_size(0.3) == 3
        assert minimum_conformal_calibration_size(1.0 / 3.0) == 3
        assert conformal_plus_ranks(9, 0.3) == (3, 7)

        # Cross-check against R stats::quantile relative fuzz (src/library/stats/R/quantile.R:27,53,73)
        # across 2-, 3-, and 4-digit decimals and sample sizes up to 10^9.
        rng = np.random.default_rng(0)
        for digits, sizes in (
            (2, list(range(1, 2000, 17))),
            (3, [int(x) for x in rng.integers(1, 10**6 + 1, size=250)]),
            (4, [int(x) for x in rng.integers(1, 10**9 + 1, size=250)]),
        ):
            all_alphas = [round(i * (10**-digits), digits) for i in range(1, 10**digits)]
            if len(all_alphas) <= 120:
                chosen = all_alphas
            else:
                pick_idx = rng.choice(len(all_alphas), size=120, replace=False)
                chosen = [all_alphas[int(k)] for k in pick_idx]
            for alpha in chosen:
                for n in sizes:
                    total = n + 1
                    lo_rank, hi_rank = conformal_plus_ranks(n, alpha)
                    assert hi_rank == conformal_quantile_rank(n, alpha)
                    assert hi_rank == total - _r_quantile_floor(total * alpha)
                    assert lo_rank == _r_quantile_floor(total * alpha)

    def test_screening_cutoff_count_large_libraries_and_validation(self) -> None:
        assert screening_cutoff_count(100, 0.07) == 7
        assert screening_cutoff_count(10_000_001, 0.01) == 100_001
        assert screening_cutoff_count(100_403, 0.33) == math.ceil(100_403 * Fraction(33, 100))
        assert screening_cutoff_count(10, 1.0) == 10
        assert screening_cutoff_count(3, 0.01) == 1

        for bad_alpha in (0.0, 1.0, -0.1, 1.2, float("nan"), jnp.asarray(1.5)):
            with pytest.raises(ValueError, match="alpha"):
                conformal_quantile_rank(10, bad_alpha)
            with pytest.raises(ValueError, match="alpha"):
                minimum_conformal_calibration_size(bad_alpha)
            with pytest.raises(ValueError, match="alpha"):
                conformal_plus_ranks(10, bad_alpha)
        with pytest.raises(ValueError, match="n must be at least 1"):
            conformal_quantile_rank(0, 0.1)
        with pytest.raises(ValueError, match="n must be at least 1"):
            conformal_plus_ranks(0, 0.1)
        with pytest.raises(ValueError, match="n_items"):
            screening_cutoff_count(0, 0.1)
        for bad_frac in (0.0, -0.1, 1.1, float("nan"), jnp.asarray(0.0)):
            with pytest.raises(ValueError, match="fraction"):
                screening_cutoff_count(10, bad_frac)
        with pytest.raises(TypeError, match="static_argnames"):
            jax.jit(lambda a: conformal_quantile_rank(10, a))(jnp.float32(0.1))

    def test_farey_floor_rational_and_discrete_bh_thresholds_exhaustive(self) -> None:
        rng = np.random.default_rng(42)
        nums = rng.integers(0, 10**18, size=150)
        dens = rng.integers(1, 10**18, size=150)
        rationals = [
            Fraction(0, 1),
            Fraction(1, 10**300),
            Fraction(3, 10),
            Fraction(9, 10),
            decimal_fraction(float(np.nextafter(np.float32(0.05), np.float32(0.0)))),
            decimal_fraction(float(np.nextafter(1.0, 0.0))),
            Fraction(3333333333333333, 10**16),
            *[Fraction(int(n), int(d)) for n, d in zip(nums, dens, strict=True)],
        ]
        for alpha in rationals:
            for max_den in (1, 2, 7, 12, 59, 60, 120):
                beta = farey_floor_rational(alpha, max_den)
                assert beta.denominator <= max_den
                assert beta <= alpha
                for d in range(1, max_den + 1):
                    assert (d * beta.numerator) // beta.denominator == (
                        (d * alpha.numerator) // alpha.denominator
                    )

        with pytest.raises(ValueError, match="max_denominator"):
            farey_floor_rational(Fraction(1, 2), 0)

        # Traced denominator under vmap agrees identically with concrete integer denominator
        # across all d in 1..65 (including large denominators and sub-ULP q).
        for q_val in (0.05, 0.3, 0.9, float(np.nextafter(np.float32(0.05), np.float32(0.0)))):
            denoms = jnp.arange(1, 66, dtype=jnp.int32)
            traced_mat = jax.jit(
                jax.vmap(
                    lambda d, qv=q_val: discrete_bh_count_thresholds(7, d, q=qv, max_denominator=65)
                )
            )(denoms)
            for idx, d_int in enumerate(range(1, 66)):
                eager_row = discrete_bh_count_thresholds(7, d_int, q=q_val)
                np.testing.assert_array_equal(np.asarray(traced_mat[idx]), np.asarray(eager_row))

        # Exercise the > 32-bit overflow-free uint32 product branch (max_k * b_num >= 2^32).
        q_large = 0.7 / 3.0
        large_traced = jax.jit(
            lambda d: discrete_bh_count_thresholds(2000, d, q=q_large, max_denominator=2000)
        )(jnp.int32(1999))
        large_eager = discrete_bh_count_thresholds(2000, 1999, q=q_large)
        np.testing.assert_array_equal(np.asarray(large_traced), np.asarray(large_eager))
        assert discrete_bh_count_thresholds(5, 0, q=0.1).tolist() == [0, 0, 0, 0, 0]
        assert jax.jit(lambda d: discrete_bh_count_thresholds(5, d, q=0.1, max_denominator=10))(
            jnp.int32(0)
        ).tolist() == [0, 0, 0, 0, 0]
        with pytest.raises(ValueError, match="max_denominator"):
            jax.jit(lambda d: discrete_bh_count_thresholds(5, d, q=0.1))(jnp.int32(3))
