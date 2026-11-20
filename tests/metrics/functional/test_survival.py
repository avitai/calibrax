"""Tests for censored survival and potency concordance and coverage bounds."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from substrax.testing import TraceCounter

from calibrax.core.models import MetricDirection
from calibrax.metrics import MetricRegistry
from calibrax.metrics.functional.survival import (
    censored_coverage_bounds,
    harrell_c_index,
    uno_c_index,
)


class TestHarrellCIndex:
    """Harrell concordance index for exact, right-, left-, and interval-censored outcomes."""

    def test_exact_right_censored_and_tied_rational_values(self) -> None:
        # T = [1, 2, 3, 4, 5], delta = [1, 0, 1, 1, 1], pred = [10, 20, 30, 40, 35].
        # Comparable pairs (T_i > T_j with delta_j == 1):
        # j=0 (T=1): i in {1, 2, 3, 4} -> 4 concordant (note i=1 censored at T=2 > 1 IS comparable)
        # j=1 (T=2, delta=0): 0 pairs (right-censored earlier time cannot upper-bound Y_1)
        # j=2 (T=3): i in {3, 4} -> 2 concordant
        # j=3 (T=4): i in {4} -> pred[4]=35 < pred[3]=40 -> 0 concordant (1 discordant)
        # Total = 6 / 7.
        times = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
        events = jnp.array([1, 0, 1, 1, 1], dtype=jnp.int32)
        preds = jnp.array([10.0, 20.0, 30.0, 40.0, 35.0], dtype=jnp.float32)
        val = harrell_c_index(preds, times, event_indicator=events)
        assert isinstance(val, jax.Array)
        assert float(val) == pytest.approx(6.0 / 7.0, abs=float(jnp.finfo(jnp.float32).eps))

        # Tied prediction scores count 0.5.
        preds_tied = jnp.array([10.0, 20.0, 30.0, 40.0, 40.0], dtype=jnp.float32)
        assert float(harrell_c_index(preds_tied, times, event_indicator=events)) == pytest.approx(
            6.5 / 7.0, abs=float(jnp.finfo(jnp.float32).eps)
        )

    def test_interval_and_left_censored_bounds(self) -> None:
        # Item 0: left-censored (-inf, 2.0]
        # Item 1: interval-censored [1.5, 3.5] (overlaps item 0 and item 2 -> incomparable with both)
        # Item 2: exact [3.0, 3.0] (strictly > item 0: 3.0 > 2.0)
        # Item 3: right-censored [4.0, +inf) (strictly > items 0, 1, 2: 4.0 > 2.0, 3.5, 3.0)
        # Comparable ordered pairs (lower[i] > upper[j]): (2,0), (3,0), (3,1), (3,2) -> 4 pairs.
        lower = jnp.array([-jnp.inf, 1.5, 3.0, 4.0], dtype=jnp.float32)
        upper = jnp.array([2.0, 3.5, 3.0, jnp.inf], dtype=jnp.float32)
        preds = jnp.array([1.0, 5.0, 2.0, 4.0], dtype=jnp.float32)
        # (2,0): 2>1 (1.0); (3,0): 4>1 (1.0); (3,1): 4<5 (0.0); (3,2): 4>2 (1.0) -> 3/4 = 0.75.
        assert float(harrell_c_index(preds, lower, upper)) == pytest.approx(0.75, abs=1e-7)

    def test_zero_comparable_pairs_returns_nan_and_invalid_inputs_raise(self) -> None:
        empty = jnp.zeros(0, dtype=jnp.float32)
        assert np.isnan(float(harrell_c_index(empty, empty)))
        assert np.isnan(float(harrell_c_index(jnp.array([1.0]), jnp.array([2.0]))))

        # All right-censored -> upper == +inf everywhere -> zero comparable pairs -> NaN.
        t = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        assert np.isnan(float(harrell_c_index(t, t, event_indicator=jnp.zeros(3))))

        with pytest.raises(ValueError, match="Cannot specify both"):
            harrell_c_index(t, t, t, event_indicator=jnp.ones(3))
        with pytest.raises(ValueError, match="Shape mismatch"):
            harrell_c_index(jnp.ones(3), jnp.ones(4))
        # Out-of-domain inverted intervals (lower > upper) return NaN under both eager and jit.
        assert np.isnan(
            float(harrell_c_index(jnp.ones(2), jnp.array([3.0, 1.0]), jnp.array([2.0, 2.0])))
        )
        assert np.isnan(
            float(
                jax.jit(harrell_c_index)(jnp.ones(2), jnp.array([3.0, 1.0]), jnp.array([2.0, 2.0]))
            )
        )

    def test_nan_and_non_binary_event_indicator_propagate_nan(self) -> None:
        times = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        preds = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
        events_i32 = jnp.array([1, 0, 1, 1], dtype=jnp.int32)
        events_bool = jnp.array([True, False, True, True])
        events_f32 = jnp.array([1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)

        assert float(harrell_c_index(preds, times, event_indicator=events_bool)) == pytest.approx(
            1.0
        )
        assert float(harrell_c_index(preds, times, event_indicator=events_f32)) == pytest.approx(
            1.0
        )

        nan_preds = preds.at[0].set(jnp.nan)
        nan_times = times.at[1].set(jnp.nan)
        nan_upper = jnp.array([1.0, jnp.nan, 3.0, 4.0], dtype=jnp.float32)
        nan_ev = events_f32.at[1].set(jnp.nan)

        for call in (
            lambda: harrell_c_index(nan_preds, times, event_indicator=events_i32),
            lambda: harrell_c_index(preds, nan_times, event_indicator=events_i32),
            lambda: harrell_c_index(preds, times, nan_upper),
            lambda: harrell_c_index(preds, times, event_indicator=nan_ev),
            lambda: harrell_c_index(preds, times, event_indicator=jnp.array([1, 2, 1, 0])),
            lambda: harrell_c_index(preds, times, event_indicator=jnp.array([-1, 0, 1, 1])),
            lambda: harrell_c_index(preds, times, event_indicator=jnp.array([1.0, 0.5, 1.0, 0.0])),
        ):
            assert np.isnan(float(call()))
            assert np.isnan(float(jax.jit(call)()))

        batch_preds = jnp.stack([preds, nan_preds])
        batch_times = jnp.stack([times, times])
        batch_ev = jnp.stack([events_i32, events_i32])
        counter = TraceCounter()
        compiled_harrell = jax.jit(
            counter.wrap(lambda p, t, e: harrell_c_index(p, t, event_indicator=e))
        )
        with counter.expect(new_traces=1):
            compiled_harrell(preds, times, events_i32)
        with counter.expect(new_traces=0):
            compiled_harrell(nan_preds, times, events_i32)
        vmapped = jax.vmap(lambda p, t, e: harrell_c_index(p, t, event_indicator=e))(
            batch_preds, batch_times, batch_ev
        )
        assert float(vmapped[0]) == pytest.approx(1.0)
        assert np.isnan(float(vmapped[1]))


class TestUnoCIndex:
    """IPCW concordance index (Uno et al., Stat. Med. 2011, Eqs. 2-5)."""

    def test_exact_ipcw_rational_value_and_horizon_truncation(self) -> None:
        # T = [1, 2, 3, 4, 5], delta = [1, 0, 1, 1, 1], scores = [10, 20, 30, 40, 35].
        # Censoring KM G_hat:
        # t=1: cens=0/5 -> G(1) = 1.0 -> w(1) = 1.0 (4 concordant pairs)
        # t=2: cens=1/4 -> G(2) = 3/4
        # t=3: cens=0/3 -> G(3) = 3/4 -> w(3) = 16/9 (2 concordant pairs)
        # t=4: cens=0/2 -> G(4) = 3/4 -> w(4) = 16/9 (1 discordant pair)
        # Numerator = 4*1 + 2*(16/9) = 68/9; Denominator = 4*1 + 3*(16/9) = 84/9 -> 17/21.
        times = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
        events = jnp.array([1, 0, 1, 1, 1], dtype=jnp.int32)
        scores = jnp.array([10.0, 20.0, 30.0, 40.0, 35.0], dtype=jnp.float32)

        val = uno_c_index(scores, times, events)
        assert float(val) == pytest.approx(17.0 / 21.0, abs=2 * float(jnp.finfo(jnp.float32).eps))

        # Truncating at horizon=3.5 excludes the T_3=4.0 discordant event -> exact 1.0.
        val_tau = uno_c_index(scores, times, events, horizon=3.5)
        assert float(val_tau) == pytest.approx(1.0, abs=float(jnp.finfo(jnp.float32).eps))

    def test_tied_censoring_times_use_simultaneous_kaplan_meier_risk_set(self) -> None:
        # Two simultaneous censorings at t=1 out of 4 at risk -> G(1) = 1 - 2/4 = 1/2 -> w = 4.
        # Event at t=2 (w=4) compared with t=3 (1 pair) + event at t=0.5 (G=1, w=1) compared with 3 later.
        times = jnp.array([0.5, 1.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
        events = jnp.array([1, 0, 0, 1, 1], dtype=jnp.int32)
        # t=0.5 (s=10) < {s=20,20,40,30}: all 4 concordant (weight 1 each -> 4*1 = 4).
        # t=2.0 (s=40) vs t=3.0 (s=30): discordant (weight 4 -> 0/4).
        # Total = 4 / (4 + 4) = 0.5.
        scores = jnp.array([10.0, 20.0, 20.0, 40.0, 30.0], dtype=jnp.float32)
        assert float(uno_c_index(scores, times, events)) == pytest.approx(0.5, abs=1e-7)

    def test_zero_ulp_agreement_with_harrell_when_uncensored(self) -> None:
        rng = np.random.default_rng(99)
        for _ in range(10):
            times = jnp.asarray(rng.uniform(0.1, 10.0, 24), dtype=jnp.float32)
            scores = jnp.asarray(rng.normal(size=24), dtype=jnp.float32)
            all_events = jnp.ones(24, dtype=jnp.int32)
            h_val = float(harrell_c_index(scores, times, event_indicator=all_events))
            u_val = float(uno_c_index(scores, times, all_events))
            assert u_val == h_val

    def test_degenerate_and_invalid_inputs_and_jit_vmap(self) -> None:
        empty = jnp.zeros(0, dtype=jnp.float32)
        assert np.isnan(float(uno_c_index(empty, empty, empty)))
        t = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        assert np.isnan(float(uno_c_index(t, t, jnp.zeros(3))))
        with pytest.raises(ValueError, match="horizon"):
            uno_c_index(t, t, jnp.ones(3), horizon=0.0)
        with pytest.raises(ValueError, match="Shape mismatch"):
            uno_c_index(jnp.ones(2), t, jnp.ones(3))

        counter = TraceCounter()
        compiled = jax.jit(counter.wrap(uno_c_index))
        batch_s = jnp.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=jnp.float32)
        batch_t = jnp.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
        batch_e = jnp.array([[1, 0, 1, 1], [1, 0, 1, 1]], dtype=jnp.int32)
        with counter.expect(new_traces=1):
            compiled(batch_s[0], batch_t[0], batch_e[0])
        with counter.expect(new_traces=0):
            compiled(batch_s[1], batch_t[1], batch_e[1])
        vmapped = jax.vmap(uno_c_index)(batch_s, batch_t, batch_e)
        np.testing.assert_allclose(np.asarray(vmapped), np.array([1.0, 0.0]), atol=1e-7)

    def test_tied_event_and_censoring_uses_left_continuous_kaplan_meier(self) -> None:
        # At t=2.0, one event (i=1) and one censoring (i=2) tie with 3 subjects at risk.
        # Zero censorings occurred strictly before t=2.0, so G_hat(2.0-) = 1.0 (weight 1.0),
        # whereas right-continuous G_hat(2.0) = 2/3 would inflate weight to 9/4 and yield 4/7.
        times = jnp.array([1.0, 2.0, 2.0, 3.0], dtype=jnp.float32)
        events = jnp.array([1, 1, 0, 1], dtype=jnp.int32)
        scores = jnp.array([10.0, 40.0, 50.0, 30.0], dtype=jnp.float32)
        assert float(uno_c_index(scores, times, events)) == pytest.approx(0.75, abs=1e-7)

        # Tie at t=1.0 (1 event, 1 censoring among 4 at risk): observed event precedes censoring,
        # so censoring hazard at t=1.0 is 1 / (4 - 1) = 1/3 -> factor = 2/3. Left-continuous
        # G_hat(1.0-) = 1.0 (w=1.0) for the 2 concordant pairs at t=1.0, while G_hat(2.0-) = 2/3
        # (w=9/4) for the 1 discordant pair at t=2.0 -> 2 / (2 + 9/4) = 8/17.
        times_early_tie = jnp.array([1.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
        events_early_tie = jnp.array([1, 0, 1, 1], dtype=jnp.int32)
        assert float(uno_c_index(scores, times_early_tie, events_early_tie)) == pytest.approx(
            8.0 / 17.0, abs=4 * float(jnp.finfo(jnp.float32).eps)
        )

        # Tied event and censoring at t_max=2.0 where right-continuous G_hat(2.0) would vanish to 0:
        # G_hat(1.0-) = 1.0 and G_hat(2.0-) = 1.0 remain strictly positive.
        times_tmax = jnp.array([1.0, 2.0, 2.0], dtype=jnp.float32)
        events_tmax = jnp.array([1, 1, 0], dtype=jnp.int32)
        scores_tmax = jnp.array([10.0, 30.0, 20.0], dtype=jnp.float32)
        assert float(uno_c_index(scores_tmax, times_tmax, events_tmax)) == pytest.approx(
            1.0, abs=float(jnp.finfo(jnp.float32).eps)
        )

    def test_nan_and_non_binary_event_indicator_propagate_nan(self) -> None:
        times = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        scores = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
        events_i32 = jnp.array([1, 0, 1, 1], dtype=jnp.int32)
        events_bool = jnp.array([True, False, True, True])
        events_f32 = jnp.array([1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)

        assert float(uno_c_index(scores, times, events_bool)) == pytest.approx(1.0)
        assert float(uno_c_index(scores, times, events_f32)) == pytest.approx(1.0)

        for call in (
            lambda: uno_c_index(scores.at[0].set(jnp.nan), times, events_i32),
            lambda: uno_c_index(scores, times.at[1].set(jnp.nan), events_i32),
            lambda: uno_c_index(scores, times, events_f32.at[1].set(jnp.nan)),
            lambda: uno_c_index(scores, times, jnp.array([1, 2, 1, 0])),
            lambda: uno_c_index(scores, times, jnp.array([1.0, 0.5, 1.0, 0.0])),
            lambda: uno_c_index(scores, times, events_i32, horizon=jnp.array(jnp.nan)),
            lambda: uno_c_index(scores, times, events_i32, horizon=float("nan")),
        ):
            assert np.isnan(float(call()))
            assert np.isnan(float(jax.jit(call)()))


class TestCensoredCoverageBounds:
    """Observable lower/upper bounds on P(L <= T*) under right-censoring."""

    def test_hand_table_and_brackets_latent_coverage_across_draws(self) -> None:
        # 4 observations:
        # i=0: L=2 <= T_obs=3 (delta=0) -> covered in LB and UB
        # i=1: L=5 >  T_obs=4 (delta=1) -> known miscoverage! Excluded from both LB and UB
        # i=2: L=5 >  T_obs=3 (delta=0) -> ambiguous (censored before L): excluded from LB, kept in UB
        # i=3: L=1 <= T_obs=2 (delta=1) -> covered in LB and UB
        # LB = 2/4 = 0.5; UB = 1 - 1/4 = 0.75.
        lower_pred = jnp.array([2.0, 5.0, 5.0, 1.0], dtype=jnp.float32)
        obs_time = jnp.array([3.0, 4.0, 3.0, 2.0], dtype=jnp.float32)
        events = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        lb, ub = censored_coverage_bounds(lower_pred, obs_time, events)
        assert float(lb) == pytest.approx(0.5)
        assert float(ub) == pytest.approx(0.75)

        # Uncensored -> LB == UB == empirical coverage.
        lb_unc, ub_unc = censored_coverage_bounds(lower_pred, obs_time, jnp.ones(4))
        assert float(lb_unc) == pytest.approx(0.5)
        assert float(ub_unc) == pytest.approx(0.5)

        # Across 50 random right-censored datasets, LB <= P_hat(L <= T_latent) <= UB identically.
        rng = np.random.default_rng(7)
        for _ in range(50):
            t_latent = rng.exponential(scale=5.0, size=64).astype(np.float32)
            c_time = rng.exponential(scale=6.0, size=64).astype(np.float32)
            t_obs = np.minimum(t_latent, c_time)
            delta = (t_latent <= c_time).astype(np.int32)
            l_pred = rng.uniform(0.0, 6.0, size=64).astype(np.float32)
            true_cov = float(np.mean(l_pred <= t_latent))
            lo_b, hi_b = censored_coverage_bounds(l_pred, t_obs, delta)
            assert float(lo_b) <= true_cov + 1e-7
            assert true_cov <= float(hi_b) + 1e-7

    def test_empty_returns_nan_pair_and_shape_mismatch_raises(self) -> None:
        empty = jnp.zeros(0, dtype=jnp.float32)
        lb, ub = censored_coverage_bounds(empty, empty, empty)
        assert np.isnan(float(lb))
        assert np.isnan(float(ub))
        with pytest.raises(ValueError, match="Shape mismatch"):
            censored_coverage_bounds(jnp.ones(2), jnp.ones(3), jnp.ones(3))

    def test_nan_and_non_binary_event_indicator_return_nan_bounds(self) -> None:
        lower_pred = jnp.array([2.0, 5.0, 5.0, 1.0], dtype=jnp.float32)
        obs_time = jnp.array([3.0, 4.0, 3.0, 2.0], dtype=jnp.float32)
        ev_bool = jnp.array([False, True, False, True])
        ev_f32 = jnp.array([0.0, 1.0, 0.0, 1.0], dtype=jnp.float32)

        lb_b, ub_b = censored_coverage_bounds(lower_pred, obs_time, ev_bool)
        assert (float(lb_b), float(ub_b)) == (pytest.approx(0.5), pytest.approx(0.75))

        for call in (
            lambda: censored_coverage_bounds(lower_pred.at[0].set(jnp.nan), obs_time, ev_f32),
            lambda: censored_coverage_bounds(lower_pred, obs_time.at[1].set(jnp.nan), ev_f32),
            lambda: censored_coverage_bounds(lower_pred, obs_time, ev_f32.at[2].set(jnp.nan)),
            lambda: censored_coverage_bounds(lower_pred, obs_time, jnp.array([0, 2, 0, 1])),
            lambda: censored_coverage_bounds(lower_pred, obs_time, jnp.array([0.0, 0.5, 0.0, 1.0])),
        ):
            lb, ub = call()
            assert np.isnan(float(lb))
            assert np.isnan(float(ub))
            lb_j, ub_j = jax.jit(call)()
            assert np.isnan(float(lb_j))
            assert np.isnan(float(ub_j))

        counter = TraceCounter()
        compiled_bounds = jax.jit(counter.wrap(censored_coverage_bounds))
        with counter.expect(new_traces=1):
            compiled_bounds(lower_pred, obs_time, ev_bool)
        with counter.expect(new_traces=0):
            compiled_bounds(lower_pred + 1.0, obs_time, ev_bool)


class TestSurvivalRegistration:
    """Survival domain registration checks."""

    def test_survival_metrics_registered(self) -> None:
        registry = MetricRegistry()
        entries = registry.list_by_domain("survival")
        assert sorted(e.name for e in entries) == ["harrell_c_index", "uno_c_index"]
        for entry in entries:
            assert entry.direction == MetricDirection.HIGHER
            assert entry.properties.is_jit_compatible is True
            assert entry.properties.is_differentiable is False
