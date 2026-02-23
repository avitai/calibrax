"""Tests for TimingCollector and TimingSample.

Verifies dataclass immutability, measure_iteration behavior,
sync_fn calling, first_batch_time capture, element counting,
warm-up exclusion, compilation time measurement, and edge cases.
"""

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from calibrax.profiling.timing import TimingCollector, TimingSample
from tests.factories import make_default_timing_sample


class TestTimingSample:
    """Tests for TimingSample frozen dataclass."""

    def test_creation_with_all_fields(self) -> None:
        sample = make_default_timing_sample()
        assert sample.wall_clock_sec == 1.5
        assert sample.per_batch_times == (0.1, 0.2, 0.3)
        assert sample.first_batch_time == 0.15
        assert sample.num_batches == 3
        assert sample.num_elements == 96

    def test_frozen_immutability(self) -> None:
        sample = TimingSample(
            wall_clock_sec=1.0,
            per_batch_times=(0.5,),
            first_batch_time=0.5,
            num_batches=1,
            num_elements=1,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample.wall_clock_sec = 2.0  # type: ignore[misc]

    def test_kw_only_enforcement(self) -> None:
        with pytest.raises(TypeError):
            TimingSample(1.0, (0.5,), 0.5, 1, 1)  # type: ignore[misc]

    def test_per_batch_times_is_tuple(self) -> None:
        sample = TimingSample(
            wall_clock_sec=1.0,
            per_batch_times=(0.1, 0.2),
            first_batch_time=0.1,
            num_batches=2,
            num_elements=2,
        )
        assert isinstance(sample.per_batch_times, tuple)

    def test_compilation_time_defaults_to_none(self) -> None:
        sample = TimingSample(
            wall_clock_sec=1.0,
            per_batch_times=(0.1,),
            first_batch_time=0.1,
            num_batches=1,
            num_elements=1,
        )
        assert sample.compilation_time_sec is None

    def test_compilation_time_field(self) -> None:
        sample = TimingSample(
            wall_clock_sec=1.0,
            per_batch_times=(0.1,),
            first_batch_time=0.1,
            num_batches=1,
            num_elements=1,
            compilation_time_sec=0.42,
        )
        assert sample.compilation_time_sec == 0.42

    def test_warmup_batches_excluded_defaults_to_zero(self) -> None:
        sample = TimingSample(
            wall_clock_sec=1.0,
            per_batch_times=(0.1,),
            first_batch_time=0.1,
            num_batches=1,
            num_elements=1,
        )
        assert sample.warmup_batches_excluded == 0

    def test_to_dict(self) -> None:
        """to_dict should serialize all fields as plain Python types."""
        sample = make_default_timing_sample(compilation_time_sec=0.42, warmup_batches_excluded=1)
        d = sample.to_dict()
        assert d["wall_clock_sec"] == 1.5
        assert d["per_batch_times"] == [0.1, 0.2, 0.3]
        assert d["first_batch_time"] == 0.15
        assert d["num_batches"] == 3
        assert d["num_elements"] == 96
        assert d["compilation_time_sec"] == 0.42
        assert d["warmup_batches_excluded"] == 1

    def test_to_dict_omits_none_compilation_time(self) -> None:
        """to_dict should omit compilation_time_sec when None."""
        sample = TimingSample(
            wall_clock_sec=1.0,
            per_batch_times=(0.1,),
            first_batch_time=0.1,
            num_batches=1,
            num_elements=1,
        )
        d = sample.to_dict()
        assert "compilation_time_sec" not in d

    def test_from_dict(self) -> None:
        """from_dict should reconstruct a TimingSample."""
        data = {
            "wall_clock_sec": 2.0,
            "per_batch_times": [0.3, 0.4],
            "first_batch_time": 0.35,
            "num_batches": 2,
            "num_elements": 64,
            "compilation_time_sec": 0.1,
            "warmup_batches_excluded": 1,
        }
        sample = TimingSample.from_dict(data)
        assert sample.wall_clock_sec == 2.0
        assert sample.per_batch_times == (0.3, 0.4)
        assert sample.first_batch_time == 0.35
        assert sample.num_batches == 2
        assert sample.num_elements == 64
        assert sample.compilation_time_sec == 0.1
        assert sample.warmup_batches_excluded == 1

    def test_from_dict_defaults(self) -> None:
        """from_dict should handle missing optional fields."""
        data = {
            "wall_clock_sec": 1.0,
            "per_batch_times": [0.5],
            "first_batch_time": 0.5,
            "num_batches": 1,
            "num_elements": 1,
        }
        sample = TimingSample.from_dict(data)
        assert sample.compilation_time_sec is None
        assert sample.warmup_batches_excluded == 0

    def test_to_dict_from_dict_round_trip(self) -> None:
        """Round-trip should produce an equivalent object."""
        original = TimingSample(
            wall_clock_sec=3.14,
            per_batch_times=(0.1, 0.2, 0.3),
            first_batch_time=0.15,
            num_batches=3,
            num_elements=42,
            compilation_time_sec=0.5,
            warmup_batches_excluded=1,
        )
        reconstructed = TimingSample.from_dict(original.to_dict())
        assert reconstructed.wall_clock_sec == original.wall_clock_sec
        assert reconstructed.per_batch_times == original.per_batch_times
        assert reconstructed.first_batch_time == original.first_batch_time
        assert reconstructed.num_batches == original.num_batches
        assert reconstructed.num_elements == original.num_elements
        assert reconstructed.compilation_time_sec == original.compilation_time_sec
        assert reconstructed.warmup_batches_excluded == original.warmup_batches_excluded

    def test_to_dict_from_dict_round_trip_no_compilation(self) -> None:
        """Round-trip should work without compilation_time_sec."""
        original = TimingSample(
            wall_clock_sec=1.0,
            per_batch_times=(0.1,),
            first_batch_time=0.1,
            num_batches=1,
            num_elements=1,
        )
        reconstructed = TimingSample.from_dict(original.to_dict())
        assert reconstructed.compilation_time_sec is None
        assert reconstructed.warmup_batches_excluded == 0


class TestTimingCollector:
    """Tests for TimingCollector."""

    def test_negative_warmup_iterations_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="warmup_iterations must be >= 0"):
            TimingCollector(warmup_iterations=-1)

    def test_measure_simple_iterator(self) -> None:
        data = [list(range(10)) for _ in range(5)]
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=5)

        assert isinstance(result, TimingSample)
        assert result.num_batches == 5
        assert len(result.per_batch_times) == 5
        assert result.wall_clock_sec > 0
        assert result.first_batch_time > 0

    def test_sync_fn_called_each_iteration(self) -> None:
        sync_fn = MagicMock()
        data = [1, 2, 3]
        collector = TimingCollector(sync_fn=sync_fn)

        collector.measure_iteration(iter(data), num_batches=3)

        assert sync_fn.call_count == 3
        assert sync_fn.call_args_list[0].args == (1,)
        assert sync_fn.call_args_list[1].args == (2,)
        assert sync_fn.call_args_list[2].args == (3,)

    def test_process_fn_result_is_synced(self) -> None:
        sync_fn = MagicMock()
        data = [1, 2, 3]
        collector = TimingCollector(sync_fn=sync_fn)

        def process_fn(batch: int) -> str:
            return f"out-{batch}"

        collector.measure_iteration(iter(data), num_batches=3, process_fn=process_fn)

        assert sync_fn.call_args_list[0].args == ("out-1",)
        assert sync_fn.call_args_list[1].args == ("out-2",)
        assert sync_fn.call_args_list[2].args == ("out-3",)

    def test_process_fn_called_each_iteration(self) -> None:
        process_fn = MagicMock(side_effect=lambda batch: batch)
        data = [1, 2, 3]
        collector = TimingCollector()

        collector.measure_iteration(iter(data), num_batches=3, process_fn=process_fn)

        assert process_fn.call_count == 3
        assert process_fn.call_args_list[0].args == (1,)
        assert process_fn.call_args_list[1].args == (2,)
        assert process_fn.call_args_list[2].args == (3,)

    def test_no_sync_fn_no_error(self) -> None:
        collector = TimingCollector()
        data = [1, 2, 3]

        result = collector.measure_iteration(iter(data), num_batches=3)

        assert result.num_batches == 3

    def test_first_batch_time_captured_separately(self) -> None:
        data = [1, 2, 3]
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=3)

        assert result.first_batch_time > 0
        assert result.first_batch_time >= result.per_batch_times[0]

    @patch("calibrax.profiling.timing.time")
    def test_uses_perf_counter(self, mock_time: MagicMock) -> None:
        mock_time.perf_counter.side_effect = [
            0.0,  # overall_start
            0.1,  # batch_start (batch 0)
            0.2,  # batch_end (batch 0)
            0.3,  # overall end
        ]
        data = [1]
        collector = TimingCollector()

        collector.measure_iteration(iter(data), num_batches=1)

        assert mock_time.perf_counter.call_count >= 3
        mock_time.time.assert_not_called()

    def test_count_fn_counts_elements(self) -> None:
        batches = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        collector = TimingCollector()

        result = collector.measure_iteration(
            iter(batches),
            num_batches=3,
            count_fn=len,
        )

        assert result.num_elements == 9

    def test_process_and_count_fns_are_separate(self) -> None:
        process_fn = MagicMock(side_effect=lambda batch: sum(batch))
        count_fn = MagicMock(side_effect=lambda batch: len(batch))
        batches = [[1, 2, 3], [4, 5], [6]]
        collector = TimingCollector()

        result = collector.measure_iteration(
            iter(batches),
            num_batches=3,
            process_fn=process_fn,
            count_fn=count_fn,
        )

        assert process_fn.call_count == 3
        assert count_fn.call_count == 3
        assert result.num_elements == 6

    def test_default_count_fn_one_per_batch(self) -> None:
        data = [1, 2, 3]
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=3)

        assert result.num_elements == 3

    def test_empty_iterator(self) -> None:
        collector = TimingCollector()

        result = collector.measure_iteration(iter([]))

        assert result.num_batches == 0
        assert result.per_batch_times == ()
        assert result.first_batch_time == 0.0
        assert result.num_elements == 0
        assert result.wall_clock_sec >= 0

    def test_iterator_shorter_than_requested(self) -> None:
        data = [1, 2]
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=10)

        assert result.num_batches == 2
        assert len(result.per_batch_times) == 2

    def test_num_batches_none_exhausts_iterator(self) -> None:
        data = [1, 2, 3, 4, 5]
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=None)

        assert result.num_batches == 5

    def test_negative_num_batches_raises_value_error(self) -> None:
        collector = TimingCollector()

        with pytest.raises(ValueError, match="num_batches must be >= 0 or None"):
            collector.measure_iteration(iter([1, 2]), num_batches=-1)

    def test_per_batch_times_all_non_negative(self) -> None:
        data = list(range(10))
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=10)

        assert all(t >= 0 for t in result.per_batch_times)

    def test_wall_clock_at_least_sum_of_batches(self) -> None:
        data = list(range(5))
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=5)

        assert result.wall_clock_sec >= sum(result.per_batch_times)


class TestWarmupIterations:
    """Tests for warmup iteration exclusion."""

    def test_warmup_excludes_leading_batches(self) -> None:
        data = list(range(10))
        collector = TimingCollector(warmup_iterations=3)

        result = collector.measure_iteration(iter(data), num_batches=10)

        assert result.num_batches == 10
        assert len(result.per_batch_times) == 7
        assert result.warmup_batches_excluded == 3

    def test_warmup_zero_is_default(self) -> None:
        data = list(range(5))
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=5)

        assert result.warmup_batches_excluded == 0
        assert len(result.per_batch_times) == 5

    def test_warmup_larger_than_data(self) -> None:
        data = [1, 2]
        collector = TimingCollector(warmup_iterations=10)

        result = collector.measure_iteration(iter(data), num_batches=10)

        assert result.num_batches == 2
        assert len(result.per_batch_times) == 0
        assert result.warmup_batches_excluded == 2

    def test_warmup_equals_data_size(self) -> None:
        data = [1, 2, 3]
        collector = TimingCollector(warmup_iterations=3)

        result = collector.measure_iteration(iter(data), num_batches=3)

        assert result.num_batches == 3
        assert len(result.per_batch_times) == 0
        assert result.warmup_batches_excluded == 3

    def test_warmup_preserves_wall_clock(self) -> None:
        """Wall clock includes warmup time."""
        data = list(range(5))
        collector = TimingCollector(warmup_iterations=2)

        result = collector.measure_iteration(iter(data), num_batches=5)

        assert result.wall_clock_sec > 0
        assert result.num_batches == 5

    def test_warmup_preserves_element_count(self) -> None:
        """All elements are counted, including warmup batches."""
        batches = [[1, 2], [3], [4, 5, 6], [7]]
        collector = TimingCollector(warmup_iterations=2)

        result = collector.measure_iteration(iter(batches), count_fn=len)

        assert result.num_elements == 7
        assert len(result.per_batch_times) == 2

    def test_warmup_with_sync_fn(self) -> None:
        """sync_fn is called for warmup batches too."""
        sync_fn = MagicMock()
        data = list(range(5))
        collector = TimingCollector(sync_fn=sync_fn, warmup_iterations=2)

        collector.measure_iteration(iter(data), num_batches=5)

        assert sync_fn.call_count == 5


class TestCompilationTime:
    """Tests for measure_compilation_time."""

    def test_measure_compilation_time_returns_positive(self) -> None:
        import jax.numpy as jnp

        def simple_fn(x: jnp.ndarray) -> jnp.ndarray:
            return x + 1

        collector = TimingCollector()
        x = jnp.ones((2, 2))
        comp_time = collector.measure_compilation_time(simple_fn, x)

        assert comp_time > 0
        assert isinstance(comp_time, float)

    def test_measure_compilation_time_uses_jit(self) -> None:
        """Verify it actually calls jax.jit -> lower -> compile."""
        import sys

        mock_compiled = MagicMock()
        mock_lowered = MagicMock()
        mock_lowered.compile.return_value = mock_compiled
        mock_jitted = MagicMock()
        mock_jitted.lower.return_value = mock_lowered

        mock_jax = MagicMock()
        mock_jax.jit.return_value = mock_jitted

        # jax is imported locally in measure_compilation_time, so we
        # temporarily replace the sys.modules entry.
        original_jax = sys.modules["jax"]
        sys.modules["jax"] = mock_jax
        try:
            collector = TimingCollector()
            comp_time = collector.measure_compilation_time(lambda x: x, "dummy_arg")
        finally:
            sys.modules["jax"] = original_jax

        mock_jax.jit.assert_called_once()
        mock_jitted.lower.assert_called_once_with("dummy_arg")
        mock_lowered.compile.assert_called_once()
        assert comp_time >= 0
