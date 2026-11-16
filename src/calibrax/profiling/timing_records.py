"""Timing records: what ``TimingCollector``, ``time_calls`` and ``compare_calls`` measure.

The records import without JAX, so reading a stored result does not load it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from substrax.records import read_record
from substrax.typing import JsonValue


@dataclass(frozen=True, slots=True, kw_only=True)
class TimingSample:
    """Result of timing an iteration through a data pipeline.

    Attributes:
        wall_clock_sec: Total wall-clock time for the iteration.
        per_batch_times: Per-batch durations in seconds (warmup batches excluded).
        first_batch_time: Time from iteration start to first batch completion.
        num_batches: Number of batches consumed (including warmup).
        num_elements: Total elements processed (via count_fn).
        compilation_time_sec: JIT compilation time, if measured separately.
        warmup_batches_excluded: Number of leading batches excluded from per_batch_times.
    """

    wall_clock_sec: float
    per_batch_times: tuple[float, ...]
    first_batch_time: float
    num_batches: int
    num_elements: int
    compilation_time_sec: float | None = None
    warmup_batches_excluded: int = 0

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        d: dict[str, JsonValue] = {
            "wall_clock_sec": float(self.wall_clock_sec),
            "per_batch_times": [float(t) for t in self.per_batch_times],
            "first_batch_time": float(self.first_batch_time),
            "num_batches": int(self.num_batches),
            "num_elements": int(self.num_elements),
            "warmup_batches_excluded": int(self.warmup_batches_excluded),
        }
        if self.compilation_time_sec is not None:
            d["compilation_time_sec"] = float(self.compilation_time_sec)
        return d

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> TimingSample:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


@dataclass(frozen=True, kw_only=True, slots=True)
class CallTiming:
    """Timing of repeated calls to one function: the samples, their median and percentiles.

    Attributes:
        samples_sec: Wall-clock seconds of each timed call, warm-up excluded, in call order.
        median_sec: Median of ``samples_sec``.
        percentiles_sec: Percentile (0-100) to seconds, for the percentiles requested.
        warmup: Calls made and discarded before timing.
    """

    samples_sec: tuple[float, ...]
    median_sec: float
    percentiles_sec: dict[int, float]
    warmup: int

    def to_dict(self) -> dict[str, JsonValue]:
        """JSON-ready form; percentile keys become strings."""
        return {
            "samples_sec": [float(sample) for sample in self.samples_sec],
            "median_sec": float(self.median_sec),
            "percentiles_sec": {str(k): float(v) for k, v in self.percentiles_sec.items()},
            "warmup": int(self.warmup),
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> CallTiming:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


@dataclass(frozen=True, kw_only=True, slots=True)
class CallComparison:
    """Counterbalanced ABBA paired comparison of two callables.

    Attributes:
        baseline_samples_sec: Per-round median seconds of ``baseline``, length ``rounds``.
        candidate_samples_sec: Per-round median seconds of ``candidate``, length ``rounds``.
        differences_sec: Paired per-round differences ``candidate - baseline`` in seconds;
            negative values mean ``candidate`` ran faster than ``baseline``.
        baseline_median_sec: Median of ``baseline_samples_sec``.
        candidate_median_sec: Median of ``candidate_samples_sec``.
        median_difference_sec: Median of ``differences_sec`` (``candidate - baseline``).
        speedup_ratio: ``baseline_median_sec / candidate_median_sec`` (``> 1.0`` when
            ``candidate`` is faster).
        p_value: Two-sided paired randomization p-value testing exchangeability of the two arms.
        rounds: Counterbalanced rounds measured (even, alternating ``AB`` and ``BA``).
        calls_per_arm: Synchronized calls executed per arm inside each round.
        warmup: Alternating warm-up pairs executed and discarded before timing.
    """

    baseline_samples_sec: tuple[float, ...]
    candidate_samples_sec: tuple[float, ...]
    differences_sec: tuple[float, ...]
    baseline_median_sec: float
    candidate_median_sec: float
    median_difference_sec: float
    speedup_ratio: float
    p_value: float
    rounds: int
    calls_per_arm: int
    warmup: int

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "baseline_samples_sec": [float(v) for v in self.baseline_samples_sec],
            "candidate_samples_sec": [float(v) for v in self.candidate_samples_sec],
            "differences_sec": [float(v) for v in self.differences_sec],
            "baseline_median_sec": float(self.baseline_median_sec),
            "candidate_median_sec": float(self.candidate_median_sec),
            "median_difference_sec": float(self.median_difference_sec),
            "speedup_ratio": float(self.speedup_ratio),
            "p_value": float(self.p_value),
            "rounds": int(self.rounds),
            "calls_per_arm": int(self.calls_per_arm),
            "warmup": int(self.warmup),
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> CallComparison:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)
