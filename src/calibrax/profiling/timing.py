"""Framework-agnostic timing with configurable result synchronization.

Provides TimingSample (frozen dataclass) and TimingCollector for
measuring iteration throughput with per-batch timing breakdown.
Uses time.perf_counter() exclusively for accurate benchmarking.
Supports warm-up iteration exclusion and JIT compilation time measurement.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from jax import block_until_ready
from substrax.typing import PyTree


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d: dict[str, Any] = {
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
    def from_dict(cls, data: dict[str, Any]) -> TimingSample:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with TimingSample fields.

        Returns:
            Reconstructed TimingSample instance.
        """
        return cls(
            wall_clock_sec=data["wall_clock_sec"],
            per_batch_times=tuple(data["per_batch_times"]),
            first_batch_time=data["first_batch_time"],
            num_batches=data["num_batches"],
            num_elements=data["num_elements"],
            compilation_time_sec=data.get("compilation_time_sec"),
            warmup_batches_excluded=data.get("warmup_batches_excluded", 0),
        )


class TimingCollector:
    """Framework-agnostic timing with configurable GPU sync support.

    Uses time.perf_counter() exclusively for accurate benchmarking.
    Supports configurable result synchronization via sync_fn and warm-up
    iteration exclusion for JIT-compiled workloads.

    JAX dispatches operations asynchronously -- the host returns
    immediately while the device is still computing.  Without an explicit
    synchronization barrier, ``perf_counter`` measures only host-side
    dispatch latency, not actual compute time. Pass a ``sync_fn`` that
    calls ``block_until_ready()`` on the workload result to force the host
    to wait for device completion before recording the timestamp.

    Example -- JAX GPU timing with warm-up:

    ```python
    import jax.numpy as jnp

    def run_step(batch):
        return jax.jit(step_fn)(batch)

    collector = TimingCollector(
        sync_fn=lambda result: result.block_until_ready(),
        warmup_iterations=2,
    )
    sample = collector.measure_iteration(data_iter, num_batches=50, process_fn=run_step)
    # sample.per_batch_times excludes the first 2 batches
    ```
    """

    def __init__(
        self,
        sync_fn: Callable[[PyTree], object] | None = None,
        warmup_iterations: int = 0,
    ) -> None:
        """Initialize TimingCollector.

        Args:
            sync_fn: Synchronization function called with each batch result;
                ``None`` waits for the whole result with ``jax.block_until_ready``.
            warmup_iterations: Number of initial batches to exclude from timing stats.

        Raises:
            ValueError: If ``warmup_iterations`` is negative.
        """
        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be >= 0")
        self._sync_fn = sync_fn or _wait_for_result
        self._warmup_iterations = warmup_iterations

    def measure_iteration[BatchT](
        self,
        iterator: Iterator[BatchT],
        num_batches: int | None = None,
        process_fn: Callable[[BatchT], PyTree] | None = None,
        count_fn: Callable[[BatchT], int] | None = None,
    ) -> TimingSample:
        """Measure timing for batches from an iterator.

        Warm-up batches (if configured) are executed but excluded from
        ``per_batch_times``. ``wall_clock_sec`` covers the entire run
        including warm-up. ``num_batches`` reflects total batches consumed.

        Args:
            iterator: Data iterator to measure.
            num_batches: Max batches to consume (including warmup). None exhausts iterator.
            process_fn: Optional per-batch function whose execution is timed.
                Defaults to identity (the yielded batch is treated as result).
            count_fn: Function to count elements per batch. Default: 1 per batch.

        Returns:
            TimingSample with timing measurements.

        Raises:
            ValueError: If ``num_batches`` is negative.
        """
        if num_batches is not None and num_batches < 0:
            raise ValueError("num_batches must be >= 0 or None")

        all_batch_times: list[float] = []
        first_batch_time = 0.0
        total_elements = 0
        process = process_fn or (lambda batch: batch)
        count = count_fn or (lambda _: 1)

        overall_start = time.perf_counter()

        for i, batch in enumerate(iterator):
            if num_batches is not None and i >= num_batches:
                break

            batch_start = time.perf_counter()
            result = process(batch)
            self._sync_fn(result)
            batch_end = time.perf_counter()

            if i == 0:
                first_batch_time = batch_end - overall_start

            all_batch_times.append(batch_end - batch_start)
            total_elements += count(batch)

        wall_clock = time.perf_counter() - overall_start

        # Exclude warmup batches from per_batch_times
        warmup_count = min(self._warmup_iterations, len(all_batch_times))
        timed_batches = all_batch_times[warmup_count:]

        return TimingSample(
            wall_clock_sec=wall_clock,
            per_batch_times=tuple(timed_batches),
            first_batch_time=first_batch_time,
            num_batches=len(all_batch_times),
            num_elements=total_elements,
            warmup_batches_excluded=warmup_count,
        )

    def measure_compilation_time(
        self,
        fn: Callable[..., PyTree],
        *args: PyTree,
    ) -> float:
        """Measure JIT compilation time for a JAX function.

        Calls ``jax.jit(fn).lower(*args).compile()`` and times it.
        This measures the XLA compilation step only, not execution.

        Args:
            fn: JAX function to compile.
            *args: Example arguments for lowering.

        Returns:
            Compilation time in seconds.
        """
        import jax

        start = time.perf_counter()
        jax.jit(fn).lower(*args).compile()
        end = time.perf_counter()
        return end - start


_PERCENTILE_MAX = 100


def _wait_for_result(result: PyTree) -> None:
    """Wait for every array in ``result`` with ``jax.block_until_ready``."""
    block_until_ready(result)


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

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form; percentile keys become strings."""
        return {
            "samples_sec": [float(sample) for sample in self.samples_sec],
            "median_sec": float(self.median_sec),
            "percentiles_sec": {str(k): float(v) for k, v in self.percentiles_sec.items()},
            "warmup": int(self.warmup),
        }


def time_calls(
    call: Callable[[], PyTree],
    *,
    warmup: int = 3,
    iterations: int = 10,
    percentiles: Sequence[int] = (50, 90, 99),
    sync: Callable[[PyTree], object] | None = None,
) -> CallTiming:
    """Time ``call()`` and report the median and percentiles.

    ``call`` takes no arguments: close over the timed function's inputs
    (``lambda: step(state, batch)``), so its keywords never meet these options'.

    Each call is followed by ``sync(result)``, ``jax.block_until_ready`` over the whole
    result pytree by default, so asynchronous dispatch is inside the measurement. The
    function is timed as given: pass ``jax.jit(f)`` to time the compiled program, and the
    warm-up calls then absorb its compilation. The median is the reported figure because
    a mean is moved by one slow call.

    Args:
        call: The zero-argument callable to time.
        warmup: Calls made and discarded before timing.
        iterations: Timed calls.
        percentiles: Percentiles (0-100) to report beside the median.
        sync: Called with each result before the clock stops; ``None`` waits for the
            whole result with ``jax.block_until_ready``.

    Returns:
        The samples, median and percentiles.

    Raises:
        ValueError: If ``iterations`` is below 1, ``warmup`` is negative, or a percentile
            is outside 0-100.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if any(not 0 <= p <= _PERCENTILE_MAX for p in percentiles):
        raise ValueError(f"every percentile must be within 0-100, got {tuple(percentiles)}")
    wait = _wait_for_result if sync is None else sync

    for _ in range(warmup):
        wait(call())

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        wait(call())
        samples.append(time.perf_counter() - start)

    values = np.asarray(samples)
    return CallTiming(
        samples_sec=tuple(samples),
        median_sec=float(np.median(values)),
        percentiles_sec={int(p): float(np.percentile(values, p)) for p in percentiles},
        warmup=warmup,
    )
