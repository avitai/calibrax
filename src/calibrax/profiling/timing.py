"""Framework-agnostic timing with configurable result synchronization.

Provides TimingCollector for measuring iteration throughput with per-batch timing breakdown,
and ``time_calls`` for the median and percentiles of repeated calls; their records,
``TimingSample`` and ``CallTiming``, live in ``calibrax.profiling.timing_records``.
Uses time.perf_counter() exclusively for accurate benchmarking.
Supports warm-up iteration exclusion and JIT compilation time measurement.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence

import numpy as np
from jax import block_until_ready, jit
from substrax.typing import PyTree

from calibrax.profiling.timing_records import CallTiming, TimingSample


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
        start = time.perf_counter()
        jit(fn).lower(*args).compile()
        end = time.perf_counter()
        return end - start


_PERCENTILE_MAX = 100


def _wait_for_result(result: PyTree) -> None:
    """Wait for every array in ``result`` with ``jax.block_until_ready``."""
    block_until_ready(result)


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
