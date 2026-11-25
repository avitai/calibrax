"""Framework-agnostic timing with configurable result synchronization.

Provides TimingCollector for measuring iteration throughput with per-batch timing breakdown,
``time_calls`` for the median and percentiles of repeated calls, and ``compare_calls`` for
counterbalanced ABBA paired A/B comparisons; their records, ``TimingSample``, ``CallTiming``
and ``CallComparison``, live in ``calibrax.profiling.timing_records``.
Uses time.perf_counter() exclusively for accurate benchmarking.
Supports warm-up iteration exclusion and JIT compilation time measurement.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Sequence

import numpy as np
from jax import block_until_ready, jit
from scipy.stats import wilcoxon
from substrax.typing import PyTree

from calibrax.profiling._jax_staging import lower_staged
from calibrax.profiling.timing_records import CallComparison, CallTiming, TimingSample


class TimingCollector:
    """Framework-agnostic timing with configurable GPU sync support.

    Uses time.perf_counter() exclusively for accurate benchmarking.
    Supports configurable result synchronization via sync_fn and warm-up
    iteration exclusion for JIT-compiled workloads.

    JAX dispatches operations asynchronously -- the host returns
    immediately while the device is still computing -- so each batch's
    clock stops only after ``sync_fn`` returns. By default that is
    ``jax.block_until_ready`` over the whole result pytree, so a batch's time
    is its computation, not its dispatch. Pass a ``sync_fn`` only for a
    result that needs some other wait.

    Example -- JAX timing with warm-up:

    ```python
    run_step = jax.jit(step_fn)

    collector = TimingCollector(warmup_iterations=2)
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
        static_argnums: int | Iterable[int] = (),
        static_argnames: str | Iterable[str] = (),
        **kwargs: PyTree,
    ) -> float:
        """Measure JIT compilation time for a JAX function or ``nnx.Module``.

        Lowers and compiles ``fn`` without executing eager mutations on live ``nnx.Module``
        state, timing the lowering and XLA compilation step only.

        Args:
            fn: JAX function, ``nnx.Module``, or bound ``nnx.Module`` method to compile.
            *args: Example positional arguments for lowering.
            static_argnums: Positional argument indices treated as static at trace time.
            static_argnames: Keyword argument names treated as static at trace time.
            **kwargs: Example keyword arguments for lowering.

        Returns:
            Compilation time in seconds.
        """
        start = time.perf_counter()
        lowered, _ = lower_staged(
            fn,
            args,
            kwargs,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            jit_fn=jit,
        )
        lowered.compile()
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


_EXACT_SIGN_FLIP_MAX_ROUNDS = 16
_MIN_COMPARISON_ROUNDS = 2


def _time_arm_median(
    call: Callable[[], PyTree],
    wait: Callable[[PyTree], object],
    calls_per_arm: int,
) -> float:
    """Run ``calls_per_arm`` synchronized calls and return their median duration."""
    durations: list[float] = []
    for _ in range(calls_per_arm):
        start = time.perf_counter()
        wait(call())
        durations.append(time.perf_counter() - start)
    return float(np.median(np.asarray(durations, dtype=np.float64)))


def _paired_p_value(differences: np.ndarray) -> float:
    """Two-sided paired p-value: exact sign-flip up to 16 rounds, Wilcoxon above."""
    if bool(np.all(differences == 0.0)):
        return 1.0
    n = int(differences.size)
    if n <= _EXACT_SIGN_FLIP_MAX_ROUNDS:
        indices = np.arange(1 << n, dtype=np.uint32)[:, None]
        shifts = np.arange(n, dtype=np.uint32)
        signs = 2.0 * ((indices >> shifts) & 1).astype(np.float64) - 1.0
        totals = np.abs(signs @ differences)
        observed = abs(float(np.sum(differences)))
        tol = n * float(np.finfo(np.float64).eps) * float(np.sum(np.abs(differences)))
        return float(np.mean(totals >= observed - tol))
    _, p_val = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    return float(p_val)


def compare_calls(
    baseline: Callable[[], PyTree],
    candidate: Callable[[], PyTree],
    *,
    rounds: int = 8,
    calls_per_arm: int = 3,
    warmup: int = 3,
    sync: Callable[[PyTree], object] | None = None,
) -> CallComparison:
    """Compare ``baseline()`` and ``candidate()`` with counterbalanced ABBA pairing.

    Running all ``baseline`` iterations before ``candidate`` -- or repeating ``(A, B)`` in one
    fixed order -- confounds a code change with thermal state, frequency scaling, and page-cache
    position effects. ``compare_calls`` warms both arms in alternating pairs, then alternates
    round order between ``(baseline, candidate)`` on even rounds and ``(candidate, baseline)``
    on odd rounds so first-in-pair and second-in-pair positions balance exactly.

    Args:
        baseline: Zero-argument reference callable (arm A).
        candidate: Zero-argument callable under evaluation (arm B).
        rounds: Even number of paired rounds (at least 2) so ``AB`` and ``BA`` balance.
        calls_per_arm: Synchronized calls per arm in each round; summarized by their median.
        warmup: Alternating ``(baseline, candidate)`` warm-up pairs discarded before timing.
        sync: Called with each result before stopping the clock; ``None`` waits for the whole
            result pytree with ``jax.block_until_ready``.

    Returns:
        ``CallComparison`` with per-round arm medians, paired differences, speedup ratio, and
        two-sided paired randomization p-value.

    Raises:
        ValueError: If ``rounds < 2``, ``rounds`` is odd, ``calls_per_arm < 1``, or
            ``warmup < 0``.
    """
    if rounds < _MIN_COMPARISON_ROUNDS:
        raise ValueError(f"rounds must be at least {_MIN_COMPARISON_ROUNDS}, got {rounds}")
    if rounds % 2 != 0:
        raise ValueError(f"rounds must be even so AB and BA pairs balance, got {rounds}")
    if calls_per_arm < 1:
        raise ValueError(f"calls_per_arm must be at least 1, got {calls_per_arm}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    wait = _wait_for_result if sync is None else sync
    for _ in range(warmup):
        wait(baseline())
        wait(candidate())

    base_samples: list[float] = []
    cand_samples: list[float] = []
    for round_idx in range(rounds):
        if round_idx % 2 == 0:
            t_base = _time_arm_median(baseline, wait, calls_per_arm)
            t_cand = _time_arm_median(candidate, wait, calls_per_arm)
        else:
            t_cand = _time_arm_median(candidate, wait, calls_per_arm)
            t_base = _time_arm_median(baseline, wait, calls_per_arm)
        base_samples.append(t_base)
        cand_samples.append(t_cand)

    base_arr = np.asarray(base_samples, dtype=np.float64)
    cand_arr = np.asarray(cand_samples, dtype=np.float64)
    diff_arr = cand_arr - base_arr

    base_med = float(np.median(base_arr))
    cand_med = float(np.median(cand_arr))
    diff_med = float(np.median(diff_arr))
    if cand_med == 0.0 and base_med == 0.0:
        speedup = 1.0
    elif cand_med == 0.0:
        speedup = float("inf")
    else:
        speedup = base_med / cand_med

    return CallComparison(
        baseline_samples_sec=tuple(base_samples),
        candidate_samples_sec=tuple(cand_samples),
        differences_sec=tuple(float(d) for d in diff_arr),
        baseline_median_sec=base_med,
        candidate_median_sec=cand_med,
        median_difference_sec=diff_med,
        speedup_ratio=speedup,
        p_value=_paired_p_value(diff_arr),
        rounds=rounds,
        calls_per_arm=calls_per_arm,
        warmup=warmup,
    )
