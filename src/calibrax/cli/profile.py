"""The profile command, and the profiling routine ``profile-gpu`` shares.

Times a no-argument function with ``TimingCollector``, which waits for every array in each
result with ``jax.block_until_ready``, and optionally counts its FLOPs and measures energy.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import click

from calibrax.core.models import Metric, Point, Run
from calibrax.profiling.energy import EnergyMonitor, EnergySummary
from calibrax.profiling.flops import FlopsCounter, FlopsResult
from calibrax.profiling.timing import TimingCollector
from calibrax.profiling.timing_records import TimingSample
from calibrax.storage.store import Store


def profile_options[F: Callable[..., None]](command: F) -> F:
    """The options ``profile`` and ``profile-gpu`` share.

    Args:
        command: The command function.

    Returns:
        The function with the options applied.
    """
    options = [
        click.option(
            "--module", required=True, help="Python module path (e.g., 'my_pkg.benchmark')."
        ),
        click.option("--function", "func_name", required=True, help="Function name to profile."),
        click.option("--warmup", default=1, type=int, help="Warmup iterations (default: 1)."),
        click.option("--iterations", default=10, type=int, help="Timing iterations (default: 10)."),
        click.option("--flops", is_flag=True, help="Enable FLOP counting."),
        click.option(
            "--data", default=None, type=click.Path(path_type=Path), help="Store directory."
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


@click.command()
@profile_options
@click.option("--energy", is_flag=True, help="Measure CPU energy through RAPL.")
def profile(
    module: str,
    func_name: str,
    warmup: int,
    iterations: int,
    flops: bool,
    data: Path | None,
    energy: bool,
) -> None:
    """Profile a JAX function: timing, and optionally FLOPs and CPU energy."""
    run_profile(
        module,
        func_name,
        warmup=warmup,
        iterations=iterations,
        count_flops=flops,
        data=data,
        energy_monitor=EnergyMonitor() if energy else None,
    )


def run_profile(
    module: str,
    func_name: str,
    *,
    warmup: int,
    iterations: int,
    count_flops: bool,
    data: Path | None,
    energy_monitor: EnergyMonitor | None,
) -> None:
    """Time the function, count its FLOPs and measure energy as asked, print, and store.

    Args:
        module: Dotted path of the module defining the function.
        func_name: The function's name; it takes no arguments.
        warmup: Calls made and excluded before timing.
        iterations: Timed calls.
        count_flops: Whether to count FLOPs through XLA's cost analysis.
        data: Store directory to save the run to, or None not to save it.
        energy_monitor: The monitor to run the timed calls under, or None.
    """
    func = _resolve_callable(module, func_name)

    print(f"Profiling {module}.{func_name}")
    print(f"  Warmup: {warmup}, Iterations: {iterations}")

    sample, energy_summary = _run_measurement(
        func, warmup, iterations, energy_monitor=energy_monitor
    )
    flops_result = _count_flops(func) if count_flops else None

    _print_profile_results(sample, flops_result, energy_summary)
    if data is not None:
        _save_profile_run(data, module, func_name, sample, flops_result, energy_summary)

    print("\nProfile complete.")


def _resolve_callable(module: str, func_name: str) -> Callable[[], object]:
    """Import the user's module and return the named callable.

    Args:
        module: Dotted Python module path (e.g. ``'my_pkg.benchmark'``).
        func_name: Attribute name to look up in the imported module.

    Returns:
        The callable object from the module.

    Raises:
        click.ClickException: If the module cannot be imported or the
            function is not found.
    """
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise click.ClickException(f"Cannot import module: {e}") from None

    func = getattr(mod, func_name, None)
    if func is None:
        raise click.ClickException(f"Function '{func_name}' not found in '{module}'") from None
    if not callable(func):
        raise click.ClickException(
            f"Attribute '{func_name}' in '{module}' is not callable"
        ) from None
    return func


def _run_measurement(
    func: Callable[[], object],
    warmup: int,
    iterations: int,
    *,
    energy_monitor: EnergyMonitor | None,
) -> tuple[TimingSample, EnergySummary | None]:
    """Time ``warmup + iterations`` calls, under the energy monitor when one is given.

    Args:
        func: A no-argument callable to profile.
        warmup: Number of warmup iterations.
        iterations: Number of timed iterations.
        energy_monitor: The monitor to run the calls under, or None.

    Returns:
        The timing sample, and the energy summary when a monitor ran.
    """
    collector = TimingCollector(warmup_iterations=warmup)
    calls = warmup + iterations

    def measure() -> TimingSample:
        return collector.measure_iteration(
            iter(range(calls)), num_batches=calls, process_fn=lambda _call: func()
        )

    if energy_monitor is None:
        return measure(), None
    with energy_monitor:
        sample = measure()
    return sample, energy_monitor.summary


def _count_flops(func: Callable[[], object]) -> FlopsResult | None:
    """The function's FLOPs, or None with the reason printed when XLA cannot count them."""
    try:
        return FlopsCounter().count(func)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        # Lowering fails for non-JAX code and tracing limits, and XLA has no cost model
        # for some custom calls (FlopsUnavailableError, a ValueError).
        print(f"\n  FLOP counting failed: {exc}")
        return None


def _print_energy_results(energy_summary: EnergySummary) -> None:
    """Print energy monitoring results.

    Args:
        energy_summary: The monitor's summary.
    """
    print("\nEnergy Results:")
    print(f"  Duration: {energy_summary.duration_sec:.4f}s")
    print(f"  Samples: {energy_summary.num_samples}")
    if energy_summary.total_gpu_energy_joules is not None:
        print(f"  GPU energy: {energy_summary.total_gpu_energy_joules:.4f} J")
    if energy_summary.total_cpu_energy_joules is not None:
        print(f"  CPU energy: {energy_summary.total_cpu_energy_joules:.4f} J")
    if energy_summary.mean_gpu_power_watts is not None:
        print(f"  Mean GPU power: {energy_summary.mean_gpu_power_watts:.2f} W")
    if energy_summary.total_combined_energy_joules is not None:
        print(f"  Total energy: {energy_summary.total_combined_energy_joules:.4f} J")


def _print_profile_results(
    sample: TimingSample,
    flops_result: FlopsResult | None,
    energy_summary: EnergySummary | None,
) -> None:
    """Print timing, FLOP, and energy profiling results.

    Args:
        sample: The timing sample.
        flops_result: The FLOP count, or None.
        energy_summary: The energy summary, or None.
    """
    print("\nTiming Results:")
    print(f"  Wall clock: {sample.wall_clock_sec:.4f}s")
    print(f"  Batches: {sample.num_batches} (warmup excluded: {sample.warmup_batches_excluded})")
    if sample.per_batch_times:
        mean_time = sum(sample.per_batch_times) / len(sample.per_batch_times)
        print(f"  Mean batch time: {mean_time:.6f}s")

    if flops_result is not None:
        print(f"\nFLOP Analysis ({flops_result.function_name}):")
        print(f"  Total FLOPs: {flops_result.total_flops:,}")
        print(f"  Transcendentals: {flops_result.transcendentals:,}")

    if energy_summary is not None:
        _print_energy_results(energy_summary)


def _save_profile_run(
    data: Path,
    module: str,
    func_name: str,
    sample: TimingSample,
    flops_result: FlopsResult | None,
    energy_summary: EnergySummary | None,
) -> None:
    """Build a Run from profiling results and persist it to the Store.

    Args:
        data: Directory for the result store.
        module: Dotted module path (stored as a tag).
        func_name: Function name (used as the point name).
        sample: The timing sample.
        flops_result: The FLOP count, or None.
        energy_summary: The energy summary, or None.
    """
    metrics: dict[str, Metric] = {"wall_clock_sec": Metric(value=sample.wall_clock_sec)}
    if sample.per_batch_times:
        mean_time = sum(sample.per_batch_times) / len(sample.per_batch_times)
        metrics["mean_batch_time_sec"] = Metric(value=mean_time)
    if flops_result is not None:
        metrics["total_flops"] = Metric(value=float(flops_result.total_flops))
    if energy_summary is not None:
        if energy_summary.total_gpu_energy_joules is not None:
            metrics["gpu_energy_joules"] = Metric(value=energy_summary.total_gpu_energy_joules)
        if energy_summary.total_cpu_energy_joules is not None:
            metrics["cpu_energy_joules"] = Metric(value=energy_summary.total_cpu_energy_joules)

    point = Point(name=func_name, scenario="profile", tags={"module": module}, metrics=metrics)
    run = Run(points=(point,))
    Store(data).save(run)
    print(f"\n  Saved profiling run {run.id} to {data}")
