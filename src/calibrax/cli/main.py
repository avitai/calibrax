"""Click-based CLI for calibrax benchmark management.

Provides commands for ingesting results, exporting to W&B, checking for
regressions, managing baselines, viewing trends, profiling, and summarizing runs.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

import click

from calibrax.storage.store import Store


@click.group()
def main() -> None:
    """Calibrax: unified benchmarking framework CLI."""


@main.command()
@click.option("--data", required=True, type=click.Path(path_type=Path), help="Store directory.")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="JSON file to ingest.",
)
def ingest(data: Path, input_file: Path) -> None:
    """Import JSON benchmark results into the store."""
    try:
        store = Store(data)
        run = store.ingest(input_file)
        print(f"Ingested run {run.id} ({len(run.points)} points)")
    except (json.JSONDecodeError, KeyError) as e:
        raise click.ClickException(f"Invalid input file: {e}") from None


@main.command()
@click.option("--data", required=True, type=click.Path(path_type=Path), help="Store directory.")
@click.option("--run", "run_id", default="latest", help="Run ID to export (default: latest).")
@click.option("--project", default=None, help="W&B project name.")
@click.option("--entity", default=None, help="W&B entity (team or user).")
def export(data: Path, run_id: str, project: str | None, entity: str | None) -> None:
    """Export a run to Weights & Biases."""
    try:
        from calibrax.exporters.wandb import WandBExporter
    except ImportError:
        raise click.ClickException("wandb is required: uv pip install 'calibrax[wandb]'") from None

    if project is None:
        raise click.ClickException("--project is required for W&B export") from None

    try:
        store = Store(data)
        run = store.load(run_id) if run_id != "latest" else store.latest()
        baseline = store.get_baseline()

        exporter = WandBExporter(project=project, entity=entity)
        url = exporter.export_run(run, finish=False)
        exporter.export_analysis(run, baseline=baseline)

        print(f"Exported run {run.id} to W&B: {url}")
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None


@main.command()
@click.option("--data", required=True, type=click.Path(path_type=Path), help="Store directory.")
@click.option("--threshold", default=0.05, type=float, help="Regression threshold (default: 0.05).")
def check(data: Path, threshold: float) -> None:
    """Check for performance regressions (CI gate)."""
    from calibrax.ci.guard import CIGuard

    try:
        store = Store(data)
        guard = CIGuard(store, threshold=threshold)
        result = guard.check()

        if result.passed:
            print(f"PASSED: No regressions detected (threshold={threshold})")
        else:
            print(f"FAILED: {len(result.regressions)} regression(s) detected")
            for r in result.regressions:
                print(f"  - {r.metric} on {r.point_name}: {r.delta_pct:+.1f}%")
            sys.exit(1)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None


@main.command()
@click.option("--data", required=True, type=click.Path(path_type=Path), help="Store directory.")
@click.option("--run", "run_id", default="latest", help="Run ID to set as baseline.")
def baseline(data: Path, run_id: str) -> None:
    """Set a run as the baseline for regression checks."""
    try:
        store = Store(data)
        if run_id == "latest":
            run = store.latest()
            actual_id = run.id
        else:
            actual_id = run_id
        store.set_baseline(actual_id)
        print(f"Baseline set to run {actual_id}")
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None


@main.command()
@click.option("--data", required=True, type=click.Path(path_type=Path), help="Store directory.")
@click.option("--metric", required=True, help="Metric name to track.")
@click.option("--point", required=True, help="Point name to match.")
@click.option("--framework", required=True, help="Framework tag to filter by.")
@click.option("--n-runs", default=None, type=int, help="Limit to N most recent runs.")
def trend(data: Path, metric: str, point: str, framework: str, n_runs: int | None) -> None:
    """Show metric trends over time."""
    try:
        store = Store(data)
        series = store.extract_trend(
            metric,
            point,
            {"framework": framework},
            n_runs=n_runs,
        )
        if not series.points:
            print("No trend data found")
            return
        print(f"Trend: {metric} for {point} ({framework})")
        print(f"{'Timestamp':<28} {'Value':>12} {'Commit':<12}")
        print("-" * 56)
        for tp in series.points:
            commit = tp.commit[:8] if tp.commit else "-"
            print(f"{str(tp.timestamp):<28} {tp.value:>12.4f} {commit:<12}")
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None


@main.command()
@click.option("--data", required=True, type=click.Path(path_type=Path), help="Store directory.")
@click.option("--run", "run_id", default="latest", help="Run ID to summarize (default: latest).")
def summary(data: Path, run_id: str) -> None:
    """Show a human-readable run summary."""
    try:
        store = Store(data)
        run = store.load(run_id) if run_id != "latest" else store.latest()

        print(f"Run: {run.id}")
        print(f"  Timestamp: {run.timestamp}")
        if run.commit:
            print(f"  Commit: {run.commit}")
        if run.branch:
            print(f"  Branch: {run.branch}")
        print(f"  Points: {len(run.points)}")
        print()

        scenarios: dict[str, list[str]] = {}
        for p in run.points:
            if p.scenario not in scenarios:
                scenarios[p.scenario] = []
            fw = p.tags.get("framework", p.name)
            metrics_str = ", ".join(f"{k}={v.value:.4f}" for k, v in sorted(p.metrics.items()))
            scenarios[p.scenario].append(f"  {fw}: {metrics_str}")

        for scenario, lines in scenarios.items():
            print(f"Scenario: {scenario}")
            for line in lines:
                print(line)
            print()
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None


def _resolve_callable(module: str, func_name: str) -> Callable[[], object]:
    """Import a module and return the named callable.

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
    enable_energy: bool,
) -> tuple[object, object]:
    """Execute timing measurement with optional energy monitoring.

    Uses lazy imports for JAX and calibrax.profiling so this module
    remains importable without JAX installed.

    Args:
        func: A no-argument callable to profile.
        warmup: Number of warmup iterations.
        iterations: Number of timed iterations.
        enable_energy: Whether to wrap the measurement with EnergyMonitor.

    Returns:
        A ``(TimingSample, EnergySummary | None)`` tuple.
    """
    from calibrax.profiling.timing import TimingCollector

    def _sync_result(result: object) -> None:
        """Block on JAX async dispatch for arrays and nested array containers."""
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()  # type: ignore[union-attr]
            return
        if isinstance(result, tuple | list):
            for item in result:
                if hasattr(item, "block_until_ready"):
                    item.block_until_ready()  # type: ignore[union-attr]

    collector = TimingCollector(
        sync_fn=_sync_result,
        warmup_iterations=warmup,
    )

    def _call_fn(batch: object) -> object:
        """Execute the target function once, ignoring the batch argument."""
        _ = batch
        return func()

    batch_indices = list(range(warmup + iterations))

    energy_summary = None
    if enable_energy:
        from calibrax.profiling.energy import EnergyMonitor

        with EnergyMonitor() as monitor:
            sample = collector.measure_iteration(
                iter(batch_indices),
                num_batches=warmup + iterations,
                process_fn=_call_fn,
            )
        energy_summary = monitor.summary
    else:
        sample = collector.measure_iteration(
            iter(batch_indices),
            num_batches=warmup + iterations,
            process_fn=_call_fn,
        )

    return sample, energy_summary


def _print_energy_results(energy_summary: object) -> None:
    """Print energy monitoring results.

    Args:
        energy_summary: An ``EnergySummary`` instance.
    """
    from calibrax.profiling.energy import EnergySummary

    if not isinstance(energy_summary, EnergySummary):
        return

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
    sample: object,
    flops_result: object,
    energy_summary: object,
) -> None:
    """Print timing, FLOP, and energy profiling results.

    Args:
        sample: A ``TimingSample`` from the timing collector.
        flops_result: A ``FlopsResult`` or ``None``.
        energy_summary: An ``EnergySummary`` or ``None``.
    """
    from calibrax.profiling.flops import FlopsResult

    print("\nTiming Results:")
    print(f"  Wall clock: {sample.wall_clock_sec:.4f}s")  # type: ignore[union-attr]
    print(f"  Batches: {sample.num_batches} (warmup excluded: {sample.warmup_batches_excluded})")  # type: ignore[union-attr]
    if sample.per_batch_times:  # type: ignore[union-attr]
        mean_time = sum(sample.per_batch_times) / len(sample.per_batch_times)  # type: ignore[union-attr]
        print(f"  Mean batch time: {mean_time:.6f}s")

    if isinstance(flops_result, FlopsResult):
        print(f"\nFLOP Analysis ({flops_result.function_name}):")
        print(f"  Total FLOPs: {flops_result.total_flops:,}")
        print(f"  Operations: {flops_result.num_operations}")
        if flops_result.flops_by_operation:
            print("  Breakdown:")
            for op, count in sorted(
                flops_result.flops_by_operation.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]:
                print(f"    {op}: {count:,}")

    _print_energy_results(energy_summary)


def _save_profile_run(
    data: Path,
    module: str,
    func_name: str,
    sample: object,
    flops_result: object,
    energy_summary: object,
) -> None:
    """Build a Run from profiling results and persist it to the Store.

    Args:
        data: Directory for the result store.
        module: Dotted module path (stored as a tag).
        func_name: Function name (used as the point name).
        sample: A ``TimingSample`` from the timing collector.
        flops_result: A ``FlopsResult`` or ``None``.
        energy_summary: An ``EnergySummary`` or ``None``.
    """
    from calibrax.core.models import Metric, Point, Run
    from calibrax.profiling.energy import EnergySummary
    from calibrax.profiling.flops import FlopsResult

    metrics: dict[str, Metric] = {
        "wall_clock_sec": Metric(value=sample.wall_clock_sec),  # type: ignore[union-attr]
    }
    if sample.per_batch_times:  # type: ignore[union-attr]
        mean_time = sum(sample.per_batch_times) / len(sample.per_batch_times)  # type: ignore[union-attr]
        metrics["mean_batch_time_sec"] = Metric(value=mean_time)
    if isinstance(flops_result, FlopsResult):
        metrics["total_flops"] = Metric(value=float(flops_result.total_flops))
    if isinstance(energy_summary, EnergySummary):
        if energy_summary.total_gpu_energy_joules is not None:
            metrics["gpu_energy_joules"] = Metric(
                value=energy_summary.total_gpu_energy_joules,
            )
        if energy_summary.total_cpu_energy_joules is not None:
            metrics["cpu_energy_joules"] = Metric(
                value=energy_summary.total_cpu_energy_joules,
            )

    point = Point(
        name=func_name,
        scenario="profile",
        tags={"module": module},
        metrics=metrics,
    )
    run = Run(points=(point,))
    store = Store(data)
    store.save(run)
    print(f"\n  Saved profiling run {run.id} to {data}")


@main.command()
@click.option("--module", required=True, help="Python module path (e.g., 'my_pkg.benchmark').")
@click.option("--function", "func_name", required=True, help="Function name to profile.")
@click.option("--warmup", default=1, type=int, help="Warmup iterations (default: 1).")
@click.option("--iterations", default=10, type=int, help="Timing iterations (default: 10).")
@click.option("--energy", is_flag=True, help="Enable energy monitoring.")
@click.option("--flops", is_flag=True, help="Enable FLOP counting.")
@click.option("--data", default=None, type=click.Path(path_type=Path), help="Store directory.")
def profile(
    module: str,
    func_name: str,
    warmup: int,
    iterations: int,
    energy: bool,
    flops: bool,
    data: Path | None,
) -> None:
    """Profile a JAX function with timing, resource, and optional energy/FLOP measurement."""
    func = _resolve_callable(module, func_name)

    print(f"Profiling {module}.{func_name}")
    print(f"  Warmup: {warmup}, Iterations: {iterations}")

    sample, energy_summary = _run_measurement(
        func,
        warmup,
        iterations,
        enable_energy=energy,
    )

    flops_result = None
    if flops:
        from calibrax.profiling.flops import FlopsCounter

        counter = FlopsCounter()
        try:
            flops_result = counter.count(func)  # type: ignore[arg-type]
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            # FlopsCounter traces via jax.make_jaxpr which can fail for many
            # reasons (unsupported ops, tracing limitations). Degrade gracefully.
            print(f"\n  FLOP counting failed: {exc}")

    _print_profile_results(sample, flops_result, energy_summary)

    if data is not None:
        _save_profile_run(data, module, func_name, sample, flops_result, energy_summary)

    print("\nProfile complete.")
