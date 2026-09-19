"""Store commands: ingest, check, baseline, trend and summary.

None of them loads JAX: they read and write the JSON store.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from calibrax.ci.guard import CIGuard
from calibrax.storage.store import Store


@click.command()
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


@click.command()
@click.option("--data", required=True, type=click.Path(path_type=Path), help="Store directory.")
@click.option("--threshold", default=0.05, type=float, help="Regression threshold (default: 0.05).")
def check(data: Path, threshold: float) -> None:
    """Check for performance regressions (CI gate)."""
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


@click.command()
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


@click.command()
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
            print(f"{tp.timestamp!s:<28} {tp.value:>12.4f} {commit:<12}")
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None


@click.command()
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
