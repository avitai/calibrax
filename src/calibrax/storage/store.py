"""JSON-per-run file backend with baseline management.

Directory layout:

```text
benchmark-data/
+-- runs/
|   +-- abc123.json
|   +-- ...
+-- baselines/
|   +-- main.json
+-- config.json
```
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from calibrax.core.models import MetricDef, Run, TrendPoint, TrendSeries


def _collect_benchmark_definitions(
    runs: list[Run],
) -> dict[str, dict[str, object]]:
    """Collect all benchmark names and build ASV benchmark definitions.

    Args:
        runs: All stored runs.

    Returns:
        ASV benchmarks dictionary keyed by benchmark name.
    """
    benchmark_names: set[str] = set()
    for run in runs:
        for point in run.points:
            for metric_name in point.metrics:
                benchmark_names.add(f"{point.scenario}.{point.name}.{metric_name}")

    benchmarks: dict[str, dict[str, object]] = {}
    for name in sorted(benchmark_names):
        benchmarks[name] = {
            "code": "",
            "name": name,
            "param_names": [],
            "params": [],
            "timeout": 60.0,
            "type": "time",
            "unit": "seconds",
        }
    return benchmarks


def _write_asv_results(runs: list[Run], results_dir: Path) -> None:
    """Write per-commit ASV result files.

    Args:
        runs: All stored runs.
        results_dir: Base results directory.
    """
    for run in runs:
        machine = run.environment.get("machine", "default")
        commit = run.commit or run.id

        machine_dir = results_dir / str(machine)
        machine_dir.mkdir(parents=True, exist_ok=True)

        result_data = _build_run_result_data(run)

        commit_file = machine_dir / f"{commit}.json"
        commit_file.write_text(
            json.dumps(
                {
                    "commit_hash": commit,
                    "date": run.timestamp.isoformat(),
                    "params": {"machine": str(machine)},
                    "results": result_data,
                    "version": 2,
                },
                indent=2,
            )
        )


def _build_run_result_data(run: Run) -> dict[str, dict[str, object]]:
    """Build ASV result data for a single run.

    Args:
        run: Benchmark run.

    Returns:
        {bench_key: {result, params, stats}}.
    """
    result_data: dict[str, dict[str, object]] = {}
    for point in run.points:
        for metric_name, metric in point.metrics.items():
            bench_key = f"{point.scenario}.{point.name}.{metric_name}"
            result_data[bench_key] = {
                "result": [float(metric.value)],
                "params": [],
                "stats": {},
            }
            if metric.samples is not None:
                result_data[bench_key]["stats"] = {
                    "samples": [float(s) for s in metric.samples],
                }
    return result_data


class Store:
    """JSON-per-run file backend with baseline management.

    Args:
        path: Root directory for storing runs, baselines, and config.
    """

    def __init__(self, path: Path | str) -> None:
        """Initialize the store and create directory structure.

        Args:
            path: Root directory for storing runs, baselines, and config.
        """
        self._path = Path(path)
        self._runs_dir = self._path / "runs"
        self._baselines_dir = self._path / "baselines"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._baselines_dir.mkdir(parents=True, exist_ok=True)

        self.metric_defs: dict[str, MetricDef] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load metric definitions from config.json if present."""
        config_path = self._path / "config.json"
        if config_path.exists():
            data = json.loads(config_path.read_text())
            for name, md_data in data.get("metric_defs", {}).items():
                self.metric_defs[name] = MetricDef.from_dict(md_data)

    def _run_path(self, run_id: str) -> Path:
        """Path for a run's JSON file."""
        return self._runs_dir / f"{run_id}.json"

    def save(self, run: Run) -> Path:
        """Save a run as JSON.

        Args:
            run: The run to persist.

        Returns:
            Path to the saved JSON file.
        """
        path = self._run_path(run.id)
        path.write_text(json.dumps(run.to_dict(), indent=2))
        return path

    def load(self, run_id: str) -> Run:
        """Load a run by ID.

        Args:
            run_id: Unique identifier of the run.

        Returns:
            The deserialized Run.

        Raises:
            FileNotFoundError: If the run does not exist.
        """
        path = self._run_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        return Run.from_dict(json.loads(path.read_text()))

    def list_runs(self, branch: str | None = None) -> list[Run]:
        """List all runs, optionally filtered by branch.

        Args:
            branch: If set, return only runs on this branch.

        Returns:
            Runs sorted by timestamp descending (newest first).
        """
        runs: list[Run] = []
        for p in self._runs_dir.glob("*.json"):
            run = Run.from_dict(json.loads(p.read_text()))
            if branch is None or run.branch == branch:
                runs.append(run)
        runs.sort(key=lambda r: r.timestamp, reverse=True)
        return runs

    def latest(self) -> Run:
        """Load the most recent run.

        Returns:
            The most recent run by timestamp.

        Raises:
            FileNotFoundError: If the store is empty.
        """
        runs = self.list_runs()
        if not runs:
            raise FileNotFoundError("No runs in store")
        return runs[0]

    def query(self, **tags: str) -> list[Run]:
        """Find runs where any point matches all given tag filters.

        Args:
            **tags: Key-value pairs that must all match on at least one point.

        Returns:
            List of matching runs.
        """
        results: list[Run] = []
        for p in self._runs_dir.glob("*.json"):
            run = Run.from_dict(json.loads(p.read_text()))
            for point in run.points:
                if all(point.tags.get(k) == v for k, v in tags.items()):
                    results.append(run)
                    break
        return results

    def set_baseline(self, run_id: str) -> None:
        """Copy a run to baselines/main.json.

        Args:
            run_id: ID of the run to set as baseline.

        Raises:
            FileNotFoundError: If the run does not exist.
        """
        src = self._run_path(run_id)
        if not src.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        shutil.copy2(src, self._baselines_dir / "main.json")

    def get_baseline(self) -> Run | None:
        """Load the current baseline, or None if not set.

        Returns:
            The baseline Run, or None if no baseline has been set.
        """
        path = self._baselines_dir / "main.json"
        if not path.exists():
            return None
        return Run.from_dict(json.loads(path.read_text()))

    def ingest(self, path: Path, format: str = "auto") -> Run:
        """Import results from an external JSON file and save to store.

        Args:
            path: Path to the external JSON file.
            format: Import format (currently only "auto" / JSON supported).

        Returns:
            The imported Run.
        """
        data = json.loads(Path(path).read_text())
        run = Run.from_dict(data)
        self.save(run)
        return run

    def export_asv(self, output_dir: Path | str) -> Path:
        """Export store data in ASV-compatible JSON format.

        Creates ``benchmarks.json`` (benchmark definitions) and per-commit
        result files in ``results/{machine}/{commit}.json``.

        Args:
            output_dir: Directory to write ASV-formatted output.

        Returns:
            Path to the output directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        runs = self.list_runs()
        if not runs:
            return out

        benchmarks = _collect_benchmark_definitions(runs)
        (out / "benchmarks.json").write_text(json.dumps(benchmarks, indent=2))

        _write_asv_results(runs, out / "results")

        return out

    def extract_trend(
        self,
        metric: str,
        point_name: str,
        tags: dict[str, str],
        *,
        n_runs: int | None = None,
    ) -> TrendSeries:
        """Extract time-series trend for a metric across stored runs.

        Scans all runs for points matching (point_name, tags), extracts
        the named metric, and returns a TrendSeries ordered oldest-first.

        Args:
            metric: Metric name to track.
            point_name: Point name to match.
            tags: Tags that must all match on the point.
            n_runs: If set, return only the N most recent data points.

        Returns:
            TrendSeries with one TrendPoint per matching run.
        """
        runs = self.list_runs()

        trend_points: list[TrendPoint] = []
        for run in runs:
            for point in run.points:
                if point.name != point_name:
                    continue
                if not all(point.tags.get(k) == v for k, v in tags.items()):
                    continue
                if metric not in point.metrics:
                    continue
                m = point.metrics[metric]
                trend_points.append(
                    TrendPoint(
                        run_id=run.id,
                        timestamp=run.timestamp,
                        value=m.value,
                        commit=run.commit,
                        lower=m.lower,
                        upper=m.upper,
                    )
                )
                break

        trend_points.sort(key=lambda tp: tp.timestamp)

        if n_runs is not None and len(trend_points) > n_runs:
            trend_points = trend_points[-n_runs:]

        return TrendSeries(
            metric=metric,
            point_name=point_name,
            tags=tags,
            points=tuple(trend_points),
        )
