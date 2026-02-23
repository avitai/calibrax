"""Tests for Store.export_asv method."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from calibrax.core.models import Metric, Point, Run
from calibrax.storage.store import Store
from tests.factories import make_matmul_run


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    """Create a Store in a temp directory."""
    return Store(tmp_path / "benchmark-data")


def _make_run(
    run_id: str = "run1",
    commit: str = "abc123",
    throughput: float = 100.0,
    latency: float = 5.0,
    timestamp: datetime | None = None,
    environment: dict | None = None,
) -> Run:
    """Helper to create a Run with standard benchmark points."""
    return make_matmul_run(
        run_id=run_id,
        throughput=throughput,
        latency=latency,
        commit=commit,
        branch="main",
        timestamp=timestamp or datetime.now(),
        environment=environment,
    )


class TestExportAsv:
    """Tests for Store.export_asv."""

    def test_creates_benchmarks_json(self, store: Store, tmp_path: Path) -> None:
        """export_asv should create benchmarks.json in the output directory."""
        run = _make_run()
        store.save(run)
        output_dir = tmp_path / "asv_output"
        store.export_asv(output_dir)

        benchmarks_path = output_dir / "benchmarks.json"
        assert benchmarks_path.exists()

    def test_creates_result_files_per_commit(self, store: Store, tmp_path: Path) -> None:
        """export_asv should create a result file for each commit."""
        run = _make_run(run_id="r1", commit="commit_aaa")
        store.save(run)
        output_dir = tmp_path / "asv_output"
        store.export_asv(output_dir)

        results_dir = output_dir / "results"
        assert results_dir.exists()
        # Find the commit file (under machine subdir)
        json_files = list(results_dir.rglob("commit_aaa.json"))
        assert len(json_files) == 1

    def test_empty_store_returns_output_dir(self, store: Store, tmp_path: Path) -> None:
        """export_asv with empty store should return output dir without crash."""
        output_dir = tmp_path / "asv_empty"
        result_path = store.export_asv(output_dir)
        assert result_path == output_dir
        assert output_dir.exists()
        # No benchmarks.json should be created for empty store
        assert not (output_dir / "benchmarks.json").exists()

    def test_benchmarks_json_content(self, store: Store, tmp_path: Path) -> None:
        """benchmarks.json should contain expected benchmark names."""
        run = _make_run()
        store.save(run)
        output_dir = tmp_path / "asv_output"
        store.export_asv(output_dir)

        benchmarks = json.loads((output_dir / "benchmarks.json").read_text())
        expected_keys = {"perf.matmul.throughput", "perf.matmul.latency"}
        assert set(benchmarks.keys()) == expected_keys

    def test_benchmarks_json_structure(self, store: Store, tmp_path: Path) -> None:
        """Each benchmark entry should have the expected ASV fields."""
        run = _make_run()
        store.save(run)
        output_dir = tmp_path / "asv_output"
        store.export_asv(output_dir)

        benchmarks = json.loads((output_dir / "benchmarks.json").read_text())
        for name, entry in benchmarks.items():
            assert entry["name"] == name
            assert "code" in entry
            assert "param_names" in entry
            assert "params" in entry
            assert "timeout" in entry
            assert "type" in entry
            assert "unit" in entry

    def test_result_file_content(self, store: Store, tmp_path: Path) -> None:
        """Result files should contain benchmark results keyed by benchmark name."""
        run = _make_run(run_id="r1", commit="abc123", throughput=42.0)
        store.save(run)
        output_dir = tmp_path / "asv_output"
        store.export_asv(output_dir)

        result_files = list((output_dir / "results").rglob("abc123.json"))
        assert len(result_files) == 1

        data = json.loads(result_files[0].read_text())
        assert data["commit_hash"] == "abc123"
        assert data["version"] == 2
        assert "perf.matmul.throughput" in data["results"]
        assert data["results"]["perf.matmul.throughput"]["result"] == [42.0]

    def test_multiple_runs_different_commits(self, store: Store, tmp_path: Path) -> None:
        """export_asv should create separate result files per commit."""
        now = datetime.now()
        store.save(_make_run(run_id="r1", commit="aaa", timestamp=now))
        store.save(
            _make_run(
                run_id="r2",
                commit="bbb",
                timestamp=now + timedelta(hours=1),
            )
        )
        output_dir = tmp_path / "asv_output"
        store.export_asv(output_dir)

        result_files = list((output_dir / "results").rglob("*.json"))
        commit_names = {f.stem for f in result_files}
        assert "aaa" in commit_names
        assert "bbb" in commit_names

    def test_machine_directory(self, store: Store, tmp_path: Path) -> None:
        """Result files should be under a machine subdirectory."""
        run = _make_run(environment={"machine": "gpu-node"})
        store.save(run)
        output_dir = tmp_path / "asv_output"
        store.export_asv(output_dir)

        machine_dir = output_dir / "results" / "gpu-node"
        assert machine_dir.exists()
        assert any(machine_dir.iterdir())

    def test_default_machine_when_missing(self, store: Store, tmp_path: Path) -> None:
        """Should use 'default' machine when not in environment."""
        run = _make_run()
        store.save(run)
        output_dir = tmp_path / "asv_output"
        store.export_asv(output_dir)

        machine_dir = output_dir / "results" / "default"
        assert machine_dir.exists()

    def test_returns_output_path(self, store: Store, tmp_path: Path) -> None:
        """export_asv should return the output directory path."""
        run = _make_run()
        store.save(run)
        output_dir = tmp_path / "asv_output"
        result = store.export_asv(output_dir)
        assert result == output_dir

    def test_samples_in_stats(self, store: Store, tmp_path: Path) -> None:
        """Result files should include samples in stats when available."""
        run = Run(
            id="r1",
            commit="abc",
            branch="main",
            points=(
                Point(
                    name="bench",
                    scenario="s1",
                    tags={},
                    metrics={
                        "time": Metric(
                            value=1.0,
                            samples=(0.9, 1.0, 1.1),
                        ),
                    },
                ),
            ),
        )
        store.save(run)
        output_dir = tmp_path / "asv_output"
        store.export_asv(output_dir)

        result_files = list((output_dir / "results").rglob("abc.json"))
        data = json.loads(result_files[0].read_text())
        bench_result = data["results"]["s1.bench.time"]
        assert "samples" in bench_result["stats"]
        assert bench_result["stats"]["samples"] == [0.9, 1.0, 1.1]
