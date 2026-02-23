"""Tests for calibrax.storage.store module."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from calibrax.core.models import Metric, MetricDirection, Point, Run
from calibrax.storage.store import Store


@pytest.fixture
def store(tmp_path) -> Store:
    """Create a Store in a temp directory."""
    return Store(tmp_path / "benchmark-data")


def _make_run(
    run_id: str = "run1",
    branch: str | None = "main",
    timestamp: datetime | None = None,
    tags: dict[str, str] | None = None,
    metrics: dict[str, Metric] | None = None,
) -> Run:
    """Helper to create a Run."""
    if metrics is None:
        metrics = {"throughput": Metric(value=100.0)}
    if tags is None:
        tags = {"framework": "jax"}
    return Run(
        id=run_id,
        branch=branch,
        timestamp=timestamp or datetime.now(),
        points=(Point(name="bench1", scenario="s1", tags=tags, metrics=metrics),),
    )


class TestSaveLoad:
    """Tests for save and load."""

    def test_round_trip(self, store: Store) -> None:
        """save + load should preserve Run data."""
        run = _make_run()
        store.save(run)
        loaded = store.load("run1")
        assert loaded.id == run.id
        assert loaded.branch == run.branch
        assert len(loaded.points) == 1
        assert loaded.points[0].name == "bench1"
        assert loaded.points[0].metrics["throughput"].value == 100.0

    def test_load_missing_raises(self, store: Store) -> None:
        """Loading a non-existent run should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Run not found"):
            store.load("nonexistent")

    def test_save_returns_path(self, store: Store) -> None:
        """save should return the path to the JSON file."""
        run = _make_run()
        path = store.save(run)
        assert path.exists()
        assert path.suffix == ".json"


class TestListRuns:
    """Tests for list_runs."""

    def test_returns_all_sorted(self, store: Store) -> None:
        """Should return all runs sorted by timestamp descending."""
        now = datetime.now()
        old = _make_run("old", timestamp=now - timedelta(hours=2))
        new = _make_run("new", timestamp=now)
        store.save(old)
        store.save(new)
        runs = store.list_runs()
        assert len(runs) == 2
        assert runs[0].id == "new"
        assert runs[1].id == "old"

    def test_filter_by_branch(self, store: Store) -> None:
        """Should filter by branch when specified."""
        main_run = _make_run("r1", branch="main")
        dev_run = _make_run("r2", branch="dev")
        store.save(main_run)
        store.save(dev_run)
        main_runs = store.list_runs(branch="main")
        assert len(main_runs) == 1
        assert main_runs[0].id == "r1"

    def test_empty_store(self, store: Store) -> None:
        """Empty store should return empty list."""
        assert store.list_runs() == []


class TestLatest:
    """Tests for latest."""

    def test_returns_most_recent(self, store: Store) -> None:
        """Should return the most recent run."""
        now = datetime.now()
        store.save(_make_run("r1", timestamp=now - timedelta(hours=1)))
        store.save(_make_run("r2", timestamp=now))
        assert store.latest().id == "r2"

    def test_empty_raises(self, store: Store) -> None:
        """Empty store should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No runs"):
            store.latest()


class TestQuery:
    """Tests for query."""

    def test_finds_matching_runs(self, store: Store) -> None:
        """Should find runs with matching tags."""
        store.save(_make_run("r1", tags={"framework": "jax"}))
        store.save(_make_run("r2", tags={"framework": "pytorch"}))
        results = store.query(framework="jax")
        assert len(results) == 1
        assert results[0].id == "r1"

    def test_no_match_empty(self, store: Store) -> None:
        """No matching tags should return empty list."""
        store.save(_make_run("r1", tags={"framework": "jax"}))
        assert store.query(framework="tensorflow") == []

    def test_multiple_tag_filters(self, store: Store) -> None:
        """All tag filters must match."""
        store.save(_make_run("r1", tags={"framework": "jax", "device": "gpu"}))
        store.save(_make_run("r2", tags={"framework": "jax", "device": "cpu"}))
        results = store.query(framework="jax", device="gpu")
        assert len(results) == 1
        assert results[0].id == "r1"


class TestBaseline:
    """Tests for set_baseline and get_baseline."""

    def test_set_and_get_round_trip(self, store: Store) -> None:
        """set_baseline + get_baseline should preserve the run."""
        run = _make_run("r1")
        store.save(run)
        store.set_baseline("r1")
        baseline = store.get_baseline()
        assert baseline is not None
        assert baseline.id == "r1"

    def test_get_baseline_none(self, store: Store) -> None:
        """get_baseline with no baseline set should return None."""
        assert store.get_baseline() is None

    def test_set_baseline_invalid_id(self, store: Store) -> None:
        """set_baseline with invalid ID should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Run not found"):
            store.set_baseline("nonexistent")


class TestIngest:
    """Tests for ingest."""

    def test_ingest_external_json(self, store: Store, tmp_path) -> None:
        """Should import an external JSON file."""
        run = _make_run("imported")
        external_file = tmp_path / "external.json"
        external_file.write_text(json.dumps(run.to_dict(), indent=2))
        imported = store.ingest(external_file)
        assert imported.id == "imported"
        loaded = store.load("imported")
        assert loaded.id == "imported"


class TestExtractTrend:
    """Tests for extract_trend."""

    def test_extracts_correct_values(self, store: Store) -> None:
        """Should extract the right metric values across runs."""
        now = datetime.now()
        for i in range(3):
            run = Run(
                id=f"r{i}",
                timestamp=now + timedelta(hours=i),
                points=(
                    Point(
                        name="bench1",
                        scenario="s1",
                        tags={"framework": "jax"},
                        metrics={"throughput": Metric(value=float(100 + i * 10))},
                    ),
                ),
            )
            store.save(run)

        trend = store.extract_trend("throughput", "bench1", {"framework": "jax"})
        assert trend.metric == "throughput"
        assert trend.point_name == "bench1"
        assert len(trend.points) == 3
        values = [tp.value for tp in trend.points]
        assert values == [100.0, 110.0, 120.0]

    def test_n_runs_limit(self, store: Store) -> None:
        """n_runs should limit to the N most recent points."""
        now = datetime.now()
        for i in range(5):
            run = Run(
                id=f"r{i}",
                timestamp=now + timedelta(hours=i),
                points=(
                    Point(
                        name="bench1",
                        scenario="s1",
                        tags={"framework": "jax"},
                        metrics={"throughput": Metric(value=float(i))},
                    ),
                ),
            )
            store.save(run)

        trend = store.extract_trend("throughput", "bench1", {"framework": "jax"}, n_runs=2)
        assert len(trend.points) == 2
        values = [tp.value for tp in trend.points]
        assert values == [3.0, 4.0]

    def test_no_matching_runs(self, store: Store) -> None:
        """No matching points should return empty trend."""
        store.save(_make_run("r1"))
        trend = store.extract_trend("throughput", "nonexistent", {"framework": "jax"})
        assert len(trend.points) == 0


class TestDirectoryStructure:
    """Tests for directory creation."""

    def test_creates_dirs_on_init(self, tmp_path) -> None:
        """Store should create runs/ and baselines/ directories on init."""
        root = tmp_path / "benchmark-data"
        Store(root)
        assert (root / "runs").is_dir()
        assert (root / "baselines").is_dir()


class TestConfig:
    """Tests for config loading."""

    def test_loads_metric_defs(self, tmp_path) -> None:
        """Should load metric_defs from config.json."""
        root = tmp_path / "benchmark-data"
        root.mkdir(parents=True)
        config = {
            "metric_defs": {
                "throughput": {
                    "name": "throughput",
                    "unit": "ops/s",
                    "direction": "higher",
                },
            },
        }
        (root / "config.json").write_text(json.dumps(config))
        s = Store(root)
        assert "throughput" in s.metric_defs
        assert s.metric_defs["throughput"].direction == MetricDirection.HIGHER
