"""Tests for BenchmarkResult unified result container.

Covers construction, frozen immutability, auto-generated timestamp,
serde round-trip with nested composed objects, and save/load file I/O.
"""

import dataclasses
import json

import jax.numpy as jnp
import pytest

from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult
from tests.factories import make_default_resource_summary, make_default_timing_sample


class TestBenchmarkResult:
    """Tests for BenchmarkResult frozen dataclass."""

    def test_minimal_construction(self) -> None:
        result = BenchmarkResult(name="test_bench")
        assert result.name == "test_bench"
        assert result.domain == ""
        assert result.tags == {}
        assert result.timing is None
        assert result.resources is None
        assert result.metrics == {}
        assert result.metadata == {}
        assert result.config == {}
        assert result.timestamp > 0

    def test_full_construction(self) -> None:
        timing = make_default_timing_sample()
        resources = make_default_resource_summary(peak_gpu_mem_mb=4096.0, mean_gpu_util=75.0)
        result = BenchmarkResult(
            name="cv_benchmark",
            domain="computer_vision",
            tags={"framework": "Datarax", "model": "resnet50"},
            timing=timing,
            resources=resources,
            metrics={
                "throughput": Metric(value=5000.0, lower=4800.0, upper=5200.0),
                "latency_p50": Metric(value=12.0),
            },
            metadata={"runner": "full"},
            config={"batch_size": 32},
        )
        assert result.name == "cv_benchmark"
        assert result.domain == "computer_vision"
        assert result.tags["framework"] == "Datarax"
        assert result.timing is not None
        assert result.timing.num_batches == 3
        assert result.resources is not None
        assert result.resources.peak_rss_mb == 512.0
        assert result.metrics["throughput"].value == 5000.0

    def test_frozen_immutability(self) -> None:
        result = BenchmarkResult(name="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.name = "other"  # type: ignore[misc]

    def test_auto_generated_timestamp(self) -> None:
        r1 = BenchmarkResult(name="a")
        r2 = BenchmarkResult(name="b")
        assert r1.timestamp > 0
        assert r2.timestamp >= r1.timestamp

    def test_to_dict_minimal(self) -> None:
        result = BenchmarkResult(name="test")
        data = result.to_dict()
        assert data["name"] == "test"
        assert data["timing"] is None
        assert data["resources"] is None
        assert data["metrics"] == {}

    def test_to_dict_from_dict_round_trip(self) -> None:
        timing = make_default_timing_sample(
            per_batch_times=(0.1, 0.2),
            num_batches=2,
            num_elements=64,
        )
        resources = make_default_resource_summary()
        original = BenchmarkResult(
            name="bench",
            domain="nlp",
            tags={"framework": "Grain"},
            timing=timing,
            resources=resources,
            metrics={"accuracy": Metric(value=0.95, lower=0.93, upper=0.97)},
            metadata={"gpu": "A100"},
            config={"epochs": 10},
            timestamp=1234567890.0,
        )
        data = original.to_dict()
        restored = BenchmarkResult.from_dict(data)

        assert restored.name == original.name
        assert restored.domain == original.domain
        assert restored.tags == original.tags
        assert restored.timestamp == original.timestamp
        assert restored.timing is not None
        assert restored.timing.wall_clock_sec == 1.5
        assert restored.timing.per_batch_times == (0.1, 0.2)
        assert restored.resources is not None
        assert restored.resources.peak_rss_mb == 512.0
        assert restored.resources.peak_gpu_mem_mb is None
        assert restored.metrics["accuracy"].value == 0.95
        assert restored.metrics["accuracy"].lower == 0.93

    def test_to_dict_from_dict_no_timing_no_resources(self) -> None:
        original = BenchmarkResult(name="minimal", timestamp=100.0)
        data = original.to_dict()
        restored = BenchmarkResult.from_dict(data)

        assert restored.name == "minimal"
        assert restored.timing is None
        assert restored.resources is None

    def test_save_and_load(self, tmp_path: "object") -> None:
        from pathlib import Path

        filepath = Path(str(tmp_path)) / "result.json"
        timing = make_default_timing_sample(
            wall_clock_sec=2.0,
            per_batch_times=(0.5, 0.5),
            first_batch_time=0.6,
            num_batches=2,
            num_elements=64,
        )
        original = BenchmarkResult(
            name="save_test",
            timing=timing,
            metrics={"loss": Metric(value=0.01)},
            timestamp=1000.0,
        )
        original.save(filepath)

        assert filepath.exists()
        loaded = BenchmarkResult.load(filepath)
        assert loaded.name == "save_test"
        assert loaded.timing is not None
        assert loaded.timing.wall_clock_sec == 2.0
        assert loaded.metrics["loss"].value == 0.01

    def test_save_creates_parent_directories(self, tmp_path: "object") -> None:
        from pathlib import Path

        filepath = Path(str(tmp_path)) / "nested" / "dir" / "result.json"
        result = BenchmarkResult(name="nested_test", timestamp=1000.0)
        result.save(filepath)
        assert filepath.exists()

    def test_tags_provide_framework_and_model_info(self) -> None:
        result = BenchmarkResult(
            name="bench",
            tags={"framework": "Datarax", "model": "resnet50"},
        )
        assert result.tags["framework"] == "Datarax"
        assert result.tags["model"] == "resnet50"

    def test_domain_field(self) -> None:
        result = BenchmarkResult(name="bench", domain="physics")
        assert result.domain == "physics"

    def test_save_produces_valid_json(self, tmp_path: "object") -> None:
        from pathlib import Path

        filepath = Path(str(tmp_path)) / "result.json"
        result = BenchmarkResult(name="json_test", timestamp=1000.0)
        result.save(filepath)

        data = json.loads(filepath.read_text())
        assert data["name"] == "json_test"

    def test_jax_scalar_metric_json_serialization(self) -> None:
        """BenchmarkResult with JAX scalar metrics must serialize to JSON."""
        result = BenchmarkResult(
            name="jax_test",
            metrics={"accuracy": Metric(value=jnp.float32(0.95))},
            timestamp=1000.0,
        )
        data = result.to_dict()
        json_str = json.dumps(data)
        assert "0.95" in json_str or "0.949" in json_str

    def test_jax_scalar_timing_json_serialization(self) -> None:
        """TimingSample with JAX-like float values in to_dict must serialize."""
        timing = make_default_timing_sample()
        result = BenchmarkResult(name="timing_test", timing=timing, timestamp=1000.0)
        data = result.to_dict()
        json_str = json.dumps(data)
        assert "1.5" in json_str

    def test_jax_scalar_resources_json_serialization(self) -> None:
        """ResourceSummary numeric fields must serialize to JSON."""
        resources = make_default_resource_summary(peak_gpu_mem_mb=4096.0, mean_gpu_util=75.0)
        result = BenchmarkResult(name="resource_test", resources=resources, timestamp=1000.0)
        data = result.to_dict()
        json_str = json.dumps(data)
        assert "512.0" in json_str

    def test_jax_scalar_in_metadata_and_config(self) -> None:
        """JAX scalars nested in metadata/config dicts must serialize."""
        result = BenchmarkResult(
            name="meta_test",
            metadata={"mean_loss": jnp.float32(0.01), "steps": jnp.int32(1000)},
            config={"learning_rate": jnp.float32(0.001)},
            timestamp=1000.0,
        )
        data = result.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["metadata"]["mean_loss"] == pytest.approx(0.01, abs=1e-5)
        assert parsed["metadata"]["steps"] == 1000
        assert parsed["config"]["learning_rate"] == pytest.approx(0.001, abs=1e-5)

    def test_save_load_with_jax_scalars(self, tmp_path: "object") -> None:
        """Full save/load round-trip with JAX scalars must not raise."""
        from pathlib import Path

        filepath = Path(str(tmp_path)) / "jax_result.json"
        result = BenchmarkResult(
            name="jax_save_test",
            metrics={
                "loss": Metric(
                    value=jnp.float32(0.01),
                    lower=jnp.float32(0.005),
                    upper=jnp.float32(0.015),
                ),
            },
            metadata={"final_lr": jnp.float32(0.0001)},
            timestamp=1000.0,
        )
        result.save(filepath)
        loaded = BenchmarkResult.load(filepath)
        assert loaded.metrics["loss"].value == pytest.approx(0.01, abs=1e-3)
        assert loaded.metadata["final_lr"] == pytest.approx(0.0001, abs=1e-5)
