"""Unified benchmark result container.

Composes TimingSample, ResourceSummary, and Metric into a single
serializable BenchmarkResult with JSON save/load support.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from calibrax.core.models import _sanitize_for_json, Metric
from calibrax.profiling.resources import ResourceSummary
from calibrax.profiling.timing import TimingSample


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkResult:
    """Framework-agnostic benchmark result.

    Composes timing, resource, and metric measurements into a single
    serializable container for cross-framework comparison.

    Attributes:
        name: Benchmark name identifier.
        domain: Domain category (e.g., "computer_vision", "nlp").
        tags: Flexible key-value tags (framework, model, variant, etc.).
        timing: Timing measurements from TimingCollector.
        resources: Resource usage from ResourceMonitor.
        metrics: Named metric values with optional confidence intervals.
        metadata: Additional metadata (system info, hyperparameters, etc.).
        config: Benchmark configuration parameters.
        timestamp: Unix timestamp of the benchmark run.
    """

    name: str
    domain: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    timing: TimingSample | None = None
    resources: ResourceSummary | None = None
    metrics: dict[str, Metric] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary representation with nested objects serialized.
        """
        return {
            "name": self.name,
            "domain": self.domain,
            "tags": dict(self.tags),
            "timing": self.timing.to_dict() if self.timing is not None else None,
            "resources": self.resources.to_dict() if self.resources is not None else None,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "metadata": _sanitize_for_json(self.metadata),
            "config": _sanitize_for_json(self.config),
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with benchmark result fields.

        Returns:
            Reconstructed BenchmarkResult instance.
        """
        timing_data = data.get("timing")
        resources_data = data.get("resources")

        return cls(
            name=data["name"],
            domain=data.get("domain", ""),
            tags=data.get("tags", {}),
            timing=TimingSample.from_dict(timing_data) if timing_data is not None else None,
            resources=(
                ResourceSummary.from_dict(resources_data) if resources_data is not None else None
            ),
            metrics={k: Metric.from_dict(v) for k, v in data.get("metrics", {}).items()},
            metadata=data.get("metadata", {}),
            config=data.get("config", {}),
            timestamp=data.get("timestamp", 0.0),
        )

    def save(self, filepath: Path) -> None:
        """Save result to a JSON file, creating parent directories.

        Args:
            filepath: Path to the output JSON file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, filepath: Path) -> BenchmarkResult:
        """Load a BenchmarkResult from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            Reconstructed BenchmarkResult instance.
        """
        data = json.loads(Path(filepath).read_text())
        return cls.from_dict(data)
