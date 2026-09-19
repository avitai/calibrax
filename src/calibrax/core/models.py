"""Core data models for the calibrax benchmarking framework.

All types are frozen dataclasses with ``to_dict()`` returning JSON values and ``from_dict()``
reading them back with ``substrax.records.read_record``, which checks every field against its
annotation. Tuples hold immutable sequences and ``StrEnum`` fixed value sets.

Numeric fields are converted to Python numbers in ``to_dict()``, so a JAX or NumPy scalar
stored in a record serialises; free-form fields hold :data:`MetadataValue`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from substrax.records import read_record
from substrax.typing import JsonValue

from calibrax.core.record_values import Metadata, metadata_to_json, require_stored


class MetricDirection(StrEnum):
    """Direction indicating whether higher or lower values are better."""

    HIGHER = "higher"
    LOWER = "lower"
    INFO = "info"


class MetricPriority(StrEnum):
    """Priority level for a metric definition."""

    PRIMARY = "primary"
    SECONDARY = "secondary"


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricDef:
    """How to interpret a metric — semantics for direction and grouping."""

    name: str
    unit: str
    direction: MetricDirection
    group: str = ""
    priority: MetricPriority = MetricPriority.SECONDARY
    description: str = ""

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction.value,
            "group": self.group,
            "priority": self.priority.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> MetricDef:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


def is_higher_better(md: MetricDef | None) -> bool:
    """Whether higher values are better for this metric.

    Returns True for "higher" or unknown (None) metrics, False for "lower".
    "info" metrics return True by convention (no ranking semantics).

    Args:
        md: Metric definition to check, or None for unknown.

    Returns:
        True if higher values are better or metric is unknown/info.
    """
    return md is None or md.direction != MetricDirection.LOWER


@dataclass(frozen=True, slots=True, kw_only=True)
class Metric:
    """Single metric value with optional confidence interval and samples."""

    value: float
    lower: float | None = None
    upper: float | None = None
    samples: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary, omitting None fields."""
        d: dict[str, JsonValue] = {"value": float(self.value)}
        if self.lower is not None:
            d["lower"] = float(self.lower)
        if self.upper is not None:
            d["upper"] = float(self.upper)
        if self.samples is not None:
            d["samples"] = [float(s) for s in self.samples]
        return d

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> Metric:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


@dataclass(frozen=True, slots=True, kw_only=True)
class Point:
    """One benchmark measurement under one configuration."""

    name: str
    scenario: str
    tags: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "scenario": self.scenario,
            "tags": dict(self.tags),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> Point:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


@dataclass(frozen=True, slots=True, kw_only=True)
class Run:
    """One execution of a benchmark suite."""

    points: tuple[Point, ...]
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    commit: str | None = None
    branch: str | None = None
    environment: Metadata = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)
    metric_defs: dict[str, MetricDef] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "commit": self.commit,
            "branch": self.branch,
            "environment": metadata_to_json(self.environment, "environment"),
            "metadata": metadata_to_json(self.metadata, "metadata"),
            "metric_defs": {k: v.to_dict() for k, v in self.metric_defs.items()},
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> Run:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit; ``id``, ``timestamp`` must be present, since their
                defaults would invent a value.
        """
        require_stored(cls, data, "id", "timestamp")
        return read_record(cls, data)


def extract_framework_metrics(
    run: Run,
    metric_names: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Extract per-framework metric values from run points.

    Iterates over each point in the run, groups by the ``framework`` tag
    (falling back to the point name), and collects values for the requested
    metrics.  Commonly used by ranking, scoring, and publication modules.

    Args:
        run: Benchmark run with points tagged by framework.
        metric_names: Metric names to extract (only keys are used if a mapping
            is passed).

    Returns:
        Mapping of ``{framework_label: {metric_name: value}}``.
    """
    frameworks: dict[str, dict[str, float]] = {}
    for point in run.points:
        fw = point.tags.get("framework", point.name)
        if fw not in frameworks:
            frameworks[fw] = {}
        for name in metric_names:
            if name in point.metrics:
                frameworks[fw][name] = point.metrics[name].value
    return frameworks


# --- Analysis result types ---


@dataclass(frozen=True, slots=True, kw_only=True)
class Regression:
    """A detected performance regression."""

    metric: str
    point_name: str
    baseline_value: float
    current_value: float
    delta_pct: float
    direction: MetricDirection

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "metric": self.metric,
            "point_name": self.point_name,
            "baseline_value": float(self.baseline_value),
            "current_value": float(self.current_value),
            "delta_pct": float(self.delta_pct),
            "direction": self.direction.value,
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> Regression:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


@dataclass(frozen=True, slots=True, kw_only=True)
class RankEntry:
    """One row in a ranking table."""

    label: str
    value: float
    rank: int
    is_best: bool
    delta_from_best: float

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "label": self.label,
            "value": float(self.value),
            "rank": int(self.rank),
            "is_best": self.is_best,
            "delta_from_best": float(self.delta_from_best),
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> RankEntry:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


@dataclass(frozen=True, slots=True, kw_only=True)
class SignificanceResult:
    """Result of a statistical significance test."""

    p_value: float
    statistic: float
    effect_size: float
    significant: bool
    method: str

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "p_value": float(self.p_value),
            "statistic": float(self.statistic),
            "effect_size": float(self.effect_size),
            "significant": self.significant,
            "method": self.method,
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> SignificanceResult:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScalingLaw:
    """Power-law fit: value = coefficient * size^exponent."""

    coefficient: float
    exponent: float
    r_squared: float
    complexity: str

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "coefficient": float(self.coefficient),
            "exponent": float(self.exponent),
            "r_squared": float(self.r_squared),
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> ScalingLaw:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


# --- Trend tracking ---


@dataclass(frozen=True, slots=True, kw_only=True)
class TrendPoint:
    """One data point in a time-series trend."""

    run_id: str
    timestamp: datetime
    value: float
    commit: str | None = None
    lower: float | None = None
    upper: float | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary, omitting None fields."""
        d: dict[str, JsonValue] = {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "value": float(self.value),
        }
        if self.commit is not None:
            d["commit"] = self.commit
        if self.lower is not None:
            d["lower"] = float(self.lower)
        if self.upper is not None:
            d["upper"] = float(self.upper)
        return d

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> TrendPoint:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrendSeries:
    """Time-series trend for a single metric across multiple runs."""

    metric: str
    point_name: str
    tags: dict[str, str] = field(default_factory=dict)
    points: tuple[TrendPoint, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "metric": self.metric,
            "point_name": self.point_name,
            "tags": dict(self.tags),
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> TrendSeries:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)
