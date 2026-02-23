"""Core data models, protocols, and abstractions."""

from calibrax.core.adapters import (
    adapt,
    AdapterRegistry,
    BenchmarkAdapter,
    NNXBenchmarkAdapter,
    register_adapter,
)
from calibrax.core.models import (
    extract_framework_metrics,
    is_higher_better,
    Metric,
    MetricDef,
    MetricDirection,
    MetricPriority,
    Point,
    RankEntry,
    Regression,
    Run,
    ScalingLaw,
    SignificanceResult,
    TrendPoint,
    TrendSeries,
)
from calibrax.core.protocols import (
    BatchableDatasetProtocol,
    BenchmarkProtocol,
    DatasetProtocol,
    MetricProtocol,
)
from calibrax.core.registry import (
    BenchmarkRegistry,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
    Registry,
)
from calibrax.core.result import BenchmarkResult


__all__ = [
    # adapters
    "AdapterRegistry",
    "BenchmarkAdapter",
    "NNXBenchmarkAdapter",
    "adapt",
    "register_adapter",
    # models
    "extract_framework_metrics",
    "Metric",
    "MetricDef",
    "MetricDirection",
    "MetricPriority",
    "Point",
    "RankEntry",
    "Regression",
    "Run",
    "ScalingLaw",
    "SignificanceResult",
    "TrendPoint",
    "TrendSeries",
    "is_higher_better",
    # protocols
    "BatchableDatasetProtocol",
    "BenchmarkProtocol",
    "DatasetProtocol",
    "MetricProtocol",
    # registry
    "BenchmarkRegistry",
    "Registry",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
    # result
    "BenchmarkResult",
]
