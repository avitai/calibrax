"""Core data models, protocols, and abstractions; each export loads on first use."""

from .adapters import (
    adapt as adapt,
    Adapter as Adapter,
    AdapterClass as AdapterClass,
    AdapterRegistry as AdapterRegistry,
    BenchmarkAdapter as BenchmarkAdapter,
    NNXBenchmarkAdapter as NNXBenchmarkAdapter,
    register_adapter as register_adapter,
)
from .models import (
    extract_framework_metrics as extract_framework_metrics,
    is_higher_better as is_higher_better,
    Metric as Metric,
    MetricDef as MetricDef,
    MetricDirection as MetricDirection,
    MetricPriority as MetricPriority,
    Point as Point,
    RankEntry as RankEntry,
    Regression as Regression,
    Run as Run,
    ScalingLaw as ScalingLaw,
    SignificanceResult as SignificanceResult,
    TrendPoint as TrendPoint,
    TrendSeries as TrendSeries,
)
from .protocols import (
    BatchableDatasetProtocol as BatchableDatasetProtocol,
    BenchmarkProtocol as BenchmarkProtocol,
    DatasetProtocol as DatasetProtocol,
    MetricLearningProtocol as MetricLearningProtocol,
    MetricProtocol as MetricProtocol,
    StatefulMetricProtocol as StatefulMetricProtocol,
)
from .record_values import (
    Metadata as Metadata,
    MetadataValue as MetadataValue,
    read_metadata as read_metadata,
    read_metadata_entry as read_metadata_entry,
    SupportsItem as SupportsItem,
)
from .registry import (
    BenchmarkRegistry as BenchmarkRegistry,
    get_benchmark as get_benchmark,
    list_benchmarks as list_benchmarks,
    register_benchmark as register_benchmark,
    Registry as Registry,
    SingletonRegistry as SingletonRegistry,
)
from .result import (
    BenchmarkResult as BenchmarkResult,
)

__all__ = [
    # adapters
    "Adapter",
    "AdapterClass",
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
    # record values
    "Metadata",
    "MetadataValue",
    "SupportsItem",
    "read_metadata",
    "read_metadata_entry",
    # protocols
    "BatchableDatasetProtocol",
    "BenchmarkProtocol",
    "DatasetProtocol",
    "MetricLearningProtocol",
    "MetricProtocol",
    "StatefulMetricProtocol",
    # registry
    "BenchmarkRegistry",
    "Registry",
    "SingletonRegistry",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
    # result
    "BenchmarkResult",
]
