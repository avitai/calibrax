"""Tests for calibrax core protocol definitions.

Verifies runtime_checkable isinstance checks for conforming and
non-conforming classes across all protocol definitions.
"""

from typing import Any

import jax

from calibrax.core.protocols import (
    BatchableDatasetProtocol,
    BenchmarkProtocol,
    DatasetProtocol,
    MetricProtocol,
)


class TestBenchmarkProtocol:
    """Tests for BenchmarkProtocol."""

    def test_conforming_class_satisfies_protocol(self) -> None:
        class GoodBenchmark:
            def setup(self) -> None:
                pass

            def run_training(self) -> dict[str, float]:
                return {}

            def run_evaluation(self) -> dict[str, float]:
                return {}

            def teardown(self) -> None:
                pass

            def get_performance_targets(self) -> dict[str, float]:
                return {}

        assert isinstance(GoodBenchmark(), BenchmarkProtocol)

    def test_non_conforming_class_fails(self) -> None:
        class BadBenchmark:
            def setup(self) -> None:
                pass

        assert not isinstance(BadBenchmark(), BenchmarkProtocol)

    def test_empty_class_fails(self) -> None:
        class Empty:
            pass

        assert not isinstance(Empty(), BenchmarkProtocol)


class TestDatasetProtocol:
    """Tests for DatasetProtocol."""

    def test_conforming_class_satisfies_protocol(self) -> None:
        class GoodDataset:
            def __len__(self) -> int:
                return 10

            def __getitem__(self, idx: int) -> Any:
                return idx

        assert isinstance(GoodDataset(), DatasetProtocol)

    def test_non_conforming_class_fails(self) -> None:
        class BadDataset:
            def __len__(self) -> int:
                return 0

        assert not isinstance(BadDataset(), DatasetProtocol)


class TestBatchableDatasetProtocol:
    """Tests for BatchableDatasetProtocol."""

    def test_conforming_class_satisfies_protocol(self) -> None:
        class GoodBatchable:
            def __len__(self) -> int:
                return 100

            def __getitem__(self, idx: int) -> Any:
                return idx

            def get_batch(self, batch_size: int, start_idx: int) -> dict[str, Any]:
                return {"data": list(range(start_idx, start_idx + batch_size))}

        assert isinstance(GoodBatchable(), BatchableDatasetProtocol)

    def test_satisfies_dataset_protocol_too(self) -> None:
        class GoodBatchable:
            def __len__(self) -> int:
                return 100

            def __getitem__(self, idx: int) -> Any:
                return idx

            def get_batch(self, batch_size: int, start_idx: int) -> dict[str, Any]:
                return {}

        instance = GoodBatchable()
        assert isinstance(instance, DatasetProtocol)
        assert isinstance(instance, BatchableDatasetProtocol)

    def test_dataset_without_get_batch_fails(self) -> None:
        class JustDataset:
            def __len__(self) -> int:
                return 10

            def __getitem__(self, idx: int) -> Any:
                return idx

        assert isinstance(JustDataset(), DatasetProtocol)
        assert not isinstance(JustDataset(), BatchableDatasetProtocol)


class TestMetricProtocol:
    """Tests for MetricProtocol."""

    def test_conforming_class_satisfies_protocol(self) -> None:
        class GoodMetric:
            @property
            def name(self) -> str:
                return "accuracy"

            @property
            def higher_is_better(self) -> bool:
                return True

            def compute(
                self,
                predictions: jax.Array,
                targets: jax.Array,
            ) -> float:
                return 0.95

            def validate_inputs(
                self,
                predictions: jax.Array,
                targets: jax.Array,
            ) -> None:
                pass

        assert isinstance(GoodMetric(), MetricProtocol)

    def test_non_conforming_class_fails(self) -> None:
        class BadMetric:
            @property
            def name(self) -> str:
                return "mse"

        assert not isinstance(BadMetric(), MetricProtocol)
