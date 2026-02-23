"""Protocol definitions for the calibrax benchmarking framework.

All protocols are runtime_checkable for structural subtyping checks.
"""

from typing import Any, Protocol, runtime_checkable

import jax


@runtime_checkable
class BenchmarkProtocol(Protocol):
    """Standard interface for benchmarks.

    Defines the lifecycle of a benchmark: setup, training, evaluation,
    teardown, and performance target retrieval.
    """

    def setup(self) -> None:
        """Set up benchmark resources before execution."""
        ...

    def run_training(self) -> dict[str, float]:
        """Execute the training phase.

        Returns:
            Dictionary of training metric name-value pairs.
        """
        ...

    def run_evaluation(self) -> dict[str, float]:
        """Execute the evaluation phase.

        Returns:
            Dictionary of evaluation metric name-value pairs.
        """
        ...

    def teardown(self) -> None:
        """Release benchmark resources after execution."""
        ...

    def get_performance_targets(self) -> dict[str, float]:
        """Return expected performance targets.

        Returns:
            Dictionary mapping metric names to target values.
        """
        ...


@runtime_checkable
class DatasetProtocol(Protocol):
    """Interface for datasets used in benchmarks."""

    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> Any:
        """Get an example by index.

        Args:
            idx: Index of the example.

        Returns:
            The example at the given index.
        """
        ...


@runtime_checkable
class BatchableDatasetProtocol(Protocol):
    """Interface for datasets that support batch retrieval.

    Extends DatasetProtocol with get_batch capability.
    """

    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> Any:
        """Get an example by index.

        Args:
            idx: Index of the example.

        Returns:
            The example at the given index.
        """
        ...

    def get_batch(self, batch_size: int, start_idx: int) -> dict[str, Any]:
        """Get a batch of data starting at the given index.

        Args:
            batch_size: Number of examples in the batch.
            start_idx: Starting index for the batch.

        Returns:
            Batch data dictionary.
        """
        ...


@runtime_checkable
class MetricProtocol(Protocol):
    """Universal metric interface for evaluation.

    Supports computing a metric from predictions and targets,
    with input validation.
    """

    @property
    def name(self) -> str:
        """Get the metric name."""
        ...

    @property
    def higher_is_better(self) -> bool:
        """Whether higher values indicate better performance."""
        ...

    def compute(self, predictions: jax.Array, targets: jax.Array) -> float:
        """Compute the metric value.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.

        Returns:
            Computed metric value.
        """
        ...

    def validate_inputs(
        self,
        predictions: jax.Array,
        targets: jax.Array,
    ) -> None:
        """Validate that inputs are compatible for metric computation.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.

        Raises:
            ValueError: If inputs are incompatible.
        """
        ...
