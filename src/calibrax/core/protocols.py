"""Protocol definitions for the calibrax benchmarking framework.

All protocols are runtime_checkable for structural subtyping checks.
"""

from typing import Protocol, runtime_checkable

import jax
from jax.typing import ArrayLike


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
class DatasetProtocol[ItemT](Protocol):
    """Interface for datasets used in benchmarks, whose examples are ``ItemT``."""

    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> ItemT:
        """Get an example by index.

        Args:
            idx: Index of the example.

        Returns:
            The example at the given index.
        """
        ...


@runtime_checkable
class BatchableDatasetProtocol[ItemT](DatasetProtocol[ItemT], Protocol):
    """A dataset that also returns batches of arrays by name."""

    def get_batch(self, batch_size: int, start_idx: int) -> dict[str, jax.Array]:
        """Get a batch of data starting at the given index.

        Args:
            batch_size: Number of examples in the batch.
            start_idx: Starting index for the batch.

        Returns:
            The batch's arrays by name.
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

    def compute(self, predictions: jax.Array, targets: jax.Array) -> jax.Array | float:
        """Compute the metric value.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.

        Returns:
            Computed metric value.
        """
        ...

    def validate_inputs(  # noqa: DOC502  # a protocol documents what implementations raise
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


@runtime_checkable
class StatefulMetricProtocol(Protocol):
    """Interface for stateful metrics with batch accumulation (Tier 1-2).

    Follows the update/compute/reset lifecycle pattern from TorchMetrics
    and Google Metrax. Metrics accumulate statistics across batches via
    update(), then produce final results via compute().

    Tier 1 (FrozenBackboneMetric): frozen pretrained backbone extracts
    features, accumulates statistics (e.g., FID mean/covariance).

    Tier 2 (LearnedMetric): trainable calibration layers on top of
    backbone features (e.g., LPIPS).
    """

    @property
    def name(self) -> str:
        """Get the metric name."""
        ...

    def update(self, **kwargs: ArrayLike) -> None:
        """Accumulate batch statistics.

        Args:
            **kwargs: Batch data (e.g., images, features, predictions).
        """
        ...

    def compute(self) -> dict[str, float]:
        """Compute final metric values from accumulated statistics.

        Returns:
            Dictionary mapping metric names to computed values.
        """
        ...

    def reset(self) -> None:
        """Reset accumulated state for a new evaluation run."""
        ...


@runtime_checkable
class MetricLearningProtocol(Protocol):
    """Interface for metric learning losses (Tier 3).

    Returns a differentiable JAX array (not a Python float) to enable
    gradient flow for training embedding spaces. The loss function IS
    the metric — it learns a distance function via backpropagation.

    Examples: ContrastiveLoss, TripletMarginLoss, ArcFaceLoss.
    """

    def __call__(self, embeddings: jax.Array, labels: jax.Array) -> jax.Array:
        """Compute the metric learning loss.

        Args:
            embeddings: Batch of embedding vectors, shape
                (batch_size, embedding_dim).
            labels: Integer class labels, shape (batch_size,).

        Returns:
            Scalar loss value as a JAX array (differentiable).
        """
        ...
