"""Record-weighted batch means composed from native Flax NNX metrics."""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class MeanAccumulator(nnx.Module):
    """Accumulate named scalar batch means without transferring them to the host.

    Each metric has its own native running total and record count. Omitted
    metrics do not advance that metric's count. State follows Flax's float32
    totals and int32 counts; callers must keep each window below int32 capacity.
    """

    def __init__(self, names: Sequence[str]) -> None:
        """Create native running averages for the fixed metric names.

        Args:
            names: Names accepted by :meth:`update`.
        """
        self.averages = nnx.Dict({name: nnx.metrics.Average() for name in names})

    def update(self, metrics: Mapping[str, jax.Array], records: jax.Array) -> None:
        """Add scalar batch means weighted by a nonnegative integer record count.

        Args:
            metrics: Scalar means for any subset of the configured metric names.
            records: Scalar integer count of the records represented by each mean.

        Raises:
            KeyError: If a metric name was not configured.
            ValueError: If a mean or the record count is not scalar.
            TypeError: If the record count is not an integer array.
        """
        unknown = set(metrics) - set(self.averages)
        if unknown:
            raise KeyError(f"Unknown metrics: {sorted(unknown)}")
        if records.ndim != 0 or any(value.ndim != 0 for value in metrics.values()):
            raise ValueError("Batch means and the record count must be scalars")
        if not jnp.issubdtype(records.dtype, jnp.integer):
            raise TypeError("The record count must have an integer dtype")
        for name, value in metrics.items():
            mean = value.astype(jnp.float32)
            # CLU's Average excludes masked values before accumulation. Native
            # NNX Average multiplies by its mask, so sanitize before that product:
            # an empty batch may have an undefined mean (NaN * 0 is still NaN).
            observed = jnp.where(records == 0, jnp.zeros_like(mean), mean)
            self.averages[name].update(values=observed, mask=records)

    def compute(self) -> dict[str, jax.Array]:
        """Return device means, with NaN for a metric with no records.

        Returns:
            Named scalar means. This call does not synchronize with the host.
        """
        return {name: average.compute() for name, average in self.averages.items()}

    def reset(self) -> None:
        """Reset all native totals and counts for a new window."""
        for average in self.averages.values():
            average.reset()
