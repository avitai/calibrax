"""Shared plotting helpers for metric classes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol


class _ComputableMetric(Protocol):
    """Protocol for stateful metric objects with scalar compute output."""

    @property
    def name(self) -> str:
        """Return the metric name."""
        ...

    def compute(self) -> Mapping[str, float]:
        """Return computed scalar metric values."""
        ...


class MetricPlotMixin:
    """Mixin that plots scalar values returned by ``compute()``."""

    def plot(
        self: _ComputableMetric,
        *,
        output_dir: str | Path = "figures",
        output_format: str = "png",
    ) -> Path | None:
        """Plot the current computed metric values.

        Args:
            output_dir: Directory where the plot should be written.
            output_format: Matplotlib output format.

        Returns:
            Path to the generated plot, or None if matplotlib is unavailable.
        """
        return _plot_stateful_metric(self, output_dir=output_dir, output_format=output_format)


def _plot_stateful_metric(
    metric: _ComputableMetric,
    *,
    output_dir: str | Path,
    output_format: str,
) -> Path | None:
    """Plot computed scalar values for a stateful metric."""
    from calibrax.exporters.publication import PublicationGenerator

    generator = PublicationGenerator(output_dir=output_dir)
    return generator.plot_metric_values(
        metric.compute(),
        title=metric.name,
        filename=metric.name,
        output_format=output_format,
    )
