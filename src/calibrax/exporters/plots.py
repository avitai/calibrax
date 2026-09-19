"""Publication plots of benchmark results, drawn with matplotlib.

This module is the matplotlib integration: it needs the ``publication`` extra, and importing it
without matplotlib raises ``ImportError`` naming the extra. Tables need no plotting library and
live in :mod:`calibrax.exporters.publication`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path


try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError as error:
    msg = 'calibrax.exporters.plots needs matplotlib: uv pip install "calibrax[publication]"'
    raise ImportError(msg) from error

from calibrax.core.models import extract_framework_metrics, Run, TrendSeries


_DPI = 150


class PlotGenerator:
    """Write comparison, scaling, convergence and scalar-value plots to a directory."""

    def __init__(self, output_dir: Path | str) -> None:
        """Create the output directory.

        Args:
            output_dir: Where plots are written.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def comparison_plot(
        self, run: Run, metrics: Sequence[str] | None = None, *, output_format: str = "png"
    ) -> Path:
        """One bar chart per metric, a bar per framework (the ``framework`` tag).

        Args:
            run: Benchmark run with points tagged by framework.
            metrics: Metrics to plot; every metric of the run by default.
            output_format: File format (png, pdf, svg).

        Returns:
            Path of the written figure.

        Raises:
            ValueError: If no point of the run holds a requested metric.
        """
        metric_names = (
            list(metrics) if metrics else sorted({mn for p in run.points for mn in p.metrics})
        )
        frameworks = extract_framework_metrics(run, metric_names)
        if not frameworks:
            msg = f"no framework in the run holds any of the metrics {metric_names}"
            raise ValueError(msg)
        fig, axes = plt.subplots(
            1, len(metric_names), figsize=(5 * len(metric_names), 5), squeeze=False
        )
        fw_names = sorted(frameworks)
        for index, metric_name in enumerate(metric_names):
            _metric_bars(axes[0, index], fw_names, frameworks, metric_name)
        fig.tight_layout()
        return self._save(fig, f"comparison.{output_format}")

    def scaling_plot(
        self,
        sizes: Sequence[float],
        values: Sequence[float],
        *,
        metric_name: str = "throughput",
        output_format: str = "png",
    ) -> Path:
        """A metric against input size.

        Args:
            sizes: Input sizes (x-axis).
            values: Metric values (y-axis).
            metric_name: The metric's name.
            output_format: File format (png, pdf, svg).

        Returns:
            Path of the written figure.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(sizes), list(values), "o-", linewidth=2, markersize=6)
        ax.set_xlabel("Size")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Scaling: {metric_name}")
        ax.grid(True, alpha=0.3)
        return self._save(fig, f"scaling_{metric_name}.{output_format}")

    def convergence_plot(self, series: TrendSeries, *, output_format: str = "png") -> Path:
        """A trend series by run index, with its confidence band when every point has one.

        Args:
            series: The series.
            output_format: File format (png, pdf, svg).

        Returns:
            Path of the written figure.

        Raises:
            ValueError: If the series has no points.
        """
        if not series.points:
            msg = f"the {series.metric!r} series of {series.point_name!r} has no points"
            raise ValueError(msg)
        values = [float(tp.value) for tp in series.points]
        indices = list(range(len(values)))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(indices, values, "o-", linewidth=2, markersize=4)
        ax.set_xlabel("Run index")
        ax.set_ylabel(series.metric)
        ax.set_title(f"Convergence: {series.metric} ({series.point_name})")
        ax.grid(True, alpha=0.3)
        _confidence_band(ax, indices, series)
        return self._save(fig, f"convergence_{series.metric}.{output_format}")

    def metric_values_plot(
        self,
        values: Mapping[str, float],
        *,
        title: str,
        filename: str,
        output_format: str = "png",
    ) -> Path:
        """One bar per scalar metric value, such as a stateful metric's ``compute()``.

        Args:
            values: Metric name to value.
            title: Plot title.
            filename: File name stem; characters outside ``[A-Za-z0-9_.-]`` become ``_``.
            output_format: File format (png, pdf, svg).

        Returns:
            Path of the written figure.

        Raises:
            ValueError: If no metric values are given.
        """
        if not values:
            msg = "metric_values_plot requires at least one metric value"
            raise ValueError(msg)
        names = list(values)
        fig, ax = plt.subplots(figsize=(max(4.0, len(names) * 1.2), 3.0))
        ax.bar(names, [float(values[name]) for name in names])
        ax.set_title(title)
        ax.set_ylabel("Value")
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
        fig.tight_layout()
        return self._save(fig, f"{_sanitize_filename(filename)}.{output_format}")

    def _save(self, fig: Figure, name: str) -> Path:
        """Write ``fig`` under the output directory and release it."""
        path = self._output_dir / name
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return path


def _metric_bars(
    ax: Axes,
    fw_names: Sequence[str],
    frameworks: Mapping[str, Mapping[str, float]],
    metric_name: str,
) -> None:
    """A bar per framework for one metric; a framework without the metric shows 0."""
    ax.bar(list(fw_names), [frameworks[fw].get(metric_name, 0) for fw in fw_names])
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)


def _confidence_band(ax: Axes, indices: Sequence[int], series: TrendSeries) -> None:
    """Shade the band between each point's bounds when every point carries both."""
    lower = [float(tp.lower) for tp in series.points if tp.lower is not None]
    upper = [float(tp.upper) for tp in series.points if tp.upper is not None]
    if len(lower) == len(indices) and len(upper) == len(indices):
        ax.fill_between(list(indices), lower, upper, alpha=0.2)


def _sanitize_filename(filename: str) -> str:
    """Return a stable filesystem-safe filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename.strip()).strip("._")
    return cleaned or "metrics"
