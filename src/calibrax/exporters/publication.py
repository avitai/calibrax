"""Publication-ready plot and table generation for benchmark results.

Generates comparison bar charts, scaling plots, convergence plots,
and formatted tables (LaTeX, HTML, CSV). Requires optional ``matplotlib``
dependency for plots; table generation works without it.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from calibrax.core.models import extract_framework_metrics, is_higher_better, Run, TrendSeries


try:
    import matplotlib.pyplot  # noqa: F401

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logger = logging.getLogger(__name__)


class PublicationGenerator:
    """Generate publication-ready plots and tables from benchmark data.

    Args:
        output_dir: Directory where generated files are saved.
    """

    def __init__(self, output_dir: Path | str) -> None:
        """Initialize the publication generator."""
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _get_pyplot(self, plot_name: str) -> Any | None:
        """Return matplotlib.pyplot or None when plotting is unavailable."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping %s", plot_name)
            return None

        import matplotlib.pyplot as plt

        return plt

    def generate_comparison_plot(
        self,
        run: Run,
        metrics: Sequence[str] | None = None,
        *,
        output_format: str = "png",
    ) -> Path | None:
        """Generate a bar chart comparing frameworks across metrics.

        Args:
            run: Benchmark run with points tagged by framework.
            metrics: Subset of metric names to plot. Defaults to all.
            output_format: File format (png, pdf, svg).

        Returns:
            Path to generated file, or None if matplotlib unavailable.
        """
        plt = self._get_pyplot("comparison plot")
        if plt is None:
            return None

        metric_names = (
            list(metrics) if metrics else sorted({mn for p in run.points for mn in p.metrics})
        )

        frameworks = extract_framework_metrics(run, metric_names)
        if not frameworks:
            return None

        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False)

        fw_names = sorted(frameworks.keys())
        for i, mn in enumerate(metric_names):
            _plot_metric_bars(axes[0, i], fw_names, frameworks, mn)

        fig.tight_layout()
        path = self._output_dir / f"comparison.{output_format}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_scaling_plot(
        self,
        sizes: Sequence[float],
        values: Sequence[float],
        *,
        metric_name: str = "throughput",
        output_format: str = "png",
    ) -> Path | None:
        """Generate a scaling plot (size vs metric value).

        Args:
            sizes: Input sizes (x-axis).
            values: Metric values (y-axis).
            metric_name: Name of the metric being plotted.
            output_format: File format (png, pdf, svg).

        Returns:
            Path to generated file, or None if matplotlib unavailable.
        """
        plt = self._get_pyplot("scaling plot")
        if plt is None:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(sizes), list(values), "o-", linewidth=2, markersize=6)
        ax.set_xlabel("Size")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Scaling: {metric_name}")
        ax.grid(True, alpha=0.3)

        path = self._output_dir / f"scaling_{metric_name}.{output_format}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_convergence_plot(
        self,
        series: TrendSeries,
        *,
        output_format: str = "png",
    ) -> Path | None:
        """Generate a convergence plot from a trend series.

        Args:
            series: Time-series trend data.
            output_format: File format (png, pdf, svg).

        Returns:
            Path to generated file, or None if matplotlib unavailable.
        """
        plt = self._get_pyplot("convergence plot")
        if plt is None:
            return None

        if not series.points:
            return None

        values = [float(tp.value) for tp in series.points]
        indices = list(range(len(values)))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(indices, values, "o-", linewidth=2, markersize=4)
        ax.set_xlabel("Run index")
        ax.set_ylabel(series.metric)
        ax.set_title(f"Convergence: {series.metric} ({series.point_name})")
        ax.grid(True, alpha=0.3)

        _add_confidence_band(ax, indices, series, len(values))

        path = self._output_dir / f"convergence_{series.metric}.{output_format}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_metric_values(
        self,
        values: Mapping[str, float],
        *,
        title: str,
        filename: str,
        output_format: str = "png",
    ) -> Path | None:
        """Plot one or more scalar metric values.

        Args:
            values: Mapping from metric name to scalar value.
            title: Plot title.
            filename: Output filename stem.
            output_format: File format (png, pdf, svg).

        Returns:
            Path to generated file, or None if matplotlib unavailable.

        Raises:
            ValueError: If no metric values are provided.
        """
        if not values:
            msg = "plot_metric_values requires at least one metric value"
            raise ValueError(msg)

        plt = self._get_pyplot("metric values plot")
        if plt is None:
            return None

        names = list(values)
        scores = [float(values[name]) for name in names]
        fig, ax = plt.subplots(figsize=(max(4.0, len(names) * 1.2), 3.0))
        ax.bar(names, scores)
        ax.set_title(title)
        ax.set_ylabel("Value")
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
        fig.tight_layout()

        path = self._output_dir / f"{_sanitize_filename(filename)}.{output_format}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_table(
        self,
        run: Run,
        metrics: Sequence[str] | None = None,
        *,
        output_format: str = "latex",
        group_by_tag: str = "framework",
    ) -> Path:
        """Generate a formatted comparison table.

        Args:
            run: Benchmark run with points and metrics.
            metrics: Subset of metrics to include. Defaults to all.
            output_format: One of "latex", "html", "csv".
            group_by_tag: Tag key used for row labels.

        Returns:
            Path to the generated table file.

        Raises:
            ValueError: If output_format is not recognized.
        """
        metric_names = (
            list(metrics) if metrics else sorted({mn for p in run.points for mn in p.metrics})
        )

        rows: list[dict[str, Any]] = []
        best_values = _find_best_per_metric(run, metric_names)

        for point in run.points:
            label = point.tags.get(group_by_tag, point.name)
            row: dict[str, Any] = {"label": label}
            for mn in metric_names:
                if mn in point.metrics:
                    row[mn] = float(point.metrics[mn].value)
                else:
                    row[mn] = None
            rows.append(row)

        if output_format == "latex":
            return self._generate_latex_table(metric_names, rows, best_values)
        if output_format == "html":
            return self._generate_html_table(metric_names, rows, best_values)
        if output_format == "csv":
            return self._generate_csv_table(metric_names, rows)

        msg = f"Unknown output format: {output_format!r}. Use 'latex', 'html', or 'csv'."
        raise ValueError(msg)

    def _generate_latex_table(
        self,
        metric_names: list[str],
        rows: list[dict[str, Any]],
        best_values: dict[str, float],
    ) -> Path:
        """Generate a LaTeX table with bold-best values."""
        col_spec = "l" + "r" * len(metric_names)
        header = " & ".join(["Framework", *metric_names])

        lines: list[str] = [
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
        ]
        for row in rows:
            cells = [row["label"]]
            for mn in metric_names:
                val = row.get(mn)
                if val is None:
                    cells.append("-")
                elif val == best_values.get(mn):
                    cells.append(f"\\textbf{{{val:.4f}}}")
                else:
                    cells.append(f"{val:.4f}")
            lines.append(" & ".join(cells) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])

        path = self._output_dir / "table.tex"
        path.write_text("\n".join(lines))
        return path

    def _generate_html_table(
        self,
        metric_names: list[str],
        rows: list[dict[str, Any]],
        best_values: dict[str, float],
    ) -> Path:
        """Generate an HTML table with bold-best values."""
        headers = "".join(f"<th>{h}</th>" for h in ["Framework", *metric_names])
        rows_html: list[str] = []
        for row in rows:
            cells = [f"<td>{row['label']}</td>"]
            for mn in metric_names:
                val = row.get(mn)
                if val is None:
                    cells.append("<td>-</td>")
                elif val == best_values.get(mn):
                    cells.append(f"<td><b>{val:.4f}</b></td>")
                else:
                    cells.append(f"<td>{val:.4f}</td>")
            rows_html.append(f"<tr>{''.join(cells)}</tr>")

        html = f"<table>\n<tr>{headers}</tr>\n{''.join(rows_html)}\n</table>"
        path = self._output_dir / "table.html"
        path.write_text(html)
        return path

    def _generate_csv_table(
        self,
        metric_names: list[str],
        rows: list[dict[str, Any]],
    ) -> Path:
        """Generate a CSV table."""
        path = self._output_dir / "table.csv"
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Framework", *metric_names])
        for row in rows:
            writer.writerow(
                [
                    row["label"],
                    *[row.get(mn, "") for mn in metric_names],
                ]
            )
        path.write_text(output.getvalue())
        return path


def _plot_metric_bars(
    ax: Any,
    fw_names: list[str],
    frameworks: dict[str, dict[str, float]],
    metric_name: str,
) -> None:
    """Plot a bar chart for a single metric on an axes.

    Args:
        ax: Matplotlib axes to plot on.
        fw_names: Sorted framework names.
        frameworks: Per-framework metric values.
        metric_name: Metric to plot.
    """
    values = [frameworks[fw].get(metric_name, 0) for fw in fw_names]
    ax.bar(fw_names, values)
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)


def _sanitize_filename(filename: str) -> str:
    """Return a stable filesystem-safe filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename.strip()).strip("._")
    return cleaned or "metrics"


def _add_confidence_band(
    ax: Any,
    indices: list[int],
    series: TrendSeries,
    n_values: int,
) -> None:
    """Add confidence band to convergence plot if bounds are available.

    Args:
        ax: Matplotlib axes to plot on.
        indices: X-axis indices.
        series: Trend series with potential lower/upper bounds.
        n_values: Number of values in the series.
    """
    if series.points[0].lower is not None and series.points[0].upper is not None:
        lower = [float(tp.lower) for tp in series.points if tp.lower is not None]
        upper = [float(tp.upper) for tp in series.points if tp.upper is not None]
        if len(lower) == n_values and len(upper) == n_values:
            ax.fill_between(indices, lower, upper, alpha=0.2)


def _find_best_per_metric(run: Run, metric_names: list[str]) -> dict[str, float]:
    """Find the best value for each metric (direction-aware).

    Args:
        run: Benchmark run to scan.
        metric_names: Metric names to consider.

    Returns:
        {metric_name: best_value}.
    """
    best: dict[str, float] = {}
    for mn in metric_names:
        md = run.metric_defs.get(mn)
        higher = is_higher_better(md)
        values = [p.metrics[mn].value for p in run.points if mn in p.metrics]
        if values:
            best[mn] = max(values) if higher else min(values)
    return best
