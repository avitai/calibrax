"""Publication-ready comparison tables (LaTeX, HTML, CSV) for benchmark results.

Tables need no plotting library; the plots live in :mod:`calibrax.exporters.plots`, which
needs matplotlib.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from calibrax.core.models import is_higher_better, Run


@dataclass(frozen=True, slots=True, kw_only=True)
class _TableRow:
    """One table row: its label and each metric's value, ``None`` where the point lacks it."""

    label: str
    values: dict[str, float | None]


class PublicationGenerator:
    """Generate publication-ready comparison tables from benchmark data."""

    def __init__(self, output_dir: Path | str) -> None:
        """Initialize the publication generator."""
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

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

        best_values = _find_best_per_metric(run, metric_names)
        rows = [
            _TableRow(
                label=point.tags.get(group_by_tag, point.name),
                values={
                    mn: float(point.metrics[mn].value) if mn in point.metrics else None
                    for mn in metric_names
                },
            )
            for point in run.points
        ]

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
        rows: list[_TableRow],
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
            cells = [row.label]
            for mn in metric_names:
                val = row.values[mn]
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
        rows: list[_TableRow],
        best_values: dict[str, float],
    ) -> Path:
        """Generate an HTML table with bold-best values."""
        headers = "".join(f"<th>{h}</th>" for h in ["Framework", *metric_names])
        rows_html: list[str] = []
        for row in rows:
            cells = [f"<td>{row.label}</td>"]
            for mn in metric_names:
                val = row.values[mn]
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
        rows: list[_TableRow],
    ) -> Path:
        """Generate a CSV table."""
        path = self._output_dir / "table.csv"
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Framework", *metric_names])
        for row in rows:
            writer.writerow(
                [
                    row.label,
                    *[row.values[mn] for mn in metric_names],
                ]
            )
        path.write_text(output.getvalue())
        return path


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
