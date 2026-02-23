"""Weights & Biases exporter for benchmark results and analysis.

Exports benchmark runs, comparisons, regressions, rankings, and trends
to W&B dashboards. Requires optional ``wandb`` dependency.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from calibrax.core.models import is_higher_better, Run
from calibrax.exporters.base import Exporter


try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    WANDB_AVAILABLE = False


logger = logging.getLogger(__name__)


def _discover_metric_names(run: Run) -> list[str]:
    """Extract all unique metric names from a run's points.

    Args:
        run: Benchmark run to scan.

    Returns:
        Sorted list of unique metric names.
    """
    names: set[str] = set()
    for point in run.points:
        names.update(point.metrics.keys())
    return sorted(names)


def _find_best_values(
    run: Run,
    metric_names: list[str],
) -> dict[str, tuple[float, bool]]:
    """Find the best value for each metric across all points.

    Args:
        run: Benchmark run to scan.
        metric_names: Metrics to consider.

    Returns:
        {metric_name: (best_value, higher_is_better)}.
    """
    best: dict[str, tuple[float, bool]] = {}
    for metric_name in metric_names:
        md = run.metric_defs.get(metric_name)
        higher = is_higher_better(md)
        values = [p.metrics[metric_name].value for p in run.points if metric_name in p.metrics]
        if values:
            best_val = max(values) if higher else min(values)
            best[metric_name] = (best_val, higher)
    return best


class WandBExporter(Exporter):
    """Export benchmark results and analysis to Weights & Biases.

    Args:
        project: W&B project name.
        entity: W&B entity (team or user). Uses default if None.
        tags: Optional tags applied to all W&B runs.

    Raises:
        ImportError: If wandb is not installed.
    """

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize the W&B exporter."""
        if not WANDB_AVAILABLE:
            msg = 'wandb is required for WandBExporter: uv pip install "calibrax[wandb]"'
            raise ImportError(msg)
        self._project = project
        self._entity = entity
        self._tags = tags or []
        self._wandb_run: Any = None

    _VALID_MODES = frozenset({"online", "offline", "disabled", "shared"})

    @staticmethod
    def _resolve_wandb_mode() -> Literal["online", "offline", "disabled", "shared"] | None:
        """Resolve the W&B mode from environment, ensuring offline mode is respected."""
        mode = os.environ.get("WANDB_MODE")
        if mode and mode in WandBExporter._VALID_MODES:
            return mode  # type: ignore[return-value]
        return None

    def check_auth(self) -> bool:
        """Check if W&B authentication is available.

        Returns:
            True if authenticated (API key, offline mode, or stored creds).
        """
        if os.environ.get("WANDB_API_KEY"):
            return True
        if os.environ.get("WANDB_MODE") == "offline":
            return True
        try:
            return wandb.api.api_key is not None  # type: ignore[union-attr]
        except (AttributeError, RuntimeError, ValueError, OSError):
            return False

    def export_run(self, run: Run, *, finish: bool = True) -> str:
        """Export a benchmark run to W&B.

        Logs all metrics with slash-grouped panel names, a comparison
        summary table, and an HTML comparison table.

        Args:
            run: Benchmark run to export.
            finish: Whether to finish the W&B run after export.

        Returns:
            URL of the W&B run.
        """
        self._wandb_run = wandb.init(  # type: ignore[union-attr]
            project=self._project,
            entity=self._entity,
            tags=self._tags,
            config={
                "run_id": run.id,
                "commit": run.commit,
                "branch": run.branch,
                **run.environment,
            },
            mode=self._resolve_wandb_mode(),
        )

        metric_names = _discover_metric_names(run)
        best_values = _find_best_values(run, metric_names)

        for point in run.points:
            fw = point.tags.get("framework", point.name)
            for metric_name, metric in point.metrics.items():
                md = run.metric_defs.get(metric_name)
                group = md.group if md else ""
                panel_key = f"{group}/{metric_name}/{fw}" if group else f"{metric_name}/{fw}"
                self._wandb_run.log({panel_key: float(metric.value)})

        self._log_comparison_table(run, metric_names, best_values)
        self._log_html_comparison(run, metric_names, best_values)

        url = self._wandb_run.url or ""
        if finish:
            self._wandb_run.finish()
            self._wandb_run = None
        return url

    def export_analysis(self, run: Run, baseline: Run | None = None) -> None:
        """Export analysis artifacts: rankings, regressions, aggregate scores, Pareto.

        Args:
            run: Current benchmark run.
            baseline: Optional baseline for regression detection.
        """
        if self._wandb_run is None:
            self._wandb_run = wandb.init(  # type: ignore[union-attr]
                project=self._project,
                entity=self._entity,
                tags=[*self._tags, "analysis"],
                mode=self._resolve_wandb_mode(),
            )

        metric_names = _discover_metric_names(run)
        self._log_rank_tables(run, metric_names)
        self._log_aggregate_scores(run, metric_names)
        self._log_pareto_front(run, metric_names)

        if baseline is not None:
            self._log_regression_alerts(run, baseline)

        self._wandb_run.finish()
        self._wandb_run = None

    def export_trends(
        self,
        store: Any,
        metric: str,
        point_name: str,
        tags: dict[str, str],
        *,
        n_runs: int | None = None,
    ) -> None:
        """Export metric trends over time to W&B.

        Args:
            store: Store instance with extract_trend method.
            metric: Metric name to track.
            point_name: Point name to match.
            tags: Tags to filter by.
            n_runs: Optional limit on number of trend points.
        """
        series = store.extract_trend(metric, point_name, tags, n_runs=n_runs)

        wb_run = wandb.init(  # type: ignore[union-attr]
            project=self._project,
            entity=self._entity,
            tags=[*self._tags, "trend"],
            mode=self._resolve_wandb_mode(),
        )

        for i, tp in enumerate(series.points):
            wb_run.log(
                {
                    f"trend/{metric}": float(tp.value),
                    "step": i,
                }
            )

        columns = ["run_id", "timestamp", metric]
        data = [[tp.run_id, str(tp.timestamp), float(tp.value)] for tp in series.points]
        table = wandb.Table(columns=columns, data=data)  # type: ignore[union-attr]
        wb_run.log({f"trend/{metric}_table": table})

        wb_run.finish()

    def log_figures(self, figures: dict[str, Any]) -> None:
        """Log matplotlib figures to W&B.

        Args:
            figures: {name: matplotlib_figure} mapping.
        """
        if self._wandb_run is None:
            return
        for name, fig in figures.items():
            self._wandb_run.log({name: wandb.Image(fig)})  # type: ignore[union-attr]

    def log_html_artifacts(self, html: dict[str, str]) -> None:
        """Log HTML strings as W&B artifacts.

        Args:
            html: {name: html_string} mapping.
        """
        if self._wandb_run is None:
            return
        for name, content in html.items():
            self._wandb_run.log({name: wandb.Html(content)})  # type: ignore[union-attr]

    def log_extra_tables(self, tables: dict[str, tuple[list[str], list[list[Any]]]]) -> None:
        """Log additional W&B tables.

        Args:
            tables: {name: (columns, rows)} mapping.
        """
        if self._wandb_run is None:
            return
        for name, (columns, rows) in tables.items():
            table = wandb.Table(columns=columns, data=rows)  # type: ignore[union-attr]
            self._wandb_run.log({name: table})

    def _log_comparison_table(
        self,
        run: Run,
        metric_names: list[str],
        best_values: dict[str, tuple[float, bool]],
    ) -> None:
        """Log a W&B Table comparing all points across metrics."""
        columns = ["point", "scenario", "framework", *metric_names]
        data: list[list[Any]] = []
        for point in run.points:
            fw = point.tags.get("framework", point.name)
            row: list[Any] = [point.name, point.scenario, fw]
            for mn in metric_names:
                if mn in point.metrics:
                    row.append(float(point.metrics[mn].value))
                else:
                    row.append(None)
            data.append(row)
        table = wandb.Table(columns=columns, data=data)  # type: ignore[union-attr]
        self._wandb_run.log({"comparison_table": table})

    def _log_html_comparison(
        self,
        run: Run,
        metric_names: list[str],
        best_values: dict[str, tuple[float, bool]],
    ) -> None:
        """Log an HTML comparison table with bold-best formatting."""
        rows_html: list[str] = []
        for point in run.points:
            fw = point.tags.get("framework", point.name)
            cells = [f"<td>{point.name}</td><td>{fw}</td>"]
            for mn in metric_names:
                if mn in point.metrics:
                    val = point.metrics[mn].value
                    best_val, _ = best_values.get(mn, (val, True))
                    formatted = f"{val:.4f}"
                    if val == best_val:
                        formatted = f"<b>{formatted}</b>"
                    cells.append(f"<td>{formatted}</td>")
                else:
                    cells.append("<td>-</td>")
            rows_html.append(f"<tr>{''.join(cells)}</tr>")

        headers = "".join(f"<th>{h}</th>" for h in ["Point", "Framework", *metric_names])
        html = f"<table><tr>{headers}</tr>{''.join(rows_html)}</table>"
        self._wandb_run.log({"comparison_html": wandb.Html(html)})  # type: ignore[union-attr]

    def _log_rank_tables(self, run: Run, metric_names: list[str]) -> None:
        """Log ranking tables for each metric."""
        from calibrax.analysis.ranking import rank_table

        for mn in metric_names:
            rankings = rank_table(run, mn)
            if not rankings:
                continue
            columns = ["rank", "label", "value", "delta_from_best_%"]
            data = [[r.rank, r.label, float(r.value), float(r.delta_from_best)] for r in rankings]
            table = wandb.Table(columns=columns, data=data)  # type: ignore[union-attr]
            self._wandb_run.log({f"rankings/{mn}": table})

    def _log_regression_alerts(self, run: Run, baseline: Run) -> None:
        """Log regression alerts if any are detected."""
        from calibrax.analysis.regression import detect_regressions

        regressions = detect_regressions(run, baseline)
        if not regressions:
            return
        columns = ["metric", "point", "baseline", "current", "delta_%"]
        data = [
            [
                r.metric,
                r.point_name,
                float(r.baseline_value),
                float(r.current_value),
                float(r.delta_pct),
            ]
            for r in regressions
        ]
        table = wandb.Table(columns=columns, data=data)  # type: ignore[union-attr]
        self._wandb_run.log({"regressions": table})

        for r in regressions:
            wandb.alert(  # type: ignore[union-attr]
                title=f"Regression: {r.metric} on {r.point_name}",
                text=(
                    f"Delta: {r.delta_pct:.1f}% "
                    f"(baseline={r.baseline_value:.4f}, current={r.current_value:.4f})"
                ),
                level=wandb.AlertLevel.WARN,  # type: ignore[union-attr]
            )

    def _log_aggregate_scores(self, run: Run, metric_names: list[str]) -> None:
        """Log aggregate scores across all metrics with equal weights."""
        from calibrax.analysis.ranking import aggregate_score

        weights = {mn: 1.0 for mn in metric_names}
        scores = aggregate_score(run, weights)
        if not scores:
            return
        columns = ["framework", "score"]
        data = [[fw, float(score)] for fw, score in sorted(scores.items(), key=lambda x: -x[1])]
        table = wandb.Table(columns=columns, data=data)  # type: ignore[union-attr]
        self._wandb_run.log({"aggregate_scores": table})

    def _log_pareto_front(self, run: Run, metric_names: list[str]) -> None:
        """Log Pareto front for the first two metrics if available."""
        if len(metric_names) < 2:
            return
        from calibrax.analysis.pareto import pareto_front

        front = pareto_front(
            list(run.points),
            metric_names[0],
            metric_names[1],
            metric_defs=run.metric_defs,
        )
        if not front:
            return
        columns = ["point", metric_names[0], metric_names[1]]
        data = [
            [
                p.name,
                float(p.metrics[metric_names[0]].value),
                float(p.metrics[metric_names[1]].value),
            ]
            for p in front
            if metric_names[0] in p.metrics and metric_names[1] in p.metrics
        ]
        table = wandb.Table(columns=columns, data=data)  # type: ignore[union-attr]
        self._wandb_run.log({"pareto_front": table})
