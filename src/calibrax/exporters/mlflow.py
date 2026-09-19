"""MLflow exporter for benchmark results and analysis.

Exports benchmark runs, comparisons, and regressions to MLflow tracking. This module is the
MLflow integration: it needs the ``mlflow`` extra, and importing it without mlflow raises
``ImportError`` naming the extra. It is not re-exported from ``calibrax.exporters``.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path


try:
    import mlflow
except ImportError as error:
    msg = 'calibrax.exporters.mlflow needs mlflow: uv pip install "calibrax[mlflow]"'
    raise ImportError(msg) from error

from calibrax.analysis.regression import detect_regressions
from calibrax.core.models import Run
from calibrax.exporters.base import Exporter


logger = logging.getLogger(__name__)


class MLflowExporter(Exporter):
    """Export benchmark results and analysis to MLflow.

    Logs metrics, parameters, and artifacts to an MLflow tracking server.
    Each benchmark run becomes an MLflow run within the specified experiment.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
    ) -> None:
        """Initialize the MLflow exporter.

        Args:
            experiment_name: MLflow experiment name.
            tracking_uri: MLflow tracking server URI.
        """
        self._experiment_name = experiment_name
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    def export_run(self, run: Run) -> str:
        """Export a benchmark run to MLflow.

        Logs each metric from each point as an MLflow metric, and logs
        environment/metadata as MLflow parameters.

        Args:
            run: Benchmark run to export.

        Returns:
            MLflow run ID.
        """
        with mlflow.start_run() as mlflow_run:
            # Log parameters
            params: dict[str, str] = {
                "run_id": run.id,
                "num_points": str(len(run.points)),
            }
            if run.commit:
                params["commit"] = run.commit
            if run.branch:
                params["branch"] = run.branch

            for key, value in run.environment.items():
                params[f"env_{key}"] = str(value)[:250]

            mlflow.log_params(params)

            # Log metrics
            for point in run.points:
                fw = point.tags.get("framework", point.name)
                for metric_name, metric in point.metrics.items():
                    mlflow_key = f"{metric_name}_{fw}".replace("/", "_")[:250]
                    mlflow.log_metric(mlflow_key, float(metric.value))

            return mlflow_run.info.run_id

    def export_analysis(self, run: Run, baseline: Run | None = None) -> None:
        """Export analysis artifacts to MLflow.

        Logs regressions as metrics and comparison data as a JSON artifact.

        Args:
            run: Current benchmark run.
            baseline: Optional baseline run for regression detection.
        """
        with mlflow.start_run():
            mlflow.log_param("analysis_run_id", run.id)

            if baseline is not None:
                self._log_regressions(run, baseline)

            # Log run summary as artifact
            with tempfile.TemporaryDirectory() as directory:
                artifact_path = Path(directory) / f"run_{run.id}.json"
                artifact_path.write_text(json.dumps(run.to_dict(), indent=2))
                mlflow.log_artifact(str(artifact_path), "benchmark_data")

    def _log_regressions(self, run: Run, baseline: Run) -> None:
        """Log regression alerts as MLflow metrics.

        Args:
            run: Current benchmark run.
            baseline: Baseline run for comparison.
        """
        regressions = detect_regressions(run, baseline)
        for regression in regressions:
            key = f"regression_{regression.metric}_{regression.point_name}"
            mlflow.log_metric(
                key.replace("/", "_")[:250],
                float(regression.delta_pct),
            )

        if regressions:
            mlflow.log_metric("regression_count", len(regressions))
