"""MLflow exporter for benchmark results and analysis.

Exports benchmark runs, comparisons, and regressions to MLflow tracking.
Requires the optional ``mlflow`` dependency (``uv pip install "calibrax[mlflow]"``).

Note: NOT re-exported from ``calibrax.exporters.__init__`` to avoid
import-time MLflow loading. Import directly::

    from calibrax.exporters.mlflow import MLflowExporter
"""

from __future__ import annotations

import json
import logging
import tempfile

from calibrax.core.models import Run
from calibrax.exporters.base import Exporter


try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


class MLflowExporter(Exporter):
    """Export benchmark results and analysis to MLflow.

    Logs metrics, parameters, and artifacts to an MLflow tracking server.
    Each benchmark run becomes an MLflow run within the specified experiment.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking server URI. Uses default if None.

    Raises:
        ImportError: If mlflow is not installed.
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

        Raises:
            ImportError: If mlflow is not installed.
        """
        if not MLFLOW_AVAILABLE:
            msg = 'mlflow is required for MLflowExporter: uv pip install "calibrax[mlflow]"'
            raise ImportError(msg)

        self._experiment_name = experiment_name
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)  # type: ignore[union-attr]

        mlflow.set_experiment(experiment_name)  # type: ignore[union-attr]

    def export_run(self, run: Run) -> str:
        """Export a benchmark run to MLflow.

        Logs each metric from each point as an MLflow metric, and logs
        environment/metadata as MLflow parameters.

        Args:
            run: Benchmark run to export.

        Returns:
            MLflow run ID.
        """
        with mlflow.start_run() as mlflow_run:  # type: ignore[union-attr]
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

            mlflow.log_params(params)  # type: ignore[union-attr]

            # Log metrics
            for point in run.points:
                fw = point.tags.get("framework", point.name)
                for metric_name, metric in point.metrics.items():
                    mlflow_key = f"{metric_name}_{fw}".replace("/", "_")[:250]
                    mlflow.log_metric(mlflow_key, float(metric.value))  # type: ignore[union-attr]

            return mlflow_run.info.run_id  # type: ignore[return-value]

    def export_analysis(self, run: Run, baseline: Run | None = None) -> None:
        """Export analysis artifacts to MLflow.

        Logs regressions as metrics and comparison data as a JSON artifact.

        Args:
            run: Current benchmark run.
            baseline: Optional baseline run for regression detection.
        """
        with mlflow.start_run():  # type: ignore[union-attr]
            mlflow.log_param("analysis_run_id", run.id)  # type: ignore[union-attr]

            if baseline is not None:
                self._log_regressions(run, baseline)

            # Log run summary as artifact
            summary = run.to_dict()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(summary, f, indent=2, default=str)
                artifact_path = f.name

            mlflow.log_artifact(artifact_path, "benchmark_data")  # type: ignore[union-attr]

    def _log_regressions(self, run: Run, baseline: Run) -> None:
        """Log regression alerts as MLflow metrics.

        Args:
            run: Current benchmark run.
            baseline: Baseline run for comparison.
        """
        from calibrax.analysis.regression import detect_regressions

        regressions = detect_regressions(run, baseline)
        for regression in regressions:
            key = f"regression_{regression.metric}_{regression.point_name}"
            mlflow.log_metric(  # type: ignore[union-attr]
                key.replace("/", "_")[:250],
                float(regression.delta_pct),
            )

        if regressions:
            mlflow.log_metric("regression_count", len(regressions))  # type: ignore[union-attr]
