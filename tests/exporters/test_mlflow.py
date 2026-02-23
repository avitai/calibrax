"""Tests for calibrax.exporters.mlflow module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from calibrax.core.models import Metric, MetricDirection, Point, Regression, Run
from tests.factories import make_matmul_run, make_throughput_only_run


def _make_run(
    run_id: str = "test123",
    throughput: float = 100.0,
    latency: float = 5.0,
) -> Run:
    """Helper to create a benchmark run for testing."""
    return make_matmul_run(
        run_id=run_id,
        throughput=throughput,
        latency=latency,
        commit="abc",
        branch="main",
    )


def _make_baseline() -> Run:
    """Helper to create a baseline run for regression comparison."""
    return make_throughput_only_run(
        throughput=200.0,
        run_id="baseline1",
        point_name="matmul",
        scenario="perf",
        commit="def",
        branch="main",
    )


@pytest.fixture()
def mock_mlflow() -> MagicMock:
    """Create a mock mlflow module with standard API surface."""
    mock = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "mlflow-run-abc"
    mock.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock.start_run.return_value.__exit__ = MagicMock(return_value=False)
    return mock


class TestMLflowExporter:
    """Tests for MLflowExporter."""

    def test_raises_import_error_when_unavailable(self) -> None:
        """Should raise ImportError when mlflow is not installed."""
        with patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", False):
            from calibrax.exporters.mlflow import MLflowExporter

            with pytest.raises(ImportError, match="mlflow is required"):
                MLflowExporter(experiment_name="test")

    def test_init_sets_experiment(self, mock_mlflow: MagicMock) -> None:
        """__init__ should call mlflow.set_experiment."""
        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            MLflowExporter(experiment_name="my-experiment")

        mock_mlflow.set_experiment.assert_called_once_with("my-experiment")

    def test_init_sets_tracking_uri(self, mock_mlflow: MagicMock) -> None:
        """__init__ should call mlflow.set_tracking_uri when provided."""
        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            MLflowExporter(experiment_name="test", tracking_uri="http://localhost:5000")

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_init_no_tracking_uri(self, mock_mlflow: MagicMock) -> None:
        """__init__ should not call set_tracking_uri when None."""
        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            MLflowExporter(experiment_name="test")

        mock_mlflow.set_tracking_uri.assert_not_called()

    def test_export_run_returns_run_id(self, mock_mlflow: MagicMock) -> None:
        """export_run should return the MLflow run ID."""
        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            exporter = MLflowExporter(experiment_name="test")
            result = exporter.export_run(_make_run())

        assert result == "mlflow-run-abc"

    def test_export_run_logs_params(self, mock_mlflow: MagicMock) -> None:
        """export_run should log parameters including run_id and commit."""
        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            exporter = MLflowExporter(experiment_name="test")
            exporter.export_run(_make_run())

        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["run_id"] == "test123"
        assert params["commit"] == "abc"
        assert params["branch"] == "main"
        assert params["num_points"] == "1"

    def test_export_run_logs_metrics(self, mock_mlflow: MagicMock) -> None:
        """export_run should log metric values for each point."""
        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            exporter = MLflowExporter(experiment_name="test")
            exporter.export_run(_make_run())

        assert mock_mlflow.log_metric.call_count >= 2
        logged_keys = [call[0][0] for call in mock_mlflow.log_metric.call_args_list]
        assert any("throughput" in k for k in logged_keys)
        assert any("latency" in k for k in logged_keys)

    def test_export_analysis_without_baseline(self, mock_mlflow: MagicMock) -> None:
        """export_analysis without baseline should log run summary artifact."""
        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            exporter = MLflowExporter(experiment_name="test")
            exporter.export_analysis(_make_run())

        mock_mlflow.log_param.assert_called_once_with("analysis_run_id", "test123")
        mock_mlflow.log_artifact.assert_called_once()
        artifact_args = mock_mlflow.log_artifact.call_args
        assert artifact_args[0][1] == "benchmark_data"

    def test_export_analysis_with_baseline(self, mock_mlflow: MagicMock) -> None:
        """export_analysis with baseline should log regression metrics."""
        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            exporter = MLflowExporter(experiment_name="test")
            exporter.export_analysis(_make_run(throughput=50.0), baseline=_make_baseline())

        # Should log at least the analysis_run_id param and the artifact
        mock_mlflow.log_param.assert_called()
        mock_mlflow.log_artifact.assert_called_once()

    def test_export_run_with_environment(self, mock_mlflow: MagicMock) -> None:
        """export_run should log environment variables as params."""
        run = Run(
            points=(
                Point(
                    name="bench",
                    scenario="s1",
                    tags={"framework": "jax"},
                    metrics={"throughput": Metric(value=100.0)},
                ),
            ),
            id="env_run",
            commit="abc",
            branch="main",
            environment={"machine": "gpu-node-1", "python_version": "3.12"},
        )

        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            exporter = MLflowExporter(experiment_name="test")
            exporter.export_run(run)

        params = mock_mlflow.log_params.call_args[0][0]
        assert params["env_machine"] == "gpu-node-1"
        assert params["env_python_version"] == "3.12"


class TestMLflowExporterAdditional:
    """Additional branch-coverage tests for MLflow exporter."""

    def test_export_run_omits_commit_and_branch_when_missing(self, mock_mlflow: MagicMock) -> None:
        """export_run should not add commit/branch params when values are None."""
        run = Run(
            points=(
                Point(
                    name="bench",
                    scenario="s1",
                    tags={"framework": "jax"},
                    metrics={"throughput": Metric(value=100.0)},
                ),
            ),
            id="no-meta",
            commit=None,
            branch=None,
        )

        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            exporter = MLflowExporter(experiment_name="test")
            exporter.export_run(run)

        params = mock_mlflow.log_params.call_args[0][0]
        assert "commit" not in params
        assert "branch" not in params

    def test_log_regressions_logs_metric_rows_and_count(self, mock_mlflow: MagicMock) -> None:
        """_log_regressions should log each regression and aggregate count."""
        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            exporter = MLflowExporter(experiment_name="test")
            fake_regressions = [
                Regression(
                    metric="throughput",
                    point_name="bench/a",
                    baseline_value=100.0,
                    current_value=80.0,
                    delta_pct=-20.0,
                    direction=MetricDirection.HIGHER,
                )
            ]
            with patch(
                "calibrax.analysis.regression.detect_regressions",
                return_value=fake_regressions,
            ):
                exporter._log_regressions(_make_run(), _make_baseline())

        calls = mock_mlflow.log_metric.call_args_list
        assert any(call.args[0] == "regression_count" and call.args[1] == 1 for call in calls)
        assert any("regression_throughput_bench_a" == call.args[0] for call in calls)

    def test_log_regressions_skips_count_when_empty(self, mock_mlflow: MagicMock) -> None:
        """_log_regressions should not log regression_count when there are none."""
        with (
            patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True),
            patch("calibrax.exporters.mlflow.mlflow", mock_mlflow),
        ):
            from calibrax.exporters.mlflow import MLflowExporter

            exporter = MLflowExporter(experiment_name="test")
            with patch("calibrax.analysis.regression.detect_regressions", return_value=[]):
                exporter._log_regressions(_make_run(), _make_baseline())

        assert mock_mlflow.log_metric.call_count == 0

    def test_module_import_sets_available_when_mlflow_present(self) -> None:
        """Module import guard should set MLFLOW_AVAILABLE=True when mlflow imports."""
        import calibrax.exporters.mlflow as mlflow_mod

        module_path = Path(mlflow_mod.__file__)
        spec = importlib.util.spec_from_file_location("mlflow_import_probe", module_path)
        assert spec is not None
        assert spec.loader is not None
        probe_module = importlib.util.module_from_spec(spec)

        fake_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            sys.modules[spec.name] = probe_module
            try:
                spec.loader.exec_module(probe_module)
            finally:
                sys.modules.pop(spec.name, None)

        assert probe_module.MLFLOW_AVAILABLE is True
        assert probe_module.mlflow is fake_mlflow
