"""Tests for calibrax.exporters.wandb module."""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from calibrax.core.models import (
    Metric,
    MetricDef,
    MetricDirection,
    Point,
    Run,
)
from tests.factories import (
    make_dual_framework_run,
    make_throughput_latency_defs,
    make_throughput_only_run,
)


def _make_run(
    metric_defs: dict[str, MetricDef] | None = None,
) -> Run:
    """Helper to create a benchmark run for testing."""
    return make_dual_framework_run(
        metric_defs=metric_defs or make_throughput_latency_defs(),
        commit="abc123",
        branch="main",
    )


def _make_baseline() -> Run:
    """Helper to create a baseline run for regression testing."""
    defs = {
        "throughput": MetricDef(
            name="throughput",
            unit="ops/s",
            direction=MetricDirection.HIGHER,
        )
    }
    return make_throughput_only_run(
        throughput=250.0,
        metric_defs=defs,
    )


@pytest.fixture()
def mock_wandb():
    """Mock wandb module for testing without real W&B connection."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}):
        import wandb as mock_wb

        mock_run = MagicMock()
        mock_run.url = "https://wandb.ai/test/run/123"
        mock_wb.init.return_value = mock_run
        mock_wb.Table = MagicMock()
        mock_wb.Html = MagicMock()
        mock_wb.Image = MagicMock()
        mock_wb.AlertLevel = MagicMock()  # pyright: ignore[reportAttributeAccessIssue]
        mock_wb.AlertLevel.WARN = "WARN"  # pyright: ignore[reportAttributeAccessIssue]
        mock_wb.alert = MagicMock()  # pyright: ignore[reportAttributeAccessIssue]
        yield mock_wb


class TestWandBExporter:
    """Tests for WandBExporter."""

    def test_import_error_without_wandb(self) -> None:
        """Should raise ImportError if wandb is not available."""
        with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", False):
            from calibrax.exporters.wandb import WandBExporter

            with pytest.raises(ImportError, match="wandb is required"):
                WandBExporter(project="test")

    def test_export_run_returns_url(self, mock_wandb: MagicMock) -> None:
        """export_run should return a W&B URL."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test-project")
                url = exporter.export_run(_make_run())
        assert url == "https://wandb.ai/test/run/123"

    def test_export_run_calls_wandb_init(self, mock_wandb: MagicMock) -> None:
        """export_run should initialize a W&B run."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="my-proj", entity="my-team")
                exporter.export_run(_make_run())
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["project"] == "my-proj"
        assert call_kwargs["entity"] == "my-team"

    def test_export_run_logs_metrics(self, mock_wandb: MagicMock) -> None:
        """export_run should log metrics for each point."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.export_run(_make_run())
        mock_run = mock_wandb.init.return_value
        assert mock_run.log.call_count > 0

    def test_export_run_finish_true(self, mock_wandb: MagicMock) -> None:
        """export_run with finish=True should call finish()."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.export_run(_make_run(), finish=True)
        mock_wandb.init.return_value.finish.assert_called_once()

    def test_export_run_finish_false(self, mock_wandb: MagicMock) -> None:
        """export_run with finish=False should not call finish()."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.export_run(_make_run(), finish=False)
        mock_wandb.init.return_value.finish.assert_not_called()

    def test_export_analysis(self, mock_wandb: MagicMock) -> None:
        """export_analysis should log rankings and scores."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.export_analysis(_make_run())
        mock_wandb.init.return_value.log.assert_called()

    def test_export_analysis_with_baseline(self, mock_wandb: MagicMock) -> None:
        """export_analysis with baseline should log regressions."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.export_analysis(_make_run(), baseline=_make_baseline())
        mock_wandb.alert.assert_called()

    def test_export_trends(self, mock_wandb: MagicMock) -> None:
        """export_trends should log trend data."""
        from calibrax.core.models import TrendPoint, TrendSeries
        from calibrax.exporters.wandb import WandBExporter

        mock_store = MagicMock()
        mock_store.extract_trend.return_value = TrendSeries(
            metric="throughput",
            point_name="bench1",
            tags={"framework": "jax"},
            points=(
                TrendPoint(
                    run_id="r1",
                    timestamp=__import__("datetime").datetime(2024, 1, 1),
                    value=100.0,
                ),
                TrendPoint(
                    run_id="r2",
                    timestamp=__import__("datetime").datetime(2024, 1, 2),
                    value=110.0,
                ),
            ),
        )

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.export_trends(mock_store, "throughput", "bench1", {"framework": "jax"})
        mock_wandb.init.return_value.log.assert_called()
        mock_wandb.init.return_value.finish.assert_called_once()

    def test_check_auth_with_api_key(
        self, mock_wandb: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """check_auth should return True when WANDB_API_KEY is set."""
        from calibrax.exporters.wandb import WandBExporter

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                assert exporter.check_auth() is True

    def test_check_auth_offline_mode(
        self, mock_wandb: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """check_auth should return True in offline mode."""
        from calibrax.exporters.wandb import WandBExporter

        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_MODE", "offline")
        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                assert exporter.check_auth() is True

    def test_log_figures_no_run(self, mock_wandb: MagicMock) -> None:
        """log_figures should no-op without active W&B run."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.log_figures({"test": MagicMock()})

    def test_log_html_artifacts(self, mock_wandb: MagicMock) -> None:
        """log_html_artifacts should log HTML content."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.export_run(_make_run(), finish=False)
                exporter.log_html_artifacts({"report": "<p>test</p>"})

    def test_log_extra_tables(self, mock_wandb: MagicMock) -> None:
        """log_extra_tables should log W&B Table objects."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.export_run(_make_run(), finish=False)
                exporter.log_extra_tables({"data": (["col1"], [["val1"]])})


class TestDiscoverMetricNames:
    """Tests for _discover_metric_names helper."""

    def test_discovers_all_metrics(self) -> None:
        """Should find all unique metric names."""
        from calibrax.exporters.wandb import _discover_metric_names

        run = _make_run()
        names = _discover_metric_names(run)
        assert set(names) == {"throughput", "latency"}

    def test_returns_sorted(self) -> None:
        """Metric names should be sorted."""
        from calibrax.exporters.wandb import _discover_metric_names

        run = _make_run()
        names = _discover_metric_names(run)
        assert names == sorted(names)


class TestFindBestValues:
    """Tests for _find_best_values helper."""

    def test_higher_is_better(self) -> None:
        """Should find max for higher-is-better metrics."""
        from calibrax.exporters.wandb import _find_best_values

        run = _make_run()
        best = _find_best_values(run, ["throughput"])
        assert best["throughput"][0] == 200.0

    def test_lower_is_better(self) -> None:
        """Should find min for lower-is-better metrics."""
        from calibrax.exporters.wandb import _find_best_values

        run = _make_run()
        best = _find_best_values(run, ["latency"])
        assert best["latency"][0] == 5.0

    def test_missing_metric_is_ignored(self) -> None:
        """Metric names with no values should be skipped."""
        from calibrax.exporters.wandb import _find_best_values

        run = _make_run()
        best = _find_best_values(run, ["throughput", "nonexistent"])
        assert "throughput" in best
        assert "nonexistent" not in best


class TestWandBExporterAdditional:
    """Additional branch-coverage tests for WandB exporter."""

    def test_export_analysis_reuses_existing_active_run(self, mock_wandb: MagicMock) -> None:
        """export_analysis should reuse existing run instead of reinitializing."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.export_run(_make_run(), finish=False)
                mock_wandb.init.reset_mock()
                exporter.export_analysis(_make_run())

        mock_wandb.init.assert_not_called()

    def test_check_auth_returns_false_when_api_lookup_fails(
        self,
        mock_wandb: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """check_auth should return False if wandb API lookup raises."""
        from calibrax.exporters.wandb import WandBExporter

        class _BadAPI:
            @property
            def api_key(self) -> str | None:
                raise RuntimeError("auth lookup failed")

        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        mock_wandb.api = _BadAPI()

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                assert exporter.check_auth() is False

    def test_resolve_wandb_mode_valid_and_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_resolve_wandb_mode should pass through only allowed values."""
        from calibrax.exporters.wandb import WandBExporter

        monkeypatch.setenv("WANDB_MODE", "offline")
        assert WandBExporter._resolve_wandb_mode() == "offline"

        monkeypatch.setenv("WANDB_MODE", "invalid")
        assert WandBExporter._resolve_wandb_mode() is None

    def test_log_figures_active_run_logs_images(self, mock_wandb: MagicMock) -> None:
        """log_figures should wrap figures as wandb.Image for active run."""
        from calibrax.exporters.wandb import WandBExporter

        fig = MagicMock()
        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.export_run(_make_run(), finish=False)
                exporter.log_figures({"plot": fig})

        mock_wandb.Image.assert_called_once_with(fig)

    def test_log_html_artifacts_no_run(self, mock_wandb: MagicMock) -> None:
        """log_html_artifacts should no-op when no run is active."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.log_html_artifacts({"report": "<p>test</p>"})

        mock_wandb.Html.assert_not_called()

    def test_log_extra_tables_no_run(self, mock_wandb: MagicMock) -> None:
        """log_extra_tables should no-op when no run is active."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.log_extra_tables({"data": (["col1"], [["val1"]])})

        mock_wandb.Table.assert_not_called()

    def test_export_run_handles_missing_metrics_in_rows_and_html(
        self,
        mock_wandb: MagicMock,
    ) -> None:
        """Comparison outputs should include placeholders for missing metrics."""
        from calibrax.exporters.wandb import WandBExporter

        defs = {
            "throughput": MetricDef(
                name="throughput",
                unit="ops/s",
                direction=MetricDirection.HIGHER,
            ),
            "latency": MetricDef(
                name="latency",
                unit="ms",
                direction=MetricDirection.LOWER,
            ),
        }
        run = Run(
            id="mixed-metrics",
            points=(
                Point(
                    name="p1",
                    scenario="s",
                    tags={"framework": "jax"},
                    metrics={"throughput": Metric(value=100.0)},
                ),
                Point(
                    name="p2",
                    scenario="s",
                    tags={"framework": "torch"},
                    metrics={"latency": Metric(value=5.0)},
                ),
            ),
            metric_defs=defs,
        )

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter.export_run(run)

        table_calls = mock_wandb.Table.call_args_list
        assert table_calls
        comparison_table = table_calls[0]
        rows = comparison_table.kwargs["data"]
        assert any(cell is None for row in rows for cell in row)

        html_arg = mock_wandb.Html.call_args[0][0]
        assert "<td>-</td>" in html_arg

    def test_rank_tables_skip_empty_rankings(self, mock_wandb: MagicMock) -> None:
        """_log_rank_tables should skip metrics with no ranking entries."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter._wandb_run = MagicMock()
                with patch("calibrax.analysis.ranking.rank_table", return_value=[]):
                    exporter._log_rank_tables(_make_run(), ["throughput"])

        mock_wandb.Table.assert_not_called()

    def test_regression_alerts_no_regressions(self, mock_wandb: MagicMock) -> None:
        """_log_regression_alerts should no-op when no regressions are found."""
        from calibrax.exporters.wandb import WandBExporter

        run = _make_run()
        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter._wandb_run = MagicMock()
                exporter._log_regression_alerts(run, run)

        mock_wandb.alert.assert_not_called()

    def test_aggregate_scores_no_scores(self, mock_wandb: MagicMock) -> None:
        """_log_aggregate_scores should no-op when aggregate_score returns empty."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter._wandb_run = MagicMock()
                with patch("calibrax.analysis.ranking.aggregate_score", return_value={}):
                    exporter._log_aggregate_scores(_make_run(), ["throughput"])

        mock_wandb.Table.assert_not_called()

    def test_pareto_front_early_returns(self, mock_wandb: MagicMock) -> None:
        """_log_pareto_front should return for too few metrics or empty front."""
        from calibrax.exporters.wandb import WandBExporter

        with patch("calibrax.exporters.wandb.wandb", mock_wandb):
            with patch("calibrax.exporters.wandb.WANDB_AVAILABLE", True):
                exporter = WandBExporter(project="test")
                exporter._wandb_run = MagicMock()
                exporter._log_pareto_front(_make_run(), ["throughput"])
                with patch("calibrax.analysis.pareto.pareto_front", return_value=[]):
                    exporter._log_pareto_front(_make_run(), ["throughput", "latency"])

        mock_wandb.Table.assert_not_called()

    def test_module_import_sets_unavailable_when_wandb_missing(self) -> None:
        """Module import guard should set WANDB_AVAILABLE=False on ImportError."""
        import calibrax.exporters.wandb as wandb_mod

        module_path = Path(wandb_mod.__file__)
        spec = importlib.util.spec_from_file_location("wandb_import_probe", module_path)
        assert spec is not None
        assert spec.loader is not None
        probe_module = importlib.util.module_from_spec(spec)

        real_import = builtins.__import__

        def _import_hook(name: str, *args: object, **kwargs: object) -> object:
            if name == "wandb":
                raise ImportError("missing wandb")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_hook):
            sys.modules[spec.name] = probe_module
            try:
                spec.loader.exec_module(probe_module)
            finally:
                sys.modules.pop(spec.name, None)

        assert probe_module.WANDB_AVAILABLE is False
        assert probe_module.wandb is None
