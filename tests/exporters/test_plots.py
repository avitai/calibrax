"""Publication plots: calibrax.exporters.plots, the module that needs matplotlib."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from substrax.testing import run_python

from calibrax.core.models import MetricDef, Run, TrendPoint, TrendSeries
from calibrax.exporters.plots import PlotGenerator
from tests.factories import make_dual_framework_run, make_throughput_latency_defs


def _make_run(metric_defs: dict[str, MetricDef] | None = None) -> Run:
    return make_dual_framework_run(metric_defs=metric_defs or make_throughput_latency_defs())


def _series(points: int, *, with_bounds: bool = False) -> TrendSeries:
    return TrendSeries(
        metric="loss",
        point_name="train",
        points=tuple(
            TrendPoint(
                run_id=f"r{i}",
                timestamp=datetime(2024, 1, i + 1),
                value=1.0 / (i + 1),
                lower=0.8 / (i + 1) if with_bounds else None,
                upper=1.2 / (i + 1) if with_bounds else None,
            )
            for i in range(points)
        ),
    )


class TestPlotGenerator:
    def test_output_dir_created(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "plots"
        PlotGenerator(out)
        assert out.is_dir()

    @pytest.mark.parametrize("output_format", ["png", "pdf"])
    def test_comparison_plot(self, tmp_path: Path, output_format: str) -> None:
        path = PlotGenerator(tmp_path).comparison_plot(_make_run(), output_format=output_format)
        assert path == tmp_path / f"comparison.{output_format}"
        assert path.stat().st_size > 0

    def test_comparison_plot_of_a_metric_subset(self, tmp_path: Path) -> None:
        path = PlotGenerator(tmp_path).comparison_plot(_make_run(), metrics=["throughput"])
        assert path.exists()

    def test_comparison_plot_of_a_run_without_points_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no framework"):
            PlotGenerator(tmp_path).comparison_plot(Run(points=()))

    def test_scaling_plot(self, tmp_path: Path) -> None:
        path = PlotGenerator(tmp_path).scaling_plot(
            [10, 100, 1000, 10000], [1.0, 8.0, 60.0, 500.0], metric_name="throughput"
        )
        assert path == tmp_path / "scaling_throughput.png"
        assert path.stat().st_size > 0

    @pytest.mark.parametrize("with_bounds", [False, True])
    def test_convergence_plot(self, tmp_path: Path, with_bounds: bool) -> None:
        path = PlotGenerator(tmp_path).convergence_plot(_series(6, with_bounds=with_bounds))
        assert path == tmp_path / "convergence_loss.png"
        assert path.stat().st_size > 0

    def test_convergence_plot_of_an_empty_series_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no points"):
            PlotGenerator(tmp_path).convergence_plot(_series(0))

    def test_metric_values_plot(self, tmp_path: Path) -> None:
        path = PlotGenerator(tmp_path).metric_values_plot(
            {"fid": 12.5, "inception_score": 4.2}, title="Image Metrics", filename="image metrics"
        )
        assert path == tmp_path / "image_metrics.png"
        assert path.stat().st_size > 0

    def test_metric_values_plot_of_nothing_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one"):
            PlotGenerator(tmp_path).metric_values_plot({}, title="Empty", filename="empty")


def test_importing_without_matplotlib_names_the_extra() -> None:
    result = run_python(
        "import sys; sys.modules['matplotlib'] = None\n"
        "try:\n"
        "    import calibrax.exporters.plots\n"
        "except ImportError as error:\n"
        "    print(error)\n",
        timeout=120,
    )

    assert "calibrax[publication]" in result.stdout


def test_tables_import_without_matplotlib() -> None:
    result = run_python(
        "import sys; sys.modules['matplotlib'] = None\n"
        "import calibrax.exporters, calibrax.metrics\n"
        "print('ok')\n",
        timeout=120,
    )

    assert result.stdout.strip() == "ok"
