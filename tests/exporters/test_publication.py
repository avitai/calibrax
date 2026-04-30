"""Tests for calibrax.exporters.publication module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import calibrax.exporters.publication as publication
from calibrax.core.models import (
    MetricDef,
    TrendPoint,
    TrendSeries,
)
from calibrax.exporters.publication import PublicationGenerator
from tests.factories import make_dual_framework_run, make_throughput_latency_defs


def _make_run(
    metric_defs: dict[str, MetricDef] | None = None,
):
    """Helper to create a benchmark run."""
    return make_dual_framework_run(
        metric_defs=metric_defs or make_throughput_latency_defs(),
    )


class TestPublicationGenerator:
    """Tests for PublicationGenerator."""

    def test_output_dir_created(self, tmp_path: Path) -> None:
        """Constructor should create output directory."""
        out = tmp_path / "publications"
        PublicationGenerator(out)
        assert out.exists()

    def test_generate_comparison_plot(self, tmp_path: Path) -> None:
        """Should generate a comparison plot PNG."""
        gen = PublicationGenerator(tmp_path)
        result = gen.generate_comparison_plot(_make_run())
        if result is not None:
            assert result.exists()
            assert result.suffix == ".png"

    def test_generate_comparison_plot_pdf(self, tmp_path: Path) -> None:
        """Should generate a PDF comparison plot."""
        gen = PublicationGenerator(tmp_path)
        result = gen.generate_comparison_plot(_make_run(), output_format="pdf")
        if result is not None:
            assert result.suffix == ".pdf"

    def test_generate_comparison_plot_subset(self, tmp_path: Path) -> None:
        """Should plot only requested metrics."""
        gen = PublicationGenerator(tmp_path)
        result = gen.generate_comparison_plot(_make_run(), metrics=["throughput"])
        if result is not None:
            assert result.exists()

    def test_generate_scaling_plot(self, tmp_path: Path) -> None:
        """Should generate a scaling plot."""
        gen = PublicationGenerator(tmp_path)
        sizes = [10, 100, 1000, 10000]
        values = [1.0, 8.0, 60.0, 500.0]
        result = gen.generate_scaling_plot(sizes, values, metric_name="throughput")
        if result is not None:
            assert result.exists()

    def test_generate_convergence_plot(self, tmp_path: Path) -> None:
        """Should generate a convergence plot."""
        gen = PublicationGenerator(tmp_path)
        series = TrendSeries(
            metric="loss",
            point_name="train",
            points=tuple(
                TrendPoint(
                    run_id=f"r{i}",
                    timestamp=datetime(2024, 1, i + 1),
                    value=1.0 / (i + 1),
                )
                for i in range(10)
            ),
        )
        result = gen.generate_convergence_plot(series)
        if result is not None:
            assert result.exists()

    def test_generate_convergence_plot_with_ci(self, tmp_path: Path) -> None:
        """Should handle confidence intervals in convergence plot."""
        gen = PublicationGenerator(tmp_path)
        series = TrendSeries(
            metric="loss",
            point_name="train",
            points=tuple(
                TrendPoint(
                    run_id=f"r{i}",
                    timestamp=datetime(2024, 1, i + 1),
                    value=1.0 / (i + 1),
                    lower=0.8 / (i + 1),
                    upper=1.2 / (i + 1),
                )
                for i in range(5)
            ),
        )
        result = gen.generate_convergence_plot(series)
        if result is not None:
            assert result.exists()

    def test_generate_convergence_plot_empty(self, tmp_path: Path) -> None:
        """Empty series should return None."""
        gen = PublicationGenerator(tmp_path)
        series = TrendSeries(metric="loss", point_name="train")
        result = gen.generate_convergence_plot(series)
        assert result is None

    def test_plot_metric_values(self, tmp_path: Path) -> None:
        """Scalar metric values should be plottable."""
        gen = PublicationGenerator(tmp_path)
        result = gen.plot_metric_values(
            {"fid": 12.5, "inception_score": 4.2},
            title="Image Metrics",
            filename="image_metrics",
        )
        assert result is not None
        assert result.exists()
        assert result.name == "image_metrics.png"

    def test_plot_metric_values_empty_raises(self, tmp_path: Path) -> None:
        """Empty scalar metric values should fail clearly."""
        gen = PublicationGenerator(tmp_path)
        with pytest.raises(ValueError, match="at least one"):
            gen.plot_metric_values({}, title="Empty", filename="empty")

    def test_plot_metric_values_without_matplotlib_returns_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unavailable matplotlib should skip scalar plots like other plots."""
        monkeypatch.setattr(publication, "MATPLOTLIB_AVAILABLE", False)
        gen = PublicationGenerator(tmp_path)
        result = gen.plot_metric_values({"fid": 12.5}, title="Image Metrics", filename="fid")
        assert result is None


class TestTableGeneration:
    """Tests for table generation (no matplotlib needed)."""

    def test_latex_table(self, tmp_path: Path) -> None:
        """Should generate a LaTeX table."""
        gen = PublicationGenerator(tmp_path)
        path = gen.generate_table(_make_run(), output_format="latex")
        assert path.suffix == ".tex"
        content = path.read_text()
        assert r"\begin{tabular}" in content
        assert r"\textbf{" in content  # Bold best value

    def test_html_table(self, tmp_path: Path) -> None:
        """Should generate an HTML table."""
        gen = PublicationGenerator(tmp_path)
        path = gen.generate_table(_make_run(), output_format="html")
        assert path.suffix == ".html"
        content = path.read_text()
        assert "<table>" in content
        assert "<b>" in content  # Bold best value

    def test_csv_table(self, tmp_path: Path) -> None:
        """Should generate a CSV table."""
        gen = PublicationGenerator(tmp_path)
        path = gen.generate_table(_make_run(), output_format="csv")
        assert path.suffix == ".csv"
        content = path.read_text()
        assert "Framework" in content

    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        """Unknown format should raise ValueError."""
        gen = PublicationGenerator(tmp_path)
        with pytest.raises(ValueError, match="Unknown output format"):
            gen.generate_table(_make_run(), output_format="xml")

    def test_latex_bold_best_higher_is_better(self, tmp_path: Path) -> None:
        """LaTeX table should bold the best (highest) throughput."""
        gen = PublicationGenerator(tmp_path)
        path = gen.generate_table(_make_run(), metrics=["throughput"], output_format="latex")
        content = path.read_text()
        assert r"\textbf{200.0000}" in content

    def test_latex_bold_best_lower_is_better(self, tmp_path: Path) -> None:
        """LaTeX table should bold the best (lowest) latency."""
        gen = PublicationGenerator(tmp_path)
        path = gen.generate_table(_make_run(), metrics=["latency"], output_format="latex")
        content = path.read_text()
        assert r"\textbf{5.0000}" in content

    def test_metric_subset(self, tmp_path: Path) -> None:
        """Should only include requested metrics."""
        gen = PublicationGenerator(tmp_path)
        path = gen.generate_table(_make_run(), metrics=["throughput"], output_format="csv")
        content = path.read_text()
        assert "throughput" in content
        assert "latency" not in content
