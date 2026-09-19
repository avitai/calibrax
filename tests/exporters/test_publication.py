"""Tests for calibrax.exporters.publication module."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrax.core.models import MetricDef
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
