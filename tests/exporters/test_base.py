"""Tests for calibrax.exporters.base module."""

from __future__ import annotations

import pytest

from calibrax.core.models import Run
from calibrax.exporters.base import Exporter


class ConcreteExporter(Exporter):
    """Minimal concrete exporter for testing the ABC."""

    def export_run(self, run: Run) -> str:
        """Return a fixed URL."""
        return "https://example.com/run"

    def export_analysis(self, run: Run, baseline: Run | None = None) -> None:
        """No-op analysis export."""


class TestExporterABC:
    """Tests for Exporter abstract base class."""

    def test_concrete_subclass_instantiable(self) -> None:
        """Concrete subclass should be instantiable."""
        exporter = ConcreteExporter()
        run = Run(points=())
        assert exporter.export_run(run) == "https://example.com/run"

    def test_abstract_methods_enforced(self) -> None:
        """Cannot instantiate Exporter directly."""
        with pytest.raises(TypeError):
            Exporter()  # type: ignore[abstract]

    def test_export_analysis_accepts_baseline(self) -> None:
        """export_analysis should accept optional baseline."""
        exporter = ConcreteExporter()
        run = Run(points=())
        exporter.export_analysis(run, baseline=run)
