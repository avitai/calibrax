"""Abstract base class for benchmark result exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from calibrax.core.models import Run


class Exporter(ABC):
    """Base class for exporting benchmark results to external systems.

    Subclasses implement export_run for raw data and export_analysis
    for computed analytics (regressions, rankings, etc.).
    """

    @abstractmethod
    def export_run(self, run: Run) -> str:
        """Export a benchmark run to an external system.

        Args:
            run: The benchmark run to export.

        Returns:
            URL or identifier of the exported artifact.
        """

    @abstractmethod
    def export_analysis(self, run: Run, baseline: Run | None = None) -> None:
        """Export analysis results (rankings, regressions, etc.).

        Args:
            run: Current benchmark run.
            baseline: Optional baseline run for comparison.
        """
