"""The export command: a stored run to Weights & Biases; needs the ``wandb`` extra."""

from __future__ import annotations

from pathlib import Path

import click

from calibrax.exporters.wandb import WandBExporter
from calibrax.storage.store import Store


@click.command()
@click.option("--data", required=True, type=click.Path(path_type=Path), help="Store directory.")
@click.option("--run", "run_id", default="latest", help="Run ID to export (default: latest).")
@click.option("--project", required=True, help="W&B project name.")
@click.option("--entity", default=None, help="W&B entity (team or user).")
def export(data: Path, run_id: str, project: str, entity: str | None) -> None:
    """Export a run to Weights & Biases."""
    try:
        store = Store(data)
        run = store.load(run_id) if run_id != "latest" else store.latest()
        baseline = store.get_baseline()

        exporter = WandBExporter(project=project, entity=entity)
        url = exporter.export_run(run, finish=False)
        exporter.export_analysis(run, baseline=baseline)

        print(f"Exported run {run.id} to W&B: {url}")
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None
