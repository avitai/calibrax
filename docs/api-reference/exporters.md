# calibrax.exporters

Export benchmark results to external systems and publication formats.

## Base Exporter

The `Exporter` ABC defines the interface for all exporters.

::: calibrax.exporters.base

## W&B Exporter

!!! warning "Import Path"

    `WandBExporter` is not re-exported from `calibrax.exporters` to avoid
    loading wandb at import time. Import directly:

    ```python
    from calibrax.exporters.wandb import WandBExporter
    ```

!!! warning "Optional Dependency"

    Requires wandb: `uv pip install "calibrax[wandb]"`

::: calibrax.exporters.wandb

## MLflow Exporter

!!! warning "Import Path"

    `MLflowExporter` is not re-exported from `calibrax.exporters` to avoid
    loading mlflow at import time. Import directly:

    ```python
    from calibrax.exporters.mlflow import MLflowExporter
    ```

!!! warning "Optional Dependency"

    Requires mlflow: `uv pip install "calibrax[mlflow]"`

::: calibrax.exporters.mlflow

## Publication Tables

::: calibrax.exporters.publication

## Plots

!!! warning "Optional Dependency"

    Requires matplotlib: `uv pip install "calibrax[publication]"`

::: calibrax.exporters.plots
