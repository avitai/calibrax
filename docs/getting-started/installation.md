# Installation

## Requirements

- Python 3.11 or higher
- JAX 0.4.0 or higher

## Install Calibrax

=== "uv (Recommended)"

    ```bash
    uv pip install calibrax
    ```

=== "From Source"

    ```bash
    git clone https://github.com/avitai/calibrax.git
    cd calibrax
    uv venv
    uv pip install -e ".[dev,test,docs]"
    uv run pre-commit install
    ```

## Optional Extras

Calibrax uses optional dependencies to keep the base install lightweight.
Install only the extras relevant to your workflow:

| Extra | Installs | Use Case |
|-------|----------|----------|
| `stats` | scipy | Significance tests (t-test, Mann-Whitney, Wilcoxon) |
| `wandb` | wandb | Weights & Biases export |
| `mlflow` | mlflow | MLflow experiment tracking export |
| `publication` | matplotlib | Publication-ready plots and figures |
| `image` | *(none — works with pre-extracted features)* | Image quality metrics (FID, Inception Score, LPIPS) |
| `text` | *(none — works with pre-computed embeddings)* | Text quality metrics (BERTScore) |
| `scientific` | *(none — pure JAX)* | Scientific domain metrics (molecular, protein) |
| `codecarbon` | codecarbon | Carbon emissions and energy tracking |
| `changepoint` | ruptures | Change point detection in benchmark trends |
| `gpu` | CUDA libraries | GPU memory profiling and energy monitoring |
| `metal` | Metal backend | Apple Silicon GPU acceleration |
| `all` | Everything above | Full feature set |

```bash
# Install with statistical testing support
uv pip install "calibrax[stats]"

# Install with W&B and publication output
uv pip install "calibrax[wandb,publication]"

# Install everything
uv pip install "calibrax[all]"
```

## Platform-Specific Setup

=== "Linux (CPU)"

    ```bash
    uv pip install calibrax
    ```

=== "Linux (CUDA GPU)"

    ```bash
    # Ensure NVIDIA drivers are installed, then:
    uv pip install "calibrax[cuda12]"
    ```

=== "macOS (Apple Silicon)"

    ```bash
    uv pip install "calibrax[metal]"
    ```

!!! tip "Verify JAX Backend"

    After installation, verify that JAX detects the expected devices:

    ```python
    import jax
    print(jax.devices())
    # CPU: [CpuDevice(id=0)]
    # GPU: [cuda(id=0)]
    # Metal: [METAL:0]
    ```

## Verify Installation

```python
import calibrax
print(calibrax.__version__)
```

??? warning "Troubleshooting"

    **JAX not found or wrong version**

    Calibrax requires JAX 0.4.0+. Upgrade with:

    ```bash
    uv pip install --upgrade jax jaxlib
    ```

    **scipy import errors when running significance tests**

    Install the `stats` extra:

    ```bash
    uv pip install "calibrax[stats]"
    ```

    **matplotlib import errors when generating plots**

    Install the `publication` extra:

    ```bash
    uv pip install "calibrax[publication]"
    ```

    **wandb authentication errors**

    Run `wandb login` before using `WandBExporter`, or set the
    `WANDB_API_KEY` environment variable.
