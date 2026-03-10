# calibrax.metrics.functional.image

Image quality metrics for comparing generated or reconstructed images
against references. Tier 0 functions include PSNR, SSIM, MS-SSIM, and
Vendi Score -- all pure JAX with no external dependencies.

::: calibrax.metrics.functional.image
    options:
      show_source: false
      show_root_heading: false
      members_order: source
      docstring_style: google
      show_signature_annotations: true

## Plugin Metrics (Tier 1-2)

!!! warning "Optional Dependency"

    FID, Inception Score, and LPIPS require pretrained backbones:
    `uv pip install "calibrax[image]"`

    Import directly from the plugin module:

    ```python
    from calibrax.metrics.plugins.image import FIDMetric, InceptionScoreMetric, LPIPSMetric
    ```

See [Stateful Metrics](stateful.md) for the base class API.
