# calibrax.metrics.stateful

Base classes for stateful metrics (Tier 1 and Tier 2). `FrozenBackboneMetric`
provides the accumulate-then-compute pattern for metrics requiring a
pre-trained feature extractor (e.g., FID, BERTScore). `LearnedMetric`
extends it with trainable calibration weights (e.g., LPIPS).

::: calibrax.metrics.stateful
    options:
      show_source: false
      show_root_heading: false
      members_order: source
      docstring_style: google
      show_signature_annotations: true

!!! note "Plugin Implementations"

    Concrete implementations live in `calibrax.metrics.plugins`:

    - **Image** (FID, InceptionScore, LPIPS): `uv pip install "calibrax[image]"`
    - **Text** (BERTScore): `uv pip install "calibrax[text]"`

    See [Image Metrics](image.md) and [Text Metrics](text.md) for details.
