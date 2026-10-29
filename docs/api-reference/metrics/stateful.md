# calibrax.metrics.stateful

Device metric accumulation and base classes for stateful metrics (Tier 1 and Tier 2).
`MeanAccumulator` composes native Flax NNX averages for scalar batch means.
`FrozenBackboneMetric`
provides the accumulate-then-compute pattern for metrics requiring a
pre-trained feature extractor (e.g., FID, BERTScore). `LearnedMetric`
extends it with trainable calibration weights (e.g., LPIPS).

`MeanAccumulator.update(metrics, records)` weights each named mean by its scalar
integer record count. A short last batch contributes only its actual records;
an omitted metric does not advance its denominator. Zero-record batches contribute
nothing, including when their mean is undefined. An empty window computes NaN.
State and computed means stay on device; converting the result to Python or NumPy
is the caller's explicit synchronization boundary.

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
