# calibrax.metrics.functional.text

Text evaluation metrics for translation, summarization, and generation.
Tier 0 functions include BLEU, ROUGE-N, ROUGE-L, perplexity, and
distinct-N -- all implemented in pure Python/JAX without external NLP
libraries.

::: calibrax.metrics.functional.text
    options:
      show_source: false
      show_root_heading: false
      members_order: source
      docstring_style: google
      show_signature_annotations: true

## Plugin Metrics (Tier 1)

!!! warning "Optional Dependency"

    BERTScore requires pretrained BERT embeddings:
    `uv pip install "calibrax[text]"`

    Import directly from the plugin module:

    ```python
    from calibrax.metrics.plugins.text import BERTScoreMetric
    ```

See [Stateful Metrics](stateful.md) for the base class API.
