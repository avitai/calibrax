# calibrax.metrics.learning

Tier 3 metric learning losses -- differentiable distance functions that
learn embedding spaces via backpropagation. All losses return JAX arrays
compatible with `jax.grad` for gradient flow. Includes contrastive,
triplet margin, NT-Xent, ArcFace, CosFace, proxy-NCA, and proxy-anchor
losses, plus hard-negative and semi-hard mining strategies.

::: calibrax.metrics.learning
    options:
      show_source: false
      show_root_heading: false
      members_order: source
      docstring_style: google
      show_signature_annotations: true
