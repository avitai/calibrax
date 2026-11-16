# calibrax.metrics.functional.seismic

Seismic / full-waveform inversion (FWI) metrics: pure-JAX misfits on seismic
gathers and velocity models. Every function returns a scalar and is
``jax.grad``/``jax.jit`` safe, so the whole set is usable inside inversion
loops. Provides L2 data misfits, relative model misfit, shift-tolerant
cross-correlation misfit, velocity-model SSIM misfit, per-trace quadratic
Wasserstein misfit (the exact quantile integral of linearly or softplus-scaled
traces), and differentiable soft-DTW misfit.

::: calibrax.metrics.functional.seismic
    options:
      show_source: false
      show_root_heading: false
      members_order: source
      docstring_style: google
      show_signature_annotations: true
