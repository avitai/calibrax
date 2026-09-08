# calibrax.profiling.flops

FLOP counting through XLA's cost analysis. `FlopsCounter.count()` lowers a JAX
function from its example arguments' shapes and reports the FLOPs and
transcendentals XLA estimates for it.

::: calibrax.profiling.flops
    options:
      show_root_heading: false
