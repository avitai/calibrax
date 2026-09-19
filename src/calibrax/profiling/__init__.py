"""Profiling: timing, resources, GPU, energy, FLOPs, hardware, roofline, compilation, complexity.

Each export loads its module on first use (scientific-python SPEC 1), so importing a light
module such as ``calibrax.profiling.resources`` or ``calibrax.profiling.timing_records`` does
not load JAX. ``__init__.pyi`` lists the exports and is what type checkers read.
"""

import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)
