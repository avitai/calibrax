"""Core data models, protocols, and abstractions.

Each export loads its module on first use (scientific-python SPEC 1), so importing a light
module such as ``calibrax.core.models`` does not load the JAX-dependent adapters and
protocols. ``__init__.pyi`` lists the exports and is what type checkers read.
"""

import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)
