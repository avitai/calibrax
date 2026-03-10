"""Domain-specific metric plugins requiring optional external dependencies.

Each plugin module is guarded with lazy import checks.
Install the corresponding extra to use:
  calibrax[image]     -- FID, InceptionScore, LPIPS
  calibrax[text]      -- BERTScore
  calibrax[scientific] -- chemical_validity, binding_affinity
"""
