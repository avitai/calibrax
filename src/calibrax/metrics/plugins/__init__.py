"""Domain-specific metric plugins requiring optional external dependencies.

Each plugin module is guarded with lazy import checks.
Install the corresponding extra to use:
  calibrax[image]     -- FID, InceptionScore, LPIPS
  calibrax[text]      -- BERTScore

Video metrics such as VMAF require system FFmpeg with libvmaf support.
"""
