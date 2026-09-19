"""The profile-gpu command: profiling with GPU energy read through NVML.

Needs the ``cuda12`` extra (nvidia-ml-py); running the command without it fails with the
install command.
"""

from __future__ import annotations

from pathlib import Path

import click

from calibrax.cli.profile import profile_options, run_profile
from calibrax.profiling.energy import EnergyMonitor
from calibrax.profiling.nvml import NvmlDevice


@click.command("profile-gpu")
@profile_options
@click.option("--gpu-index", default=0, type=int, help="NVML index of the GPU (default: 0).")
def profile_gpu(
    module: str,
    func_name: str,
    warmup: int,
    iterations: int,
    flops: bool,
    data: Path | None,
    gpu_index: int,
) -> None:
    """Profile a JAX function with GPU and CPU energy, the GPU read through NVML."""
    with NvmlDevice(gpu_index) as gpu:
        run_profile(
            module,
            func_name,
            warmup=warmup,
            iterations=iterations,
            count_flops=flops,
            data=data,
            energy_monitor=EnergyMonitor(gpu=gpu),
        )
