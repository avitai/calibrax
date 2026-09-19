"""Profiling: timing, resources, GPU, energy, FLOPs, hardware, roofline, compilation, complexity."""

from calibrax.profiling.compilation import (
    CompilationProfiler,
    CompilationResult,
    XLAOptimizationResult,
)
from calibrax.profiling.complexity import analyze_complexity, ComplexityResult
from calibrax.profiling.energy import EnergyMonitor, EnergySample, EnergySummary, GpuPowerSource
from calibrax.profiling.flops import FlopsCounter, FlopsResult
from calibrax.profiling.gpu import (
    AdaptiveOperation,
    analyze_memory_pattern,
    GPUMemoryProfiler,
    HardwareConfig,
    MemoryAnalysis,
    MemoryOptimizer,
)
from calibrax.profiling.hardware import (
    detect_hardware_specs,
    HARDWARE_SPECS,
    HardwareSpec,
    spec_for_device_kind,
    UnknownHardwareError,
)
from calibrax.profiling.resources import (
    GpuClocks,
    GpuMemory,
    GpuPower,
    GPUProfilerProtocol,
    ResourceMonitor,
    ResourceSample,
    ResourceSummary,
)
from calibrax.profiling.roofline import RooflineAnalyzer, RooflineResult
from calibrax.profiling.timing import CallTiming, time_calls, TimingCollector, TimingSample
from calibrax.profiling.tracing import TraceLinker, TraceReference


__all__ = [
    # compilation
    "CompilationProfiler",
    "CompilationResult",
    "XLAOptimizationResult",
    # complexity
    "ComplexityResult",
    "analyze_complexity",
    # energy
    "EnergyMonitor",
    "EnergySample",
    "EnergySummary",
    "GpuPowerSource",
    # flops
    "FlopsCounter",
    "FlopsResult",
    # gpu
    "AdaptiveOperation",
    "GPUMemoryProfiler",
    "HardwareConfig",
    "MemoryAnalysis",
    "MemoryOptimizer",
    "analyze_memory_pattern",
    # hardware
    "HARDWARE_SPECS",
    "HardwareSpec",
    "UnknownHardwareError",
    "detect_hardware_specs",
    "spec_for_device_kind",
    # resources
    "GPUProfilerProtocol",
    "GpuClocks",
    "GpuMemory",
    "GpuPower",
    "ResourceMonitor",
    "ResourceSample",
    "ResourceSummary",
    # roofline
    "RooflineAnalyzer",
    "RooflineResult",
    # timing
    "CallTiming",
    "TimingCollector",
    "TimingSample",
    "time_calls",
    # tracing
    "TraceLinker",
    "TraceReference",
]
