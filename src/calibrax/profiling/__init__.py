"""Profiling: timing, resources, GPU, energy, FLOPs, hardware, roofline, compilation, complexity."""

from calibrax.profiling.compilation import (
    CompilationProfiler,
    CompilationResult,
    XLAOptimizationResult,
)
from calibrax.profiling.complexity import analyze_complexity, ComplexityResult
from calibrax.profiling.energy import EnergyMonitor, EnergySample, EnergySummary
from calibrax.profiling.flops import FlopsCounter, FlopsResult
from calibrax.profiling.gpu import (
    AdaptiveOperation,
    GPUMemoryProfiler,
    HardwareConfig,
    MemoryAnalysis,
    MemoryOptimizer,
)
from calibrax.profiling.hardware import (
    detect_hardware_specs,
    HARDWARE_SPECS,
)
from calibrax.profiling.resources import (
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
    # flops
    "FlopsCounter",
    "FlopsResult",
    # gpu
    "AdaptiveOperation",
    "GPUMemoryProfiler",
    "HardwareConfig",
    "MemoryAnalysis",
    "MemoryOptimizer",
    # hardware
    "HARDWARE_SPECS",
    "detect_hardware_specs",
    # resources
    "GPUProfilerProtocol",
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
