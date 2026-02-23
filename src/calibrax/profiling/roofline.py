"""Roofline analysis for JAX operations.

Identifies whether operations are compute-bound or memory-bound by
comparing arithmetic intensity against hardware roofline limits,
and generates optimization recommendations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax


logger = logging.getLogger(__name__)

from calibrax.profiling.hardware import detect_hardware_specs, measure_execution_time


@dataclass(frozen=True, slots=True, kw_only=True)
class RooflineResult:
    """Result of a roofline analysis on a JAX operation.

    Attributes:
        arithmetic_intensity: Achieved FLOPs per byte of memory traffic.
        critical_intensity: Hardware's ridge point (FLOPs/byte).
        memory_bandwidth_utilization: Fraction of peak memory bandwidth used.
        flops_utilization: Fraction of peak FLOPs achieved.
        bottleneck: Either "memory_bandwidth" or "compute".
        efficiency: Utilization of the binding resource.
        execution_time_ms: Measured execution time in milliseconds.
        recommendations: Optimization suggestions.
    """

    arithmetic_intensity: float
    critical_intensity: float
    memory_bandwidth_utilization: float
    flops_utilization: float
    bottleneck: str
    efficiency: float
    execution_time_ms: float
    recommendations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "arithmetic_intensity": float(self.arithmetic_intensity),
            "critical_intensity": float(self.critical_intensity),
            "memory_bandwidth_utilization": float(self.memory_bandwidth_utilization),
            "flops_utilization": float(self.flops_utilization),
            "bottleneck": self.bottleneck,
            "efficiency": float(self.efficiency),
            "execution_time_ms": float(self.execution_time_ms),
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RooflineResult:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with roofline result fields.

        Returns:
            Reconstructed RooflineResult instance.
        """
        return cls(
            arithmetic_intensity=data["arithmetic_intensity"],
            critical_intensity=data["critical_intensity"],
            memory_bandwidth_utilization=data["memory_bandwidth_utilization"],
            flops_utilization=data["flops_utilization"],
            bottleneck=data["bottleneck"],
            efficiency=data["efficiency"],
            execution_time_ms=data["execution_time_ms"],
            recommendations=tuple(data.get("recommendations", ())),
        )


def _extract_flops_from_cost(cost: dict[str, Any] | list[dict[str, Any]] | None) -> int | None:
    """Extract FLOP count from XLA cost analysis result.

    Args:
        cost: Result from ``lowered.cost_analysis()`` (dict or list of dicts).

    Returns:
        Positive FLOP count, or None if not available.
    """
    if cost and isinstance(cost, dict):
        flops = cost.get("flops")
        if flops is not None and flops > 0:
            return int(flops)
    if cost and isinstance(cost, list) and len(cost) > 0:
        flops = cost[0].get("flops")
        if flops is not None and flops > 0:
            return int(flops)
    return None


def _try_xla_cost_analysis(func: Callable[..., Any], inputs: list[jax.Array]) -> int | None:
    """Attempt to extract FLOPs from XLA cost analysis.

    Args:
        func: JAX function.
        inputs: Input arrays.

    Returns:
        FLOP count from XLA, or None if unavailable.
    """
    try:
        lowered = jax.jit(func).lower(*inputs)
        return _extract_flops_from_cost(lowered.cost_analysis())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        logger.debug("XLA cost analysis unavailable, falling back to heuristic")
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class RooflineAnalyzer:
    """Analyzes operation performance against hardware roofline limits.

    Uses measured execution time and estimated FLOPs to determine whether
    an operation is compute-bound or memory-bound, and how efficiently
    it uses the available hardware resources.

    Attributes:
        hardware_specs: Hardware specification dictionary (auto-detected if not provided).
    """

    hardware_specs: dict[str, Any] = field(default_factory=detect_hardware_specs)

    def analyze_operation(
        self,
        func: Callable[..., Any],
        inputs: list[jax.Array],
        *,
        flops_override: int | None = None,
    ) -> RooflineResult:
        """Perform roofline analysis on a JAX operation.

        Args:
            func: JAX function to analyze.
            inputs: Input arrays for the function.
            flops_override: If provided, use this FLOP count instead of estimating.
                For accurate results, pass the output of ``FlopsCounter.count()``.

        Returns:
            RooflineResult with bottleneck classification and recommendations.
        """
        execution_time = measure_execution_time(func, inputs)
        theoretical_flops = flops_override or self._estimate_flops(func, inputs)
        memory_traffic = self._estimate_memory_traffic(func, inputs)

        achieved_flops = theoretical_flops / execution_time if execution_time > 0 else 0.0
        memory_bw = memory_traffic / execution_time if execution_time > 0 else 0.0
        arithmetic_intensity = theoretical_flops / memory_traffic if memory_traffic > 0 else 0.0

        peak_flops = self.hardware_specs["peak_flops"]
        peak_bandwidth = self.hardware_specs["memory_bandwidth"]
        critical_intensity = self.hardware_specs["critical_intensity"]

        flops_util = achieved_flops / peak_flops if peak_flops > 0 else 0.0
        bw_util = memory_bw / peak_bandwidth if peak_bandwidth > 0 else 0.0

        if arithmetic_intensity < critical_intensity:
            bottleneck = "memory_bandwidth"
            efficiency = bw_util
        else:
            bottleneck = "compute"
            efficiency = flops_util

        recommendations = self._generate_recommendations(
            arithmetic_intensity, efficiency, bottleneck, inputs, achieved_flops
        )

        return RooflineResult(
            arithmetic_intensity=arithmetic_intensity,
            critical_intensity=critical_intensity,
            memory_bandwidth_utilization=bw_util,
            flops_utilization=flops_util,
            bottleneck=bottleneck,
            efficiency=efficiency,
            execution_time_ms=execution_time * 1000,
            recommendations=tuple(recommendations),
        )

    def _estimate_flops(self, func: Callable[..., Any], inputs: list[jax.Array]) -> int:
        """Estimate FLOPs using XLA cost analysis when possible.

        Falls back to a simple heuristic if cost_analysis is unavailable.

        Args:
            func: JAX function.
            inputs: Input arrays.

        Returns:
            Estimated FLOP count.
        """
        xla_flops = _try_xla_cost_analysis(func, inputs)
        if xla_flops is not None:
            return xla_flops

        # Fallback: rough heuristic based on input sizes
        total_elements = sum(x.size for x in inputs)
        return total_elements * 10

    def _estimate_memory_traffic(self, func: Callable[..., Any], inputs: list[jax.Array]) -> int:
        """Estimate total memory traffic (input + output bytes).

        Args:
            func: JAX function.
            inputs: Input arrays.

        Returns:
            Estimated bytes of memory traffic.
        """
        memory_traffic = sum(x.nbytes for x in inputs)

        try:
            output = func(*inputs)
            if isinstance(output, tuple | list):
                for out in output:
                    if hasattr(out, "nbytes"):
                        memory_traffic += out.nbytes
            elif hasattr(output, "nbytes"):
                memory_traffic += output.nbytes
        except (AttributeError, TypeError, ValueError, RuntimeError):
            # If we can't run the function, estimate output = input size
            memory_traffic *= 2

        return memory_traffic

    def _generate_recommendations(
        self,
        arithmetic_intensity: float,
        efficiency: float,
        bottleneck: str,
        inputs: list[jax.Array],
        achieved_flops: float,
    ) -> list[str]:
        """Generate optimization recommendations based on roofline analysis.

        Args:
            arithmetic_intensity: Achieved FLOPs per byte.
            efficiency: Utilization of the binding resource.
            bottleneck: "memory_bandwidth" or "compute".
            inputs: Input arrays.
            achieved_flops: Achieved FLOP/s.

        Returns:
            List of recommendation strings.
        """
        recommendations: list[str] = []

        if bottleneck == "memory_bandwidth":
            recommendations.extend(
                [
                    (
                        f"Memory bound (intensity: {arithmetic_intensity:.2f} < "
                        f"{self.hardware_specs['critical_intensity']:.2f}). "
                        f"Optimize memory access patterns."
                    ),
                    "Increase batch size to improve arithmetic intensity.",
                    "Use operation fusion to reduce memory traffic.",
                ]
            )
        else:
            recommendations.extend(
                [
                    (
                        f"Compute bound (intensity: {arithmetic_intensity:.2f} > "
                        f"{self.hardware_specs['critical_intensity']:.1f}). "
                        "Optimize FLOPs."
                    ),
                    "Check for inefficient math operations.",
                    "Ensure high-precision matrix units (MXU/TensorCore) are utilized.",
                ]
            )

        if efficiency < 0.2:
            attained_gflops = achieved_flops / 1e9
            recommendations.extend(
                [
                    (
                        f"Low performance ({attained_gflops:.2f} GFLOPS). "
                        "Check for bottlenecks other than compute/memory."
                    ),
                    "Consider kernel launch overhead (too many small ops).",
                    "Check data alignment.",
                ]
            )
        elif efficiency < 0.5:
            recommendations.extend(
                [
                    f"Moderate efficiency ({efficiency:.2%}). Potential improvements:",
                    "Optimize tensor layouts for memory access patterns.",
                    "Consider hardware-specific optimizations.",
                ]
            )

        if inputs:
            alignment_score = _calculate_alignment_score(inputs[0].shape)
            if alignment_score < 1.0:
                recommendations.append(
                    f"Poor tensor alignment (score: {alignment_score:.2f}). "
                    "Pad dimensions to multiples of 128/256."
                )

        return recommendations


def _calculate_alignment_score(shape: tuple[int, ...]) -> float:
    """Calculate how well tensor shape aligns with hardware requirements.

    Checks last dimension alignment with common hardware tile sizes.

    Args:
        shape: Tensor shape to evaluate.

    Returns:
        Score between 0 and 1 (1.0 = perfectly aligned).
    """
    if not shape:
        return 1.0

    last_dim = shape[-1]

    if last_dim % 128 == 0:
        return 1.0
    if last_dim % 32 == 0:
        return 0.8
    if last_dim % 8 == 0:
        return 0.5
    return 0.2
