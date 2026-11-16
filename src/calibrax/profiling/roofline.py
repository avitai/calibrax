"""Roofline analysis for JAX operations.

Identifies whether operations are compute-bound or memory-bound by
comparing arithmetic intensity against hardware roofline limits,
and generates optimization recommendations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import jax
from substrax.devices import detect_devices
from substrax.records import read_record
from substrax.typing import JsonValue, PyTree

from calibrax.profiling._jax_staging import (
    contains_nnx_graph_node,
    lower_staged,
    normalize_static_argnames,
    normalize_static_argnums,
    prepare_kernel_invocation,
    Shaped,
    shaped_nbytes,
    unbind_nnx_call,
)
from calibrax.profiling.flops import FlopsCounter
from calibrax.profiling.hardware import detect_hardware_specs, HardwareSpec, UnknownHardwareError
from calibrax.profiling.timing import time_calls


# Attained-over-attainable fractions behind the recommendations.
_LOW_EFFICIENCY = 0.2
_MODERATE_EFFICIENCY = 0.5


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

    def to_dict(self) -> dict[str, JsonValue]:
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
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> RooflineResult:
        """Read the record from the JSON object ``to_dict`` writes.

        Args:
            data: The JSON object.

        Returns:
            The record.

        Raises:
            pydantic.ValidationError: If a field is missing or holds a value its annotation
                does not admit.
        """
        return read_record(cls, data)


@dataclass(frozen=True, slots=True, kw_only=True)
class RooflineAnalyzer:
    """Analyzes operation performance against hardware roofline limits.

    Uses measured execution time and estimated FLOPs to determine whether
    an operation is compute-bound or memory-bound, and how efficiently
    it uses the available hardware resources.

    Attributes:
        hardware_specs: The accelerator's figures; detected from JAX's device kind when not
            given, and ``None`` when the accelerator is not in ``HARDWARE_SPECS``.
    """

    hardware_specs: HardwareSpec | None = field(default_factory=detect_hardware_specs)

    def analyze_operation(  # noqa: DOC503  # FlopsUnavailableError is raised by FlopsCounter
        self,
        func: Callable[..., PyTree],
        inputs: Sequence[PyTree],
        *,
        flops_override: int | None = None,
        static_argnums: int | Iterable[int] = (),
        static_argnames: str | Iterable[str] = (),
        **kwargs: PyTree,
    ) -> RooflineResult:
        """Perform roofline analysis on a JAX operation or ``flax.nnx.Module``.

        Args:
            func: JAX function, ``nnx.Module``, or bound ``nnx.Module`` method to analyze.
            inputs: Positional input arrays or PyTrees for the function.
            flops_override: The operation's FLOP count, used in place of XLA's estimate.
            static_argnums: Positional argument indices treated as static at trace time.
            static_argnames: Keyword argument names treated as static at trace time.
            **kwargs: Keyword arguments forwarded to ``func``.

        Returns:
            RooflineResult with bottleneck classification and recommendations.

        Raises:
            UnknownHardwareError: If no spec was given and the accelerator is not in
                ``HARDWARE_SPECS``.
            FlopsUnavailableError: If no override is given and XLA cannot estimate the cost.
        """
        spec = self.hardware_specs
        if spec is None:
            kinds = ", ".join(sorted(set(detect_devices().device_kinds)))
            msg = (
                f"no roofline figures for the accelerator in use ({kinds}); pass "
                "RooflineAnalyzer(hardware_specs=resolve_hardware_spec(dtype=...)) to measure "
                "its ceilings, or a HardwareSpec with its peak FLOP/s and memory bandwidth"
            )
            raise UnknownHardwareError(msg)

        norm_nums = normalize_static_argnums(static_argnums)
        norm_names = normalize_static_argnames(static_argnames)
        pos_args = tuple(inputs)
        invoke_kernel = prepare_kernel_invocation(
            func,
            pos_args,
            kwargs,
            static_argnums=norm_nums,
            static_argnames=norm_names,
            jit_fn=jax.jit,
        )
        execution_time = time_calls(invoke_kernel).median_sec
        counted = FlopsCounter().count(
            func,
            *pos_args,
            static_argnums=norm_nums,
            static_argnames=norm_names,
            optimized=True,
            **kwargs,
        )
        theoretical_flops = flops_override if flops_override is not None else counted.total_flops
        memory_traffic = counted.bytes_accessed or self._estimate_memory_traffic(
            func,
            inputs,
            static_argnums=norm_nums,
            static_argnames=norm_names,
            **kwargs,
        )

        achieved_flops = theoretical_flops / execution_time if execution_time > 0 else 0.0
        memory_bw = memory_traffic / execution_time if execution_time > 0 else 0.0
        arithmetic_intensity = theoretical_flops / memory_traffic if memory_traffic > 0 else 0.0

        peak_flops = spec.peak_flops
        peak_bandwidth = spec.memory_bandwidth
        critical_intensity = spec.critical_intensity

        flops_util = achieved_flops / peak_flops if peak_flops > 0 else 0.0
        bw_util = memory_bw / peak_bandwidth if peak_bandwidth > 0 else 0.0

        if arithmetic_intensity < critical_intensity:
            bottleneck = "memory_bandwidth"
            efficiency = bw_util
        else:
            bottleneck = "compute"
            efficiency = flops_util

        recommendations = _recommendations(
            spec, arithmetic_intensity, efficiency, bottleneck, inputs, achieved_flops
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

    def _estimate_memory_traffic(
        self,
        func: Callable[..., PyTree],
        inputs: Sequence[PyTree],
        *,
        static_argnums: int | Iterable[int] = (),
        static_argnames: str | Iterable[str] = (),
        **kwargs: PyTree,
    ) -> int:
        """The bytes read and written, when XLA reports none: input and output leaves once.

        XLA's ``bytes accessed`` is what ``analyze_operation`` uses; it counts the
        intermediates materialised between kernels, which this lower bound does not.

        Args:
            func: JAX function or ``nnx.Module``.
            inputs: Positional input arrays or PyTrees.
            static_argnums: Positional argument indices treated as static at trace time.
            static_argnames: Keyword argument names treated as static at trace time.
            **kwargs: Keyword arguments forwarded to ``func``.

        Returns:
            Input bytes plus output bytes across all shaped leaves.
        """
        pos_args = tuple(inputs)
        _, _, _, _, was_unbound = unbind_nnx_call(func, pos_args, kwargs)
        if not was_unbound and not contains_nnx_graph_node((pos_args, kwargs)) and not kwargs:
            norm_nums = normalize_static_argnums(static_argnums)
            norm_names = normalize_static_argnames(static_argnames)
            if not norm_nums and not norm_names:
                out_info: object = jax.eval_shape(func, *pos_args)
            else:
                _, out_info = lower_staged(
                    func,
                    pos_args,
                    kwargs,
                    static_argnums=norm_nums,
                    static_argnames=norm_names,
                    abstract_inputs=True,
                )
        else:
            _, out_info = lower_staged(
                func,
                pos_args,
                kwargs,
                static_argnums=static_argnums,
                static_argnames=static_argnames,
                abstract_inputs=True,
            )
        outputs = [leaf for leaf in jax.tree.leaves(out_info) if isinstance(leaf, Shaped)]
        output_bytes = sum(shaped_nbytes(leaf) for leaf in outputs)
        input_leaves = [
            leaf for leaf in jax.tree.leaves((pos_args, kwargs)) if isinstance(leaf, Shaped)
        ]
        input_bytes = sum(shaped_nbytes(leaf) for leaf in input_leaves)
        return input_bytes + output_bytes


def _recommendations(
    spec: HardwareSpec,
    arithmetic_intensity: float,
    efficiency: float,
    bottleneck: str,
    inputs: Sequence[PyTree],
    achieved_flops: float,
) -> list[str]:
    """Generate optimization recommendations based on roofline analysis.

    Args:
        spec: The accelerator's figures.
        arithmetic_intensity: Achieved FLOPs per byte.
        efficiency: Utilization of the binding resource.
        bottleneck: "memory_bandwidth" or "compute".
        inputs: Input arrays or PyTrees.
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
                    f"{spec.critical_intensity:.2f}). "
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
                    f"{spec.critical_intensity:.1f}). "
                    "Optimize FLOPs."
                ),
                "Check for inefficient math operations.",
                "Ensure high-precision matrix units (MXU/TensorCore) are utilized.",
            ]
        )

    if efficiency < _LOW_EFFICIENCY:
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
    elif efficiency < _MODERATE_EFFICIENCY:
        recommendations.extend(
            [
                f"Moderate efficiency ({efficiency:.2%}). Potential improvements:",
                "Optimize tensor layouts for memory access patterns.",
                "Consider hardware-specific optimizations.",
            ]
        )

    shaped_inputs = [leaf for leaf in jax.tree.leaves(inputs) if isinstance(leaf, Shaped)]
    if shaped_inputs:
        alignment_score = _calculate_alignment_score(shaped_inputs[0].shape)
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
