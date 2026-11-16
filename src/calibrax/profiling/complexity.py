"""Model complexity analysis for Flax NNX modules via XLA lowering.

Provides parameter counts, memory traffic estimates, XLA operation counts,
and dominant operation classification for any NNX module without executing
eager forward passes or mutating live module state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from substrax.records import read_record
from substrax.typing import JsonValue

from calibrax.profiling._jax_staging import lower_staged, Shaped, shaped_nbytes
from calibrax.profiling.compilation import _extract_hlo_instructions
from calibrax.profiling.flops import cost_mapping, FlopsCounter


_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
class ComplexityResult:
    """Result of model complexity analysis.

    Attributes:
        total_parameters: Total number of trainable parameters.
        parameter_memory_mb: Memory consumed by parameters in MiB.
        largest_layer_name: Name of the layer with the most parameters.
        largest_layer_params: Parameter count of the largest layer.
        input_shape: Shape of the analyzed input.
        estimated_memory_mb: Estimated memory footprint in MiB from parameter, I/O,
            and XLA buffer traffic analysis.
        total_estimated_operations: Total operations (FLOPs + transcendentals) from
            XLA's ``HloCostAnalysis`` of the lowered module.
        dominant_complexity: Dominant HLO operation family (``"fft_operations"``,
            ``"convolution_operations"``, ``"linear_operations"``, or
            ``"elementwise_operations"``).
        scaling_characteristics: Mapping of operation type to complexity class.
    """

    total_parameters: int
    parameter_memory_mb: float
    largest_layer_name: str
    largest_layer_params: int
    input_shape: tuple[int, ...]
    estimated_memory_mb: float
    total_estimated_operations: int
    dominant_complexity: str
    scaling_characteristics: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "total_parameters": int(self.total_parameters),
            "parameter_memory_mb": float(self.parameter_memory_mb),
            "largest_layer_name": self.largest_layer_name,
            "largest_layer_params": int(self.largest_layer_params),
            "input_shape": list(self.input_shape),
            "estimated_memory_mb": float(self.estimated_memory_mb),
            "total_estimated_operations": int(self.total_estimated_operations),
            "dominant_complexity": self.dominant_complexity,
            "scaling_characteristics": dict(self.scaling_characteristics),
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> ComplexityResult:
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


def analyze_complexity(
    model: nnx.Module,
    input_shape: tuple[int, ...],
    *,
    dtype: DTypeLike = jnp.float32,
) -> ComplexityResult:
    """Analyze complexity of a Flax NNX module via abstract XLA lowering.

    Counts parameters directly from ``nnx.Param`` state and lowers ``model`` on an
    abstract ``jax.ShapeDtypeStruct(input_shape, dtype)`` so zero device arrays are
    allocated and stateful variables (``BatchNorm``, ``Rngs``) are never mutated.

    Args:
        model: Flax NNX model to analyze.
        input_shape: Shape of input data (including batch dimension).
        dtype: Element dtype of the abstract input tensor (default ``jnp.float32``).

    Returns:
        ComplexityResult with parameter, memory, XLA operation, and HLO metrics.
    """
    abstract_input = jax.ShapeDtypeStruct(input_shape, jnp.dtype(dtype))
    parameters = _analyze_parameters(model)
    lowered, out_info = lower_staged(model, (abstract_input,), {}, abstract_inputs=True)
    counted = FlopsCounter().count(model, abstract_input)

    memory_mb = _estimate_memory_mb(lowered, abstract_input, out_info, parameters.total_bytes)
    dominant = _dominant_hlo_family(lowered.as_text("hlo"))
    scaling = _analyze_scaling_characteristics()

    return ComplexityResult(
        total_parameters=parameters.total,
        parameter_memory_mb=parameters.memory_mb,
        largest_layer_name=parameters.largest_name,
        largest_layer_params=parameters.largest,
        input_shape=input_shape,
        estimated_memory_mb=memory_mb,
        total_estimated_operations=counted.total_flops + counted.transcendentals,
        dominant_complexity=dominant,
        scaling_characteristics=scaling,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _ParameterCounts:
    """A model's parameter count and bytes, and its largest parameter array."""

    total: int
    total_bytes: int
    memory_mb: float
    largest_name: str
    largest: int


def _analyze_parameters(model: nnx.Module) -> _ParameterCounts:
    """Count a model's parameters and their bytes, by each array's dtype."""
    params = nnx.state(model, nnx.Param)
    params_tree = nnx.to_tree(params)
    flat_with_path = jax.tree_util.tree_leaves_with_path(params_tree)

    total_params = 0
    total_bytes = 0
    largest_name = ""
    largest_count = 0

    for path, value in flat_with_path:
        if not isinstance(value, Shaped):
            continue

        key = "/".join(str(getattr(k, "key", k)) for k in path)
        param_bytes = shaped_nbytes(value)
        itemsize = getattr(value.dtype, "itemsize", None) or jnp.dtype(value.dtype).itemsize
        param_count = param_bytes // int(itemsize) if int(itemsize) > 0 else 0
        total_params += param_count
        total_bytes += param_bytes

        if param_count > largest_count:
            largest_count = param_count
            largest_name = key

    return _ParameterCounts(
        total=total_params,
        total_bytes=total_bytes,
        memory_mb=total_bytes / _BYTES_PER_MB,
        largest_name=largest_name,
        largest=largest_count,
    )


def _estimate_memory_mb(
    lowered: jax.stages.Lowered,
    abstract_input: jax.ShapeDtypeStruct,
    out_info: object,
    param_bytes: int,
) -> float:
    """Estimate memory footprint in MiB from parameter, I/O, and XLA buffer traffic."""
    input_bytes = shaped_nbytes(abstract_input)
    output_leaves = [leaf for leaf in jax.tree.leaves(out_info) if isinstance(leaf, Shaped)]
    output_bytes = sum(shaped_nbytes(leaf) for leaf in output_leaves)
    minimum_bytes = param_bytes + input_bytes + output_bytes

    compiled_cost = cost_mapping(lowered.compile().cost_analysis())
    unopt_cost = cost_mapping(lowered.cost_analysis()) if compiled_cost is None else compiled_cost
    xla_bytes = 0
    if unopt_cost is not None:
        raw_bytes = unopt_cost.get("bytes accessed", 0.0)
        if raw_bytes > 0:
            xla_bytes = int(raw_bytes)

    return max(minimum_bytes, xla_bytes) / _BYTES_PER_MB


def _dominant_hlo_family(hlo_text: str) -> str:
    """Classify dominant operation family from lowered HLO instruction opcodes."""
    counts = {
        "fft_operations": 0,
        "convolution_operations": 0,
        "linear_operations": 0,
    }
    for opcode, _ in _extract_hlo_instructions(hlo_text):
        if "fft" in opcode:
            counts["fft_operations"] += 1
        elif "convolution" in opcode:
            counts["convolution_operations"] += 1
        elif "dot" in opcode:
            counts["linear_operations"] += 1

    best_family, best_count = max(counts.items(), key=lambda item: item[1])
    if best_count > 0:
        return best_family
    return "elementwise_operations"


def _analyze_scaling_characteristics() -> dict[str, str]:
    """Return reference scaling characteristics for neural network operator families."""
    return {
        "fft_scaling": "O(N log N)",
        "convolution_scaling": "O(N^2)",
        "memory_scaling": "O(N)",
        "parameter_scaling": "O(1)",
    }
