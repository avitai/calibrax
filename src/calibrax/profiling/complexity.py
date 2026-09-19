"""Model complexity analysis for Flax NNX modules.

Provides parameter counts, memory usage estimates, computational
complexity analysis, and scaling characteristics for any NNX module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import jax
from flax import nnx
from substrax.records import read_record
from substrax.typing import JsonValue


# Spectral methods take FFTs over at least two spatial dimensions.
_MIN_FFT_DIMS = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class ComplexityResult:
    """Result of model complexity analysis.

    Attributes:
        total_parameters: Total number of trainable parameters.
        parameter_memory_mb: Memory consumed by parameters (float32).
        largest_layer_name: Name of the layer with the most parameters.
        largest_layer_params: Parameter count of the largest layer.
        input_shape: Shape of the analyzed input.
        estimated_memory_mb: Estimated total memory (params + activations).
        total_estimated_operations: Estimated total operations count.
        dominant_complexity: Name of the dominant operation type.
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


def analyze_complexity(model: nnx.Module, input_shape: tuple[int, ...]) -> ComplexityResult:
    """Analyze complexity of a Flax NNX module.

    Performs parameter counting, memory estimation, computational
    complexity analysis, and scaling characterization.

    Args:
        model: Flax NNX model to analyze.
        input_shape: Shape of input data (including batch dimension).

    Returns:
        ComplexityResult with detailed complexity metrics.
    """
    key = jax.random.PRNGKey(42)
    sample_input = jax.random.normal(key, input_shape)

    parameters = _analyze_parameters(model)
    memory_mb = _analyze_memory_usage(model, sample_input, parameters.memory_mb)
    operations = _analyze_computational_complexity(input_shape)
    scaling = _analyze_scaling_characteristics()

    return ComplexityResult(
        total_parameters=parameters.total,
        parameter_memory_mb=parameters.memory_mb,
        largest_layer_name=parameters.largest_name,
        largest_layer_params=parameters.largest,
        input_shape=input_shape,
        estimated_memory_mb=memory_mb,
        total_estimated_operations=operations.total,
        dominant_complexity=operations.dominant,
        scaling_characteristics=scaling,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _ParameterCounts:
    """A model's parameter count and bytes, and its largest parameter array."""

    total: int
    memory_mb: float
    largest_name: str
    largest: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _OperationCounts:
    """The estimated operation count and the operation type that dominates it."""

    total: int
    dominant: str


def _analyze_parameters(model: nnx.Module) -> _ParameterCounts:
    """Count a model's parameters and their bytes, by each array's dtype.

    Args:
        model: Flax NNX model.

    Returns:
        The total count, its size in MiB, and the largest parameter array's path and count.
    """
    params = nnx.state(model, nnx.Param)
    params_tree = nnx.to_tree(params)
    flat_with_path = jax.tree_util.tree_leaves_with_path(params_tree)

    total_params = 0
    total_bytes = 0
    largest_name = ""
    largest_count = 0

    for path, value in flat_with_path:
        if not hasattr(value, "shape"):
            continue

        key = "/".join(str(getattr(k, "key", k)) for k in path)
        param_count = int(value.size)
        total_params += param_count
        total_bytes += param_count * value.dtype.itemsize

        if param_count > largest_count:
            largest_count = param_count
            largest_name = key

    return _ParameterCounts(
        total=total_params,
        memory_mb=total_bytes / (1024 * 1024),
        largest_name=largest_name,
        largest=largest_count,
    )


def _analyze_memory_usage(
    model: nnx.Module, sample_input: jax.Array, param_memory_mb: float
) -> float:
    """Estimate total memory usage during forward pass.

    Args:
        model: Flax NNX model.
        sample_input: Sample input array.
        param_memory_mb: Memory used by parameters.

    Returns:
        Estimated total memory in MB.
    """
    input_memory_mb = sample_input.nbytes / (1024 * 1024)

    try:
        output = model(sample_input)  # type: ignore[misc]
        output_memory_mb = output.nbytes / (1024 * 1024)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        output_memory_mb = input_memory_mb

    # Rough estimate: intermediates are ~3x input size
    estimated_intermediate_mb = input_memory_mb * 3
    return param_memory_mb + input_memory_mb + output_memory_mb + estimated_intermediate_mb


def _analyze_computational_complexity(input_shape: tuple[int, ...]) -> _OperationCounts:
    """Estimate the operation count from the input shape.

    Args:
        input_shape: Input shape (batch, *spatial_dims).

    Returns:
        The total estimated operations and the operation type with the most.
    """
    spatial_dims = input_shape[1:]
    spatial_size = math.prod(spatial_dims)

    ops: dict[str, int] = {}

    # FFT-based operations (common in spectral methods)
    if len(spatial_dims) >= _MIN_FFT_DIMS and spatial_size > 1:
        fft_ops = int(spatial_size * math.log2(max(spatial_size, 2)))
        ops["fft_operations"] = fft_ops

    # Convolution operations
    conv_ops = spatial_size**2
    ops["convolution_operations"] = conv_ops

    # Linear operations
    ops["linear_operations"] = spatial_size

    dominant = max(ops, key=lambda name: ops[name]) if ops else "unknown"
    return _OperationCounts(total=sum(ops.values()), dominant=dominant)


def _analyze_scaling_characteristics() -> dict[str, str]:
    """Return standard scaling characteristics for neural network operations.

    Returns:
        Mapping of operation type to complexity class string.
    """
    return {
        "fft_scaling": "O(N log N)",
        "convolution_scaling": "O(N^2)",
        "memory_scaling": "O(N)",
        "parameter_scaling": "O(1)",
    }
