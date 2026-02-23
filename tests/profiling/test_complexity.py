"""Tests for calibrax.profiling.complexity module."""

from __future__ import annotations

import dataclasses

import pytest
from flax import nnx

from calibrax.profiling.complexity import analyze_complexity, ComplexityResult


class TestComplexityResult:
    """Tests for ComplexityResult frozen dataclass."""

    def _make_result(self) -> ComplexityResult:
        """Create a sample ComplexityResult for testing."""
        return ComplexityResult(
            total_parameters=1024,
            parameter_memory_mb=0.004,
            largest_layer_name="dense/kernel",
            largest_layer_params=512,
            input_shape=(1, 32),
            estimated_memory_mb=0.01,
            total_estimated_operations=2048,
            dominant_complexity="linear_operations",
            scaling_characteristics={"fft_scaling": "O(N log N)"},
        )

    def test_creation(self) -> None:
        """Should create ComplexityResult with all fields."""
        result = self._make_result()
        assert result.total_parameters == 1024
        assert result.parameter_memory_mb == 0.004
        assert result.largest_layer_name == "dense/kernel"
        assert result.largest_layer_params == 512
        assert result.input_shape == (1, 32)
        assert result.estimated_memory_mb == 0.01
        assert result.total_estimated_operations == 2048
        assert result.dominant_complexity == "linear_operations"
        assert result.scaling_characteristics == {"fft_scaling": "O(N log N)"}

    def test_creation_default_scaling(self) -> None:
        """Should create ComplexityResult with empty scaling_characteristics."""
        result = ComplexityResult(
            total_parameters=10,
            parameter_memory_mb=0.001,
            largest_layer_name="w",
            largest_layer_params=10,
            input_shape=(1, 2),
            estimated_memory_mb=0.002,
            total_estimated_operations=20,
            dominant_complexity="linear_operations",
        )
        assert result.scaling_characteristics == {}

    def test_frozen_immutability(self) -> None:
        """Should raise FrozenInstanceError on attribute mutation."""
        result = self._make_result()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.total_parameters = 0  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.dominant_complexity = "other"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """to_dict should return a JSON-compatible dictionary."""
        result = self._make_result()
        d = result.to_dict()
        assert d["total_parameters"] == 1024
        assert d["parameter_memory_mb"] == 0.004
        assert d["largest_layer_name"] == "dense/kernel"
        assert d["largest_layer_params"] == 512
        assert d["input_shape"] == [1, 32]
        assert d["estimated_memory_mb"] == 0.01
        assert d["total_estimated_operations"] == 2048
        assert d["dominant_complexity"] == "linear_operations"
        assert d["scaling_characteristics"] == {"fft_scaling": "O(N log N)"}

    def test_from_dict(self) -> None:
        """from_dict should reconstruct a ComplexityResult."""
        data = {
            "total_parameters": 256,
            "parameter_memory_mb": 0.001,
            "largest_layer_name": "kernel",
            "largest_layer_params": 128,
            "input_shape": [1, 8],
            "estimated_memory_mb": 0.005,
            "total_estimated_operations": 512,
            "dominant_complexity": "convolution_operations",
            "scaling_characteristics": {"memory_scaling": "O(N)"},
        }
        result = ComplexityResult.from_dict(data)
        assert result.total_parameters == 256
        assert result.input_shape == (1, 8)
        assert result.scaling_characteristics == {"memory_scaling": "O(N)"}

    def test_from_dict_missing_scaling(self) -> None:
        """from_dict should default to empty dict for missing scaling_characteristics."""
        data = {
            "total_parameters": 10,
            "parameter_memory_mb": 0.0,
            "largest_layer_name": "w",
            "largest_layer_params": 10,
            "input_shape": [1, 2],
            "estimated_memory_mb": 0.0,
            "total_estimated_operations": 20,
            "dominant_complexity": "linear_operations",
        }
        result = ComplexityResult.from_dict(data)
        assert result.scaling_characteristics == {}

    def test_to_dict_from_dict_round_trip(self) -> None:
        """to_dict/from_dict should produce an equivalent object."""
        original = self._make_result()
        reconstructed = ComplexityResult.from_dict(original.to_dict())
        assert reconstructed.total_parameters == original.total_parameters
        assert reconstructed.parameter_memory_mb == original.parameter_memory_mb
        assert reconstructed.largest_layer_name == original.largest_layer_name
        assert reconstructed.largest_layer_params == original.largest_layer_params
        assert reconstructed.input_shape == original.input_shape
        assert reconstructed.estimated_memory_mb == original.estimated_memory_mb
        assert reconstructed.total_estimated_operations == original.total_estimated_operations
        assert reconstructed.dominant_complexity == original.dominant_complexity
        assert reconstructed.scaling_characteristics == original.scaling_characteristics


class TestAnalyzeComplexity:
    """Tests for analyze_complexity function."""

    def test_linear_model(self) -> None:
        """Should analyze a simple nnx.Linear model correctly."""
        model = nnx.Linear(4, 8, rngs=nnx.Rngs(0))
        result = analyze_complexity(model, (1, 4))
        assert isinstance(result, ComplexityResult)
        assert result.total_parameters > 0
        # Linear(4, 8) has 4*8 kernel + 8 bias = 40 parameters
        assert result.total_parameters == 40
        assert result.input_shape == (1, 4)

    def test_parameter_memory_positive(self) -> None:
        """Parameter memory should be positive for a non-empty model."""
        model = nnx.Linear(16, 32, rngs=nnx.Rngs(0))
        result = analyze_complexity(model, (1, 16))
        assert result.parameter_memory_mb > 0

    def test_estimated_memory_includes_params(self) -> None:
        """Estimated memory should be at least as large as parameter memory."""
        model = nnx.Linear(8, 16, rngs=nnx.Rngs(0))
        result = analyze_complexity(model, (1, 8))
        assert result.estimated_memory_mb >= result.parameter_memory_mb

    def test_largest_layer_info(self) -> None:
        """Should identify the largest layer by parameter count."""
        model = nnx.Linear(4, 8, rngs=nnx.Rngs(0))
        result = analyze_complexity(model, (1, 4))
        assert result.largest_layer_name != ""
        assert result.largest_layer_params > 0
        # Kernel (4*8=32) should be larger than bias (8)
        assert result.largest_layer_params == 32

    def test_scaling_characteristics(self) -> None:
        """Should return standard scaling characteristics."""
        model = nnx.Linear(4, 8, rngs=nnx.Rngs(0))
        result = analyze_complexity(model, (1, 4))
        assert "fft_scaling" in result.scaling_characteristics
        assert "memory_scaling" in result.scaling_characteristics
        assert result.scaling_characteristics["memory_scaling"] == "O(N)"

    def test_operations_positive(self) -> None:
        """Total estimated operations should be positive."""
        model = nnx.Linear(4, 8, rngs=nnx.Rngs(0))
        result = analyze_complexity(model, (1, 4))
        assert result.total_estimated_operations > 0

    def test_dominant_complexity_is_string(self) -> None:
        """Dominant complexity should be a non-empty string."""
        model = nnx.Linear(4, 8, rngs=nnx.Rngs(0))
        result = analyze_complexity(model, (1, 4))
        assert isinstance(result.dominant_complexity, str)
        assert len(result.dominant_complexity) > 0

    def test_batch_dimension(self) -> None:
        """Should handle different batch sizes in input_shape."""
        model = nnx.Linear(4, 8, rngs=nnx.Rngs(0))
        result_b1 = analyze_complexity(model, (1, 4))
        result_b4 = analyze_complexity(model, (4, 4))
        # Same model, so same parameter count
        assert result_b1.total_parameters == result_b4.total_parameters
        assert result_b4.input_shape == (4, 4)
