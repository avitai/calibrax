"""Tests for roofline analysis.

Verifies RooflineResult dataclass immutability and serialization,
RooflineAnalyzer operation analysis with flops_override, bottleneck
classification, and alignment score calculation.
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from calibrax.profiling.flops import cost_mapping, FlopsCounter, FlopsUnavailableError
from calibrax.profiling.hardware import detect_hardware_specs, HardwareSpec
from calibrax.profiling.roofline import (
    _calculate_alignment_score,
    _recommendations,
    RooflineAnalyzer,
    RooflineResult,
    UnknownHardwareError,
)


# 1 TFLOP/s over 200 GB/s: a ridge point of 5 FLOPs per byte.
SPEC = HardwareSpec(name="test", peak_flops=1e12, memory_bandwidth=200e9)


class TestRooflineResult:
    """Tests for RooflineResult frozen dataclass."""

    def test_creation_with_all_fields(self) -> None:
        result = RooflineResult(
            arithmetic_intensity=5.0,
            critical_intensity=10.0,
            memory_bandwidth_utilization=0.3,
            flops_utilization=0.1,
            bottleneck="memory_bandwidth",
            efficiency=0.3,
            execution_time_ms=1.5,
            recommendations=("Use larger batch size.",),
        )
        assert result.arithmetic_intensity == 5.0
        assert result.critical_intensity == 10.0
        assert result.memory_bandwidth_utilization == 0.3
        assert result.flops_utilization == 0.1
        assert result.bottleneck == "memory_bandwidth"
        assert result.efficiency == 0.3
        assert result.execution_time_ms == 1.5
        assert result.recommendations == ("Use larger batch size.",)

    def test_frozen_immutability(self) -> None:
        result = RooflineResult(
            arithmetic_intensity=5.0,
            critical_intensity=10.0,
            memory_bandwidth_utilization=0.3,
            flops_utilization=0.1,
            bottleneck="memory_bandwidth",
            efficiency=0.3,
            execution_time_ms=1.5,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.arithmetic_intensity = 99.0  # type: ignore[misc]

    def test_kw_only_enforcement(self) -> None:
        with pytest.raises(TypeError):
            RooflineResult(5.0, 10.0, 0.3, 0.1, "compute", 0.1, 1.0)  # type: ignore[misc]

    def test_recommendations_default_empty_tuple(self) -> None:
        result = RooflineResult(
            arithmetic_intensity=5.0,
            critical_intensity=10.0,
            memory_bandwidth_utilization=0.3,
            flops_utilization=0.1,
            bottleneck="compute",
            efficiency=0.1,
            execution_time_ms=1.0,
        )
        assert result.recommendations == ()

    def test_to_dict(self) -> None:
        result = RooflineResult(
            arithmetic_intensity=5.0,
            critical_intensity=10.0,
            memory_bandwidth_utilization=0.3,
            flops_utilization=0.1,
            bottleneck="memory_bandwidth",
            efficiency=0.3,
            execution_time_ms=1.5,
            recommendations=("rec1", "rec2"),
        )
        d = result.to_dict()
        assert d["arithmetic_intensity"] == 5.0
        assert d["bottleneck"] == "memory_bandwidth"
        assert d["recommendations"] == ["rec1", "rec2"]
        assert isinstance(d["recommendations"], list)

    def test_from_dict(self) -> None:
        data = {
            "arithmetic_intensity": 5.0,
            "critical_intensity": 10.0,
            "memory_bandwidth_utilization": 0.3,
            "flops_utilization": 0.1,
            "bottleneck": "compute",
            "efficiency": 0.1,
            "execution_time_ms": 2.0,
            "recommendations": ["optimize"],
        }
        result = RooflineResult.from_dict(data)
        assert result.bottleneck == "compute"
        assert result.recommendations == ("optimize",)

    def test_to_dict_from_dict_round_trip(self) -> None:
        original = RooflineResult(
            arithmetic_intensity=12.5,
            critical_intensity=150.0,
            memory_bandwidth_utilization=0.65,
            flops_utilization=0.42,
            bottleneck="compute",
            efficiency=0.42,
            execution_time_ms=3.7,
            recommendations=("batch more", "fuse ops"),
        )
        restored = RooflineResult.from_dict(original.to_dict())
        assert restored.arithmetic_intensity == original.arithmetic_intensity
        assert restored.critical_intensity == original.critical_intensity
        assert restored.bottleneck == original.bottleneck
        assert restored.efficiency == original.efficiency
        assert restored.execution_time_ms == original.execution_time_ms
        assert restored.recommendations == original.recommendations

    def test_from_dict_missing_recommendations_defaults_empty(self) -> None:
        data = {
            "arithmetic_intensity": 1.0,
            "critical_intensity": 10.0,
            "memory_bandwidth_utilization": 0.1,
            "flops_utilization": 0.05,
            "bottleneck": "memory_bandwidth",
            "efficiency": 0.1,
            "execution_time_ms": 0.5,
        }
        result = RooflineResult.from_dict(data)
        assert result.recommendations == ()


class TestRooflineAnalyzer:
    """Tests for RooflineAnalyzer operation analysis."""

    def test_analyze_with_flops_override(self) -> None:
        analyzer = RooflineAnalyzer(hardware_specs=SPEC)
        x = jnp.ones((64, 64))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=1_000_000)

        assert isinstance(result, RooflineResult)
        assert result.execution_time_ms > 0
        assert result.critical_intensity == 5.0

    def test_bottleneck_is_valid_value(self) -> None:
        analyzer = RooflineAnalyzer(hardware_specs=SPEC)
        x = jnp.ones((32, 32))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=100)

        assert result.bottleneck in ("memory_bandwidth", "compute")

    def test_memory_bound_when_low_intensity(self) -> None:
        """Low flops_override with large arrays -> low arithmetic intensity -> memory bound."""
        # Very high threshold: 1000 FLOPs per byte.
        spec = HardwareSpec(name="test", peak_flops=1e12, memory_bandwidth=1e9)
        analyzer = RooflineAnalyzer(hardware_specs=spec)
        x = jnp.ones((128, 128))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=1)

        assert result.bottleneck == "memory_bandwidth"

    def test_compute_bound_when_high_intensity(self) -> None:
        """Very high flops_override with small arrays -> high arithmetic intensity -> compute."""
        # Very low threshold: 0.001 FLOPs per byte.
        spec = HardwareSpec(name="test", peak_flops=1e12, memory_bandwidth=1e15)
        analyzer = RooflineAnalyzer(hardware_specs=spec)
        x = jnp.ones((2,))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=10_000_000)

        assert result.bottleneck == "compute"

    def test_result_has_recommendations(self) -> None:
        analyzer = RooflineAnalyzer(hardware_specs=SPEC)
        x = jnp.ones((16, 16))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=100)

        assert isinstance(result.recommendations, tuple)
        assert len(result.recommendations) > 0

    def test_utilization_values_non_negative(self) -> None:
        analyzer = RooflineAnalyzer(hardware_specs=SPEC)
        x = jnp.ones((8, 8))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=500)

        assert result.memory_bandwidth_utilization >= 0
        assert result.flops_utilization >= 0
        assert result.efficiency >= 0

    def test_default_hardware_specs_auto_detected(self) -> None:
        """Default factory calls detect_hardware_specs for auto-detection."""

        assert RooflineAnalyzer().hardware_specs == detect_hardware_specs()

    def test_flops_come_from_xla(self) -> None:
        x = jnp.ones((8, 8))
        cost = cost_mapping(jax.jit(jnp.matmul).lower(x, x).compile().cost_analysis())
        assert cost is not None

        result = RooflineAnalyzer(hardware_specs=SPEC).analyze_operation(jnp.matmul, [x, x])

        assert cost["flops"] == 2 * 8 * 8 * 8
        assert result.arithmetic_intensity == pytest.approx(
            cost["flops"] / cost["bytes accessed"], rel=1e-6
        )

    def test_a_function_xla_cannot_cost_is_refused_not_guessed(self) -> None:
        def with_callback(x: jax.Array) -> jax.Array:
            return jax.pure_callback(lambda v: v * 2, jax.ShapeDtypeStruct((4,), jnp.float32), x)

        analyzer = RooflineAnalyzer(hardware_specs=SPEC)
        with pytest.raises(FlopsUnavailableError):
            analyzer.analyze_operation(with_callback, [jnp.ones(4)])

    def test_memory_traffic_counts_every_output_leaf_without_running(self) -> None:
        calls: list[int] = []

        def split(x: jax.Array) -> dict[str, jax.Array | tuple[jax.Array, jax.Array]]:
            calls.append(1)
            return {"sum": x + 1, "parts": (x[:2], x.astype(jnp.float16))}

        analyzer = RooflineAnalyzer(hardware_specs=SPEC)
        x = jnp.ones((4,), jnp.float32)
        # 16 input bytes; outputs 16 + 8 + 8 bytes.
        assert analyzer._estimate_memory_traffic(split, [x]) == 16 + 16 + 8 + 8
        assert calls == [1]  # traced once by eval_shape, never executed eagerly

    def test_a_zero_flop_override_is_used(self) -> None:
        analyzer = RooflineAnalyzer(hardware_specs=SPEC)
        x = jnp.ones((64, 64))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=0)
        assert result.arithmetic_intensity == 0.0
        assert result.flops_utilization == 0.0

    def test_unknown_hardware_is_refused_at_analysis(self) -> None:
        analyzer = RooflineAnalyzer(hardware_specs=None)
        with pytest.raises(UnknownHardwareError, match="HardwareSpec"):
            analyzer.analyze_operation(jnp.add, [jnp.ones(4), jnp.ones(4)])

    def test_generate_recommendations_includes_moderate_efficiency_message(self) -> None:
        x = jnp.ones((2, 32))
        recs = _recommendations(
            SPEC,
            arithmetic_intensity=10.0,
            efficiency=0.3,
            bottleneck="compute",
            inputs=[x],
            achieved_flops=1e9,
        )
        assert any("Moderate efficiency" in rec for rec in recs)

    def test_generate_recommendations_adds_alignment_hint_for_unaligned_inputs(self) -> None:
        x = jnp.ones((2, 7))
        recs = _recommendations(
            SPEC,
            arithmetic_intensity=2.0,
            efficiency=0.9,
            bottleneck="memory_bandwidth",
            inputs=[x],
            achieved_flops=1e9,
        )
        assert any("Poor tensor alignment" in rec for rec in recs)

    def test_generate_recommendations_skips_alignment_for_empty_inputs(self) -> None:
        recs = _recommendations(
            SPEC,
            arithmetic_intensity=2.0,
            efficiency=0.9,
            bottleneck="memory_bandwidth",
            inputs=[],
            achieved_flops=1e9,
        )
        assert all("Poor tensor alignment" not in rec for rec in recs)


class TestCalculateAlignmentScore:
    """Tests for _calculate_alignment_score helper."""

    def test_empty_shape_returns_one(self) -> None:
        assert _calculate_alignment_score(()) == 1.0

    def test_divisible_by_128_returns_one(self) -> None:
        assert _calculate_alignment_score((32, 128)) == 1.0
        assert _calculate_alignment_score((10, 256)) == 1.0
        assert _calculate_alignment_score((1, 512)) == 1.0

    def test_divisible_by_32_returns_0_8(self) -> None:
        assert _calculate_alignment_score((10, 32)) == 0.8
        assert _calculate_alignment_score((5, 64)) == 0.8
        assert _calculate_alignment_score((3, 96)) == 0.8

    def test_divisible_by_8_returns_0_5(self) -> None:
        assert _calculate_alignment_score((10, 8)) == 0.5
        assert _calculate_alignment_score((5, 24)) == 0.5
        assert _calculate_alignment_score((3, 40)) == 0.5

    def test_unaligned_returns_0_2(self) -> None:
        assert _calculate_alignment_score((10, 7)) == 0.2
        assert _calculate_alignment_score((5, 13)) == 0.2
        assert _calculate_alignment_score((3, 3)) == 0.2

    def test_uses_last_dimension(self) -> None:
        # Last dim is 128 -> 1.0, regardless of earlier dims
        assert _calculate_alignment_score((7, 13, 128)) == 1.0
        # Last dim is 7 -> 0.2, regardless of earlier dims
        assert _calculate_alignment_score((128, 256, 7)) == 0.2

    def test_single_dimension(self) -> None:
        assert _calculate_alignment_score((128,)) == 1.0
        assert _calculate_alignment_score((32,)) == 0.8
        assert _calculate_alignment_score((8,)) == 0.5
        assert _calculate_alignment_score((5,)) == 0.2


class TestMemoryTrafficIsXlasOwnFigure:
    """The intensity is XLA's FLOPs over XLA's bytes accessed, which counts intermediates."""

    @staticmethod
    def _cost(fn: Callable[..., Any], *inputs: jax.Array) -> Mapping[str, float]:
        cost = cost_mapping(jax.jit(fn).lower(*inputs).compile().cost_analysis())
        assert cost is not None
        return cost

    @pytest.mark.parametrize(
        "name",
        ["matmul", "fused_chain", "softmax"],
    )
    def test_intensity_matches_the_cost_analysis(self, name: str) -> None:
        a = jnp.ones((256, 256))
        b = jnp.ones((256, 256))
        functions = {
            "matmul": (lambda x, y: x @ y, (a, b)),
            "fused_chain": (lambda x, y: jnp.tanh(x @ y) @ y, (a, b)),
            "softmax": (lambda x, y: jax.nn.softmax(x @ y, axis=-1), (a, b)),
        }
        fn, inputs = functions[name]
        cost = self._cost(fn, *inputs)
        expected = cost["flops"] / cost["bytes accessed"]

        result = RooflineAnalyzer(
            hardware_specs=HardwareSpec(name="t", peak_flops=1.0e12, memory_bandwidth=1.0e11)
        ).analyze_operation(fn, list(inputs))

        assert result.arithmetic_intensity == pytest.approx(expected, rel=1e-6)

    def test_the_flops_counter_reports_the_bytes_too(self) -> None:
        a = jnp.ones((128, 128))
        counted = FlopsCounter().count(lambda x: jnp.tanh(x @ x) @ x, a, optimized=True)

        assert counted.bytes_accessed == pytest.approx(
            self._cost(lambda x: jnp.tanh(x @ x) @ x, a)["bytes accessed"]
        )
