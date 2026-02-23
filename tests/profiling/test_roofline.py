"""Tests for roofline analysis.

Verifies RooflineResult dataclass immutability and serialization,
RooflineAnalyzer operation analysis with flops_override, bottleneck
classification, and alignment score calculation.
"""

import dataclasses

import jax.numpy as jnp
import pytest

from calibrax.profiling.roofline import (
    _calculate_alignment_score,
    _extract_flops_from_cost,
    _try_xla_cost_analysis,
    RooflineAnalyzer,
    RooflineResult,
)


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
        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 5.0,
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        x = jnp.ones((64, 64))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=1_000_000)

        assert isinstance(result, RooflineResult)
        assert result.execution_time_ms > 0
        assert result.critical_intensity == 5.0

    def test_bottleneck_is_valid_value(self) -> None:
        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 5.0,
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        x = jnp.ones((32, 32))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=100)

        assert result.bottleneck in ("memory_bandwidth", "compute")

    def test_memory_bound_when_low_intensity(self) -> None:
        """Low flops_override with large arrays -> low arithmetic intensity -> memory bound."""
        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 1000.0,  # Very high threshold
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        x = jnp.ones((128, 128))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=1)

        assert result.bottleneck == "memory_bandwidth"

    def test_compute_bound_when_high_intensity(self) -> None:
        """Very high flops_override with small arrays -> high arithmetic intensity -> compute."""
        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 0.001,  # Very low threshold
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        x = jnp.ones((2,))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=10_000_000)

        assert result.bottleneck == "compute"

    def test_result_has_recommendations(self) -> None:
        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 5.0,
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        x = jnp.ones((16, 16))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=100)

        assert isinstance(result.recommendations, tuple)
        assert len(result.recommendations) > 0

    def test_utilization_values_non_negative(self) -> None:
        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 5.0,
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        x = jnp.ones((8, 8))
        result = analyzer.analyze_operation(jnp.add, [x, x], flops_override=500)

        assert result.memory_bandwidth_utilization >= 0
        assert result.flops_utilization >= 0
        assert result.efficiency >= 0

    def test_default_hardware_specs_auto_detected(self) -> None:
        """Default factory calls detect_hardware_specs for auto-detection."""
        from calibrax.profiling.hardware import detect_hardware_specs

        analyzer = RooflineAnalyzer()
        expected = detect_hardware_specs()
        assert analyzer.hardware_specs["peak_flops"] == expected["peak_flops"]
        assert analyzer.hardware_specs["memory_bandwidth"] == expected["memory_bandwidth"]
        assert analyzer.hardware_specs["critical_intensity"] == expected["critical_intensity"]

    def test_extract_flops_from_cost_handles_dict_and_list(self) -> None:
        assert _extract_flops_from_cost({"flops": 123.0}) == 123
        assert _extract_flops_from_cost([{"flops": 456.0}]) == 456
        assert _extract_flops_from_cost({"flops": 0.0}) is None
        assert _extract_flops_from_cost([{"flops": -1.0}]) is None
        assert _extract_flops_from_cost([]) is None
        assert _extract_flops_from_cost(None) is None

    def test_try_xla_cost_analysis_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Lowered:
            def cost_analysis(self) -> dict[str, float]:
                return {"flops": 321.0}

        class _Jitted:
            def lower(self, *_args: object) -> _Lowered:
                return _Lowered()

        monkeypatch.setattr("calibrax.profiling.roofline.jax.jit", lambda _func: _Jitted())
        flops = _try_xla_cost_analysis(lambda x: x, [jnp.ones((1,))])
        assert flops == 321

    def test_try_xla_cost_analysis_fallback_on_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _Jitted:
            def lower(self, *_args: object) -> object:
                raise RuntimeError("lowering failed")

        monkeypatch.setattr("calibrax.profiling.roofline.jax.jit", lambda _func: _Jitted())
        flops = _try_xla_cost_analysis(lambda x: x, [jnp.ones((1,))])
        assert flops is None

    def test_estimate_flops_uses_heuristic_when_xla_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _FakeArray:
            def __init__(self, size: int) -> None:
                self.size = size

        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 5.0,
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        monkeypatch.setattr("calibrax.profiling.roofline._try_xla_cost_analysis", lambda *_: None)

        flops = analyzer._estimate_flops(lambda *_: None, [_FakeArray(3), _FakeArray(7)])  # type: ignore[arg-type]
        assert flops == 100

    def test_estimate_memory_traffic_handles_tuple_outputs(self) -> None:
        class _FakeArray:
            def __init__(self, *, nbytes: int, shape: tuple[int, ...] = (16,)) -> None:
                self.nbytes = nbytes
                self.shape = shape

        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 5.0,
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        inputs = [_FakeArray(nbytes=8), _FakeArray(nbytes=12)]

        def _func(*_args: object) -> tuple[object, object, object]:
            return _FakeArray(nbytes=20), object(), _FakeArray(nbytes=4)

        traffic = analyzer._estimate_memory_traffic(_func, inputs)  # type: ignore[arg-type]
        assert traffic == 44

    def test_estimate_memory_traffic_fallback_when_func_raises(self) -> None:
        class _FakeArray:
            def __init__(self, *, nbytes: int, shape: tuple[int, ...] = (16,)) -> None:
                self.nbytes = nbytes
                self.shape = shape

        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 5.0,
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        inputs = [_FakeArray(nbytes=8), _FakeArray(nbytes=12)]

        def _func(*_args: object) -> object:
            raise RuntimeError("cannot execute")

        traffic = analyzer._estimate_memory_traffic(_func, inputs)  # type: ignore[arg-type]
        assert traffic == 40

    def test_generate_recommendations_includes_moderate_efficiency_message(self) -> None:
        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 5.0,
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        x = jnp.ones((2, 32))
        recs = analyzer._generate_recommendations(
            arithmetic_intensity=10.0,
            efficiency=0.3,
            bottleneck="compute",
            inputs=[x],
            achieved_flops=1e9,
        )
        assert any("Moderate efficiency" in rec for rec in recs)

    def test_generate_recommendations_adds_alignment_hint_for_unaligned_inputs(self) -> None:
        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 5.0,
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        x = jnp.ones((2, 7))
        recs = analyzer._generate_recommendations(
            arithmetic_intensity=2.0,
            efficiency=0.9,
            bottleneck="memory_bandwidth",
            inputs=[x],
            achieved_flops=1e9,
        )
        assert any("Poor tensor alignment" in rec for rec in recs)

    def test_generate_recommendations_skips_alignment_for_empty_inputs(self) -> None:
        specs = {
            "peak_flops": 1e12,
            "memory_bandwidth": 200e9,
            "critical_intensity": 5.0,
        }
        analyzer = RooflineAnalyzer(hardware_specs=specs)
        recs = analyzer._generate_recommendations(
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
