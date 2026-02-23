"""Tests for JIT compilation profiler.

Verifies CompilationResult and XLAOptimizationResult dataclass immutability
and serialization, CompilationProfiler cache hit/miss tracking, reset
behavior, and XLA optimization estimation.
"""

import builtins
import dataclasses
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from calibrax.profiling.compilation import (
    _block_result,
    _parse_hlo_instruction,
    _safe_ratio,
    CompilationProfiler,
    CompilationResult,
    XLAOptimizationResult,
)


class TestCompilationResult:
    """Tests for CompilationResult frozen dataclass."""

    def test_creation_with_all_fields(self) -> None:
        result = CompilationResult(
            cache_hit_rate=0.75,
            total_calls=20,
            cache_hits=15,
            cache_misses=5,
            avg_compilation_time_ms=120.0,
            max_compilation_time_ms=350.0,
            unique_signatures=5,
            health_score=0.85,
            health_level="excellent",
            recommendations=("All good.",),
        )
        assert result.cache_hit_rate == 0.75
        assert result.total_calls == 20
        assert result.cache_hits == 15
        assert result.cache_misses == 5
        assert result.avg_compilation_time_ms == 120.0
        assert result.max_compilation_time_ms == 350.0
        assert result.unique_signatures == 5
        assert result.health_score == 0.85
        assert result.health_level == "excellent"
        assert result.recommendations == ("All good.",)

    def test_frozen_immutability(self) -> None:
        result = CompilationResult(
            cache_hit_rate=0.5,
            total_calls=10,
            cache_hits=5,
            cache_misses=5,
            avg_compilation_time_ms=100.0,
            max_compilation_time_ms=200.0,
            unique_signatures=3,
            health_score=0.6,
            health_level="good",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.cache_hit_rate = 1.0  # type: ignore[misc]

    def test_kw_only_enforcement(self) -> None:
        with pytest.raises(TypeError):
            CompilationResult(0.5, 10, 5, 5, 100.0, 200.0, 3, 0.6, "good")  # type: ignore[misc]

    def test_recommendations_default_empty_tuple(self) -> None:
        result = CompilationResult(
            cache_hit_rate=0.9,
            total_calls=100,
            cache_hits=90,
            cache_misses=10,
            avg_compilation_time_ms=50.0,
            max_compilation_time_ms=150.0,
            unique_signatures=10,
            health_score=0.9,
            health_level="excellent",
        )
        assert result.recommendations == ()

    def test_to_dict(self) -> None:
        result = CompilationResult(
            cache_hit_rate=0.8,
            total_calls=50,
            cache_hits=40,
            cache_misses=10,
            avg_compilation_time_ms=80.0,
            max_compilation_time_ms=250.0,
            unique_signatures=8,
            health_score=0.75,
            health_level="good",
            recommendations=("rec1",),
        )
        d = result.to_dict()
        assert d["cache_hit_rate"] == 0.8
        assert d["total_calls"] == 50
        assert d["health_level"] == "good"
        assert d["recommendations"] == ["rec1"]
        assert isinstance(d["recommendations"], list)

    def test_from_dict(self) -> None:
        data = {
            "cache_hit_rate": 0.6,
            "total_calls": 30,
            "cache_hits": 18,
            "cache_misses": 12,
            "avg_compilation_time_ms": 200.0,
            "max_compilation_time_ms": 500.0,
            "unique_signatures": 12,
            "health_score": 0.55,
            "health_level": "moderate",
            "recommendations": ["fix shapes"],
        }
        result = CompilationResult.from_dict(data)
        assert result.cache_hit_rate == 0.6
        assert result.total_calls == 30
        assert result.health_level == "moderate"
        assert result.recommendations == ("fix shapes",)

    def test_to_dict_from_dict_round_trip(self) -> None:
        original = CompilationResult(
            cache_hit_rate=0.85,
            total_calls=40,
            cache_hits=34,
            cache_misses=6,
            avg_compilation_time_ms=95.0,
            max_compilation_time_ms=300.0,
            unique_signatures=6,
            health_score=0.82,
            health_level="excellent",
            recommendations=("good job", "keep going"),
        )
        restored = CompilationResult.from_dict(original.to_dict())
        assert restored.cache_hit_rate == original.cache_hit_rate
        assert restored.total_calls == original.total_calls
        assert restored.cache_hits == original.cache_hits
        assert restored.cache_misses == original.cache_misses
        assert restored.health_level == original.health_level
        assert restored.recommendations == original.recommendations

    def test_from_dict_missing_recommendations_defaults_empty(self) -> None:
        data = {
            "cache_hit_rate": 0.0,
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_compilation_time_ms": 0.0,
            "max_compilation_time_ms": 0.0,
            "unique_signatures": 0,
            "health_score": 0.5,
            "health_level": "moderate",
        }
        result = CompilationResult.from_dict(data)
        assert result.recommendations == ()


class TestXLAOptimizationResult:
    """Tests for XLAOptimizationResult frozen dataclass."""

    def test_creation_with_all_fields(self) -> None:
        result = XLAOptimizationResult(
            optimization_score=0.7,
            fusion_ratio=0.4,
            arithmetic_ratio=0.5,
            memory_ratio=0.1,
            total_kernels=20,
            recommendations=("Good fusion.",),
        )
        assert result.optimization_score == 0.7
        assert result.fusion_ratio == 0.4
        assert result.arithmetic_ratio == 0.5
        assert result.memory_ratio == 0.1
        assert result.total_kernels == 20
        assert result.recommendations == ("Good fusion.",)

    def test_frozen_immutability(self) -> None:
        result = XLAOptimizationResult(
            optimization_score=0.5,
            fusion_ratio=0.3,
            arithmetic_ratio=0.4,
            memory_ratio=0.3,
            total_kernels=10,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.optimization_score = 1.0  # type: ignore[misc]

    def test_recommendations_default_empty_tuple(self) -> None:
        result = XLAOptimizationResult(
            optimization_score=0.5,
            fusion_ratio=0.3,
            arithmetic_ratio=0.4,
            memory_ratio=0.3,
            total_kernels=10,
        )
        assert result.recommendations == ()

    def test_to_dict(self) -> None:
        result = XLAOptimizationResult(
            optimization_score=0.65,
            fusion_ratio=0.35,
            arithmetic_ratio=0.45,
            memory_ratio=0.2,
            total_kernels=15,
            recommendations=("rec1", "rec2"),
        )
        d = result.to_dict()
        assert d["optimization_score"] == 0.65
        assert d["total_kernels"] == 15
        assert d["recommendations"] == ["rec1", "rec2"]

    def test_from_dict(self) -> None:
        data = {
            "optimization_score": 0.8,
            "fusion_ratio": 0.5,
            "arithmetic_ratio": 0.6,
            "memory_ratio": 0.1,
            "total_kernels": 25,
            "recommendations": ["optimize more"],
        }
        result = XLAOptimizationResult.from_dict(data)
        assert result.optimization_score == 0.8
        assert result.total_kernels == 25
        assert result.recommendations == ("optimize more",)

    def test_to_dict_from_dict_round_trip(self) -> None:
        original = XLAOptimizationResult(
            optimization_score=0.72,
            fusion_ratio=0.45,
            arithmetic_ratio=0.55,
            memory_ratio=0.15,
            total_kernels=30,
            recommendations=("fuse more", "batch ops"),
        )
        restored = XLAOptimizationResult.from_dict(original.to_dict())
        assert restored.optimization_score == original.optimization_score
        assert restored.fusion_ratio == original.fusion_ratio
        assert restored.total_kernels == original.total_kernels
        assert restored.recommendations == original.recommendations

    def test_from_dict_missing_recommendations_defaults_empty(self) -> None:
        data = {
            "optimization_score": 0.0,
            "fusion_ratio": 0.0,
            "arithmetic_ratio": 0.0,
            "memory_ratio": 0.0,
            "total_kernels": 0,
        }
        result = XLAOptimizationResult.from_dict(data)
        assert result.recommendations == ()


class TestCompilationProfiler:
    """Tests for CompilationProfiler JIT profiling."""

    def test_profile_jit_compilation_returns_callable(self) -> None:
        profiler = CompilationProfiler()
        fn = profiler.profile_jit_compilation(lambda x: x + 1)
        assert callable(fn)

    def test_profiled_function_produces_correct_result(self) -> None:
        profiler = CompilationProfiler()
        fn = profiler.profile_jit_compilation(lambda x: x + 1)
        x = jnp.array([1.0, 2.0, 3.0])
        result = fn(x)

        expected = jnp.array([2.0, 3.0, 4.0])
        assert jnp.allclose(result, expected)

    def test_cache_miss_on_first_call(self) -> None:
        profiler = CompilationProfiler()
        fn = profiler.profile_jit_compilation(lambda x: x * 2)
        x = jnp.ones((4,))
        fn(x)

        result = profiler.get_result()
        assert isinstance(result, CompilationResult)
        assert result.cache_misses >= 1
        assert result.total_calls >= 1

    def test_cache_hit_on_second_call_same_shape(self) -> None:
        profiler = CompilationProfiler()
        fn = profiler.profile_jit_compilation(lambda x: x + 1)
        x = jnp.ones((4,))

        fn(x)  # First call -> cache miss
        fn(x)  # Second call -> cache hit (same shape/dtype)

        result = profiler.get_result()
        assert result.cache_hits >= 1
        assert result.cache_misses >= 1
        assert result.total_calls >= 2

    def test_cache_miss_on_different_shapes(self) -> None:
        profiler = CompilationProfiler()
        fn = profiler.profile_jit_compilation(lambda x: x + 1)

        fn(jnp.ones((4,)))  # shape (4,)
        fn(jnp.ones((8,)))  # shape (8,) -> different signature

        result = profiler.get_result()
        assert result.cache_misses >= 2
        assert result.unique_signatures >= 2

    def test_get_result_returns_compilation_result(self) -> None:
        profiler = CompilationProfiler()
        fn = profiler.profile_jit_compilation(lambda x: x + 1)
        fn(jnp.ones((2,)))

        result = profiler.get_result()
        assert isinstance(result, CompilationResult)
        assert result.total_calls > 0
        assert result.health_score >= 0
        assert result.health_level in ("excellent", "good", "moderate", "poor")

    def test_get_result_empty_profiler(self) -> None:
        profiler = CompilationProfiler()
        result = profiler.get_result()

        assert result.total_calls == 0
        assert result.cache_hits == 0
        assert result.cache_misses == 0
        assert result.cache_hit_rate == 0.0

    def test_reset_clears_state(self) -> None:
        profiler = CompilationProfiler()
        fn = profiler.profile_jit_compilation(lambda x: x + 1)
        fn(jnp.ones((4,)))
        fn(jnp.ones((4,)))

        pre_reset = profiler.get_result()
        assert pre_reset.total_calls >= 2

        profiler.reset()

        post_reset = profiler.get_result()
        assert post_reset.total_calls == 0
        assert post_reset.cache_hits == 0
        assert post_reset.cache_misses == 0
        assert post_reset.unique_signatures == 0

    def test_cache_hit_rate_calculation(self) -> None:
        profiler = CompilationProfiler()
        fn = profiler.profile_jit_compilation(lambda x: x * 3)
        x = jnp.ones((4,))

        fn(x)  # miss
        fn(x)  # hit
        fn(x)  # hit
        fn(x)  # hit

        result = profiler.get_result()
        assert result.cache_hit_rate == pytest.approx(0.75)

    def test_compilation_time_non_negative(self) -> None:
        profiler = CompilationProfiler()
        fn = profiler.profile_jit_compilation(lambda x: x + 1)
        fn(jnp.ones((4,)))

        result = profiler.get_result()
        assert result.avg_compilation_time_ms >= 0
        assert result.max_compilation_time_ms >= 0

    def test_health_score_between_zero_and_one(self) -> None:
        profiler = CompilationProfiler()
        fn = profiler.profile_jit_compilation(lambda x: x + 1)
        fn(jnp.ones((4,)))
        fn(jnp.ones((4,)))

        result = profiler.get_result()
        assert 0.0 <= result.health_score <= 1.0

    def test_warmup_runtime_error_logs_and_reraises(self, caplog: pytest.LogCaptureFixture) -> None:
        profiler = CompilationProfiler()
        with patch("calibrax.profiling.compilation.jax") as mock_jax:
            mock_jax.jit.return_value = MagicMock(side_effect=RuntimeError("warmup compile failed"))
            fn = profiler.profile_jit_compilation(lambda x: x)

            with caplog.at_level("WARNING"):
                with pytest.raises(RuntimeError, match="warmup compile failed"):
                    fn(jnp.ones((2,)))

        assert any("Compilation failed after" in msg for msg in caplog.messages)

    def test_warmup_unexpected_error_propagates_without_wrapper_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class CatastrophicCompileFailure(Exception):
            pass

        profiler = CompilationProfiler()
        with patch("calibrax.profiling.compilation.jax") as mock_jax:
            mock_jax.jit.return_value = MagicMock(
                side_effect=CatastrophicCompileFailure("unexpected compile failure")
            )
            fn = profiler.profile_jit_compilation(lambda x: x)

            with caplog.at_level("WARNING"):
                with pytest.raises(
                    CatastrophicCompileFailure,
                    match="unexpected compile failure",
                ):
                    fn(jnp.ones((2,)))

        assert all("Compilation failed after" not in msg for msg in caplog.messages)


class TestCompilationProfilerXLAOptimization:
    """Tests for estimate_xla_optimization."""

    def test_returns_xla_optimization_result(self) -> None:
        profiler = CompilationProfiler()
        x = jnp.ones((4, 4))
        result = profiler.estimate_xla_optimization(lambda a: a + 1, x)

        assert isinstance(result, XLAOptimizationResult)

    def test_total_kernels_positive_for_valid_function(self) -> None:
        profiler = CompilationProfiler()
        x = jnp.ones((4, 4))
        result = profiler.estimate_xla_optimization(lambda a: a + 1, x)

        assert result.total_kernels > 0

    def test_scores_between_zero_and_one(self) -> None:
        profiler = CompilationProfiler()
        x = jnp.ones((8, 8))
        result = profiler.estimate_xla_optimization(lambda a: a * 2 + 1, x)

        assert 0.0 <= result.optimization_score <= 1.0
        assert 0.0 <= result.fusion_ratio <= 1.0
        assert 0.0 <= result.arithmetic_ratio <= 1.0
        assert 0.0 <= result.memory_ratio <= 1.0

    def test_has_recommendations(self) -> None:
        profiler = CompilationProfiler()
        x = jnp.ones((4,))
        result = profiler.estimate_xla_optimization(lambda a: a + 1, x)

        assert isinstance(result.recommendations, tuple)
        assert len(result.recommendations) > 0

    def test_graceful_fallback_on_failure(self) -> None:
        profiler = CompilationProfiler()

        with patch("calibrax.profiling.compilation.jax") as mock_jax:
            mock_jit = MagicMock()
            mock_jit.lower.side_effect = RuntimeError("XLA failure")
            mock_jax.jit.return_value = mock_jit

            result = profiler.estimate_xla_optimization(lambda a: a, "bad_input")

        assert result.optimization_score == 0.0
        assert result.total_kernels == 0
        assert len(result.recommendations) > 0

    def test_unexpected_xla_error_is_not_swallowed(self) -> None:
        class CatastrophicXLAFailure(Exception):
            pass

        profiler = CompilationProfiler()

        with patch("calibrax.profiling.compilation.jax") as mock_jax:
            mock_jit = MagicMock()
            mock_jit.lower.side_effect = CatastrophicXLAFailure("unexpected XLA failure")
            mock_jax.jit.return_value = mock_jit

            with pytest.raises(CatastrophicXLAFailure, match="unexpected XLA failure"):
                profiler.estimate_xla_optimization(lambda a: a, "bad_input")

    def test_analyze_hlo_counts_only_instruction_lines(self) -> None:
        """Kernel counts should reflect HLO instructions, not all text lines."""
        profiler = CompilationProfiler()
        hlo_text = """
HloModule test

ENTRY main {
  %x = f32[4]{0} parameter(0)
  %y = f32[4]{0} parameter(1)
  %add = f32[4]{0} add(%x, %y)
  ROOT %out = f32[4]{0} copy(%add)
}
"""
        result = profiler._analyze_hlo(hlo_text)

        assert result.total_kernels == 4
        assert result.arithmetic_ratio == pytest.approx(0.25)
        assert result.memory_ratio == pytest.approx(0.25)


class TestCompilationProfilerAdditionalBranches:
    """Additional branch-coverage tests for compilation helpers."""

    def test_parse_hlo_instruction_returns_none_for_empty_rhs(self) -> None:
        """Lines with no right-hand instruction should be ignored."""
        assert _parse_hlo_instruction("%x = // comment-only") is None

    def test_safe_ratio_with_zero_total_returns_zero(self) -> None:
        """_safe_ratio should guard division by zero totals."""
        assert _safe_ratio(3, 0) == 0.0

    def test_create_function_signature_uses_python_type_for_non_arrays(self) -> None:
        """Non-array args should be represented by type name in signature."""
        profiler = CompilationProfiler()

        sig_obj = profiler._create_function_signature(lambda x: x, (object(),), {})
        sig_str = profiler._create_function_signature(lambda x: x, ("value",), {})

        assert sig_obj != sig_str

    def test_generate_recommendations_moderate_paths(self) -> None:
        """Moderate cache hit and compile time should generate moderate advice."""
        profiler = CompilationProfiler()
        recs = profiler._generate_recommendations(cache_hit_rate=0.7, avg_compilation_time=3.0)

        assert any("Moderate cache hit rate" in rec for rec in recs)
        assert any("Moderate compilation time" in rec for rec in recs)

    def test_generate_recommendations_high_paths(self) -> None:
        """Low cache hit and high compile time should trigger stronger advice."""
        profiler = CompilationProfiler()
        recs = profiler._generate_recommendations(cache_hit_rate=0.2, avg_compilation_time=6.0)

        assert any("Low cache hit rate" in rec for rec in recs)
        assert any("High average compilation time" in rec for rec in recs)

    def test_generate_recommendations_empty_for_healthy_inputs(self) -> None:
        """Healthy cache hit and compile time should not emit recommendations."""
        profiler = CompilationProfiler()
        recs = profiler._generate_recommendations(cache_hit_rate=0.95, avg_compilation_time=0.5)

        assert recs == []

    def test_assess_health_reports_poor_at_low_score(self) -> None:
        """Very low score should map to the 'poor' health label."""
        profiler = CompilationProfiler()
        score, level = profiler._assess_health(cache_hit_rate=0.0, avg_compilation_time=10.0)

        assert score == pytest.approx(0.0)
        assert level == "poor"

    def test_generate_xla_recommendations_includes_all_issue_signals(self) -> None:
        """Low fusion/intensity and high memory ratio should produce all warnings."""
        profiler = CompilationProfiler()
        recs = profiler._generate_xla_recommendations(
            fusion_ratio=0.1,
            arithmetic_ratio=0.2,
            memory_ratio=0.8,
        )

        assert any("Low fusion ratio" in rec for rec in recs)
        assert any("Low arithmetic intensity" in rec for rec in recs)
        assert any("High memory operation ratio" in rec for rec in recs)

    def test_generate_xla_recommendations_effective_default(self) -> None:
        """Good ratios should return the default positive recommendation."""
        profiler = CompilationProfiler()
        recs = profiler._generate_xla_recommendations(
            fusion_ratio=0.5,
            arithmetic_ratio=0.5,
            memory_ratio=0.2,
        )

        assert recs == ["XLA optimizations appear effective."]

    def test_block_result_materializes_sequence_items(self) -> None:
        """_block_result should recurse through sequence outputs."""

        class _Blockable:
            def __init__(self) -> None:
                self.calls = 0

            def block_until_ready(self) -> None:
                self.calls += 1

        first = _Blockable()
        second = object()

        _block_result([first, second])
        _block_result(123)

        assert first.calls == 1

    def test_module_import_falls_back_without_jax_runtime_error_attr(self) -> None:
        """Import guard should use RuntimeError when jax.errors is unavailable."""
        import calibrax.profiling.compilation as compilation_mod

        module_path = Path(compilation_mod.__file__)
        spec = importlib.util.spec_from_file_location("compilation_import_probe", module_path)
        assert spec is not None
        assert spec.loader is not None
        probe_module = importlib.util.module_from_spec(spec)

        fake_jax = types.ModuleType("jax")
        real_import = builtins.__import__

        def _import_hook(name: str, *args: object, **kwargs: object) -> object:
            if name == "jax":
                return fake_jax
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_hook):
            sys.modules[spec.name] = probe_module
            try:
                spec.loader.exec_module(probe_module)
            finally:
                sys.modules.pop(spec.name, None)

        assert probe_module._JAX_RUNTIME_ERROR is RuntimeError
