"""JIT compilation profiler for JAX.

Analyzes JIT compilation efficiency, cache hit rates, XLA optimization
effectiveness, and provides recommendations for compilation optimization.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax


logger = logging.getLogger(__name__)

_HLO_OPCODE_RE = re.compile(r"\b([A-Za-z_][\w.\-]*)\s*\(")
_FUSION_KIND_PATTERNS = ("kloop", "kinput", "koutput")
_MEMORY_OPS = ("copy", "transpose", "reshape", "broadcast")
_ARITHMETIC_OPS = ("add", "multiply", "dot", "convolution", "reduce")

try:
    _JAX_RUNTIME_ERROR: type[BaseException] = jax.errors.JaxRuntimeError
except AttributeError:
    _JAX_RUNTIME_ERROR = RuntimeError

_RECOVERABLE_COMPILATION_ERRORS = (
    _JAX_RUNTIME_ERROR,
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _parse_hlo_instruction(line: str) -> tuple[str, str] | None:
    """Parse an HLO instruction line into (opcode, rhs_lower).

    Returns None for non-instruction lines (module headers, braces, etc.).
    """
    stripped = line.strip()
    if not stripped or "=" not in stripped:
        return None

    # Drop trailing comments and analyze RHS of assignment.
    rhs = stripped.split("=", 1)[1].split("//", 1)[0].strip()
    if not rhs:
        return None

    match = _HLO_OPCODE_RE.search(rhs)
    if match is None:
        return None

    return match.group(1).lower(), rhs.lower()


def _extract_hlo_instructions(hlo_text: str) -> list[tuple[str, str]]:
    """Extract parsed HLO instructions from raw text."""
    instructions: list[tuple[str, str]] = []
    for line in hlo_text.splitlines():
        parsed = _parse_hlo_instruction(line)
        if parsed is not None:
            instructions.append(parsed)
    return instructions


def _classify_instruction(opcode: str, rhs_lower: str) -> tuple[int, int, int]:
    """Classify one HLO instruction into fused/memory/arithmetic counters."""
    is_fused = int(
        "fusion" in opcode or any(kind in rhs_lower for kind in _FUSION_KIND_PATTERNS),
    )
    is_memory = int(any(op in opcode for op in _MEMORY_OPS))
    is_arithmetic = int(any(op in opcode for op in _ARITHMETIC_OPS))
    return is_fused, is_memory, is_arithmetic


def _safe_ratio(count: int, total: int) -> float:
    """Return count/total, or 0.0 when total is zero."""
    if total == 0:
        return 0.0
    return count / total


@dataclass(frozen=True, slots=True, kw_only=True)
class CompilationResult:
    """Result of compilation profiling analysis.

    Attributes:
        cache_hit_rate: Fraction of calls that hit the compilation cache.
        total_calls: Total number of profiled function calls.
        cache_hits: Number of cache hits.
        cache_misses: Number of cache misses (triggering compilation).
        avg_compilation_time_ms: Average compilation time in milliseconds.
        max_compilation_time_ms: Maximum compilation time in milliseconds.
        unique_signatures: Number of unique function signatures compiled.
        health_score: Overall compilation health score (0-1).
        health_level: Human-readable health level.
        recommendations: Optimization recommendations.
    """

    cache_hit_rate: float
    total_calls: int
    cache_hits: int
    cache_misses: int
    avg_compilation_time_ms: float
    max_compilation_time_ms: float
    unique_signatures: int
    health_score: float
    health_level: str
    recommendations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "cache_hit_rate": float(self.cache_hit_rate),
            "total_calls": int(self.total_calls),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "avg_compilation_time_ms": float(self.avg_compilation_time_ms),
            "max_compilation_time_ms": float(self.max_compilation_time_ms),
            "unique_signatures": int(self.unique_signatures),
            "health_score": float(self.health_score),
            "health_level": self.health_level,
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompilationResult:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with compilation result fields.

        Returns:
            Reconstructed CompilationResult instance.
        """
        return cls(
            cache_hit_rate=data["cache_hit_rate"],
            total_calls=data["total_calls"],
            cache_hits=data["cache_hits"],
            cache_misses=data["cache_misses"],
            avg_compilation_time_ms=data["avg_compilation_time_ms"],
            max_compilation_time_ms=data["max_compilation_time_ms"],
            unique_signatures=data["unique_signatures"],
            health_score=data["health_score"],
            health_level=data["health_level"],
            recommendations=tuple(data.get("recommendations", ())),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class XLAOptimizationResult:
    """Result of XLA optimization effectiveness analysis.

    Attributes:
        optimization_score: Overall optimization score (0-1).
        fusion_ratio: Fraction of fused kernels.
        arithmetic_ratio: Fraction of arithmetic operations.
        memory_ratio: Fraction of memory operations.
        total_kernels: Total number of HLO kernels.
        recommendations: Optimization recommendations.
    """

    optimization_score: float
    fusion_ratio: float
    arithmetic_ratio: float
    memory_ratio: float
    total_kernels: int
    recommendations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "optimization_score": float(self.optimization_score),
            "fusion_ratio": float(self.fusion_ratio),
            "arithmetic_ratio": float(self.arithmetic_ratio),
            "memory_ratio": float(self.memory_ratio),
            "total_kernels": int(self.total_kernels),
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> XLAOptimizationResult:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with XLA optimization result fields.

        Returns:
            Reconstructed XLAOptimizationResult instance.
        """
        return cls(
            optimization_score=data["optimization_score"],
            fusion_ratio=data["fusion_ratio"],
            arithmetic_ratio=data["arithmetic_ratio"],
            memory_ratio=data["memory_ratio"],
            total_kernels=data["total_kernels"],
            recommendations=tuple(data.get("recommendations", ())),
        )


class CompilationProfiler:
    """Analyzes JAX JIT compilation performance and optimization.

    Instruments JIT-compiled functions to track compilation cache hits/misses,
    compilation times, and input shape consistency. Use ``profile_jit_compilation``
    to wrap a function, then call ``get_result()`` for aggregated analysis.
    """

    def __init__(self) -> None:
        """Initialize the compilation profiler with empty tracking state."""
        self._compilation_cache: dict[str, dict[str, Any]] = {}
        self._compilation_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._cache_hit_count = 0
        self._cache_miss_count = 0

    def profile_jit_compilation(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Create an instrumented wrapper that profiles JIT compilation.

        The returned callable tracks cache hits/misses, compilation times,
        and input shape patterns. Results accumulate in this profiler instance.

        Args:
            func: JAX function to instrument.

        Returns:
            Instrumented function with identical signature.
        """

        def instrumented_func(*args: Any, **kwargs: Any) -> Any:
            """Wrapper that tracks compilation cache hits and timing."""
            signature = self._create_function_signature(func, args, kwargs)
            compilation_start = time.perf_counter()

            if signature in self._compilation_cache:
                compiled_func = self._compilation_cache[signature]["compiled_func"]
                self._cache_hit_count += 1
                compilation_time = 0.0
            else:
                compiled_func = jax.jit(func)
                try:
                    warmup_result = compiled_func(*args, **kwargs)
                    _block_result(warmup_result)
                except _RECOVERABLE_COMPILATION_ERRORS:
                    compilation_time = time.perf_counter() - compilation_start
                    logger.warning(
                        "Compilation failed after %.3fs for signature %s",
                        compilation_time,
                        signature[:16],
                    )
                    raise

                compilation_time = time.perf_counter() - compilation_start
                self._cache_miss_count += 1

                self._compilation_cache[signature] = {
                    "compiled_func": compiled_func,
                    "compilation_time": compilation_time,
                    "input_shapes": [getattr(arg, "shape", None) for arg in args],
                    "input_dtypes": [getattr(arg, "dtype", None) for arg in args],
                }

                self._compilation_stats[signature].append(
                    {
                        "compilation_time": compilation_time,
                        "timestamp": time.perf_counter(),
                    }
                )

            result = compiled_func(*args, **kwargs)
            _block_result(result)
            return result

        return instrumented_func

    def get_result(self) -> CompilationResult:
        """Get aggregated compilation profiling results.

        Returns:
            CompilationResult with cache statistics, timing, and recommendations.
        """
        total_calls = self._cache_hit_count + self._cache_miss_count
        cache_hit_rate = self._cache_hit_count / total_calls if total_calls > 0 else 0.0

        all_times: list[float] = []
        for stats_list in self._compilation_stats.values():
            all_times.extend(s["compilation_time"] for s in stats_list)

        avg_time_ms = (sum(all_times) / len(all_times) * 1000) if all_times else 0.0
        max_time_ms = (max(all_times) * 1000) if all_times else 0.0

        recommendations = self._generate_recommendations(
            cache_hit_rate,
            avg_time_ms / 1000 if all_times else 0.0,
        )

        health_score, health_level = self._assess_health(cache_hit_rate, avg_time_ms / 1000)

        return CompilationResult(
            cache_hit_rate=cache_hit_rate,
            total_calls=total_calls,
            cache_hits=self._cache_hit_count,
            cache_misses=self._cache_miss_count,
            avg_compilation_time_ms=avg_time_ms,
            max_compilation_time_ms=max_time_ms,
            unique_signatures=len(self._compilation_cache),
            health_score=health_score,
            health_level=health_level,
            recommendations=tuple(recommendations),
        )

    def estimate_xla_optimization(
        self, func: Callable[..., Any], *sample_args: Any
    ) -> XLAOptimizationResult:
        """Estimate XLA optimization effectiveness by analyzing HLO text.

        Args:
            func: JAX function to analyze.
            *sample_args: Example arguments for lowering/compiling.

        Returns:
            XLAOptimizationResult with HLO analysis metrics.
        """
        try:
            jit_func = jax.jit(func)
            lowered = jit_func.lower(*sample_args)
            hlo_text = lowered.compile().as_text() or ""
            return self._analyze_hlo(hlo_text)
        except _RECOVERABLE_COMPILATION_ERRORS as e:
            logger.warning("Failed to analyze XLA optimizations: %s", e)
            return XLAOptimizationResult(
                optimization_score=0.0,
                fusion_ratio=0.0,
                arithmetic_ratio=0.0,
                memory_ratio=0.0,
                total_kernels=0,
                recommendations=("Unable to analyze XLA optimizations.",),
            )

    def reset(self) -> None:
        """Reset all profiling state."""
        self._compilation_cache.clear()
        self._compilation_stats.clear()
        self._cache_hit_count = 0
        self._cache_miss_count = 0

    def _create_function_signature(
        self, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> str:
        """Create a unique signature for function + argument shapes.

        Args:
            func: The function being called.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            MD5 hex digest identifying the function + input signature.
        """
        func_id = getattr(func, "__name__", str(func))

        arg_parts: list[str] = []
        for arg in args:
            if hasattr(arg, "shape") and hasattr(arg, "dtype"):
                arg_parts.append(f"{arg.shape}:{arg.dtype}")
            else:
                arg_parts.append(str(type(arg).__name__))

        static_kwargs = {k: v for k, v in kwargs.items() if not hasattr(v, "shape")}
        signature_str = f"{func_id}({','.join(arg_parts)}){static_kwargs}"
        return hashlib.md5(signature_str.encode(), usedforsecurity=False).hexdigest()

    def _generate_recommendations(
        self, cache_hit_rate: float, avg_compilation_time: float
    ) -> list[str]:
        """Generate compilation optimization recommendations.

        Args:
            cache_hit_rate: Fraction of cache hits.
            avg_compilation_time: Average compilation time in seconds.

        Returns:
            List of recommendation strings.
        """
        recommendations: list[str] = []

        if cache_hit_rate < 0.5:
            recommendations.extend(
                [
                    f"Low cache hit rate ({cache_hit_rate:.2%}).",
                    "Ensure consistent shapes and static arguments.",
                    "Use static_argnums for non-array arguments in jax.jit.",
                ]
            )
        elif cache_hit_rate < 0.8:
            recommendations.append(
                f"Moderate cache hit rate ({cache_hit_rate:.2%}). "
                "Fine-tune input preprocessing for better shape consistency."
            )

        if avg_compilation_time > 5.0:
            recommendations.extend(
                [
                    f"High average compilation time ({avg_compilation_time:.2f}s).",
                    "Break down large functions into smaller, composable parts.",
                    "Consider pre-compilation for critical paths.",
                ]
            )
        elif avg_compilation_time > 2.0:
            recommendations.append(
                f"Moderate compilation time ({avg_compilation_time:.2f}s). "
                "Monitor for complex control flow that may slow compilation."
            )

        return recommendations

    def _assess_health(
        self, cache_hit_rate: float, avg_compilation_time: float
    ) -> tuple[float, str]:
        """Assess overall compilation health.

        Args:
            cache_hit_rate: Fraction of cache hits.
            avg_compilation_time: Average compilation time in seconds.

        Returns:
            Tuple of (health_score, health_level).
        """
        score = 0.0
        score += cache_hit_rate * 0.5
        speed_score = max(0.0, 1.0 - (avg_compilation_time / 10.0))
        score += speed_score * 0.5
        score = min(score, 1.0)

        if score > 0.8:
            level = "excellent"
        elif score > 0.6:
            level = "good"
        elif score > 0.4:
            level = "moderate"
        else:
            level = "poor"

        return score, level

    def _analyze_hlo(self, hlo_text: str) -> XLAOptimizationResult:
        """Analyze HLO text for optimization patterns.

        Args:
            hlo_text: HLO text from compiled XLA module.

        Returns:
            XLAOptimizationResult with analysis metrics.
        """
        instructions = _extract_hlo_instructions(hlo_text)
        total_kernels = len(instructions)
        fused_kernels = 0
        memory_op_count = 0
        arithmetic_op_count = 0

        for opcode, rhs_lower in instructions:
            is_fused, is_memory, is_arithmetic = _classify_instruction(opcode, rhs_lower)
            fused_kernels += is_fused
            memory_op_count += is_memory
            arithmetic_op_count += is_arithmetic

        fusion_ratio = _safe_ratio(fused_kernels, total_kernels)
        arithmetic_ratio = _safe_ratio(arithmetic_op_count, total_kernels)
        memory_ratio = _safe_ratio(memory_op_count, total_kernels)

        opt_score = self._calculate_optimization_score(fusion_ratio, arithmetic_ratio, memory_ratio)

        recommendations = self._generate_xla_recommendations(
            fusion_ratio, arithmetic_ratio, memory_ratio
        )

        return XLAOptimizationResult(
            optimization_score=opt_score,
            fusion_ratio=fusion_ratio,
            arithmetic_ratio=arithmetic_ratio,
            memory_ratio=memory_ratio,
            total_kernels=total_kernels,
            recommendations=tuple(recommendations),
        )

    def _calculate_optimization_score(
        self, fusion_ratio: float, arithmetic_ratio: float, memory_ratio: float
    ) -> float:
        """Calculate an optimization effectiveness score (0-1).

        Args:
            fusion_ratio: Fraction of fused kernels.
            arithmetic_ratio: Fraction of arithmetic operations.
            memory_ratio: Fraction of memory operations.

        Returns:
            Score between 0 and 1.
        """
        score = 0.0
        score += min(fusion_ratio * 2, 1.0) * 0.4
        score += arithmetic_ratio * 0.3
        score += max(0.0, 1.0 - memory_ratio) * 0.3
        return min(score, 1.0)

    def _generate_xla_recommendations(
        self, fusion_ratio: float, arithmetic_ratio: float, memory_ratio: float
    ) -> list[str]:
        """Generate XLA optimization recommendations.

        Args:
            fusion_ratio: Fraction of fused kernels.
            arithmetic_ratio: Fraction of arithmetic operations.
            memory_ratio: Fraction of memory operations.

        Returns:
            List of recommendation strings.
        """
        recommendations: list[str] = []

        if fusion_ratio < 0.2:
            recommendations.extend(
                [
                    "Low fusion ratio - operations may not be well-fused.",
                    "Combine operations to enable better fusion.",
                ]
            )

        if arithmetic_ratio < 0.3:
            recommendations.extend(
                [
                    "Low arithmetic intensity.",
                    "Consider batching to improve arithmetic intensity.",
                ]
            )

        if memory_ratio > 0.5:
            recommendations.extend(
                [
                    "High memory operation ratio.",
                    "Reduce unnecessary data movement.",
                ]
            )

        if not recommendations:
            recommendations.append("XLA optimizations appear effective.")

        return recommendations


def _block_result(result: Any) -> None:
    """Block until a JAX result is materialized.

    Args:
        result: JAX computation result.
    """
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    elif isinstance(result, tuple | list):
        for item in result:
            if hasattr(item, "block_until_ready"):
                item.block_until_ready()
