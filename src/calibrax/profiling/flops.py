"""FLOP counting via JAX's jaxpr tracing.

Provides FlopsCounter for estimating FLOPs of JAX functions
by analyzing their Jaxpr intermediate representation.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
from jax._src import core as jax_core


logger = logging.getLogger(__name__)

# Elementwise operations that take 1 FLOP per element
_ELEMENTWISE_OPS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "max",
        "min",
        "sign",
        "floor",
        "ceil",
        "round",
        "clamp",
        "rem",
        "add_any",
        "mul_p",
        "integer_pow",
    }
)

# Transcendental ops (also 1 FLOP per element for counting purposes)
_TRANSCENDENTAL_OPS = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "tanh",
        "sinh",
        "cosh",
        "asin",
        "acos",
        "atan",
        "log1p",
        "expm1",
        "rsqrt",
        "erf",
        "erfc",
        "logistic",
    }
)

# Comparison ops (1 FLOP per element)
_COMPARISON_OPS = frozenset(
    {
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "select_n",
    }
)

# Reduction ops (product of input shape)
_REDUCTION_OPS = frozenset(
    {
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "reduce_and",
        "reduce_or",
    }
)

# Zero-FLOP structural operations
_ZERO_FLOP_OPS = frozenset(
    {
        "broadcast_in_dim",
        "convert_element_type",
        "reshape",
        "transpose",
        "concatenate",
        "slice",
        "squeeze",
        "iota",
    }
)

# Operations with nested Jaxprs
_NESTED_OPS = frozenset({"pjit", "xla_call", "scan", "while", "cond"})


@dataclass(frozen=True, slots=True, kw_only=True)
class FlopsResult:
    """Result of FLOP counting for a function.

    Attributes:
        total_flops: Total estimated FLOPs.
        flops_by_operation: Breakdown by primitive operation name.
        num_operations: Number of JAX primitives in the trace.
        function_name: Name of the analyzed function.
    """

    total_flops: int
    flops_by_operation: dict[str, int]
    num_operations: int
    function_name: str


class FlopsCounter:
    """Count FLOPs of JAX functions via jaxpr analysis.

    Uses ``jax.make_jaxpr`` to trace the function and counts FLOPs
    for each primitive based on operation-specific rules.

    For NNX models that use stochastic operations (dropout, etc.),
    use ``flax.nnx.tabulate(model, *args, compute_flops=True)``
    instead — it handles NNX state management internally.
    """

    def count(
        self,
        fn: Callable[..., Any],
        *args: Any,
        static_argnums: tuple[int, ...] = (),
    ) -> FlopsResult:
        """Count FLOPs for a function with given example arguments.

        Args:
            fn: JAX function to analyze.
            *args: Example arguments for tracing.
            static_argnums: Argument indices to treat as static.

        Returns:
            FlopsResult with FLOP count and breakdown.
        """
        trace_args = [arg for i, arg in enumerate(args) if i not in static_argnums]
        static_vals = {i: args[i] for i in static_argnums}

        if static_vals:

            def wrapped(*dynamic_args: Any) -> Any:
                """Re-insert static arguments and call the original function."""
                full_args = list(dynamic_args)
                for idx in sorted(static_vals.keys()):
                    full_args.insert(idx, static_vals[idx])
                return fn(*full_args)

            jaxpr = jax.make_jaxpr(wrapped)(*trace_args)
        else:
            jaxpr = jax.make_jaxpr(fn)(*trace_args)

        flops_by_op: dict[str, int] = {}
        total = self._count_jaxpr(jaxpr.jaxpr, flops_by_op)

        return FlopsResult(
            total_flops=total,
            flops_by_operation=dict(flops_by_op),
            num_operations=len(jaxpr.jaxpr.eqns),
            function_name=fn.__name__,
        )

    def _count_jaxpr(
        self,
        jaxpr: jax_core.Jaxpr,
        flops_by_op: dict[str, int],
    ) -> int:
        """Recursively count FLOPs in a Jaxpr.

        Args:
            jaxpr: The Jaxpr to analyze.
            flops_by_op: Accumulator for per-operation FLOP counts.

        Returns:
            Total FLOPs in this Jaxpr.
        """
        total = 0
        for eqn in jaxpr.eqns:
            flops = self._count_eqn(eqn, flops_by_op)
            total += flops
        return total

    def _count_eqn(
        self,
        eqn: jax_core.JaxprEqn,
        flops_by_op: dict[str, int],
    ) -> int:
        """Count FLOPs for a single Jaxpr equation.

        Args:
            eqn: The equation to analyze.
            flops_by_op: Accumulator for per-operation FLOP counts.

        Returns:
            FLOPs for this equation.
        """
        name = eqn.primitive.name
        flops = self._classify_primitive_flops(name, eqn, flops_by_op)

        if flops > 0:
            flops_by_op[name] = flops_by_op.get(name, 0) + flops

        return flops

    def _classify_primitive_flops(
        self,
        name: str,
        eqn: jax_core.JaxprEqn,
        flops_by_op: dict[str, int],
    ) -> int:
        """Dispatch primitive to appropriate FLOP counting strategy.

        Args:
            name: Primitive operation name.
            eqn: The Jaxpr equation.
            flops_by_op: Accumulator for nested operations.

        Returns:
            Estimated FLOPs for this primitive.
        """
        if name == "dot_general":
            return self._count_dot_general(eqn)
        if name == "conv_general_dilated":
            return self._count_conv(eqn)
        if name in _ELEMENTWISE_OPS or name in _TRANSCENDENTAL_OPS or name in _COMPARISON_OPS:
            return self._output_size(eqn)
        if name in _REDUCTION_OPS:
            return self._input_size(eqn)
        if name in _ZERO_FLOP_OPS:
            return 0
        if name in _NESTED_OPS:
            return self._count_nested_jaxpr(eqn, flops_by_op)
        logger.warning("Unknown primitive '%s' — counting as 0 FLOPs", name)
        return 0

    def _count_dot_general(self, eqn: jax_core.JaxprEqn) -> int:
        """Count FLOPs for dot_general (matmul).

        For (M, K) @ (K, N) -> 2 * M * K * N.
        """
        out_aval = eqn.outvars[0].aval
        if not hasattr(out_aval, "shape"):
            return 0

        dim_numbers = eqn.params.get("dimension_numbers")
        if dim_numbers is None:
            return 0

        lhs_contract, rhs_contract = dim_numbers[0]
        lhs_aval = eqn.invars[0].aval
        if not hasattr(lhs_aval, "shape"):
            return 0

        k_dim = 1
        for idx in lhs_contract:
            k_dim *= lhs_aval.shape[idx]  # type: ignore[union-attr]

        output_elements = math.prod(out_aval.shape)  # type: ignore[union-attr]
        return 2 * k_dim * output_elements

    def _count_conv(self, eqn: jax_core.JaxprEqn) -> int:
        """Count FLOPs for conv_general_dilated."""
        out_aval = eqn.outvars[0].aval
        if not hasattr(out_aval, "shape"):
            return 0

        rhs_aval = eqn.invars[1].aval
        if not hasattr(rhs_aval, "shape"):
            return 0

        output_elements = math.prod(out_aval.shape)  # type: ignore[union-attr]
        kernel_elements = math.prod(rhs_aval.shape)  # type: ignore[union-attr]
        return 2 * output_elements * kernel_elements

    def _count_nested_jaxpr(
        self,
        eqn: jax_core.JaxprEqn,
        flops_by_op: dict[str, int],
    ) -> int:
        """Count FLOPs in nested Jaxprs (pjit, scan, etc.)."""
        total = 0
        for sub in eqn.params.values():
            if isinstance(sub, jax_core.Jaxpr):
                total += self._count_jaxpr(sub, flops_by_op)
            elif isinstance(sub, jax_core.ClosedJaxpr):
                total += self._count_jaxpr(sub.jaxpr, flops_by_op)
        return total

    def _output_size(self, eqn: jax_core.JaxprEqn) -> int:
        """Product of output shape dimensions."""
        if not eqn.outvars:
            return 0
        aval = eqn.outvars[0].aval
        if not hasattr(aval, "shape"):
            return 0
        return math.prod(aval.shape)  # type: ignore[union-attr]

    def _input_size(self, eqn: jax_core.JaxprEqn) -> int:
        """Product of first input shape dimensions."""
        if not eqn.invars:
            return 0
        aval = eqn.invars[0].aval
        if not hasattr(aval, "shape"):
            return 0
        return math.prod(aval.shape)  # type: ignore[union-attr]
