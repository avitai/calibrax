"""FLOP counting through XLA's cost analysis.

``FlopsCounter.count`` lowers the function with ``jax.jit`` from the shapes and
dtypes of the example arguments (no data is copied and nothing is executed) and reads
``jax.stages.Lowered.cost_analysis()``, XLA's ``HloCostAnalysis`` of the unoptimised
HLO. It is the estimate ``flax.nnx.tabulate(..., compute_flops=True)`` reports and
JAX's ahead-of-time documentation shows, so the numbers agree across the stack and
follow jax's primitive set without a table here.

Conventions of ``HloCostAnalysis`` a reader should know: a matmul ``(M, K) @ (K, N)``
is ``2 * M * K * N``; an elementwise op is one FLOP per output element; ``sin``,
``exp`` and friends are reported as transcendentals, not FLOPs; a reduction over
``n`` elements is ``n - 1``; a conditional costs its most expensive branch; a loop body
is counted once, because the trip count is not part of the HLO.

The lowering targets the CPU backend when jax has one, so accelerator custom calls
(a cuDNN convolution, for example) cannot hide the arithmetic behind a cost XLA does
not estimate. When jax was started without a CPU backend (``JAX_PLATFORMS=cuda``) the
function is lowered for the default device and, if that lowering carries no analysis,
the compiled executable's analysis is used, as ``nnx.tabulate`` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import jax


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class FlopsUnavailableError(ValueError):
    """XLA could not estimate the cost of the function.

    Raised when the analysis is negative, which is how ``HloCostAnalysis`` reports a
    custom call it has no model for (``jax.pure_callback``, some linear-algebra
    kernels, Pallas kernels), or when no backend provides an analysis at all.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class FlopsResult:
    """Cost of one function, as XLA estimates it.

    Attributes:
        total_flops: Floating-point operations, excluding transcendentals.
        transcendentals: Transcendental operations (``sin``, ``exp``, ``tanh``, ...).
        function_name: Name of the analyzed function.
    """

    total_flops: int
    transcendentals: int
    function_name: str


def _abstract(leaf: Any) -> Any:
    """Replace an array-like leaf by its shape and dtype; leave other leaves as they are."""
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
    return leaf


def _cost_field(cost: Mapping[str, float], key: str) -> int:
    """Read one field of a cost analysis; a missing field is a zero cost."""
    value = cost.get(key)
    return 0 if value is None else int(value)


def _analyse(jitted: Any, spec: tuple[Any, ...]) -> Mapping[str, float]:
    """Return XLA's cost analysis for ``jitted`` applied to ``spec``.

    Prefers the analysis of the HLO lowered for the CPU backend. Without a CPU
    backend, or when that lowering carries no analysis, falls back to the default
    device and then to the compiled executable.

    Raises:
        FlopsUnavailableError: If no analysis is available on any path.
    """
    try:
        cpu = jax.devices("cpu")[0]
    except RuntimeError:  # jax was started without a CPU backend
        cpu = None
    if cpu is not None:
        with jax.default_device(cpu):
            cost = jitted.lower(*spec).cost_analysis()
        if cost is not None:
            return cost
    lowered = jitted.lower(*spec)
    cost = lowered.cost_analysis()
    if cost is None:
        cost = lowered.compile().cost_analysis()
    if cost is None:
        raise FlopsUnavailableError(
            f"XLA returned no cost analysis on backend {jax.default_backend()!r}"
        )
    return cost


class FlopsCounter:
    """Count the FLOPs of a JAX function from XLA's cost analysis of its lowering.

    Works for pure JAX functions and for functions that close over or take Flax NNX
    state; nothing is executed, so stochastic modules are fine as long as their keys
    are arguments or captured state.
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
            *args: Example arguments; only their shapes and dtypes are used, except
                for the static ones.
            static_argnums: Argument indices ``jax.jit`` treats as static.

        Returns:
            FlopsResult with the FLOP and transcendental counts.

        Raises:
            FlopsUnavailableError: If XLA cannot estimate the cost, typically because
                the function contains a custom call it has no model for.
        """
        spec = tuple(
            arg if index in static_argnums else jax.tree.map(_abstract, arg)
            for index, arg in enumerate(args)
        )
        cost = _analyse(jax.jit(fn, static_argnums=static_argnums), spec)
        name = getattr(fn, "__name__", type(fn).__name__)
        total_flops = _cost_field(cost, "flops")
        transcendentals = _cost_field(cost, "transcendentals")
        if total_flops < 0 or transcendentals < 0:
            raise FlopsUnavailableError(
                f"XLA cannot estimate the cost of {name!r}: it contains a custom call "
                f"(a callback, a Pallas kernel or a library kernel) with no cost model "
                f"(flops={total_flops}, transcendentals={transcendentals})."
            )
        return FlopsResult(
            total_flops=total_flops,
            transcendentals=transcendentals,
            function_name=name,
        )
