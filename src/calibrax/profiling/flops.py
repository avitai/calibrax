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

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import jax
from substrax.records import read_record
from substrax.typing import JsonValue, PyTree

from calibrax.profiling._jax_staging import lower_staged


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
        bytes_accessed: Bytes the operation reads and writes, XLA's ``bytes accessed``. It
            counts the intermediates materialised between kernels, which the inputs and
            outputs alone do not.
        function_name: Name of the analyzed function.
    """

    total_flops: int
    transcendentals: int
    bytes_accessed: int
    function_name: str

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "total_flops": int(self.total_flops),
            "transcendentals": int(self.transcendentals),
            "bytes_accessed": int(self.bytes_accessed),
            "function_name": self.function_name,
        }

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> FlopsResult:
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


def cost_mapping(cost: object) -> Mapping[str, float] | None:
    """The numeric fields of an XLA cost analysis, or ``None`` when there is none.

    ``Lowered.cost_analysis()`` and ``Compiled.cost_analysis()`` are typed ``Any`` by jax,
    which documents their structure as arbitrary: a mapping of property names to numbers,
    a list holding one such mapping on some backends, or ``None``.

    Args:
        cost: What ``cost_analysis()`` returned.

    Returns:
        Each numeric field as a float, or ``None`` for anything other than a mapping or a
        list of exactly one mapping.
    """
    if isinstance(cost, list) and len(cost) == 1:
        (cost,) = cost
    if not isinstance(cost, Mapping):
        return None
    return {
        str(name): float(value)
        for name, value in cost.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _cost_field(cost: Mapping[str, float], key: str) -> int:
    """Read one field of a cost analysis; a missing field is a zero cost."""
    value = cost.get(key)
    return 0 if value is None else int(value)


def _analyse(
    fn: Callable[..., PyTree],
    args: tuple[PyTree, ...],
    kwargs: Mapping[str, PyTree],
    *,
    static_argnums: int | Iterable[int] = (),
    static_argnames: str | Iterable[str] = (),
    optimized: bool = False,
) -> Mapping[str, float]:
    """Return XLA's cost analysis for ``fn`` lowered on abstract ``(args, kwargs)``.

    Prefers the analysis of the HLO lowered for the CPU backend. Without a CPU
    backend, or when that lowering carries no analysis, falls back to the default
    device and then to the compiled executable. ``optimized`` reads the compiled
    executable instead: the HLO that runs, whose fusions decide how many bytes move.

    Args:
        fn: Callable or ``nnx.Module`` to analyze.
        args: Positional example arguments.
        kwargs: Keyword example arguments.
        static_argnums: Static positional argument indices.
        static_argnames: Static keyword argument names.
        optimized: Read the compiled executable's analysis.

    Returns:
        The analysis mapping, with at least the keys XLA populated.

    Raises:
        FlopsUnavailableError: If no analysis is available on any path.
    """
    try:
        cpu = jax.devices("cpu")[0]
    except RuntimeError:  # jax was started without a CPU backend
        cpu = None
    if cpu is not None:
        with jax.default_device(cpu):
            lowering, _ = lower_staged(
                fn,
                args,
                kwargs,
                static_argnums=static_argnums,
                static_argnames=static_argnames,
                abstract_inputs=True,
            )
            cost = cost_mapping(
                lowering.compile().cost_analysis() if optimized else lowering.cost_analysis()
            )
        if cost is not None:
            return cost
    lowered, _ = lower_staged(
        fn,
        args,
        kwargs,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        abstract_inputs=True,
    )
    cost = cost_mapping(lowered.cost_analysis())
    if cost is None:
        cost = cost_mapping(lowered.compile().cost_analysis())
    if cost is None:
        raise FlopsUnavailableError(
            f"XLA returned no cost analysis on backend {jax.default_backend()!r}"
        )
    return cost


class FlopsCounter:
    """Count the FLOPs of a JAX function from XLA's cost analysis of its lowering.

    Works for pure JAX functions, PyTree arguments, keyword arguments, and stateful or
    stateless Flax NNX modules and bound methods without mutating live module state.
    """

    def count(
        self,
        fn: Callable[..., PyTree],
        *args: PyTree,
        static_argnums: int | Iterable[int] = (),
        static_argnames: str | Iterable[str] = (),
        optimized: bool = False,
        **kwargs: PyTree,
    ) -> FlopsResult:
        """Count FLOPs for a function or ``nnx.Module`` with given example arguments.

        Args:
            fn: JAX function, ``nnx.Module``, or bound ``nnx.Module`` method to analyze.
            *args: Example positional arguments; only their shapes and dtypes are used,
                except for static ones.
            static_argnums: Positional argument indices treated as static at trace time.
            static_argnames: Keyword argument names treated as static at trace time.
            optimized: Analyse the compiled executable, whose fusions decide how many bytes
                move, instead of the unoptimised HLO.
            **kwargs: Example keyword arguments passed to ``fn``.

        Returns:
            FlopsResult with the FLOP, transcendental and byte counts.

        Raises:
            FlopsUnavailableError: If XLA cannot estimate the cost, typically because
                the function contains a custom call it has no model for.
        """
        cost = _analyse(
            fn,
            args,
            kwargs,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            optimized=optimized,
        )
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
            bytes_accessed=_cost_field(cost, "bytes accessed"),
            function_name=name,
        )
