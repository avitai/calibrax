"""Internal JAX and Flax NNX staging helpers shared across ``calibrax.profiling``.

Handles PyTree leaf abstraction, static positional and keyword arguments, and unbinding or
splitting ``flax.nnx.Module`` instances and bound methods so static lowerings never mutate
caller state while live execution propagates stateful ``nnx`` updates cleanly.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx


_NNX_PURE_TUPLE_LEN = 2


@runtime_checkable
class Shaped(Protocol):
    """An array-like leaf carrying ``shape`` and ``dtype``."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> jax.typing.DTypeLike: ...


def abstract_leaf(leaf: object) -> object:
    """Replace an array-like leaf with ``jax.ShapeDtypeStruct``; keep other leaves intact."""
    if isinstance(leaf, Shaped):
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
    return leaf


def shaped_nbytes(leaf: Shaped) -> int:
    """Byte size of a shaped leaf, supporting both NumPy/JAX dtypes and typed PRNG keys."""
    itemsize = getattr(leaf.dtype, "itemsize", None)
    if itemsize is None:
        itemsize = jnp.dtype(leaf.dtype).itemsize
    return int(math.prod(leaf.shape) * int(itemsize))


def normalize_static_argnums(static_argnums: int | Iterable[int] | None) -> tuple[int, ...]:
    """Normalize ``static_argnums`` to a tuple of argument indices."""
    if static_argnums is None:
        return ()
    if isinstance(static_argnums, int):
        return (static_argnums,)
    return tuple(static_argnums)


def normalize_static_argnames(static_argnames: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize ``static_argnames`` to a tuple of keyword names."""
    if static_argnames is None:
        return ()
    if isinstance(static_argnames, str):
        return (static_argnames,)
    return tuple(static_argnames)


def unbind_nnx_call(
    fn: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    static_argnums: tuple[int, ...] = (),
) -> tuple[Callable[..., Any], tuple[object, ...], dict[str, object], tuple[int, ...], bool]:
    """Unbind a callable ``nnx.Module`` or bound ``nnx.Module`` method into call-time arguments.

    ``nnx.jit`` and ``nnx.split`` require mutable ``nnx.Module`` instances to arrive as call
    arguments rather than closed over inside the transformed callable.

    Args:
        fn: Callable to invoke (function, ``nnx.Module``, bound method, or ``functools.partial``).
        args: Positional call arguments.
        kwargs: Keyword call arguments.
        static_argnums: Static positional indices relative to ``args``.

    Returns:
        Tuple ``(target_fn, full_args, full_kwargs, shifted_static_argnums, was_unbound)``.
    """
    partials: list[tuple[tuple[object, ...], Mapping[str, object]]] = []
    current: object = fn
    while isinstance(current, functools.partial):
        partials.append((current.args or (), current.keywords or {}))
        current = current.func

    prepended: list[object] = []
    if isinstance(current, nnx.Module):
        target: Callable[..., Any] = type(current).__call__
        prepended.append(current)
    elif inspect.ismethod(current) and isinstance(getattr(current, "__self__", None), nnx.Module):
        target = current.__func__
        prepended.append(current.__self__)
    else:
        return fn, args, dict(kwargs), static_argnums, False

    merged_kwargs: dict[str, object] = {}
    for p_args, p_kwargs in reversed(partials):
        prepended.extend(p_args)
        merged_kwargs.update(p_kwargs)
    merged_kwargs.update(kwargs)

    shift = len(prepended)
    shifted_static = tuple(idx + shift for idx in static_argnums)
    return target, (*prepended, *args), merged_kwargs, shifted_static, True


def contains_nnx_graph_node(obj: object) -> bool:
    """Return ``True`` when ``obj`` or any PyTree container inside it holds an NNX graph node."""
    leaves = jax.tree.leaves(obj, is_leaf=nnx.graphlib.is_graph_node)
    return any(nnx.graphlib.is_graph_node(leaf) for leaf in leaves)


def compile_callable(
    target: Callable[..., Any],
    *,
    use_nnx: bool,
    static_argnums: tuple[int, ...] = (),
    static_argnames: tuple[str, ...] = (),
    jit_fn: Callable[..., Any] = jax.jit,
) -> Callable[..., Any]:
    """Wrap ``target`` with ``jax.jit`` or ``nnx.jit``, passing static options only when set."""
    argnums_opt = static_argnums or None
    argnames_opt = static_argnames or None
    if use_nnx:
        if argnums_opt is None and argnames_opt is None:
            return nnx.jit(target)
        return nnx.jit(target, static_argnums=argnums_opt, static_argnames=argnames_opt)
    if argnums_opt is None and argnames_opt is None:
        return jit_fn(target)
    return jit_fn(target, static_argnums=argnums_opt, static_argnames=argnames_opt)


def lower_staged(
    fn: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    *,
    static_argnums: int | Iterable[int] | None = (),
    static_argnames: str | Iterable[str] | None = (),
    abstract_inputs: bool = False,
    jit_fn: Callable[..., Any] = jax.jit,
) -> tuple[jax.stages.Lowered, object]:
    """Lower ``fn(*args, **kwargs)`` without mutating live ``nnx.Module`` state.

    Args:
        fn: Function, ``nnx.Module``, or bound ``nnx.Module`` method.
        args: Positional example arguments.
        kwargs: Keyword example arguments.
        static_argnums: Static positional argument indices.
        static_argnames: Static keyword argument names.
        abstract_inputs: Replace dynamic array leaves with ``ShapeDtypeStruct`` before lowering.
        jit_fn: ``jit`` constructor (defaults to ``jax.jit``).

    Returns:
        Tuple ``(lowered, out_info)`` where ``out_info`` describes the abstract output of ``fn``.
    """
    norm_nums = normalize_static_argnums(static_argnums)
    norm_names = normalize_static_argnames(static_argnames)
    target, full_args, full_kwargs, shifted_nums, was_unbound = unbind_nnx_call(
        fn, args, kwargs, norm_nums
    )

    if not was_unbound and not contains_nnx_graph_node((full_args, full_kwargs)):
        num_set = set(norm_nums)
        name_set = set(norm_names)
        if abstract_inputs:
            call_args = tuple(
                arg if i in num_set else jax.tree.map(abstract_leaf, arg)
                for i, arg in enumerate(args)
            )
            call_kwargs = {
                k: v if k in name_set else jax.tree.map(abstract_leaf, v) for k, v in kwargs.items()
            }
        else:
            call_args = args
            call_kwargs = dict(kwargs)
        argnums_opt = norm_nums or None
        argnames_opt = norm_names or None
        jitted: jax.stages.Wrapped = (
            jit_fn(fn)
            if argnums_opt is None and argnames_opt is None
            else jit_fn(fn, static_argnums=argnums_opt, static_argnames=argnames_opt)
        )
        lowered = jitted.lower(*call_args, **call_kwargs)
        return lowered, getattr(lowered, "out_info", None)

    pure_nnx, lower_state, other_tuple = _build_pure_nnx_bundle(
        target,
        full_args,
        full_kwargs,
        shifted_nums,
        norm_names,
        abstract_inputs=abstract_inputs,
    )
    lowered = jit_fn(pure_nnx).lower(lower_state, other_tuple)
    out_info = getattr(lowered, "out_info", None)
    user_out_info = (
        out_info[0]
        if isinstance(out_info, tuple) and len(out_info) == _NNX_PURE_TUPLE_LEN
        else out_info
    )
    return lowered, user_out_info


def _build_pure_nnx_bundle(
    target: Callable[..., Any],
    full_args: tuple[object, ...],
    full_kwargs: Mapping[str, object],
    shifted_nums: tuple[int, ...],
    norm_names: tuple[str, ...],
    *,
    abstract_inputs: bool,
) -> tuple[
    Callable[[nnx.State, tuple[object, ...]], tuple[object, nnx.State]],
    nnx.State,
    tuple[object, ...],
]:
    """Split NNX graph nodes once and return ``(pure_fn, state, non_nnx_leaves)``."""
    num_set = set(shifted_nums)
    name_set = set(norm_names)
    static_pos = {i: arg for i, arg in enumerate(full_args) if i in num_set}
    dyn_pos = [arg for i, arg in enumerate(full_args) if i not in num_set]
    static_kw = {k: v for k, v in full_kwargs.items() if k in name_set}
    dyn_kw = {k: v for k, v in full_kwargs.items() if k not in name_set}
    total_pos = len(full_args)

    dyn_bundle = (tuple(dyn_pos), dyn_kw)
    leaves, outer_treedef = jax.tree.flatten(dyn_bundle, is_leaf=nnx.graphlib.is_graph_node)
    is_gn = [nnx.graphlib.is_graph_node(leaf) for leaf in leaves]
    gn_tuple = tuple(leaf for leaf, flag in zip(leaves, is_gn, strict=True) if flag)
    other_tuple = tuple(
        (abstract_leaf(leaf) if abstract_inputs else leaf)
        for leaf, flag in zip(leaves, is_gn, strict=True)
        if not flag
    )

    graphdef, state = nnx.split(gn_tuple)
    lower_state = jax.tree.map(abstract_leaf, state) if abstract_inputs else state

    def _pure_nnx(st: nnx.State, others: tuple[object, ...]) -> tuple[object, nnx.State]:
        merged_gn = nnx.merge(graphdef, st)
        gn_iter = iter(merged_gn)
        ot_iter = iter(others)
        rebuilt_leaves = [next(gn_iter) if flag else next(ot_iter) for flag in is_gn]
        merged_pos, merged_kw = jax.tree.unflatten(outer_treedef, rebuilt_leaves)
        pos_iter = iter(merged_pos)
        rebuilt_args = tuple(
            static_pos[i] if i in static_pos else next(pos_iter) for i in range(total_pos)
        )
        out = target(*rebuilt_args, **merged_kw, **static_kw)
        _, st_out = nnx.split(merged_gn)
        return out, st_out

    return _pure_nnx, lower_state, other_tuple


def prepare_kernel_invocation(
    fn: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    *,
    static_argnums: int | Iterable[int] | None = (),
    static_argnames: str | Iterable[str] | None = (),
    jit_fn: Callable[..., Any] = jax.jit,
) -> Callable[[], object]:
    """Build a zero-argument compiled kernel invocation without per-call ``nnx.split`` overhead.

    Splits any ``flax.nnx`` graph nodes once outside the timing loop so kernel microbenchmarks
    and roofline measurements time the compiled XLA executable rather than Python graph
    extraction, while leaving caller module state unmutated.

    Args:
        fn: Function, ``nnx.Module``, or bound ``nnx.Module`` method.
        args: Positional call arguments.
        kwargs: Keyword call arguments.
        static_argnums: Static positional argument indices.
        static_argnames: Static keyword argument names.
        jit_fn: ``jit`` constructor (defaults to ``jax.jit``).

    Returns:
        Zero-argument callable invoking the compiled XLA executable.
    """
    norm_nums = normalize_static_argnums(static_argnums)
    norm_names = normalize_static_argnames(static_argnames)
    target, full_args, full_kwargs, shifted_nums, was_unbound = unbind_nnx_call(
        fn, args, kwargs, norm_nums
    )
    if not was_unbound and not contains_nnx_graph_node((full_args, full_kwargs)):
        compiled = compile_callable(
            fn,
            use_nnx=False,
            static_argnums=norm_nums,
            static_argnames=norm_names,
            jit_fn=jit_fn,
        )
        call_kwargs = dict(kwargs)
        return lambda: compiled(*args, **call_kwargs)

    pure_nnx, state, other_tuple = _build_pure_nnx_bundle(
        target,
        full_args,
        full_kwargs,
        shifted_nums,
        norm_names,
        abstract_inputs=False,
    )
    compiled_pure = jit_fn(pure_nnx)
    return lambda: compiled_pure(state, other_tuple)


def _leaf_token(leaf: object) -> str:
    """Format one dynamic PyTree leaf for dispatch signature hashing."""
    if isinstance(leaf, Shaped):
        return f"arr:{leaf.shape}:{leaf.dtype}"
    if isinstance(leaf, bool | int | float | complex):
        return f"scalar:{type(leaf).__qualname__}"
    return f"obj:{type(leaf).__qualname__}"


def call_signature_digest(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    *,
    static_argnums: tuple[int, ...] = (),
    static_argnames: tuple[str, ...] = (),
) -> str:
    """Compute a digest matching JAX/NNX JIT trace-cache dispatch keys.

    Captures PyTree structure and every dynamic leaf's ``(shape, dtype)`` across both positional
    and keyword arguments (splitting NNX graph nodes when present), alongside values of static
    positional and static keyword arguments.
    """
    num_set = set(static_argnums)
    name_set = set(static_argnames)

    static_parts = [
        f"spos:{i}:{type(args[i]).__qualname__}:{args[i]!r}"
        for i in sorted(num_set)
        if 0 <= i < len(args)
    ]
    for name in sorted(name_set):
        if name in kwargs:
            val = kwargs[name]
            static_parts.append(f"skw:{name}:{type(val).__qualname__}:{val!r}")

    dyn_args = tuple(arg for i, arg in enumerate(args) if i not in num_set)
    dyn_kwargs = {k: kwargs[k] for k in sorted(kwargs) if k not in name_set}

    if contains_nnx_graph_node((dyn_args, dyn_kwargs)):
        graphdef, state = nnx.split((dyn_args, dyn_kwargs))
        leaves, treedef = jax.tree.flatten(state)
        structure_tag = f"nnx:{graphdef!r}:{treedef!r}"
    else:
        leaves, treedef = jax.tree.flatten((dyn_args, dyn_kwargs))
        structure_tag = f"pytree:{treedef!r}"

    leaf_parts = [_leaf_token(leaf) for leaf in leaves]
    payload = f"{structure_tag}|{'|'.join(static_parts)}|{'|'.join(leaf_parts)}"
    return hashlib.md5(payload.encode(), usedforsecurity=False).hexdigest()
