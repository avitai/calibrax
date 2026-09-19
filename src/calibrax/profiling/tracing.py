"""XLA trace linking for connecting JAX profiler output to benchmark runs.

Provides a simple context manager wrapping ``jax.profiler.trace()``
that records the trace file path for association with Store run metadata.
Does not parse trace files — only links file paths to benchmark results.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import jax
from substrax.records import read_record
from substrax.typing import JsonValue


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class TraceReference:
    """Reference to a JAX profiler trace output.

    Attributes:
        trace_dir: Directory where the trace files were written.
        run_id: Optional benchmark run ID to link the trace to.
    """

    trace_dir: str
    run_id: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-compatible dictionary."""
        d: dict[str, JsonValue] = {"trace_dir": self.trace_dir}
        if self.run_id is not None:
            d["run_id"] = self.run_id
        return d

    @classmethod
    def from_dict(  # noqa: DOC502  # raised by read_record
        cls, data: Mapping[str, JsonValue]
    ) -> TraceReference:
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


class TraceLinker:
    """Links JAX profiler traces to benchmark runs.

    Usage:

    ```python
    linker = TraceLinker()
    with linker.trace("/tmp/my_trace") as ref:
        # ... run workload ...
    print(ref.trace_dir)  # "/tmp/my_trace"
    ```
    """

    @contextmanager
    def trace(
        self,
        log_dir: str | Path,
        *,
        run_id: str | None = None,
        create_perfetto_link: bool = False,
        create_perfetto_trace: bool = False,
    ) -> Iterator[TraceReference]:
        """Start an XLA profiling session and record output metadata.

        Wraps ``jax.profiler.trace()`` and records the output directory
        path as a ``TraceReference`` for downstream Store linkage.

        Args:
            log_dir: Directory for trace output files.
            run_id: Optional benchmark run ID to associate with the trace.
            create_perfetto_link: Whether to create a Perfetto link (passed to JAX).
            create_perfetto_trace: Whether to create a Perfetto trace (passed to JAX).

        Yields:
            TraceReference with the trace directory and optional run ID.
        """
        log_dir_str = str(log_dir)
        ref = TraceReference(trace_dir=log_dir_str, run_id=run_id)

        logger.info("Starting JAX profiler trace in %s", log_dir_str)

        with jax.profiler.trace(
            log_dir_str,
            create_perfetto_link=create_perfetto_link,
            create_perfetto_trace=create_perfetto_trace,
        ):
            yield ref

        logger.info("Completed JAX profiler trace in %s", log_dir_str)
