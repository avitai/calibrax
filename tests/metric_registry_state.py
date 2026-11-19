"""Snapshot and restore of the process-wide metric registry, through its public API."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from calibrax.metrics import MetricRegistry


@contextmanager
def restored_metric_registry() -> Iterator[None]:
    """Restore the metric registry's names and entries on exit.

    The snapshot maps each registered name to its entry, in registration order. On exit the
    registry is cleared and each entry registered again, so registrations, removals and
    replacements made inside do not outlive the block.

    Yields:
        Nothing; the snapshot is taken on entry and restored on exit.
    """
    registry = MetricRegistry()
    snapshot = {name: registry.get(name) for name in registry}
    try:
        yield
    finally:
        registry.clear()
        for name, entry in snapshot.items():
            registry.register(name, entry)
