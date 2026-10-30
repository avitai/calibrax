"""Shared test fixtures for calibrax."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from calibrax.metrics._registry import MetricRegistry


@pytest.fixture(autouse=True)
def _restore_the_metric_registry() -> Iterator[None]:
    """Leave the metric registry as it was found, for every test.

    The registry is a process-wide singleton, so a test that registers a metric changes
    what every later test sees. Three things make that more than theoretical here:
    ``tests/test_doc_examples.py`` executes the documentation, and
    ``docs/user-guide/metrics-overview.md`` contains a worked example that registers a
    ``weighted_mse`` metric into the ``general`` domain; ``pytest-randomly`` reorders the
    suite on every run; and several tests assert what the registry holds.

    The result was a test that failed one full run and passed the next with no change in
    between, twice over -- once for an assertion on a domain's size, and again for one on a
    domain's membership, because narrowing the assertion was never the fix. Any statement
    about the registry's contents is unreliable while a neighbour can add to it.

    ``scripts/derive_status.py`` reaches the same conclusion from the other side and
    measures the shipped counts in a subprocess, saying so in its own docstring. This
    fixture gives the in-process tests the same guarantee: the registry is snapshotted
    before each test and restored after it, so registrations cannot outlive the test that
    made them.

    Yields:
        Nothing; the snapshot is taken on entry and restored on exit.
    """
    registry = MetricRegistry()
    # The singleton exposes no snapshot API, so its backing dict is copied directly.
    snapshot = dict(registry._items)
    try:
        yield
    finally:
        registry._items.clear()
        registry._items.update(snapshot)
