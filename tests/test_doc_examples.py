"""The documentation's Python code blocks: every one collected, each page run in order.

The blocks themselves run as Sybil items, configured in :mod:`tests.doc_examples` and collected
through the root conftest; these tests hold that configuration to what the pages contain.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import pytest

from calibrax.metrics import MetricRegistry, register_metric
from tests.doc_examples import (
    collected_examples,
    DOCS_DIR,
    GENERATED_EXAMPLE_PAGES,
    page_runner,
    page_sybils,
)


def _pages() -> list[Path]:
    """Every Markdown page below ``docs/``."""
    return sorted(DOCS_DIR.rglob("*.md"))


def _is_generated(page: Path) -> bool:
    """Whether ``substrax-notebooks page`` wrote ``page``."""
    return page.relative_to(DOCS_DIR).parts[0] == GENERATED_EXAMPLE_PAGES


@pytest.mark.doctest
def test_each_page_is_run_by_one_sybil_and_a_generated_page_by_none() -> None:
    """A page run by two Sybils would run twice; a generated page repeats its example's code,
    which ``substrax-notebooks test`` runs in its own invocation."""
    runs = {str(page.relative_to(DOCS_DIR)): len(page_sybils(page)) for page in _pages()}
    expected = {
        str(page.relative_to(DOCS_DIR)): 0 if _is_generated(page) else 1 for page in _pages()
    }

    assert sum(expected.values()) > 20, "found almost no pages; check DOCS_DIR"
    assert any(_is_generated(page) for page in _pages()), "no generated page to leave out"
    assert runs == expected


# A python fence's opening line at any indentation: in an admonition, a tab or a list item too.
PYTHON_FENCE_LINE = re.compile(r"^[ \t]*```python[ \t]*$", re.MULTILINE)


@pytest.mark.doctest
def test_every_python_fence_in_the_docs_is_run() -> None:
    """Every python fence the pages hold is run, counted independently of the extractor.

    A fence the collector does not recognise is neither run nor reported, so an accounting built on
    the collector's own output cannot see it; counting the fences' opening lines can. The earlier
    extractor matched fences at column 0 only and never ran the ten indented in admonitions.
    """
    on_pages = Counter(
        {
            str(page.relative_to(DOCS_DIR)): len(PYTHON_FENCE_LINE.findall(page.read_text()))
            for page in _pages()
            if not _is_generated(page)
        }
    )
    collected = Counter(
        {str(page.relative_to(DOCS_DIR)): collected_examples(page) for page in _pages()}
    )

    assert sum(on_pages.values()) > 100, "the fence count found almost nothing; check the pattern"
    assert +collected == +on_pages


_ORDERED_PAGE = """\
```python
total = 1
```

!!! note

    ```python
    total += 1
    ```

```python
assert total == 2, total
```
"""
"""Three blocks, the second indented in an admonition, each needing the one before it."""

_ORDER_CONFTEST = """\
from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser

from tests.doc_examples import keep_pages_in_order

pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser()], path="docs", patterns=["*.md"]
).pytest()
{hook}
"""
_ORDER_HOOK = """
def pytest_collection_modifyitems(items):
    keep_pages_in_order(items)
"""


def _run_ordered_pages(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, *, hook: str, seed: int
) -> pytest.RunResult:
    """Run two dependent pages under pytest-randomly with ``seed``, the order hook as given.

    The run is a subprocess, so this suite's warning filters and plugins do not reach it, and
    finds ``tests.doc_examples`` through the repository root on its path.
    """
    monkeypatch.setenv("PYTHONPATH", str(DOCS_DIR.parent))
    pytester.makeconftest(_ORDER_CONFTEST.format(hook=hook))
    (pytester.path / "docs").mkdir(exist_ok=True)
    for name in ("first.md", "second.md"):
        (pytester.path / "docs" / name).write_text(_ORDERED_PAGE, encoding="utf-8")
    return pytester.runpytest_subprocess(
        "-p", "no:cacheprovider", f"--randomly-seed={seed}", "docs"
    )


@pytest.mark.doctest
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_page_s_blocks_run_in_page_order_under_pytest_randomly(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    result = _run_ordered_pages(pytester, monkeypatch, hook=_ORDER_HOOK, seed=seed)

    result.assert_outcomes(passed=6)


@pytest.mark.doctest
def test_pytest_randomly_reorders_a_page_s_blocks_without_the_hook(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The premise of :func:`~tests.doc_examples.keep_pages_in_order`: without it, a seed breaks
    a page. Should pytest-randomly stop reordering Sybil's items, this fails and the hook goes."""
    outcomes = [
        _run_ordered_pages(pytester, monkeypatch, hook="", seed=seed).parseoutcomes()
        for seed in range(1, 6)
    ]

    # Every run collected its six blocks and none errored, so a failure is the blocks' own: a
    # name used before the block defining it ran, not a run that never started.
    assert all(sum(outcome.values()) >= 6 and "errors" not in outcome for outcome in outcomes)
    assert any(outcome.get("failed", 0) for outcome in outcomes)


@pytest.mark.doctest
def test_a_page_leaves_the_metric_registry_as_it_found_it() -> None:
    """A page that stops before removing what it registered leaves nothing for the next one.

    ``tests/conftest.py`` restores the registry around each test under ``tests/``; the pages run
    under ``docs/``, out of its reach, so each page restores it when it ends.
    """
    setup, teardown = page_runner(())
    namespace: dict[str, object] = {}
    setup(namespace)
    register_metric("a_page_s_own_metric", domain="general")(
        lambda predictions, targets: float(jnp.mean(predictions - targets))
    )
    assert MetricRegistry().has("a_page_s_own_metric")

    teardown(namespace)

    assert not MetricRegistry().has("a_page_s_own_metric")
