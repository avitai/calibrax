"""The documentation's Python code blocks, collected and run by Sybil.

Every ``python`` fence under ``docs/``, at any indentation (in an admonition or a list item too),
is one test item. A page's blocks share one namespace and run in page order, as a reader runs
them; pages are independent. The pages ``substrax-notebooks page`` writes under ``docs/examples``
are left out: each repeats an example's code, which ``substrax-notebooks test`` runs in its own
invocation.

Each page starts from :func:`preamble`, in a temporary working directory, with its page's entry
in :data:`PAGE_PATCHES` applied; its figures are closed, its patches undone and the metric
registry restored when it ends. The registry is restored per page, not per block, because a page's
later blocks use what its earlier ones registered (``docs/user-guide/metrics-overview.md``
registers ``weighted_mse``); ``tests/conftest.py``'s per-test restore does not reach ``docs/``.

Sybil (https://github.com/simplistix/sybil) replaced a hand-written extractor whose pattern only
matched fences at column 0, so the blocks indented inside admonitions never ran.
"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest
from flax import nnx
from matplotlib.figure import Figure
from sybil import Sybil
from sybil.integration.pytest import SybilItem
from sybil.parsers.markdown import PythonCodeBlockParser
from sybil.sybil import SybilCollection

from tests.factories import make_fake_nvml
from tests.metric_registry_state import restored_metric_registry


DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
# The directory of docs/ that `substrax-notebooks page` writes, one page per example.
GENERATED_EXAMPLE_PAGES = "examples"

_CONTEXT = "__calibrax_doc_context__"
"""The namespace key holding a page's working directory, patches and other context."""


def _returning(target: str, value: object) -> Callable[[], contextlib.AbstractContextManager[Any]]:
    """A patch of ``target`` that returns ``value``, made when the page starts."""
    return lambda: patch(target, return_value=value)


def _replacing(
    target: str, new: Callable[[], object]
) -> Callable[[], contextlib.AbstractContextManager[Any]]:
    """A patch replacing ``target`` with ``new()``, made when the page starts."""
    return lambda: patch(target, new())


_WANDB = "calibrax.exporters.wandb.WandBExporter"
_MLFLOW = "calibrax.exporters.mlflow.MLflowExporter"

PAGE_PATCHES: dict[str, tuple[Callable[[], contextlib.AbstractContextManager[Any]], ...]] = {
    # The exporters' network calls and the MLflow client go to mocks, so the page needs no
    # account, server or socket.
    "user-guide/exporters.md": (
        _returning(f"{_WANDB}.check_auth", True),
        _returning(f"{_WANDB}.export_run", "https://wandb.local/mock-run"),
        _returning(f"{_WANDB}.export_analysis", None),
        _returning(f"{_WANDB}.export_trends", None),
        _returning(f"{_WANDB}.log_images", None),
        _returning(f"{_WANDB}.log_html_artifacts", None),
        _returning(f"{_WANDB}.log_extra_tables", None),
        _replacing("calibrax.exporters.mlflow.mlflow", MagicMock),
        _returning(f"{_MLFLOW}.export_run", "mock-run-id"),
        _returning(f"{_MLFLOW}.export_analysis", None),
    ),
    # The NVML examples read a GPU; a fake NVML answers where no driver is present.
    "user-guide/profiling.md": (_replacing("calibrax.profiling.nvml.pynvml", make_fake_nvml),),
}
"""Each page's patches of external side effects, keyed by its path below ``docs/``."""


def preamble() -> dict[str, object]:
    """The names every page starts with: common imports and the stubs its examples assume."""
    mock_model = MagicMock()
    mock_model.forward = MagicMock(return_value=jnp.ones((4, 4)))

    return {
        # Common imports used across doc examples
        "jax": jax,
        "jnp": jnp,
        # Common stub functions used in doc examples
        "train": lambda *_a, **_kw: None,
        "train_step": lambda *_a, **_kw: None,
        "pipeline_fn": lambda *_a, **_kw: None,
        # Common data stubs
        "data": [{"image": jnp.ones((4, 8))} for _ in range(10)],
        "data_loader": [{"image": jnp.ones((4, 8))} for _ in range(100)],
        "data_iterator": iter([jnp.ones((4, 8)) for _ in range(100)]),
        "sample_data": jnp.ones((4, 8)),
        "x": jnp.ones((32, 128)),
        "batch": jnp.ones((4, 8)),
        # Metric evaluation stubs (predictions/targets for regression, classification, etc.)
        "predictions": jnp.array([1.1, 1.9, 3.2, 3.8, 5.1]),
        "targets": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        # Metric learning stubs
        "embeddings": jnp.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        "labels": jnp.array([0, 0, 1, 1]),
        # Model stubs
        "model": nnx.Linear(8, 4, rngs=nnx.Rngs(0)),
        "my_model": mock_model,
        "my_nnx_model": nnx.Linear(8, 4, rngs=nnx.Rngs(0)),
        "my_pytorch_model": mock_model,
        # Function stubs
        "fn": lambda x: jnp.dot(x, x.T),
        "sample_args": (jnp.ones((4, 4)),),
        # Artifact stubs
        "fig": _example_figure(),
        "html_string": "<h1>test</h1>",
        # Distributed stubs
        "pjit_function": lambda x: x,
        "sharded_input": jnp.ones((4, 4)),
    }


def _example_figure() -> Figure:
    """A small pyplot figure, the kind a user logs or saves."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    return fig


def page_runner(
    page_patches: tuple[Callable[[], contextlib.AbstractContextManager[Any]], ...],
) -> tuple[Callable[[dict[str, Any]], None], Callable[[dict[str, Any]], None]]:
    """The setup and teardown of a page that runs with ``page_patches`` applied."""

    def setup(namespace: dict[str, Any]) -> None:
        # Entered inside `with`, so a context that fails to enter undoes the ones before it; the
        # page holds them only once all have entered.
        with contextlib.ExitStack() as context:
            context.enter_context(restored_metric_registry())
            context.enter_context(
                contextlib.chdir(context.enter_context(tempfile.TemporaryDirectory()))
            )
            for make_patch in page_patches:
                context.enter_context(make_patch())
            namespace[_CONTEXT] = context.pop_all()
        namespace.update(preamble())

    def teardown(namespace: dict[str, Any]) -> None:
        # A reader's session ends with its figures closed.
        plt.close("all")
        namespace.pop(_CONTEXT).close()

    return setup, teardown


def _generated_pages() -> list[str]:
    """Every page under ``docs/examples``, by its path below ``docs/``."""
    generated = DOCS_DIR / GENERATED_EXAMPLE_PAGES
    return [str(page.relative_to(DOCS_DIR)) for page in sorted(generated.rglob("*.md"))]


def _sybil(name: str, *, patterns: list[str], excludes: list[str]) -> Sybil:
    """A Sybil over ``docs/`` running ``patterns`` minus ``excludes`` with their page patches."""
    page_patches = tuple(make for page in patterns for make in PAGE_PATCHES.get(page, ()))
    setup, teardown = page_runner(page_patches)
    return Sybil(
        parsers=[PythonCodeBlockParser()],
        path=str(DOCS_DIR),
        patterns=patterns,
        excludes=excludes,
        setup=setup,
        teardown=teardown,
        name=name,
    )


DOCS = SybilCollection(
    [
        _sybil("docs", patterns=["*.md"], excludes=[*PAGE_PATCHES, *_generated_pages()]),
        *(_sybil(Path(page).stem, patterns=[page], excludes=[]) for page in PAGE_PATCHES),
    ]
)
"""Every page below ``docs/`` but the generated example pages, each run by exactly one Sybil."""


def keep_pages_in_order(items: list[pytest.Item]) -> None:
    """Put each page's examples back together, in page order, after the run was reordered.

    pytest-randomly groups items by their ``module``, which a Sybil item does not have, so it sorts
    every page's examples together by a hash of their node ids; a page's later block would then
    run before the block that defines its names. The documentation's examples move to the end of
    the run, grouped by page and in page order; the other tests keep the order they were given.

    Args:
        items: The collected items, reordered in place.
    """
    examples = sorted(
        (item for item in items if isinstance(item, SybilItem)),
        key=lambda item: (str(item.path), item.example.line, item.example.column),
    )
    items[:] = [item for item in items if not isinstance(item, SybilItem)] + examples


def page_sybils(page: Path) -> list[Sybil]:
    """The Sybils that would run ``page``: one for a page under ``docs/`` that is not generated."""
    return [sybil for sybil in DOCS if sybil.should_parse(page)]


def collected_examples(page: Path) -> int:
    """How many examples the Sybils collect from ``page``."""
    return sum(len(list(sybil.parse(page).examples())) for sybil in page_sybils(page))
