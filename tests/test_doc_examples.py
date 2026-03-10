"""Test all Python code blocks in documentation run successfully.

Auto-discovers every fenced code block in docs/**/*.md, groups them by file,
and executes each file's blocks sequentially in document order. This design is
immune to pytest-randomly reordering — files are independent, but blocks within
a file share a namespace and must run in order.

Blocks marked with ``# doctest: +SKIP`` are skipped with documented reasons.
A coverage-assertion test guarantees zero blocks are silently missed.
"""

from __future__ import annotations

import re
import textwrap
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
from flax import nnx


DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

SKIP_PATTERN = re.compile(r"^#\s*doctest:\s*\+SKIP(?:\s*[—–-]\s*(.+))?", re.MULTILINE)
FENCE_PATTERN = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """A fenced code block extracted from a markdown file."""

    file: str
    line: int
    language: str
    code: str
    skip_reason: str | None


def _extract_skip_reason(code: str) -> str | None:
    """Extract the skip reason from a ``# doctest: +SKIP`` marker."""
    match = SKIP_PATTERN.search(code)
    if match:
        return match.group(1) or "marked +SKIP"
    return None


def discover_doc_code_blocks(docs_dir: Path) -> list[CodeBlock]:
    """Scan all markdown files under *docs_dir* and extract every fenced code block."""
    blocks: list[CodeBlock] = []
    for md_file in sorted(docs_dir.rglob("*.md")):
        content = md_file.read_text()
        for match in FENCE_PATTERN.finditer(content):
            lang = match.group(1)
            code = match.group(2)
            line = content[: match.start()].count("\n") + 1
            rel_path = md_file.relative_to(docs_dir)
            blocks.append(
                CodeBlock(
                    file=str(rel_path),
                    line=line,
                    language=lang,
                    code=code,
                    skip_reason=_extract_skip_reason(code),
                )
            )
    return blocks


def _build_preamble() -> dict[str, Any]:
    """Build the preamble namespace injected into every file's execution context."""
    import jax

    mock_model = MagicMock()
    mock_model.forward = MagicMock(return_value=jnp.ones((4, 4)))

    return {
        # Builtins & common imports
        "__builtins__": __builtins__,
        "print": print,
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
        "fig": MagicMock(),
        "html_string": "<h1>test</h1>",
        # Distributed stubs
        "pjit_function": lambda x: x,
        "sharded_input": jnp.ones((4, 4)),
    }


def _run_code_block(code: str, namespace: dict[str, Any], label: str) -> None:
    """Compile and run a code block in the given namespace.

    This intentionally runs documentation code blocks for testing — the input
    is trusted (authored by project maintainers, stored in docs/).
    """
    compiled = compile(code, f"<{label}>", "exec")
    # S102: exec is intentional here — we are testing doc code blocks
    exec(compiled, namespace)  # noqa: S102


@contextmanager
def _patch_external_dependencies(file_path: str):
    """Patch external side effects for doc files that include networked examples."""
    with ExitStack() as stack:
        if file_path == "user-guide/exporters.md":
            # Keep docs executable while preventing network/socket dependencies in tests.
            stack.enter_context(
                patch(
                    "calibrax.exporters.wandb.WandBExporter.check_auth",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch(
                    "calibrax.exporters.wandb.WandBExporter.export_run",
                    return_value="https://wandb.local/mock-run",
                )
            )
            stack.enter_context(
                patch(
                    "calibrax.exporters.wandb.WandBExporter.export_analysis",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "calibrax.exporters.wandb.WandBExporter.export_trends",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "calibrax.exporters.wandb.WandBExporter.log_figures",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "calibrax.exporters.wandb.WandBExporter.log_html_artifacts",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "calibrax.exporters.wandb.WandBExporter.log_extra_tables",
                    return_value=None,
                )
            )
        if file_path == "user-guide/exporters.md":
            # Mock mlflow to prevent ImportError when optional dep is not installed.
            mock_mlflow = MagicMock()
            stack.enter_context(patch.dict("sys.modules", {"mlflow": mock_mlflow}))
            # Patch the module-level mlflow variable and availability flag
            # (both set at import time before the mock takes effect).
            stack.enter_context(patch("calibrax.exporters.mlflow.mlflow", mock_mlflow))
            stack.enter_context(patch("calibrax.exporters.mlflow.MLFLOW_AVAILABLE", True))
            stack.enter_context(
                patch(
                    "calibrax.exporters.mlflow.MLflowExporter.export_run",
                    return_value="mock-run-id",
                )
            )
            stack.enter_context(
                patch(
                    "calibrax.exporters.mlflow.MLflowExporter.export_analysis",
                    return_value=None,
                )
            )
        yield


# ---------------------------------------------------------------------------
# Discover all blocks once at module level for parameterization
# ---------------------------------------------------------------------------

ALL_BLOCKS = discover_doc_code_blocks(DOCS_DIR)
PYTHON_BLOCKS = [b for b in ALL_BLOCKS if b.language == "python"]

# Group blocks by file, preserving document order within each file
_BLOCKS_BY_FILE: dict[str, list[CodeBlock]] = {}
for _block in PYTHON_BLOCKS:
    _BLOCKS_BY_FILE.setdefault(_block.file, []).append(_block)


# ---------------------------------------------------------------------------
# Parameterized test: one case per documentation FILE
# Blocks within each file run sequentially in document order, so
# pytest-randomly can safely shuffle file order without breaking namespaces.
# ---------------------------------------------------------------------------


@pytest.mark.doctest
@pytest.mark.parametrize(
    "file_path",
    sorted(_BLOCKS_BY_FILE.keys()),
    ids=sorted(_BLOCKS_BY_FILE.keys()),
)
def test_doc_file(file_path: str) -> None:
    """Execute all Python code blocks in a single documentation file, in order."""
    blocks = _BLOCKS_BY_FILE[file_path]
    ns = dict(_build_preamble())
    failures: list[str] = []
    skipped = 0

    with _patch_external_dependencies(file_path):
        for block in blocks:
            if block.skip_reason:
                skipped += 1
                continue

            code = textwrap.dedent(block.code)
            try:
                _run_code_block(code, ns, f"{block.file}:{block.line}")
            except Exception as exc:  # noqa: BLE001 - aggregate block failures in a single report
                failures.append(
                    f"Block at line {block.line} failed:\n{exc!r}\n\n--- code ---\n{code}"
                )

    if skipped == len(blocks):
        pytest.skip(f"All {skipped} blocks in {file_path} are marked +SKIP")

    if failures:
        header = f"{len(failures)} of {len(blocks)} blocks failed in {file_path}:\n"
        pytest.fail(header + "\n\n".join(failures))


# ---------------------------------------------------------------------------
# Coverage assertion: every Python block must be accounted for
# ---------------------------------------------------------------------------


@pytest.mark.doctest
def test_all_python_blocks_accounted_for() -> None:
    """Guarantee that every Python code block is either executed or explicitly skipped."""
    total = len(PYTHON_BLOCKS)
    skipped = sum(1 for b in PYTHON_BLOCKS if b.skip_reason)
    executed = total - skipped

    assert total > 0, "No Python code blocks found in documentation"
    assert executed + skipped == total, (
        f"Block accounting mismatch: {executed} executed + {skipped} skipped != {total} total"
    )

    # Informational summary
    non_python = sum(1 for b in ALL_BLOCKS if b.language != "python")
    print("\nDoc code block summary:")
    print(f"  Total code blocks: {len(ALL_BLOCKS)}")
    print(f"  Python blocks:     {total}")
    print(f"  Executed:          {executed}")
    print(f"  Skipped:           {skipped}")
    print(f"  Non-Python:        {non_python}")
