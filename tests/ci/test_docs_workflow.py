"""The documentation is built as a check; Read the Docs builds and hosts it.

``.readthedocs.yaml`` publishes the site. No workflow deploys to GitHub Pages, and ``docs.yml``
only proves the site builds strictly, on a pull request and on a push to ``main``, with a
read-only token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
DOCS_WORKFLOW = WORKFLOWS / "docs.yml"
PAGES_ACTIONS = (
    "peaceiris/actions-gh-pages",
    "actions/deploy-pages",
    "actions/upload-pages-artifact",
)
READ_ONLY = {"contents": "read"}


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(document: dict[str, Any]) -> dict[str, Any]:
    """The ``on`` mapping; PyYAML reads the bare key ``on`` as the boolean ``True``."""
    keys: dict[Any, Any] = document
    return keys.get("on") or keys.get(True) or {}


def _permission_blocks(document: dict[str, Any]) -> list[Any]:
    """Every ``permissions`` block: the workflow's own, then each job's."""
    blocks = [document["permissions"]] if "permissions" in document else []
    blocks += [job["permissions"] for job in document["jobs"].values() if "permissions" in job]
    return blocks


def _steps(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in document["jobs"].values() for step in job.get("steps", [])]


def _workflow_paths() -> list[Path]:
    return sorted(WORKFLOWS.glob("*.yml"))


def test_the_contract_reads_the_workflows() -> None:
    """A positive control: the scan below must see docs.yml and its steps."""
    assert DOCS_WORKFLOW in _workflow_paths()
    assert _steps(_load(DOCS_WORKFLOW))


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda path: path.name)
def test_no_workflow_deploys_to_github_pages(path: Path) -> None:
    document = _load(path)
    uses = [str(step.get("uses", "")) for step in _steps(document)]

    assert [use for use in uses if use.startswith(PAGES_ACTIONS)] == [], path.name
    assert [block for block in _permission_blocks(document) if "pages" in block] == [], path.name


def test_the_docs_workflow_holds_a_read_only_token() -> None:
    blocks = _permission_blocks(_load(DOCS_WORKFLOW))

    assert blocks, "docs.yml inherits the default token permissions"
    assert all(block == READ_ONLY for block in blocks), blocks


@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_the_docs_build_runs_on_pull_requests_and_pushes_to_main(event: str) -> None:
    triggers = _triggers(_load(DOCS_WORKFLOW))

    assert event in triggers
    assert triggers[event]["branches"] == ["main"]
    assert triggers[event]["paths"] == triggers["push"]["paths"]


def test_the_docs_build_is_strict() -> None:
    commands = [str(step.get("run", "")) for step in _steps(_load(DOCS_WORKFLOW))]
    builds = [command for command in commands if "mkdocs build" in command]

    assert builds, "docs.yml never builds the site"
    assert all("--strict" in command for command in builds), builds
