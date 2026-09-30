"""A merge onto ``main`` does not repeat the jobs its pull request already ran over the same tree.

A squash or rebase merge onto an unmoved ``main`` carries the tree its pull request tested, so the
jobs that would repeat that run stand down when every check of that pull request succeeded. The
compare is substrax's ``already-tested`` action, pinned by commit. A schedule or a manual run
re-measures on purpose and is never skipped.

``ci.yml`` is not gated: its test matrix adds the macOS legs only on a push to ``main``, so a
merge is the one run that measures macOS, and the lint job it waits on must run for it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
GATE_JOB = "already_tested"
GATE_CONDITION = f"needs.{GATE_JOB}.outputs.skip != 'true'"
GATE_ACTION = re.compile(r"^avitai/substrax/\.github/actions/already-tested@[0-9a-f]{40}$")
GATED_WORKFLOWS = ("quality-checks.yml",)


def _jobs(name: str) -> dict[str, Any]:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    workflow = yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)  # noqa: S506  # BaseLoader keeps `on:` a string
    return workflow["jobs"]


def _compare_step(name: str) -> dict[str, Any]:
    return next(step for step in _jobs(name)[GATE_JOB]["steps"] if step.get("id") == "compare")


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_the_gate_compares_only_on_a_push(name: str) -> None:
    """A schedule or a manual run re-measures on purpose; only a merge repeats a pull request."""
    gate = _jobs(name)[GATE_JOB]
    steps = [step for step in gate["steps"] if "already-tested" in str(step.get("uses", ""))]

    assert [step.get("if") for step in steps] == ["github.event_name == 'push'"]
    assert steps[0]["id"] in gate["outputs"]["skip"]


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_the_gate_is_the_shared_action_pinned_to_a_commit(name: str) -> None:
    """The compare is substrax's already-tested action, pinned by a full commit SHA.

    The action finds the pull request a push merged (squash or rebase) and skips only when that
    pull request tested this tree and every one of its checks succeeded; its rules are tested in
    substrax. A full commit SHA pins exactly the code that runs.
    """
    compare = _compare_step(name)

    assert GATE_ACTION.match(compare.get("uses", "")), compare.get("uses")
    assert "run" not in compare, "the gate runs the shared action, not an inline script"


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_every_job_that_repeats_the_pull_request_consults_the_gate(name: str) -> None:
    jobs = _jobs(name)
    ungated = sorted(
        job_name
        for job_name, job in jobs.items()
        if job_name != GATE_JOB
        and (job.get("if") != GATE_CONDITION or GATE_JOB not in job.get("needs", []))
    )

    assert ungated == [], f"{name}: these repeat the pull request without the gate: {ungated}"


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_an_unanswered_gate_leaves_the_work_running(name: str) -> None:
    """An empty output (no compare, or a lookup that failed) reads as "test it"."""
    for job_name, job in _jobs(name).items():
        if job_name == GATE_JOB:
            continue
        text = yaml.safe_dump(job)
        assert "outputs.skip == " not in text, f"{job_name} tests the gate for equality"
        assert "outputs.skip != 'false'" not in text, f"{job_name} runs only on an explicit false"


def test_the_merge_that_measures_macos_is_never_gated() -> None:
    """The macOS test legs run only on a push to main, so no CI job may stand down on a merge."""
    jobs = _jobs("ci.yml")

    assert "macos" in str(jobs["test"]["strategy"]["matrix"]["os"])
    assert "refs/heads/main" in str(jobs["test"]["strategy"]["matrix"]["os"])
    assert GATE_JOB not in jobs
    assert [name for name, job in jobs.items() if "if" in job] == []
