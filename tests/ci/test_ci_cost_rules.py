"""A merge onto ``main`` does not repeat the jobs its pull request already ran over the same tree.

A squash or rebase merge onto an unmoved ``main`` carries the tree its pull request tested, so the
jobs that would repeat that run stand down when every check of that pull request succeeded. The
compare is substrax's ``already-tested`` action, pinned by commit. A schedule or a manual run
re-measures on purpose and is never skipped.

In ``ci.yml`` the merge is also the one run that measures macOS: pull requests test on ubuntu only,
and the macOS legs join the matrix on a push to ``main``. So the test matrix narrows to macOS on an
already-tested tree rather than standing down, and the lint job it waits on is never gated, since
GitHub skips a job whose dependency was skipped.
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
GATED_WORKFLOWS = ("quality-checks.yml", "ci.yml")
# The jobs of each workflow that run on an already-tested tree. ci.yml's lint job is waited on by
# the test job, whose macOS legs no pull request ran; the test job narrows rather than stands down.
RUN_ON_A_TESTED_TREE = {
    "quality-checks.yml": frozenset({GATE_JOB}),
    "ci.yml": frozenset({GATE_JOB, "lint", "test"}),
}


def _jobs(name: str) -> dict[str, Any]:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    workflow = yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)  # noqa: S506  # BaseLoader keeps `on:` a string
    return workflow["jobs"]


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _consults_the_gate(job: dict[str, Any]) -> bool:
    return f"needs.{GATE_JOB}.outputs.skip" in yaml.safe_dump(job, width=10_000)


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
def test_every_job_that_repeats_the_pull_request_stands_down(name: str) -> None:
    jobs = _jobs(name)
    ungated = sorted(
        job_name
        for job_name, job in jobs.items()
        if job_name not in RUN_ON_A_TESTED_TREE[name]
        and (job.get("if") != GATE_CONDITION or GATE_JOB not in _needs(job))
    )

    assert ungated == [], f"{name}: these repeat the pull request without the gate: {ungated}"


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_the_jobs_that_run_on_a_tested_tree_are_never_skipped_by_the_gate(name: str) -> None:
    """A job the merge still needs has no gate condition, and neither has anything it waits on."""
    jobs = _jobs(name)
    for job_name in RUN_ON_A_TESTED_TREE[name] - {GATE_JOB}:
        waited_on = [job_name, *_needs(jobs[job_name])]
        skipped = sorted(dep for dep in waited_on if GATE_CONDITION in str(jobs[dep].get("if", "")))
        assert skipped == [], f"{name}: {job_name} must run on a tested tree but waits on {skipped}"


def test_an_already_tested_merge_runs_only_the_platform_no_pull_request_ran() -> None:
    """A merge of a tree its pull request tested repeats ubuntu; macOS it has never measured.

    Pull requests run ubuntu only and a push to main adds macOS, so an already-tested merge
    narrows the test matrix to macOS. This is artifex's ``unit_tests`` matrix.
    """
    test = _jobs("ci.yml")["test"]

    assert GATE_JOB in _needs(test)
    assert test["strategy"]["matrix"]["os"] == (
        "${{ fromJSON(needs.already_tested.outputs.skip == 'true' && '[\"macos-15\"]'"
        " || (github.ref == 'refs/heads/main' && '[\"ubuntu-latest\", \"macos-15\"]'"
        " || '[\"ubuntu-latest\"]')) }}"
    )


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_an_unanswered_gate_leaves_the_work_running(name: str) -> None:
    """Where the gate answers nothing, every job runs as it would without it.

    The compare step does not run for a pull request, a schedule or a manual run, and its lookups
    answer ``unknown`` rather than failing, so an empty output is ordinary. A consumer may stand
    work down or narrow it only on ``'true'``.
    """
    pattern = rf"needs\.{GATE_JOB}\.outputs\.skip\s*(==|!=)\s*'([a-z]*)'"
    for job_name, job in _jobs(name).items():
        if job_name == GATE_JOB or not _consults_the_gate(job):
            continue
        assert GATE_JOB in _needs(job), f"{name}: {job_name} reads the gate without waiting on it"
        assert job.get("if") in (None, GATE_CONDITION), f"{job_name} runs only when {job.get('if')}"

        compared = set(re.findall(pattern, yaml.safe_dump(job, width=10_000)))
        assert {value for _, value in compared} == {"true"}, f"{job_name} compares {compared}"
