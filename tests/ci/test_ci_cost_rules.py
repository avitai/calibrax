"""CI runs only what a change needs: a merge does not repeat its pull request, and macOS runs nightly.

A squash or rebase merge onto an unmoved ``main`` carries the tree its pull request tested, so the
jobs that would repeat that run stand down when every check of that pull request succeeded. The
compare is substrax's ``already-tested`` action, pinned by commit. A schedule or a manual run
re-measures on purpose and is never skipped.

macOS runners are scarce, so no push or pull request runs on them. ``macos.yml`` runs the suite and
the build on macOS nightly when ``main`` has moved since the last green night (substrax's
``main-moved`` action), on demand, and on the release commit before its tag, by calling ``ci.yml``
and ``build-verification.yml`` with ``os: macos-15``.
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

NIGHTLY = "macos.yml"
MOVED_JOB = "main_moved"
MOVED_CONDITION = f"needs.{MOVED_JOB}.outputs.unchanged != 'true'"
MOVED_ACTION = re.compile(r"^avitai/substrax/\.github/actions/main-moved@[0-9a-f]{40}$")
MACOS_RUNNER = "macos-15"
# The workflows the nightly runs on macOS; each builds on ubuntu for a push or pull request.
CALLED_WORKFLOWS = ("ci.yml", "build-verification.yml")
CALLER_OR_UBUNTU = "${{ inputs.os || 'ubuntu-latest' }}"


def _workflow(name: str) -> dict[str, Any]:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)  # noqa: S506  # BaseLoader keeps `on:` a string


def _jobs(name: str) -> dict[str, Any]:
    return _workflow(name)["jobs"]


def _triggers(name: str) -> set[str]:
    triggers = _workflow(name)["on"]
    return {triggers} if isinstance(triggers, str) else set(triggers)


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _consults_the_gate(job: dict[str, Any]) -> bool:
    return f"needs.{GATE_JOB}.outputs.skip" in yaml.safe_dump(job, width=10_000)


def _compare_step(name: str, job: str) -> dict[str, Any]:
    return next(step for step in _jobs(name)[job]["steps"] if step.get("id") == "compare")


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
    compare = _compare_step(name, GATE_JOB)

    assert GATE_ACTION.match(compare.get("uses", "")), compare.get("uses")
    assert "run" not in compare, "the gate runs the shared action, not an inline script"


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_every_job_that_repeats_the_pull_request_stands_down(name: str) -> None:
    jobs = _jobs(name)
    ungated = sorted(
        job_name
        for job_name, job in jobs.items()
        if job_name != GATE_JOB and (job.get("if") != GATE_CONDITION or GATE_JOB not in _needs(job))
    )

    assert ungated == [], f"{name}: these repeat the pull request without the gate: {ungated}"


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_an_unanswered_gate_leaves_the_work_running(name: str) -> None:
    """Where the gate answers nothing, every job runs as it would without it.

    The compare step does not run for a pull request, a schedule, a manual run or a call from the
    nightly, and its lookups answer ``unknown`` rather than failing, so an empty output is
    ordinary. A consumer may stand work down only on ``'true'``.
    """
    pattern = rf"needs\.{GATE_JOB}\.outputs\.skip\s*(==|!=)\s*'([a-z]*)'"
    for job_name, job in _jobs(name).items():
        if job_name == GATE_JOB or not _consults_the_gate(job):
            continue
        assert GATE_JOB in _needs(job), f"{name}: {job_name} reads the gate without waiting on it"
        assert job.get("if") == GATE_CONDITION, f"{job_name} runs only when {job.get('if')}"

        compared = set(re.findall(pattern, yaml.safe_dump(job, width=10_000)))
        assert {value for _, value in compared} == {"true"}, f"{job_name} compares {compared}"


def test_no_push_or_pull_request_runs_on_macos() -> None:
    """macOS runners queue for hours; a push or a pull request is measured on ubuntu alone."""
    on_macos = sorted(
        path.name
        for path in WORKFLOWS.glob("*.yml")
        if _triggers(path.name) & {"push", "pull_request"}
        and "macos" in yaml.safe_dump(_jobs(path.name), width=10_000).lower()
    )

    assert on_macos == [], f"these run on macOS for a push or pull request: {on_macos}"


def test_macos_runs_nightly_and_on_demand_only() -> None:
    assert _triggers(NIGHTLY) == {"schedule", "workflow_dispatch"}


def test_a_quiet_night_stands_down() -> None:
    """A night on which main has not moved since the last green night runs nothing on macOS.

    The compare is substrax's main-moved action, pinned by a full commit SHA; it compares only on
    a schedule, so a manual run (the release check) always runs.
    """
    jobs = _jobs(NIGHTLY)
    compare = _compare_step(NIGHTLY, MOVED_JOB)

    assert MOVED_ACTION.match(compare.get("uses", "")), compare.get("uses")
    assert compare.get("with", {}).get("workflow") == NIGHTLY
    assert compare["id"] in jobs[MOVED_JOB]["outputs"]["unchanged"]
    for job_name, job in jobs.items():
        if job_name == MOVED_JOB:
            continue
        assert MOVED_JOB in _needs(job), job_name
        assert job.get("if") == MOVED_CONDITION, f"{job_name} runs when {job.get('if')}"


def test_the_nightly_runs_the_suite_and_the_build_on_macos() -> None:
    """The nightly calls the workflows a pull request runs, on macOS, rather than copying them."""
    called = {
        job["uses"]: job.get("with", {}).get("os")
        for job in _jobs(NIGHTLY).values()
        if "uses" in job
    }

    assert called == {f"./.github/workflows/{name}": MACOS_RUNNER for name in CALLED_WORKFLOWS}


def test_the_nightly_holds_at_most_two_macos_runners() -> None:
    """The build waits for the suite, so the nightly's two-leg matrices never overlap."""
    jobs = _jobs(NIGHTLY)
    by_workflow = {Path(job["uses"]).name: name for name, job in jobs.items() if "uses" in job}

    assert by_workflow["ci.yml"] in _needs(jobs[by_workflow["build-verification.yml"]])
    for name in CALLED_WORKFLOWS:
        legs = [job["strategy"]["matrix"]["python-version"] for job in _macos_jobs(name)]
        assert [len(versions) for versions in legs] == [2], f"{name}: {legs}"


def _macos_jobs(name: str) -> list[dict[str, Any]]:
    """The jobs of a called workflow that run on the runner the caller names."""
    return [job for job in _jobs(name).values() if job.get("runs-on") == CALLER_OR_UBUNTU]


@pytest.mark.parametrize("name", CALLED_WORKFLOWS)
def test_a_called_workflow_runs_on_the_callers_runner_or_ubuntu(name: str) -> None:
    workflow = _workflow(name)

    assert workflow["on"]["workflow_call"]["inputs"]["os"]["default"] == "ubuntu-latest"
    assert _macos_jobs(name), f"{name} has no job on the caller's runner"


@pytest.mark.parametrize("name", CALLED_WORKFLOWS)
def test_a_called_workflow_keeps_its_own_concurrency_group(name: str) -> None:
    """Called from the nightly, ``github.workflow`` names the caller.

    A called workflow whose group is the caller's own cancels the call as a deadlock, so each
    group carries the called file's name.
    """
    group = _workflow(name)["concurrency"]["group"]

    assert group == f"${{{{ github.workflow }}}}-{Path(name).stem}-${{{{ github.ref }}}}"


def test_the_nightly_grants_the_reads_its_called_jobs_ask_for() -> None:
    """A called job may ask only for permissions its caller holds, or the run fails to start."""
    granted = _workflow(NIGHTLY)["permissions"]
    asked = {
        (scope, level)
        for name in CALLED_WORKFLOWS
        for job in _jobs(name).values()
        for scope, level in job.get("permissions", {}).items()
    }

    missing = {(scope, level) for scope, level in asked if granted.get(scope) != level}
    assert missing == set(), f"{NIGHTLY} does not grant what its called jobs ask for: {missing}"


def test_a_release_runs_the_macos_workflow_before_its_tag() -> None:
    """The release commit is measured on macOS before the tag that publishes it."""
    checklist = (ROOT / "RELEASING.md").read_text()

    assert "gh workflow run macos.yml" in checklist
    assert checklist.index("gh workflow run macos.yml") < checklist.index("git tag -a")
