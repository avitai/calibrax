"""One required check answers for every check a pull request runs.

The main branch ruleset names a single status check, ``required``. It is the last job of
``ci.yml``, it needs every other job there, and ``ci.yml`` calls the Security, Quality Checks and
Build Verification workflows, so a red audit, lint or package build blocks a merge as a red test
does. The documentation build, which runs only when its paths change, and CodSpeed, which compares
rather than gates, stay outside it: a required check that sometimes never reports would block every
pull request that did not trigger it.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
REQUIRED_JOB = "required"
GATE_JOB = "already_tested"
GATE_CONDITION = "needs.already_tested.outputs.skip != 'true'"
ALLS_GREEN = re.compile(r"^re-actors/alls-green@[0-9a-f]{40}$")
ALLOWED_SKIPS = re.compile(
    r"^\$\{\{ github\.event_name != 'pull_request' && '([^']*)' \|\| '([^']*)' \}\}$"
)
# Called from ci.yml and nowhere on their own pull_request or push trigger.
GATED_WORKFLOWS = ("security.yml", "quality-checks.yml", "build-verification.yml")
# The pull-request workflows deliberately outside the gate, each for the reason in the docstring.
UNGATED_PULL_REQUEST_WORKFLOWS = {"docs.yml", "codspeed.yml"}


def _load_workflow(name: str) -> dict:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)  # noqa: S506  # BaseLoader keeps `on:` a string


def _ci_jobs() -> dict[str, dict]:
    return _load_workflow("ci.yml")["jobs"]


def _needs(job: dict) -> set[str]:
    needs = job.get("needs", [])
    return {needs} if isinstance(needs, str) else set(needs)


def _names(listed: str) -> set[str]:
    return {name.strip() for name in listed.split(",") if name.strip()}


def _triggers(name: str) -> dict:
    triggers = _load_workflow(name)["on"]
    return dict.fromkeys(triggers) if isinstance(triggers, list) else triggers


def test_one_required_check_reports_every_ci_job() -> None:
    """``required`` runs whatever the others did, needs each of them, and asks alls-green."""
    jobs = _ci_jobs()
    required = jobs[REQUIRED_JOB]
    steps = required["steps"]

    assert required.get("name") == REQUIRED_JOB, "the ruleset matches the check by this name"
    assert "always()" in required.get("if", ""), "a failed or skipped job must still be reported"
    assert _needs(required) == set(jobs) - {REQUIRED_JOB}
    assert [bool(ALLS_GREEN.match(step.get("uses", ""))) for step in steps] == [True]
    assert steps[0]["with"]["jobs"] == "${{ toJSON(needs) }}"


def test_the_gated_workflows_run_only_through_ci() -> None:
    """Each gated workflow is a reusable workflow that ci.yml calls, so it reports to ``required``.

    Keeping its own pull_request or push trigger would run it twice, and its standalone run would
    be a second, unrequired check.
    """
    called = {
        Path(job["uses"]).name
        for job in _ci_jobs().values()
        if job.get("uses", "").startswith("./.github/workflows/")
    }

    for name in GATED_WORKFLOWS:
        triggers = _triggers(name)
        assert "workflow_call" in triggers, name
        assert "pull_request" not in triggers, name
        assert "push" not in triggers, name
        assert name in called, f"ci.yml does not call {name}"


def test_every_pull_request_workflow_is_gated_or_named_as_outside_the_gate() -> None:
    """A new pull-request workflow joins the gate or is listed here with its reason."""
    pull_request_workflows = {
        path.name for path in WORKFLOWS.glob("*.yml") if "pull_request" in _triggers(path.name)
    }

    assert pull_request_workflows == {"ci.yml"} | UNGATED_PULL_REQUEST_WORKFLOWS


def test_called_workflows_keep_distinct_concurrency_groups() -> None:
    """A called workflow's ``github.workflow`` names its caller, so each group needs a suffix.

    Two calls in one group would cancel each other, and a call in ci.yml's own group would cancel
    the run that made it.
    """
    groups = [_load_workflow("ci.yml")["concurrency"]["group"]]
    groups += [_load_workflow(name)["concurrency"]["group"] for name in GATED_WORKFLOWS]

    assert len(set(groups)) == len(groups), groups


def test_a_pull_request_may_skip_nothing_and_a_push_only_what_the_gate_stands_down() -> None:
    """The skips are derived from each job's own condition, not written down a second time."""
    jobs = _ci_jobs()
    stood_down = {
        name
        for name, job in jobs.items()
        if GATE_CONDITION in job.get("if", "")
        or job.get("with", {}).get("already-tested") == "${{ needs.already_tested.outputs.skip }}"
        or Path(job.get("uses", "")).name == "quality-checks.yml"
    }

    expression = ALLOWED_SKIPS.match(jobs[REQUIRED_JOB]["steps"][0]["with"]["allowed-skips"])

    assert expression, jobs[REQUIRED_JOB]["steps"][0]["with"]["allowed-skips"]
    assert _names(expression.group(2)) == set(), "a pull request may skip no job"
    assert _names(expression.group(1)) == stood_down, "what a push may skip"


def test_the_gated_calls_stay_off_the_macos_run() -> None:
    """macos.yml calls ci.yml for the suite; the gate and its calls belong to ubuntu runs only."""
    jobs = _ci_jobs()
    ubuntu_only = "(inputs.os || 'ubuntu-latest') == 'ubuntu-latest'"
    gated = [
        name
        for name, job in jobs.items()
        if name == REQUIRED_JOB or Path(job.get("uses", "")).name in GATED_WORKFLOWS
    ]

    assert len(gated) == len(GATED_WORKFLOWS) + 1, gated
    for name in gated:
        assert ubuntu_only in jobs[name].get("if", ""), name
