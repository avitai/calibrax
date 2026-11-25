"""CI fails when coverage drops below the floor, and uploads coverage nowhere."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
CAP_OVERRIDES = ("--no-cov", "-o addopts", "--override-ini")
# The already-tested gate's condition: the one condition the test job may carry. The gate's own
# rules are checked once for the whole stack, not by a copy in this repository.
GATE_CONDITION = "needs.already_tested.outputs.skip != 'true'"


def _load_workflow(path: Path) -> dict:
    # BaseLoader keeps every scalar a string, so the `on:` key stays `on` rather than True.
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)  # noqa: S506  # BaseLoader keeps `on:` a string


def _pytest_commands(job: dict) -> list[str]:
    return [step["run"] for step in job.get("steps", []) if "pytest" in step.get("run", "")]


def _needs(job: dict) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _combined_floors(coverage_job: dict, pyproject: dict) -> tuple[list[str], list[int]]:
    """Return the coverage job's ``coverage report`` commands and the floors they apply."""
    runs = "\n".join(step.get("run", "") for step in coverage_job.get("steps", []))
    reports = re.findall(r"coverage report[^\n]*", runs)
    # A --fail-under on the report command overrides [tool.coverage.report] fail_under.
    floors = [
        int(cap) for report in reports for cap in re.findall(r"--fail-under[= ](\d+)", report)
    ]
    configured = pyproject["tool"].get("coverage", {}).get("report", {}).get("fail_under")
    if not floors and configured is not None:
        floors = [int(configured)]
    return reports, floors


def coverage_floor_violations(workflow: dict, pyproject: dict) -> list[str]:
    """Return why CI would not fail when the suite's coverage drops below the floor, if it would not.

    The test job runs the suite in shards that each measure part of it, so no shard holds the
    floor: the coverage job combines every shard's data and holds it over the whole suite.
    """
    jobs = workflow["jobs"]
    test, coverage = jobs["test"], jobs.get("coverage")
    problems = []
    if not {"push", "pull_request"} <= set(workflow["on"]):
        problems.append(f"CI runs on {sorted(workflow['on'])}, not on both push and pull_request")
    # The one condition allowed is the already-tested gate: it skips a merge only when its pull
    # request ran these jobs, floor included, over the same tree with every check green.
    for name, job in (("test", test), ("coverage", coverage or {})):
        if job.get("if", GATE_CONDITION) != GATE_CONDITION:
            problems.append(f"the {name} job only runs when {job['if']}")
    commands = _pytest_commands(test)
    if not commands:
        problems.append("the test job runs no pytest command")
    problems += [
        f"a pytest command stops measuring coverage with {override}"
        for command in commands
        for override in CAP_OVERRIDES
        if override in command
    ]
    if coverage is None:
        return [*problems, "no coverage job combines the test job's coverage data"]
    if "test" not in _needs(coverage):
        problems.append(f"the coverage job needs {_needs(coverage)}, not the test job")
    if not any("coverage combine" in step.get("run", "") for step in coverage.get("steps", [])):
        problems.append("the coverage job does not combine the shards' data")
    reports, floors = _combined_floors(coverage, pyproject)
    if not reports:
        problems.append("the coverage job runs no `coverage report`")
    elif not floors or min(floors) < 80:
        problems.append(f"the combined coverage floor is {floors}, not at least 80")
    return problems


def test_ci_fails_below_the_coverage_floor_over_the_whole_suite() -> None:
    """Every tree that reaches main passed a coverage floor of at least 80% over the whole suite.

    CI runs it on every pull request, and on every push except a merge whose pull request already
    ran it over the same tree.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert coverage_floor_violations(_load_workflow(WORKFLOWS / "ci.yml"), pyproject) == []


def test_the_floor_contract_refuses_a_lowered_floor_and_a_missing_combine() -> None:
    """The contract fails on the two ways the floor could silently stop applying."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    workflow = _load_workflow(WORKFLOWS / "ci.yml")
    tool = pyproject["tool"]
    lowered = {**pyproject, "tool": {**tool, "coverage": {**tool["coverage"], "report": {}}}}
    lowered["tool"]["coverage"]["report"]["fail_under"] = 70
    without_combine = {
        **workflow,
        "jobs": {name: job for name, job in workflow["jobs"].items() if name != "coverage"},
    }

    assert any("not at least 80" in p for p in coverage_floor_violations(workflow, lowered))
    assert "no coverage job combines the test job's coverage data" in coverage_floor_violations(
        without_combine, pyproject
    )


def test_ci_uploads_no_coverage_to_codecov() -> None:
    """coverage.py in the test job is the coverage gate; no workflow uploads to Codecov."""
    uses = [
        step.get("uses", "")
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for job in _load_workflow(path).get("jobs", {}).values()
        for step in job.get("steps", [])
    ]

    assert any(action.startswith("actions/checkout@") for action in uses)
    assert [action for action in uses if action.startswith("codecov/")] == []


def _cov_sources(command_line: str) -> list[str]:
    return re.findall(r"--cov=(\S+)", command_line)


def test_pytest_measures_coverage_by_source_directory() -> None:
    """Every ``--cov``, in addopts and in the CI test command, names a directory.

    coverage treats a ``--cov`` value that is not a directory as a package and matches executed
    code to it by module name, so a package file run as a script (``__main__``, no module spec)
    never counts toward the package and reports 0%.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    addopts = pyproject["tool"]["pytest"]["ini_options"].get("addopts", "")
    addopts = " ".join(addopts) if isinstance(addopts, list) else addopts
    workflow = _load_workflow(WORKFLOWS / "ci.yml")
    places = [("addopts", addopts)] + [
        (f"ci.yml test command {index}", command)
        for index, command in enumerate(_pytest_commands(workflow["jobs"]["test"]))
    ]

    assert len(places) > 1, "ci.yml's test job runs no pytest command"
    for where, command_line in places:
        sources = _cov_sources(command_line)
        assert sources, f"{where} measures no coverage source"
        assert [source for source in sources if not (ROOT / source).is_dir()] == [], where


def test_every_uv_cache_is_pruned_before_it_is_saved() -> None:
    """A saved uv cache holds only what uv built, not every wheel it downloaded.

    setup-uv prunes only when asked (``prune-cache`` defaults to false from v9); unpruned, the
    caches of the heavy extras grow to gigabytes each and evict the repository's other caches.
    """
    github = ROOT / ".github"
    documents = [
        *sorted(github.glob("workflows/*.yml")),
        *sorted(github.glob("actions/*/action.yml")),
    ]
    checked = 0
    unpruned: list[str] = []
    for path in documents:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        owners = {**document.get("jobs", {}), "runs": document.get("runs") or {}}
        for owner, body in owners.items():
            for step in body.get("steps", []):
                if not str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
                    continue
                checked += 1
                if (step.get("with") or {}).get("prune-cache") is not True:
                    unpruned.append(f"{path.relative_to(ROOT)}:{owner}")

    assert checked, "no setup-uv step found; the contract is reading the wrong files"
    assert unpruned == [], f"setup-uv steps saving an unpruned cache: {unpruned}"
