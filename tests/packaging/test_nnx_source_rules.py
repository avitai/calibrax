"""Contract checks over library source rules, framework premises, and import-time side effects."""

from __future__ import annotations

from pathlib import Path

from substrax.testing.nnx_premises import check_premises
from substrax.testing.nnx_rules import findings
from substrax.testing.source_scans import (
    configures_logging,
    documented_texts,
    import_time_lines,
    python_files,
    unguarded_make_mesh_calls,
    writes_environment,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY = ("src/calibrax",)


def test_nnx_premises_hold_on_installed_frameworks() -> None:
    check_premises()


def test_library_code_has_no_unexempted_nnx_rule_site() -> None:
    found = findings(_REPO_ROOT, _LIBRARY)
    assert found == [], "\n".join(f"{f.path}:{f.line} [{f.rule}] {f.source}" for f in found)


def test_a_site_planted_in_a_copy_of_the_library_is_found(tmp_path: Path) -> None:
    module = Path("src/calibrax/metrics/learning/angular.py")
    original = (_REPO_ROOT / module).read_text(encoding="utf-8")
    planted = tmp_path / module
    planted.parent.mkdir(parents=True)
    planted.write_text(
        original
        + "\n\nclass _Planted(nnx.Module):\n    def __init__(self) -> None:\n"
        + "        super().__init__()\n",
        encoding="utf-8",
    )
    planted_line = len(original.splitlines()) + 5
    found = findings(tmp_path, _LIBRARY, rules=("noop-super-init",))
    assert (module.as_posix(), planted_line) in {(f.path, f.line) for f in found}


def test_library_modules_have_no_import_time_side_effects(tmp_path: Path) -> None:
    env_hits = {
        path.relative_to(_REPO_ROOT).as_posix(): lines
        for path in python_files(_REPO_ROOT, _LIBRARY)
        if (lines := import_time_lines(path, writes_environment))
    }
    log_hits = {
        path.relative_to(_REPO_ROOT).as_posix(): lines
        for path in python_files(_REPO_ROOT, _LIBRARY)
        if (lines := import_time_lines(path, configures_logging))
    }
    assert env_hits == {}
    assert log_hits == {}

    probe = tmp_path / "probe.py"
    probe.write_text(
        'import logging\nimport os\nos.environ["X"] = "1"\nlogging.basicConfig()\n',
        encoding="utf-8",
    )
    assert import_time_lines(probe, writes_environment) == [3]
    assert import_time_lines(probe, configures_logging) == [4]


def test_documented_texts_guard_every_make_mesh_call() -> None:
    texts = documented_texts(
        _REPO_ROOT,
        package="calibrax",
        pages=("README.md",),
        page_directories=("docs",),
        script_directories=("examples",),
    )
    unguarded = {
        key: calls for key, text in texts.items() if (calls := unguarded_make_mesh_calls(text))
    }
    assert unguarded == {}
    assert unguarded_make_mesh_calls("jax.make_mesh((2,), ('data',))") == [
        "make_mesh((2,), ('data',))"
    ]
    assert unguarded_make_mesh_calls("jax.make_mesh((2,), ('data',), axis_types=(auto,))") == []
