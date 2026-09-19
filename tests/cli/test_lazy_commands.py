"""The CLI loads each command's module when the command runs, so help needs none of them."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from substrax.testing import run_python

from calibrax.cli.main import COMMANDS, LazyCommand, main


_LOADED_PROBE = (
    "import sys\n"
    "from click.testing import CliRunner\n"
    "from calibrax.cli.main import main\n"
    "result = CliRunner().invoke(main, {args!r})\n"
    "print(result.exit_code)\n"
    "print(sorted(name for name in ('jax', 'flax', 'wandb', 'pynvml') if name in sys.modules))\n"
)


def test_help_lists_every_command_without_loading_one() -> None:
    result = run_python(_LOADED_PROBE.format(args=["--help"]), timeout=120)

    assert result.stdout.split("\n")[:2] == ["0", "[]"]


def test_help_names_every_registered_command() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    listed = " ".join(result.output.split())  # help wraps long summaries
    for name, command in COMMANDS.items():
        assert name in listed
        assert command.summary in listed


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_each_summary_is_the_command_docstring_first_line(name: str) -> None:
    # Read from source, so commands whose extra is not installed are checked too.
    module, _, function = COMMANDS[name].target.partition(":")
    spec = importlib.util.find_spec(module)
    assert spec is not None
    assert spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text())
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function
    ]

    assert len(definitions) == 1
    docstring = ast.get_docstring(definitions[0])
    assert docstring is not None
    assert docstring.splitlines()[0] == COMMANDS[name].summary


def test_an_unknown_command_is_a_usage_error() -> None:
    result = CliRunner().invoke(main, ["no-such-command"])

    assert result.exit_code == 2
    assert "No such command" in result.output


def test_a_command_whose_extra_is_missing_names_the_extra() -> None:
    probe = (
        "import sys; sys.modules['pynvml'] = None\n"
        + _LOADED_PROBE.format(args=["profile-gpu", "--module", "m", "--function", "f"])
        + "print(result.output)\n"
    )

    result = run_python(probe, timeout=120)

    assert result.stdout.split("\n")[0] == "1"
    assert "calibrax[cuda12]" in result.stdout


def test_list_commands_is_the_registry() -> None:
    assert main.list_commands(click.Context(main)) == sorted(COMMANDS)


def test_a_registered_target_that_is_not_a_command_is_a_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        COMMANDS, "broken", LazyCommand(target="calibrax.cli.main:LazyCommand", summary="Broken.")
    )

    with pytest.raises(TypeError, match="is not a click command"):
        main.get_command(click.Context(main), "broken")
