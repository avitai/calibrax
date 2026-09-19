"""Click-based CLI for calibrax benchmark management.

Commands for ingesting results, exporting to W&B, checking for regressions, managing
baselines, viewing trends, profiling and summarizing runs. Each command lives in a module
that imports what it needs at the top, and the group loads that module only when the command
runs (click's lazily loaded subcommands), so ``calibrax --help`` and the store commands do not
load JAX, and a command whose extra is missing fails with the install command.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import click


@dataclass(frozen=True, slots=True, kw_only=True)
class LazyCommand:
    """Where a command is defined, and the summary ``--help`` lists before loading it.

    Attributes:
        target: ``"module:function"`` of the ``click.Command``.
        summary: The first line of the command's docstring.
    """

    target: str
    summary: str


COMMANDS: dict[str, LazyCommand] = {
    "baseline": LazyCommand(
        target="calibrax.cli.store_commands:baseline",
        summary="Set a run as the baseline for regression checks.",
    ),
    "check": LazyCommand(
        target="calibrax.cli.store_commands:check",
        summary="Check for performance regressions (CI gate).",
    ),
    "export": LazyCommand(
        target="calibrax.cli.export:export",
        summary="Export a run to Weights & Biases.",
    ),
    "ingest": LazyCommand(
        target="calibrax.cli.store_commands:ingest",
        summary="Import JSON benchmark results into the store.",
    ),
    "profile": LazyCommand(
        target="calibrax.cli.profile:profile",
        summary="Profile a JAX function: timing, and optionally FLOPs and CPU energy.",
    ),
    "profile-gpu": LazyCommand(
        target="calibrax.cli.profile_gpu:profile_gpu",
        summary="Profile a JAX function with GPU and CPU energy, the GPU read through NVML.",
    ),
    "summary": LazyCommand(
        target="calibrax.cli.store_commands:summary",
        summary="Show a human-readable run summary.",
    ),
    "trend": LazyCommand(
        target="calibrax.cli.store_commands:trend",
        summary="Show metric trends over time.",
    ),
}


class _LazyGroup(click.Group):
    """A group whose commands load from ``COMMANDS`` when invoked."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Every registered command name."""
        del ctx
        return sorted(COMMANDS)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Load the command's module and return the command.

        Args:
            ctx: The click context.
            cmd_name: The command's registered name.

        Returns:
            The command, or None for a name the registry does not hold.

        Raises:
            click.ClickException: If the command's module needs an extra that is not installed.
            TypeError: If the registered target is not a ``click.Command``.
        """
        del ctx
        entry = COMMANDS.get(cmd_name)
        if entry is None:
            return None
        module_name, _, function = entry.target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise click.ClickException(str(error)) from error
        command = getattr(module, function)
        if not isinstance(command, click.Command):
            msg = f"{entry.target} is not a click command"
            raise TypeError(msg)
        return command

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """List the commands from their registered summaries, loading none of them."""
        del ctx
        with formatter.section("Commands"):
            formatter.write_dl([(name, entry.summary) for name, entry in sorted(COMMANDS.items())])


@click.group(cls=_LazyGroup)
def main() -> None:
    """Calibrax: unified benchmarking framework CLI."""
