"""Tests for the profile CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from calibrax.cli.main import main


def _make_mock_sample() -> MagicMock:
    """Create a mock TimingSample for testing."""
    sample = MagicMock()
    sample.wall_clock_sec = 0.1234
    sample.num_batches = 11
    sample.warmup_batches_excluded = 1
    sample.per_batch_times = [0.01] * 10
    return sample


class TestProfileCommand:
    """Tests for the 'profile' CLI command."""

    def test_nonexistent_module_shows_error(self) -> None:
        """profile with a non-existent module should show an error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["profile", "--module", "nonexistent_module_xyz", "--function", "foo"],
        )
        assert result.exit_code != 0
        assert "Cannot import module" in result.output

    def test_nonexistent_function_shows_error(self) -> None:
        """profile with a valid module but non-existent function should show an error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["profile", "--module", "math", "--function", "nonexistent_func_xyz"],
        )
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_valid_module_and_function(self) -> None:
        """profile with a valid module and function should run successfully."""
        mock_sample = _make_mock_sample()
        mock_collector_cls = MagicMock()
        mock_collector_cls.return_value.measure_iteration.return_value = mock_sample

        mock_module = MagicMock()
        mock_module.my_func = MagicMock()

        with (
            patch("calibrax.profiling.timing.TimingCollector", mock_collector_cls),
            patch("calibrax.cli.main.importlib.import_module", return_value=mock_module),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "profile",
                    "--module",
                    "my_pkg.benchmark",
                    "--function",
                    "my_func",
                    "--warmup",
                    "1",
                    "--iterations",
                    "10",
                ],
            )

        assert result.exit_code == 0
        assert "Profiling my_pkg.benchmark.my_func" in result.output
        assert "Profile complete" in result.output

    def test_missing_module_option(self) -> None:
        """profile without --module should show usage error."""
        runner = CliRunner()
        result = runner.invoke(main, ["profile", "--function", "foo"])
        assert result.exit_code != 0
        assert "module" in result.output.lower()

    def test_missing_function_option(self) -> None:
        """profile without --function should show usage error."""
        runner = CliRunner()
        result = runner.invoke(main, ["profile", "--module", "math"])
        assert result.exit_code != 0
        assert "function" in result.output.lower()

    def test_profile_shows_timing_results(self) -> None:
        """profile output should include timing result details."""
        mock_sample = MagicMock()
        mock_sample.wall_clock_sec = 2.5
        mock_sample.num_batches = 6
        mock_sample.warmup_batches_excluded = 1
        mock_sample.per_batch_times = [0.5] * 5

        mock_collector_cls = MagicMock()
        mock_collector_cls.return_value.measure_iteration.return_value = mock_sample

        mock_module = MagicMock()
        mock_module.run = MagicMock()

        with (
            patch("calibrax.profiling.timing.TimingCollector", mock_collector_cls),
            patch("calibrax.cli.main.importlib.import_module", return_value=mock_module),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "profile",
                    "--module",
                    "my_pkg",
                    "--function",
                    "run",
                    "--iterations",
                    "5",
                ],
            )

        assert result.exit_code == 0
        assert "Wall clock: 2.5000s" in result.output
        assert "Mean batch time" in result.output

    def test_profile_help(self) -> None:
        """profile --help should show command documentation."""
        runner = CliRunner()
        result = runner.invoke(main, ["profile", "--help"])
        assert result.exit_code == 0
        assert "--module" in result.output
        assert "--function" in result.output
        assert "--warmup" in result.output
        assert "--iterations" in result.output
        assert "--energy" in result.output
        assert "--flops" in result.output

    def test_profile_with_energy_flag(self) -> None:
        """profile with --energy should include energy results in output."""
        mock_sample = _make_mock_sample()
        mock_collector_cls = MagicMock()
        mock_collector_cls.return_value.measure_iteration.return_value = mock_sample

        mock_module = MagicMock()
        mock_module.fn = MagicMock()

        mock_energy_summary = MagicMock()
        mock_energy_summary.duration_sec = 0.1
        mock_energy_summary.num_samples = 5
        mock_energy_summary.total_gpu_energy_joules = None
        mock_energy_summary.total_cpu_energy_joules = None
        mock_energy_summary.mean_gpu_power_watts = None
        mock_energy_summary.total_combined_energy_joules = None

        mock_monitor = MagicMock()
        mock_monitor.__enter__ = MagicMock(return_value=mock_monitor)
        mock_monitor.__exit__ = MagicMock(return_value=False)
        mock_monitor.summary = mock_energy_summary
        mock_energy_cls = MagicMock(return_value=mock_monitor)

        with (
            patch("calibrax.profiling.timing.TimingCollector", mock_collector_cls),
            patch("calibrax.cli.main.importlib.import_module", return_value=mock_module),
            patch("calibrax.profiling.energy.EnergyMonitor", mock_energy_cls),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "profile",
                    "--module",
                    "pkg",
                    "--function",
                    "fn",
                    "--energy",
                ],
            )

        assert result.exit_code == 0
        assert "Energy Results" in result.output

    def test_profile_with_flops_flag(self) -> None:
        """profile with --flops should include FLOP counting note in output."""
        mock_sample = _make_mock_sample()
        mock_collector_cls = MagicMock()
        mock_collector_cls.return_value.measure_iteration.return_value = mock_sample

        mock_module = MagicMock()
        mock_module.fn = MagicMock()

        with (
            patch("calibrax.profiling.timing.TimingCollector", mock_collector_cls),
            patch("calibrax.cli.main.importlib.import_module", return_value=mock_module),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "profile",
                    "--module",
                    "pkg",
                    "--function",
                    "fn",
                    "--flops",
                ],
            )

        assert result.exit_code == 0
        assert "FLOP counting" in result.output
