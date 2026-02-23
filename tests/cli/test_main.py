"""Tests for calibrax.cli.main module."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from calibrax.cli.main import (
    _print_energy_results,
    _print_profile_results,
    _resolve_callable,
    _run_measurement,
    _save_profile_run,
    main,
)
from calibrax.core.models import Run
from calibrax.profiling.energy import EnergySummary
from calibrax.profiling.flops import FlopsResult
from calibrax.storage.store import Store
from tests.factories import make_single_framework_run, make_throughput_latency_defs


def _make_run(
    run_id: str = "testrun",
    throughput: float = 100.0,
    latency: float = 5.0,
) -> Run:
    """Helper to create a benchmark run."""
    return make_single_framework_run(
        run_id=run_id,
        throughput=throughput,
        latency=latency,
        commit="abc123",
        branch="main",
        metric_defs=make_throughput_latency_defs(),
    )


def _setup_store(tmp_path: Path, run_id: str = "testrun") -> Store:
    """Helper to create a Store with one run."""
    store = Store(tmp_path / "data")
    run = _make_run(run_id=run_id)
    store.save(run)
    return store


class TestIngest:
    """Tests for the ingest command."""

    def test_ingest_valid_json(self, tmp_path: Path) -> None:
        """Should ingest a valid JSON file."""
        runner = CliRunner()
        data_dir = tmp_path / "data"
        input_file = tmp_path / "results.json"
        input_file.write_text(json.dumps(_make_run().to_dict()))

        result = runner.invoke(
            main,
            [
                "ingest",
                "--data",
                str(data_dir),
                "--input",
                str(input_file),
            ],
        )
        assert result.exit_code == 0
        assert "Ingested run" in result.output

    def test_ingest_invalid_json(self, tmp_path: Path) -> None:
        """Should report error on invalid JSON."""
        runner = CliRunner()
        data_dir = tmp_path / "data"
        input_file = tmp_path / "bad.json"
        input_file.write_text("not json")

        result = runner.invoke(
            main,
            [
                "ingest",
                "--data",
                str(data_dir),
                "--input",
                str(input_file),
            ],
        )
        assert result.exit_code != 0

    def test_ingest_missing_file(self, tmp_path: Path) -> None:
        """Should report error on missing file."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "ingest",
                "--data",
                str(tmp_path),
                "--input",
                str(tmp_path / "nope.json"),
            ],
        )
        assert result.exit_code != 0


class TestCheck:
    """Tests for the check command."""

    def test_check_passes(self, tmp_path: Path) -> None:
        """Check should pass when no regressions."""
        store = _setup_store(tmp_path)
        store.set_baseline("testrun")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "check",
                "--data",
                str(tmp_path / "data"),
            ],
        )
        assert result.exit_code == 0
        assert "PASSED" in result.output

    def test_check_fails_on_regression(self, tmp_path: Path) -> None:
        """Check should fail when regression detected."""
        store = _setup_store(tmp_path, run_id="baseline")
        store.set_baseline("baseline")
        # Save a regressed run
        regressed = _make_run(run_id="regressed", throughput=50.0)
        store.save(regressed)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "check",
                "--data",
                str(tmp_path / "data"),
            ],
        )
        assert result.exit_code == 1
        assert "FAILED" in result.output

    def test_check_no_baseline(self, tmp_path: Path) -> None:
        """Check without baseline should report error."""
        _setup_store(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "check",
                "--data",
                str(tmp_path / "data"),
            ],
        )
        assert result.exit_code != 0
        assert "baseline" in result.output.lower() or "Error" in result.output

    def test_check_custom_threshold(self, tmp_path: Path) -> None:
        """Custom threshold should be respected."""
        store = _setup_store(tmp_path, run_id="baseline")
        store.set_baseline("baseline")
        # 15% regression
        regressed = _make_run(run_id="regressed", throughput=85.0)
        store.save(regressed)

        runner = CliRunner()
        # Lenient threshold
        result = runner.invoke(
            main,
            [
                "check",
                "--data",
                str(tmp_path / "data"),
                "--threshold",
                "0.20",
            ],
        )
        assert result.exit_code == 0


class TestExport:
    """Tests for the export command."""

    def test_export_missing_project(self, tmp_path: Path) -> None:
        """Should error when --project is not provided."""
        _setup_store(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "export",
                "--data",
                str(tmp_path / "data"),
            ],
        )
        assert result.exit_code != 0
        assert "project" in result.output.lower()

    def test_export_missing_wandb(self, tmp_path: Path) -> None:
        """Should error when wandb is not installed."""
        from unittest.mock import patch

        _setup_store(tmp_path)
        runner = CliRunner()
        with patch(
            "calibrax.exporters.wandb.WANDB_AVAILABLE",
            False,
        ):
            result = runner.invoke(
                main,
                [
                    "export",
                    "--data",
                    str(tmp_path / "data"),
                    "--project",
                    "test",
                ],
            )
        assert result.exit_code != 0

    def test_export_missing_store(self, tmp_path: Path) -> None:
        """Should error when store has no runs."""
        Store(tmp_path / "data")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "export",
                "--data",
                str(tmp_path / "data"),
                "--project",
                "test",
                "--run",
                "nonexistent",
            ],
        )
        assert result.exit_code != 0

    def test_export_success(self, tmp_path: Path) -> None:
        """Should export run and print destination URL."""
        _setup_store(tmp_path)
        runner = CliRunner()

        fake_exporter = MagicMock()
        fake_exporter.export_run.return_value = "https://wandb.ai/test/run"
        with patch("calibrax.exporters.wandb.WandBExporter", return_value=fake_exporter):
            result = runner.invoke(
                main,
                [
                    "export",
                    "--data",
                    str(tmp_path / "data"),
                    "--project",
                    "proj",
                    "--entity",
                    "team",
                ],
            )

        assert result.exit_code == 0
        assert "Exported run" in result.output
        fake_exporter.export_run.assert_called_once()
        fake_exporter.export_analysis.assert_called_once()

    def test_export_import_error_reports_install_hint(self, tmp_path: Path) -> None:
        """Should provide install command when exporter import fails."""
        _setup_store(tmp_path)
        runner = CliRunner()
        real_import = builtins.__import__

        def _import_hook(name: str, *args: object, **kwargs: object) -> object:
            if name == "calibrax.exporters.wandb":
                raise ImportError("missing exporter module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_hook):
            result = runner.invoke(
                main,
                [
                    "export",
                    "--data",
                    str(tmp_path / "data"),
                    "--project",
                    "proj",
                ],
            )

        assert result.exit_code != 0
        assert "uv pip install 'calibrax[wandb]'" in result.output


class TestBaseline:
    """Tests for the baseline command."""

    def test_set_baseline(self, tmp_path: Path) -> None:
        """Should set baseline for a run."""
        _setup_store(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "baseline",
                "--data",
                str(tmp_path / "data"),
                "--run",
                "testrun",
            ],
        )
        assert result.exit_code == 0
        assert "Baseline set" in result.output

    def test_set_baseline_latest(self, tmp_path: Path) -> None:
        """Should set baseline to latest run."""
        _setup_store(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "baseline",
                "--data",
                str(tmp_path / "data"),
            ],
        )
        assert result.exit_code == 0

    def test_set_baseline_missing_run(self, tmp_path: Path) -> None:
        """Should error on nonexistent run."""
        _setup_store(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "baseline",
                "--data",
                str(tmp_path / "data"),
                "--run",
                "nonexistent",
            ],
        )
        assert result.exit_code != 0


class TestTrend:
    """Tests for the trend command."""

    def test_trend_with_data(self, tmp_path: Path) -> None:
        """Should display trend data."""
        _setup_store(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "trend",
                "--data",
                str(tmp_path / "data"),
                "--metric",
                "throughput",
                "--point",
                "bench1",
                "--framework",
                "jax",
            ],
        )
        assert result.exit_code == 0
        assert "100.0000" in result.output

    def test_trend_no_data(self, tmp_path: Path) -> None:
        """Should handle empty trend gracefully."""
        Store(tmp_path / "data")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "trend",
                "--data",
                str(tmp_path / "data"),
                "--metric",
                "nonexistent",
                "--point",
                "bench1",
                "--framework",
                "jax",
            ],
        )
        assert result.exit_code == 0
        assert "No trend data" in result.output

    def test_trend_missing_store(self, tmp_path: Path) -> None:
        """FileNotFoundError from Store should be reported as click error."""
        runner = CliRunner()
        with patch("calibrax.cli.main.Store") as mock_store_cls:
            mock_store_cls.return_value.extract_trend.side_effect = FileNotFoundError(
                "missing store"
            )
            result = runner.invoke(
                main,
                [
                    "trend",
                    "--data",
                    str(tmp_path / "data"),
                    "--metric",
                    "throughput",
                    "--point",
                    "bench1",
                    "--framework",
                    "jax",
                ],
            )
        assert result.exit_code != 0
        assert "missing store" in result.output.lower()


class TestSummary:
    """Tests for the summary command."""

    def test_summary_specific_run(self, tmp_path: Path) -> None:
        """Should display run summary."""
        _setup_store(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "summary",
                "--data",
                str(tmp_path / "data"),
                "--run",
                "testrun",
            ],
        )
        assert result.exit_code == 0
        assert "testrun" in result.output
        assert "jax" in result.output

    def test_summary_latest(self, tmp_path: Path) -> None:
        """Should display latest run summary."""
        _setup_store(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "summary",
                "--data",
                str(tmp_path / "data"),
            ],
        )
        assert result.exit_code == 0
        assert "Points: 1" in result.output

    def test_summary_missing_run(self, tmp_path: Path) -> None:
        """Should error on nonexistent run."""
        Store(tmp_path / "data")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "summary",
                "--data",
                str(tmp_path / "data"),
                "--run",
                "nonexistent",
            ],
        )
        assert result.exit_code != 0

    def test_summary_shows_commit(self, tmp_path: Path) -> None:
        """Summary should include commit info."""
        _setup_store(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "summary",
                "--data",
                str(tmp_path / "data"),
                "--run",
                "testrun",
            ],
        )
        assert "abc123" in result.output

    def test_summary_shows_metrics(self, tmp_path: Path) -> None:
        """Summary should include metric values."""
        _setup_store(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "summary",
                "--data",
                str(tmp_path / "data"),
                "--run",
                "testrun",
            ],
        )
        assert "throughput" in result.output

    def test_summary_omits_commit_and_branch_when_unset(self, tmp_path: Path) -> None:
        """Summary should not print commit/branch lines when values are missing."""
        store = Store(tmp_path / "data")
        run = make_single_framework_run(
            run_id="nocommit",
            throughput=10.0,
            latency=1.0,
            commit=None,
            branch=None,
            metric_defs=make_throughput_latency_defs(),
        )
        store.save(run)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "summary",
                "--data",
                str(tmp_path / "data"),
                "--run",
                "nocommit",
            ],
        )

        assert result.exit_code == 0
        assert "Commit:" not in result.output
        assert "Branch:" not in result.output


class TestMainGroup:
    """Tests for the main CLI group."""

    def test_help(self) -> None:
        """--help should show all available commands."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output
        assert "export" in result.output
        assert "check" in result.output
        assert "baseline" in result.output
        assert "trend" in result.output
        assert "summary" in result.output


class TestProfileHelpers:
    """Tests for profiling helper functions."""

    def test_run_measurement_syncs_profiled_result(self) -> None:
        """_run_measurement should block on the result returned by func()."""

        class _Result:
            def __init__(self, tracker: dict[str, int]) -> None:
                self._tracker = tracker

            def block_until_ready(self) -> None:
                self._tracker["synced"] += 1

        tracker = {"called": 0, "synced": 0}

        def _func() -> _Result:
            tracker["called"] += 1
            return _Result(tracker)

        sample, energy_summary = _run_measurement(
            _func,
            warmup=2,
            iterations=3,
            enable_energy=False,
        )

        assert energy_summary is None
        assert sample.num_batches == 5  # type: ignore[union-attr]
        assert tracker["called"] == 5
        assert tracker["synced"] == 5

    def test_run_measurement_uses_energy_monitor_when_enabled(self) -> None:
        """_run_measurement should wrap execution with EnergyMonitor when requested."""

        fake_summary = object()
        fake_monitor = MagicMock()
        fake_monitor.summary = fake_summary
        fake_monitor.__enter__.return_value = fake_monitor
        fake_monitor.__exit__.return_value = None

        with patch("calibrax.profiling.energy.EnergyMonitor", return_value=fake_monitor):
            _sample, energy_summary = _run_measurement(
                lambda: 1,
                warmup=1,
                iterations=2,
                enable_energy=True,
            )

        assert fake_monitor.__enter__.call_count == 1
        assert fake_monitor.__exit__.call_count == 1
        assert energy_summary is fake_summary

    def test_run_measurement_syncs_nested_results(self) -> None:
        """_run_measurement should sync each array-like value in tuple/list results."""

        class _Result:
            def __init__(self, tracker: dict[str, int]) -> None:
                self._tracker = tracker

            def block_until_ready(self) -> None:
                self._tracker["synced"] += 1

        tracker = {"called": 0, "synced": 0}

        def _func() -> tuple[_Result, _Result, object]:
            tracker["called"] += 1
            return _Result(tracker), _Result(tracker), object()

        sample, _ = _run_measurement(
            _func,
            warmup=1,
            iterations=2,
            enable_energy=False,
        )

        assert sample.num_batches == 3  # type: ignore[union-attr]
        assert tracker["called"] == 3
        assert tracker["synced"] == 6


class TestResolveCallable:
    """Tests for dynamic callable resolution."""

    def test_resolve_callable_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module_path = tmp_path / "tmp_profile_module.py"
        module_path.write_text("def benchmark_fn():\n    return 42\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        fn = _resolve_callable("tmp_profile_module", "benchmark_fn")
        assert callable(fn)
        assert fn() == 42

    def test_resolve_callable_missing_module_raises_click_exception(self) -> None:
        with pytest.raises(click.ClickException, match="Cannot import module"):
            _resolve_callable("module_that_does_not_exist_xyz", "fn")

    def test_resolve_callable_missing_function_raises_click_exception(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module_path = tmp_path / "tmp_profile_missing_fn.py"
        module_path.write_text("value = 1\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(click.ClickException, match="not found"):
            _resolve_callable("tmp_profile_missing_fn", "benchmark_fn")

    def test_resolve_callable_non_callable_attribute_raises_click_exception(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module_path = tmp_path / "tmp_profile_not_callable.py"
        module_path.write_text("benchmark_fn = 123\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(click.ClickException, match="not callable"):
            _resolve_callable("tmp_profile_not_callable", "benchmark_fn")


class TestProfileOutputHelpers:
    """Tests for profile-output formatting and run persistence helpers."""

    def test_print_energy_results_noop_for_non_summary(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _print_energy_results(object())
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_energy_results_prints_available_fields(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        summary = EnergySummary(
            total_gpu_energy_joules=2.5,
            total_cpu_energy_joules=1.2,
            total_combined_energy_joules=3.7,
            mean_gpu_power_watts=10.0,
            peak_gpu_power_watts=20.0,
            duration_sec=0.5,
            num_samples=4,
        )
        _print_energy_results(summary)
        output = capsys.readouterr().out
        assert "Energy Results:" in output
        assert "GPU energy: 2.5000 J" in output
        assert "CPU energy: 1.2000 J" in output
        assert "Mean GPU power: 10.00 W" in output
        assert "Total energy: 3.7000 J" in output

    def test_print_profile_results_includes_timing_flops_and_energy(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        sample = SimpleNamespace(
            wall_clock_sec=0.4,
            num_batches=3,
            warmup_batches_excluded=1,
            per_batch_times=(0.1, 0.2),
        )
        flops_result = FlopsResult(
            total_flops=1234,
            flops_by_operation={"add": 1000, "mul": 234},
            num_operations=2,
            function_name="bench_fn",
        )
        energy_summary = EnergySummary(
            total_gpu_energy_joules=1.0,
            total_cpu_energy_joules=None,
            total_combined_energy_joules=1.0,
            mean_gpu_power_watts=8.0,
            peak_gpu_power_watts=9.0,
            duration_sec=0.4,
            num_samples=5,
        )

        _print_profile_results(sample, flops_result, energy_summary)
        output = capsys.readouterr().out
        assert "Timing Results:" in output
        assert "FLOP Analysis (bench_fn):" in output
        assert "Breakdown:" in output
        assert "Energy Results:" in output

    def test_save_profile_run_persists_all_optional_metrics(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        sample = SimpleNamespace(wall_clock_sec=1.5, per_batch_times=(0.5, 0.7))
        flops_result = FlopsResult(
            total_flops=5000,
            flops_by_operation={},
            num_operations=0,
            function_name="bench_fn",
        )
        energy_summary = EnergySummary(
            total_gpu_energy_joules=2.0,
            total_cpu_energy_joules=3.0,
            total_combined_energy_joules=5.0,
            mean_gpu_power_watts=10.0,
            peak_gpu_power_watts=12.0,
            duration_sec=1.5,
            num_samples=5,
        )

        _save_profile_run(
            tmp_path / "data",
            "tmp_module",
            "bench_fn",
            sample,
            flops_result,
            energy_summary,
        )

        run = Store(tmp_path / "data").latest()
        metrics = run.points[0].metrics
        assert "wall_clock_sec" in metrics
        assert "mean_batch_time_sec" in metrics
        assert "total_flops" in metrics
        assert "gpu_energy_joules" in metrics
        assert "cpu_energy_joules" in metrics
        assert "Saved profiling run" in capsys.readouterr().out

    def test_save_profile_run_without_optional_metrics(self, tmp_path: Path) -> None:
        sample = SimpleNamespace(wall_clock_sec=1.0, per_batch_times=())

        _save_profile_run(
            tmp_path / "data",
            "tmp_module",
            "bench_fn",
            sample,
            flops_result=None,
            energy_summary=None,
        )

        run = Store(tmp_path / "data").latest()
        metrics = run.points[0].metrics
        assert set(metrics.keys()) == {"wall_clock_sec"}


class TestProfileCommand:
    """Integration-style tests for the profile CLI command."""

    def test_profile_with_flops_and_data_saves_run(self, tmp_path: Path) -> None:
        runner = CliRunner()
        sample = SimpleNamespace(
            wall_clock_sec=0.2,
            num_batches=2,
            warmup_batches_excluded=0,
            per_batch_times=(0.1,),
        )
        flops_result = FlopsResult(
            total_flops=100,
            flops_by_operation={"add": 100},
            num_operations=1,
            function_name="bench_fn",
        )

        with (
            patch("calibrax.cli.main._resolve_callable", return_value=lambda: 1) as resolve_mock,
            patch("calibrax.cli.main._run_measurement", return_value=(sample, None)) as run_mock,
            patch("calibrax.cli.main._print_profile_results") as print_mock,
            patch("calibrax.cli.main._save_profile_run") as save_mock,
            patch("calibrax.profiling.flops.FlopsCounter") as counter_cls,
        ):
            counter_cls.return_value.count.return_value = flops_result
            result = runner.invoke(
                main,
                [
                    "profile",
                    "--module",
                    "m",
                    "--function",
                    "f",
                    "--warmup",
                    "1",
                    "--iterations",
                    "1",
                    "--flops",
                    "--data",
                    str(tmp_path / "data"),
                ],
            )

        assert result.exit_code == 0
        assert "Profile complete." in result.output
        resolve_mock.assert_called_once_with("m", "f")
        run_mock.assert_called_once()
        print_mock.assert_called_once_with(sample, flops_result, None)
        save_mock.assert_called_once()

    def test_profile_flops_failure_prints_error_and_continues(self) -> None:
        runner = CliRunner()
        sample = SimpleNamespace(
            wall_clock_sec=0.2,
            num_batches=2,
            warmup_batches_excluded=0,
            per_batch_times=(0.1,),
        )

        with (
            patch("calibrax.cli.main._resolve_callable", return_value=lambda: 1),
            patch("calibrax.cli.main._run_measurement", return_value=(sample, None)),
            patch("calibrax.cli.main._print_profile_results") as print_mock,
            patch("calibrax.profiling.flops.FlopsCounter") as counter_cls,
        ):
            counter_cls.return_value.count.side_effect = RuntimeError("flops failed")
            result = runner.invoke(
                main,
                [
                    "profile",
                    "--module",
                    "m",
                    "--function",
                    "f",
                    "--flops",
                ],
            )

        assert result.exit_code == 0
        assert "FLOP counting failed: flops failed" in result.output
        assert "Profile complete." in result.output
        print_mock.assert_called_once_with(sample, None, None)
