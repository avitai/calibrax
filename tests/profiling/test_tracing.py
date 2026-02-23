"""Tests for calibrax.profiling.tracing module."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from calibrax.profiling.tracing import TraceLinker, TraceReference


class TestTraceReference:
    """Tests for TraceReference frozen dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Should create TraceReference with default run_id=None."""
        ref = TraceReference(trace_dir="/tmp/trace")
        assert ref.trace_dir == "/tmp/trace"
        assert ref.run_id is None

    def test_creation_with_run_id(self) -> None:
        """Should create TraceReference with explicit run_id."""
        ref = TraceReference(trace_dir="/tmp/trace", run_id="run123")
        assert ref.trace_dir == "/tmp/trace"
        assert ref.run_id == "run123"

    def test_frozen_immutability(self) -> None:
        """Should raise FrozenInstanceError on attribute mutation."""
        ref = TraceReference(trace_dir="/tmp/trace", run_id="run1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.trace_dir = "/other"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.run_id = "other"  # type: ignore[misc]

    def test_to_dict_without_run_id(self) -> None:
        """to_dict should omit run_id when None."""
        ref = TraceReference(trace_dir="/tmp/trace")
        d = ref.to_dict()
        assert d == {"trace_dir": "/tmp/trace"}
        assert "run_id" not in d

    def test_to_dict_with_run_id(self) -> None:
        """to_dict should include run_id when set."""
        ref = TraceReference(trace_dir="/tmp/trace", run_id="r1")
        d = ref.to_dict()
        assert d == {"trace_dir": "/tmp/trace", "run_id": "r1"}

    def test_from_dict_without_run_id(self) -> None:
        """from_dict should handle missing run_id as None."""
        ref = TraceReference.from_dict({"trace_dir": "/tmp/trace"})
        assert ref.trace_dir == "/tmp/trace"
        assert ref.run_id is None

    def test_from_dict_with_run_id(self) -> None:
        """from_dict should restore run_id."""
        ref = TraceReference.from_dict({"trace_dir": "/tmp/trace", "run_id": "r2"})
        assert ref.trace_dir == "/tmp/trace"
        assert ref.run_id == "r2"

    def test_to_dict_from_dict_round_trip(self) -> None:
        """to_dict/from_dict should produce an equivalent object."""
        original = TraceReference(trace_dir="/tmp/my_trace", run_id="abc123")
        reconstructed = TraceReference.from_dict(original.to_dict())
        assert reconstructed.trace_dir == original.trace_dir
        assert reconstructed.run_id == original.run_id

    def test_to_dict_from_dict_round_trip_no_run_id(self) -> None:
        """Round-trip should work for TraceReference without run_id."""
        original = TraceReference(trace_dir="/data/traces")
        reconstructed = TraceReference.from_dict(original.to_dict())
        assert reconstructed.trace_dir == original.trace_dir
        assert reconstructed.run_id is None


class TestTraceLinker:
    """Tests for TraceLinker context manager."""

    @patch("calibrax.profiling.tracing.jax.profiler.trace")
    def test_trace_yields_trace_reference(self, mock_trace: MagicMock) -> None:
        """trace() should yield a TraceReference with the correct trace_dir."""
        mock_trace.return_value.__enter__ = MagicMock(return_value=None)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        linker = TraceLinker()
        with linker.trace("/tmp/my_trace") as ref:
            assert isinstance(ref, TraceReference)
            assert ref.trace_dir == "/tmp/my_trace"
            assert ref.run_id is None

    @patch("calibrax.profiling.tracing.jax.profiler.trace")
    def test_trace_with_run_id(self, mock_trace: MagicMock) -> None:
        """trace() should pass run_id to TraceReference."""
        mock_trace.return_value.__enter__ = MagicMock(return_value=None)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        linker = TraceLinker()
        with linker.trace("/tmp/trace_dir", run_id="run42") as ref:
            assert ref.run_id == "run42"
            assert ref.trace_dir == "/tmp/trace_dir"

    @patch("calibrax.profiling.tracing.jax.profiler.trace")
    def test_trace_calls_jax_profiler(self, mock_trace: MagicMock) -> None:
        """trace() should delegate to jax.profiler.trace with correct args."""
        mock_trace.return_value.__enter__ = MagicMock(return_value=None)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        linker = TraceLinker()
        with linker.trace("/tmp/prof", create_perfetto_link=True):
            pass

        mock_trace.assert_called_once_with(
            "/tmp/prof",
            create_perfetto_link=True,
            create_perfetto_trace=False,
        )

    @patch("calibrax.profiling.tracing.jax.profiler.trace")
    def test_trace_with_path_object(self, mock_trace: MagicMock) -> None:
        """trace() should accept Path objects and convert to str."""
        from pathlib import Path

        mock_trace.return_value.__enter__ = MagicMock(return_value=None)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        linker = TraceLinker()
        with linker.trace(Path("/tmp/path_trace")) as ref:
            assert ref.trace_dir == "/tmp/path_trace"

    @patch("calibrax.profiling.tracing.jax.profiler.trace")
    def test_trace_perfetto_options(self, mock_trace: MagicMock) -> None:
        """trace() should forward both Perfetto flags to jax.profiler.trace."""
        mock_trace.return_value.__enter__ = MagicMock(return_value=None)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        linker = TraceLinker()
        with linker.trace(
            "/tmp/t",
            create_perfetto_link=True,
            create_perfetto_trace=True,
        ):
            pass

        mock_trace.assert_called_once_with(
            "/tmp/t",
            create_perfetto_link=True,
            create_perfetto_trace=True,
        )
