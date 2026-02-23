"""Tests for GPU profiling: HardwareConfig, AdaptiveOperation, GPUMemoryProfiler.

All GPU/hardware access is mocked — no hardware dependency.
"""

import builtins
import dataclasses
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from calibrax.profiling.gpu import (
    AdaptiveOperation,
    GPUMemoryProfiler,
    MemoryAnalysis,
    MemoryOptimizer,
)
from calibrax.profiling.resources import GPUProfilerProtocol
from tests.factories import make_cpu_hardware_config


class TestHardwareConfig:
    """Tests for HardwareConfig frozen dataclass."""

    def test_construction(self) -> None:
        cfg = make_cpu_hardware_config()
        assert cfg.platform == "cpu"
        assert cfg.tile_size == 64

    def test_frozen_immutability(self) -> None:
        cfg = make_cpu_hardware_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.platform = "gpu"  # type: ignore[misc]


class TestAdaptiveOperation:
    """Tests for AdaptiveOperation hardware detection + shape optimization."""

    @patch("calibrax.profiling.gpu.jax")
    def test_cpu_default_config(self, mock_jax: MagicMock) -> None:
        mock_jax.default_backend.return_value = "cpu"
        op = AdaptiveOperation()
        assert op.config.platform == "cpu"
        assert op.config.precision == "float32"
        assert op.config.tile_size == 64

    @patch("calibrax.profiling.gpu.jax")
    def test_tpu_config(self, mock_jax: MagicMock) -> None:
        mock_jax.default_backend.return_value = "tpu"
        op = AdaptiveOperation()
        assert op.config.platform == "tpu"
        assert op.config.precision == "bfloat16"
        assert op.config.tile_size == 128
        assert op.config.use_vmem_optimization is True

    @patch("calibrax.profiling.gpu.jax")
    def test_gpu_modern_a100(self, mock_jax: MagicMock) -> None:
        mock_jax.default_backend.return_value = "gpu"
        mock_dev = MagicMock()
        mock_dev.device_kind = "NVIDIA A100"
        mock_jax.devices.return_value = [mock_dev]
        op = AdaptiveOperation()
        assert op.config.platform == "gpu_modern"
        assert op.config.precision == "bfloat16"
        assert op.config.tile_size == 16

    @patch("calibrax.profiling.gpu.jax")
    def test_gpu_modern_h100(self, mock_jax: MagicMock) -> None:
        mock_jax.default_backend.return_value = "gpu"
        mock_dev = MagicMock()
        mock_dev.device_kind = "NVIDIA H100"
        mock_jax.devices.return_value = [mock_dev]
        op = AdaptiveOperation()
        assert op.config.platform == "gpu_modern"

    @patch("calibrax.profiling.gpu.jax")
    def test_gpu_legacy(self, mock_jax: MagicMock) -> None:
        mock_jax.default_backend.return_value = "gpu"
        mock_dev = MagicMock()
        mock_dev.device_kind = "NVIDIA RTX 3090"
        mock_jax.devices.return_value = [mock_dev]
        op = AdaptiveOperation()
        assert op.config.platform == "gpu_legacy"
        assert op.config.precision == "float32"
        assert op.config.tile_size == 32

    @patch("calibrax.profiling.gpu.jax")
    def test_gpu_detection_exception_falls_back(
        self,
        mock_jax: MagicMock,
    ) -> None:
        mock_jax.default_backend.return_value = "gpu"
        mock_jax.devices.side_effect = RuntimeError("no GPU")
        op = AdaptiveOperation()
        assert op.config.platform == "cpu"

    @patch("calibrax.profiling.gpu.jax")
    def test_gpu_backend_with_no_devices_falls_back_to_cpu(self, mock_jax: MagicMock) -> None:
        mock_jax.default_backend.return_value = "gpu"
        mock_jax.devices.return_value = []

        op = AdaptiveOperation()

        assert op.config.platform == "cpu"

    def test_optimize_shapes_pads_to_tile_size(self) -> None:
        op = AdaptiveOperation()
        tile = op.config.tile_size
        shapes = op.optimize_shapes((10, 100), (5, 5))
        for shape in shapes:
            for dim in shape:
                assert dim % tile == 0

    def test_optimize_shapes_rank1_skips_second_last_dimension(self) -> None:
        op = AdaptiveOperation()
        tile = op.config.tile_size

        (shape,) = op.optimize_shapes((tile + 1,))

        assert len(shape) == 1
        assert shape[0] % tile == 0

    def test_optimize_shapes_already_aligned(self) -> None:
        op = AdaptiveOperation()
        tile = op.config.tile_size
        result = op.optimize_shapes((tile, tile * 2))
        assert result == [(tile, tile * 2)]


class TestGPUMemoryProfiler:
    """Tests for GPUMemoryProfiler."""

    @patch("calibrax.profiling.gpu.jax")
    def test_has_gpu_false_on_cpu(self, mock_jax: MagicMock) -> None:
        mock_jax.devices.side_effect = RuntimeError("No GPU backend")
        profiler = GPUMemoryProfiler()
        assert profiler.has_gpu is False

    @patch("calibrax.profiling.gpu.jax")
    def test_has_gpu_true_with_devices(self, mock_jax: MagicMock) -> None:
        mock_jax.devices.return_value = [MagicMock()]
        profiler = GPUMemoryProfiler()
        assert profiler.has_gpu is True

    @patch("calibrax.profiling.gpu.jax")
    def test_has_gpu_runtime_error(self, mock_jax: MagicMock) -> None:
        mock_jax.devices.side_effect = RuntimeError("no GPU")
        profiler = GPUMemoryProfiler()
        assert profiler.has_gpu is False

    def test_get_memory_no_gpu(self) -> None:
        profiler = GPUMemoryProfiler()
        profiler.has_gpu = False
        mem = profiler.get_memory_usage()
        assert mem == {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}

    @patch("calibrax.profiling.gpu.jax")
    def test_get_memory_with_memory_stats(
        self,
        mock_jax: MagicMock,
    ) -> None:
        mock_device = MagicMock()
        mock_device.memory_stats.return_value = {
            "bytes_in_use": 2048 * 1024 * 1024,
            "bytes_limit": 8192 * 1024 * 1024,
        }
        mock_jax.devices.return_value = [mock_device]

        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True
        mem = profiler.get_memory_usage()
        assert mem["gpu_memory_used_mb"] == pytest.approx(2048.0)
        assert mem["gpu_memory_total_mb"] == pytest.approx(8192.0)

    def test_get_memory_runtime_error_fallback(self) -> None:
        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True
        with patch(
            "calibrax.profiling.gpu.jax",
        ) as mock_jax:
            mock_jax.devices.side_effect = RuntimeError("backend unavailable")
            mem = profiler.get_memory_usage()
        assert mem == {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}

    @patch("calibrax.profiling.gpu.jax")
    def test_get_memory_without_memory_stats_method_falls_back(
        self,
        mock_jax: MagicMock,
    ) -> None:
        mock_jax.devices.return_value = [object()]

        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True
        mem = profiler.get_memory_usage()

        assert mem == {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}

    @patch("calibrax.profiling.gpu.jax")
    def test_get_memory_with_empty_memory_stats_falls_back(self, mock_jax: MagicMock) -> None:
        mock_device = MagicMock()
        mock_device.memory_stats.return_value = {}
        mock_jax.devices.return_value = [mock_device]

        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True
        mem = profiler.get_memory_usage()

        assert mem == {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}

    def test_get_memory_unexpected_error_propagates(self) -> None:
        class CatastrophicMemoryQueryError(Exception):
            pass

        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True
        with patch(
            "calibrax.profiling.gpu.jax",
        ) as mock_jax:
            mock_jax.devices.side_effect = CatastrophicMemoryQueryError("catastrophic")
            with pytest.raises(CatastrophicMemoryQueryError):
                profiler.get_memory_usage()

    def test_get_utilization_no_gpu(self) -> None:
        profiler = GPUMemoryProfiler()
        profiler.has_gpu = False
        assert profiler.get_utilization() == 0.0

    def test_safe_nvml_query_returns_fallback_on_nvml_error(self) -> None:
        class FakeNVMLError(Exception):
            pass

        fake_nvml = types.SimpleNamespace(
            NVMLError=FakeNVMLError,
            nvmlInit=MagicMock(side_effect=FakeNVMLError("init failed")),
            nvmlDeviceGetHandleByIndex=MagicMock(),
        )
        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True

        with patch("calibrax.profiling.gpu.PYNVML_AVAILABLE", True):
            with patch("calibrax.profiling.gpu.pynvml", fake_nvml):
                result = profiler._safe_nvml_query(lambda _handle: {"ok": 1.0}, {"ok": 0.0})

        assert result == {"ok": 0.0}

    def test_get_clock_info_reads_nvml_when_available(self) -> None:
        class FakeNVMLError(Exception):
            pass

        handle = object()
        fake_nvml = types.SimpleNamespace(
            NVMLError=FakeNVMLError,
            NVML_CLOCK_GRAPHICS=1,
            NVML_CLOCK_MEM=2,
            nvmlInit=MagicMock(),
            nvmlDeviceGetHandleByIndex=MagicMock(return_value=handle),
            nvmlDeviceGetClockInfo=MagicMock(side_effect=[1500, 2100]),
        )
        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True

        with patch("calibrax.profiling.gpu.PYNVML_AVAILABLE", True):
            with patch("calibrax.profiling.gpu.pynvml", fake_nvml):
                clocks = profiler.get_clock_info()

        assert clocks == {"gpu_clock_mhz": 1500.0, "mem_clock_mhz": 2100.0}

    def test_get_power_info_reads_nvml_when_available(self) -> None:
        class FakeNVMLError(Exception):
            pass

        handle = object()
        fake_nvml = types.SimpleNamespace(
            NVMLError=FakeNVMLError,
            nvmlInit=MagicMock(),
            nvmlDeviceGetHandleByIndex=MagicMock(return_value=handle),
            nvmlDeviceGetPowerUsage=MagicMock(return_value=240000),
            nvmlDeviceGetPowerManagementLimit=MagicMock(return_value=300000),
        )
        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True

        with patch("calibrax.profiling.gpu.PYNVML_AVAILABLE", True):
            with patch("calibrax.profiling.gpu.pynvml", fake_nvml):
                power = profiler.get_power_info()

        assert power == {"power_draw_w": 240.0, "power_limit_w": 300.0}

    def test_satisfies_gpu_profiler_protocol(self) -> None:
        profiler = GPUMemoryProfiler()
        assert isinstance(profiler, GPUProfilerProtocol)

    def test_analyze_memory_pattern_empty(self) -> None:
        profiler = GPUMemoryProfiler()
        assert profiler.analyze_memory_pattern([]) == []

    def test_analyze_memory_pattern_leak_detection(self) -> None:
        profiler = GPUMemoryProfiler()
        measurements = [
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.3},
            {"gpu_memory_used_mb": 200, "gpu_memory_utilization": 0.3},
            {"gpu_memory_used_mb": 300, "gpu_memory_utilization": 0.3},
        ]
        suggestions = profiler.analyze_memory_pattern(measurements)
        assert any("memory leak" in s.lower() for s in suggestions)

    def test_analyze_memory_pattern_high_utilization(self) -> None:
        profiler = GPUMemoryProfiler()
        measurements = [
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.95},
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.5},
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.5},
        ]
        suggestions = profiler.analyze_memory_pattern(measurements)
        assert any("90%" in s for s in suggestions)

    def test_analyze_memory_pattern_no_suggestions(self) -> None:
        profiler = GPUMemoryProfiler()
        measurements = [
            {"gpu_memory_used_mb": 50, "gpu_memory_utilization": 0.1},
            {"gpu_memory_used_mb": 51, "gpu_memory_utilization": 0.1},
            {"gpu_memory_used_mb": 50, "gpu_memory_utilization": 0.1},
        ]
        assert profiler.analyze_memory_pattern(measurements) == []

    def test_analyze_memory_pattern_high_average_utilization_warns(self) -> None:
        profiler = GPUMemoryProfiler()
        measurements = [
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.82},
            {"gpu_memory_used_mb": 102, "gpu_memory_utilization": 0.85},
            {"gpu_memory_used_mb": 101, "gpu_memory_utilization": 0.83},
        ]

        suggestions = profiler.analyze_memory_pattern(measurements)

        assert any("80%" in suggestion for suggestion in suggestions)
        assert all("90%" not in suggestion for suggestion in suggestions)

    def test_analyze_memory_pattern_short_series_skips_leak_trend(self) -> None:
        profiler = GPUMemoryProfiler()
        measurements = [
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.2},
            {"gpu_memory_used_mb": 300, "gpu_memory_utilization": 0.2},
        ]

        suggestions = profiler.analyze_memory_pattern(measurements)

        assert all("memory leak" not in suggestion.lower() for suggestion in suggestions)


class TestMemoryAnalysis:
    """Tests for MemoryAnalysis frozen dataclass."""

    def test_construction(self) -> None:
        analysis = MemoryAnalysis(
            baseline_memory_mb=100.0,
            peak_memory_mb=200.0,
            peak_usage_mb=100.0,
            retained_memory_mb=10.0,
            memory_efficiency=0.9,
            suggestions=("Use smaller batches",),
        )
        assert analysis.memory_efficiency == 0.9

    def test_frozen_immutability(self) -> None:
        analysis = MemoryAnalysis(
            baseline_memory_mb=100.0,
            peak_memory_mb=200.0,
            peak_usage_mb=100.0,
            retained_memory_mb=10.0,
            memory_efficiency=0.9,
            suggestions=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            analysis.peak_memory_mb = 0.0  # type: ignore[misc]


class TestMemoryOptimizer:
    """Tests for MemoryOptimizer."""

    def test_analyze_pipeline_returns_memory_analysis(self) -> None:
        optimizer = MemoryOptimizer()
        result = optimizer.analyze_pipeline_memory(
            lambda _: None,
            sample_data=None,
        )
        assert isinstance(result, MemoryAnalysis)

    def test_analyze_pipeline_exception_returns_error(self) -> None:
        optimizer = MemoryOptimizer()

        def failing_fn(_: object) -> None:
            msg = "pipeline exploded"
            raise ValueError(msg)

        result = optimizer.analyze_pipeline_memory(
            failing_fn,
            sample_data=None,
        )
        assert result is None

    def test_analyze_pipeline_unexpected_exception_propagates(self) -> None:
        class CatastrophicPipelineError(Exception):
            pass

        optimizer = MemoryOptimizer()

        def catastrophic_fn(_: object) -> None:
            raise CatastrophicPipelineError("unexpected pipeline failure")

        with pytest.raises(CatastrophicPipelineError):
            optimizer.analyze_pipeline_memory(
                catastrophic_fn,
                sample_data=None,
            )

    def test_generate_suggestions_includes_high_peak_and_low_efficiency(self) -> None:
        optimizer = MemoryOptimizer()

        suggestions = optimizer._generate_suggestions(peak_usage=1500.0, retained_memory=1200.0)

        assert any("High memory usage detected" in suggestion for suggestion in suggestions)
        assert any("Low memory efficiency" in suggestion for suggestion in suggestions)


class TestGpuModuleImportGuards:
    """Import guard behavior for optional pynvml dependency."""

    def test_module_import_sets_pynvml_unavailable_when_missing(self) -> None:
        import calibrax.profiling.gpu as gpu_mod

        module_path = Path(gpu_mod.__file__)
        spec = importlib.util.spec_from_file_location("gpu_import_probe", module_path)
        assert spec is not None
        assert spec.loader is not None
        probe_module = importlib.util.module_from_spec(spec)

        real_import = builtins.__import__

        def _import_hook(name: str, *args: object, **kwargs: object) -> object:
            if name == "pynvml":
                raise ImportError("missing pynvml")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_hook):
            sys.modules[spec.name] = probe_module
            try:
                spec.loader.exec_module(probe_module)
            finally:
                sys.modules.pop(spec.name, None)

        assert probe_module.PYNVML_AVAILABLE is False
        assert probe_module.pynvml is None
