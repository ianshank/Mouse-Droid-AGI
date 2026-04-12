"""Smoke tests verifying training module imports and key symbols.

Ensures all training modules can be imported without errors and that
key classes, functions, and protocols are accessible.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# pipeline_orchestrator
# ---------------------------------------------------------------------------


def test_pipeline_orchestrator_imports() -> None:
    """pipeline_orchestrator module imports and exposes expected symbols."""
    from mousedroid.training.pipeline_orchestrator import (
        PipelineOrchestrator,
        async_main,
        main,
    )

    assert callable(PipelineOrchestrator)
    assert callable(async_main)
    assert callable(main)


def test_pipeline_orchestrator_has_run_method() -> None:
    """PipelineOrchestrator exposes an async run() method."""
    from mousedroid.training.pipeline_orchestrator import PipelineOrchestrator

    assert hasattr(PipelineOrchestrator, "run")


# ---------------------------------------------------------------------------
# gpu_monitor
# ---------------------------------------------------------------------------


def test_gpu_monitor_imports() -> None:
    """gpu_monitor module imports and exposes expected symbols."""
    from mousedroid.training.gpu_monitor import (
        GPUMonitorProtocol,
        JetsonGPUMonitor,
    )

    assert callable(JetsonGPUMonitor)
    # Protocol should be runtime-checkable.
    assert hasattr(GPUMonitorProtocol, "__protocol_attrs__") or hasattr(
        GPUMonitorProtocol, "__abstractmethods__"
    )


def test_gpu_monitor_protocol_methods() -> None:
    """GPUMonitorProtocol defines expected async methods."""
    from mousedroid.training.gpu_monitor import GPUMonitorProtocol

    for method in ("get_temperature", "get_vram_free_mb", "should_pause"):
        assert hasattr(GPUMonitorProtocol, method)


# ---------------------------------------------------------------------------
# batch_tuner
# ---------------------------------------------------------------------------


def test_batch_tuner_imports() -> None:
    """batch_tuner module imports and exposes expected symbols."""
    from mousedroid.training.batch_tuner import (
        BatchTunerProtocol,
        VRAMBatchTuner,
    )

    assert callable(VRAMBatchTuner)
    assert hasattr(BatchTunerProtocol, "tune_batch_size")


# ---------------------------------------------------------------------------
# training __init__
# ---------------------------------------------------------------------------


def test_training_package_imports() -> None:
    """The training package itself can be imported."""
    import mousedroid.training  # noqa: F401


# ---------------------------------------------------------------------------
# config schema: TrainingPipelineConfig
# ---------------------------------------------------------------------------


def test_training_pipeline_config_imports() -> None:
    """TrainingPipelineConfig imports from config schema."""
    from mousedroid.config.schema import TrainingPipelineConfig

    config = TrainingPipelineConfig()
    assert isinstance(config.phases, list)
    assert len(config.phases) == 4
    assert config.thermal_limit_celsius > 0
    assert config.vram_headroom_mb > 0


# ---------------------------------------------------------------------------
# Orchestrator delegates phases (relationship documentation)
# ---------------------------------------------------------------------------


def test_orchestrator_phase_dispatch_covers_all_defaults() -> None:
    """Orchestrator's phase dispatch table covers all default phases."""
    from unittest.mock import AsyncMock, MagicMock

    from mousedroid.config.schema import Settings, TrainingPipelineConfig
    from mousedroid.training.pipeline_orchestrator import PipelineOrchestrator

    config = TrainingPipelineConfig()
    orch = PipelineOrchestrator(
        settings=Settings(mock_hardware=True, ultrasonic=None),
        pipeline_config=config,
        gpu_monitor=AsyncMock(),
        batch_tuner=MagicMock(),
    )
    for phase in config.phases:
        runner = orch._get_phase_runner(phase)
        assert callable(runner), f"No runner for phase '{phase}'"
