"""Tests for PipelineOrchestrator wiring to real training functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mousedroid.config.schema import Settings, TrainingPipelineConfig
from mousedroid.training.pipeline_orchestrator import PipelineOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(
    *,
    phases: list[str] | None = None,
    settings: Settings | None = None,
) -> PipelineOrchestrator:
    """Build an orchestrator with mock GPU monitor and batch tuner."""
    cfg = settings or Settings(mock_hardware=True)
    pipeline_cfg = TrainingPipelineConfig(
        phases=phases or ["rssm", "warmstart", "bdi", "constitutional_rl"],
    )

    gpu_monitor = MagicMock()
    gpu_monitor.should_pause = AsyncMock(return_value=False)

    batch_tuner = MagicMock()
    batch_tuner.tune_batch_size = MagicMock(side_effect=lambda _phase, base: base)

    return PipelineOrchestrator(
        settings=cfg,
        pipeline_config=pipeline_cfg,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )


# ---------------------------------------------------------------------------
# Phase dispatch
# ---------------------------------------------------------------------------


def test_get_phase_runner_known_phases() -> None:
    """All four standard phases resolve to async callables."""
    orch = _make_orchestrator()
    for phase in ("rssm", "warmstart", "bdi", "constitutional_rl"):
        runner = orch._get_phase_runner(phase)
        assert callable(runner), f"Runner for '{phase}' is not callable"


def test_get_phase_runner_unknown_raises() -> None:
    """Unknown phase name raises ValueError."""
    orch = _make_orchestrator()
    with pytest.raises(ValueError, match="Unknown training phase"):
        orch._get_phase_runner("nonexistent_phase")


# ---------------------------------------------------------------------------
# Delegation to real training functions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_train_rssm_delegates_to_run_pipeline(tmp_path: Path) -> None:
    """_train_rssm calls run_phase_1_rssm via asyncio.to_thread."""
    cfg = Settings(mock_hardware=True)
    orch = _make_orchestrator(settings=cfg)
    fake_checkpoint = tmp_path / "rssm" / "final.pt"
    fake_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    with patch(
        "mousedroid.training.pipeline_orchestrator.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=fake_checkpoint,
    ) as mock_to_thread:
        await orch._train_rssm(orch._settings, 16, orch._checkpoint_dir)

    mock_to_thread.assert_awaited_once()
    call_args = mock_to_thread.call_args
    # First positional arg is the function
    fn = call_args[0][0]
    assert fn.__name__ == "run_phase_1_rssm"


@pytest.mark.asyncio
async def test_train_warmstart_delegates_to_run_pipeline() -> None:
    """_train_warmstart calls run_phase_2_warmstart via asyncio.to_thread."""
    orch = _make_orchestrator()

    with patch(
        "mousedroid.training.pipeline_orchestrator.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_to_thread:
        await orch._train_warmstart(orch._settings, 32, orch._checkpoint_dir)

    mock_to_thread.assert_awaited_once()
    fn = mock_to_thread.call_args[0][0]
    assert fn.__name__ == "run_phase_2_warmstart"


@pytest.mark.asyncio
async def test_train_bdi_delegates_to_run_pipeline() -> None:
    """_train_bdi calls run_phase_3_bdi via asyncio.to_thread."""
    orch = _make_orchestrator()

    with patch(
        "mousedroid.training.pipeline_orchestrator.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=Path("weights/bdi"),
    ) as mock_to_thread:
        await orch._train_bdi(orch._settings, 32, orch._checkpoint_dir)

    mock_to_thread.assert_awaited_once()
    fn = mock_to_thread.call_args[0][0]
    assert fn.__name__ == "run_phase_3_bdi"


@pytest.mark.asyncio
async def test_train_constitutional_rl_delegates_to_run_pipeline() -> None:
    """_train_constitutional_rl calls run_phase_4_constitutional_rl."""
    orch = _make_orchestrator()

    with patch(
        "mousedroid.training.pipeline_orchestrator.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=Path("weights/constitutional_rl"),
    ) as mock_to_thread:
        await orch._train_constitutional_rl(orch._settings, 64, orch._checkpoint_dir)

    mock_to_thread.assert_awaited_once()
    fn = mock_to_thread.call_args[0][0]
    assert fn.__name__ == "run_phase_4_constitutional_rl"


# ---------------------------------------------------------------------------
# Batch size propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_size_propagated_to_settings_copy() -> None:
    """Tuned batch_size reaches the training function via settings copy."""
    orch = _make_orchestrator()
    tuned_batch = 24

    with patch(
        "mousedroid.training.pipeline_orchestrator.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=Path("fake.pt"),
    ) as mock_to_thread:
        await orch._train_rssm(orch._settings, tuned_batch, orch._checkpoint_dir)

    # Second positional arg is the updated_settings
    updated_settings = mock_to_thread.call_args[0][1]
    assert updated_settings.training.batch_size == tuned_batch


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def test_build_training_pipeline_returns_orchestrator() -> None:
    """build_training_pipeline returns a PipelineOrchestrator instance."""
    from mousedroid.factory import build_training_pipeline

    cfg = Settings(mock_hardware=True)
    orch = build_training_pipeline(cfg)
    assert isinstance(orch, PipelineOrchestrator)


def test_build_training_pipeline_uses_config() -> None:
    """build_training_pipeline honours the training_pipeline config."""
    from mousedroid.factory import build_training_pipeline

    pipeline_cfg = TrainingPipelineConfig(phases=["rssm", "bdi"])
    cfg = Settings(mock_hardware=True, training_pipeline=pipeline_cfg)
    orch = build_training_pipeline(cfg)
    assert orch._config.phases == ["rssm", "bdi"]


def test_build_training_pipeline_default_config() -> None:
    """build_training_pipeline uses default TrainingPipelineConfig when None."""
    from mousedroid.factory import build_training_pipeline

    cfg = Settings(mock_hardware=True)
    assert cfg.training_pipeline is None
    orch = build_training_pipeline(cfg)
    assert orch._config.phases == ["rssm", "warmstart", "bdi", "constitutional_rl"]


# ---------------------------------------------------------------------------
# Config field defaults — no hardcoded values
# ---------------------------------------------------------------------------


def test_training_config_new_fields_have_defaults() -> None:
    """sequences_filename and bdi_subdir have backwards-compatible defaults."""
    from mousedroid.config.schema import Settings

    cfg = Settings(mock_hardware=True)
    assert cfg.training.sequences_filename == "sequences.pt"
    assert cfg.training.bdi_subdir == "bdi"


def test_training_config_fields_are_overridable() -> None:
    """sequences_filename and bdi_subdir can be overridden via config."""
    from mousedroid.config.schema import Settings, TrainingConfig

    training_cfg = TrainingConfig(sequences_filename="custom.pt", bdi_subdir="custom_bdi")
    cfg = Settings(mock_hardware=True, training=training_cfg)
    assert cfg.training.sequences_filename == "custom.pt"
    assert cfg.training.bdi_subdir == "custom_bdi"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_training_phase_protocol_is_runtime_checkable() -> None:
    """TrainingPhaseProtocol supports runtime structural isinstance checks."""
    from mousedroid.training.protocol import TrainingPhaseProtocol

    class _ConformingPhase:
        """Minimal concrete implementation satisfying the Protocol."""

        @property
        def name(self) -> str:
            return "test_phase"

        async def run(
            self,
            cfg: object,
            batch_size: int,
            checkpoint_dir: object,
        ) -> object:
            return object()

    phase = _ConformingPhase()
    assert isinstance(phase, TrainingPhaseProtocol)


# ---------------------------------------------------------------------------
# Integration: run() orchestrates phases in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_orchestrates_phases_in_order(tmp_path: Path) -> None:
    """run() calls phases sequentially, creating checkpoint markers."""
    phases_called: list[str] = []

    cfg = Settings(mock_hardware=True)
    pipeline_cfg = TrainingPipelineConfig(
        phases=["rssm", "bdi"],
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    gpu_monitor = MagicMock()
    gpu_monitor.should_pause = AsyncMock(return_value=False)

    batch_tuner = MagicMock()
    batch_tuner.tune_batch_size = MagicMock(side_effect=lambda _phase, base: base)

    orch = PipelineOrchestrator(
        settings=cfg,
        pipeline_config=pipeline_cfg,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )

    async def _fake_rssm(cfg: object, batch_size: int, checkpoint_dir: object) -> None:
        phases_called.append("rssm")

    async def _fake_bdi(cfg: object, batch_size: int, checkpoint_dir: object) -> None:
        phases_called.append("bdi")

    orch._train_rssm = _fake_rssm  # type: ignore[assignment]
    orch._train_bdi = _fake_bdi  # type: ignore[assignment]

    await orch.run()

    assert phases_called == ["rssm", "bdi"]
    # Checkpoint markers should exist
    assert (tmp_path / "checkpoints" / "rssm.done").exists()
    assert (tmp_path / "checkpoints" / "bdi.done").exists()


@pytest.mark.asyncio
async def test_run_resume_skips_completed_phases(tmp_path: Path) -> None:
    """run() with resume_from_phase skips earlier phases."""
    phases_called: list[str] = []

    cfg = Settings(mock_hardware=True)
    pipeline_cfg = TrainingPipelineConfig(
        phases=["rssm", "warmstart", "bdi"],
        checkpoint_dir=str(tmp_path / "checkpoints"),
        resume_from_phase="bdi",
    )

    # Create warmstart checkpoint marker (prev phase before bdi, so validation passes)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "warmstart.done").write_text("phase=warmstart\n")

    gpu_monitor = MagicMock()
    gpu_monitor.should_pause = AsyncMock(return_value=False)

    batch_tuner = MagicMock()
    batch_tuner.tune_batch_size = MagicMock(side_effect=lambda _phase, base: base)

    orch = PipelineOrchestrator(
        settings=cfg,
        pipeline_config=pipeline_cfg,
        gpu_monitor=gpu_monitor,
        batch_tuner=batch_tuner,
    )

    async def _track_phase(name: str) -> None:
        phases_called.append(name)

    orch._train_rssm = lambda cfg, bs, cd: _track_phase("rssm")  # type: ignore[assignment]
    orch._train_warmstart = lambda cfg, bs, cd: _track_phase("warmstart")  # type: ignore[assignment]
    orch._train_bdi = lambda cfg, bs, cd: _track_phase("bdi")  # type: ignore[assignment]

    await orch.run()

    # Only BDI should have run (rssm and warmstart skipped)
    assert phases_called == ["bdi"]
