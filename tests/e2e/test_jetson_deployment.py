"""End-to-end tests for Jetson deployment readiness.

Loads the Jetson production config with mock hardware overrides and exercises
the full component pipeline: factory wiring, orchestrator lifecycle, telemetry
publishing, health checks, and graceful shutdown. All tests run with
``mock_hardware=True`` so they execute in CI without real hardware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from mousedroid.config.loader import load_yaml
from mousedroid.config.schema import Settings
from mousedroid.factory import (
    build_camera,
    build_distance_sensor,
    build_esp32_driver,
    build_health_monitor,
    build_microphone,
    build_orchestrator,
    build_safety_monitor,
    build_telemetry_publisher,
    build_world_model,
)

pytestmark = pytest.mark.smoke

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_JETSON_PRODUCTION_YAML = _CONFIG_DIR / "jetson_production.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_mock_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override mock_hardware for all tests in this module."""
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_jetson_cfg() -> Settings:
    """Load jetson_production.yaml with mock_hardware forced to True for CI.

    Also reconciles model.audio_dim with microphone feature extractor output
    so tensor shapes match in the encoder.
    """
    base = load_yaml(_CONFIG_DIR / "default.yaml")
    overlay = load_yaml(_JETSON_PRODUCTION_YAML)
    merged = _deep_merge(base, overlay)
    merged["mock_hardware"] = True

    # Reconcile audio dimensions: the feature extractor output dim is
    # 3 * n_mels (mel + delta + delta-delta), which must match model.audio_dim.
    mic_cfg = merged.get("microphone", {})
    if mic_cfg and mic_cfg.get("enabled", False):
        n_mels = mic_cfg.get("n_mels", 64)
        audio_feature_dim = n_mels * 3  # mel + delta + delta-delta
        merged.setdefault("model", {})["audio_dim"] = audio_feature_dim

    return Settings(**merged)


def _patch_imagine_step(cfg: Settings) -> Any:
    """Create a deterministic imagine_step for stable MCTS rollouts."""

    def _fixed_imagine_step(
        action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return (
            torch.zeros(1, cfg.model.hidden_dim),
            torch.zeros(1, cfg.model.latent_dim),
            torch.tensor([[0.0]]),
        )

    return _fixed_imagine_step


# ---------------------------------------------------------------------------
# 1. Config loads correctly
# ---------------------------------------------------------------------------


def test_jetson_production_config_loads() -> None:
    """Jetson production YAML loads and produces valid Settings."""
    cfg = _load_jetson_cfg()
    assert cfg.platform.value == "mouse_droid"
    assert cfg.mock_hardware is True  # overridden by env var


def test_jetson_config_has_telemetry_enabled() -> None:
    """Production config enables telemetry."""
    cfg = _load_jetson_cfg()
    assert cfg.telemetry.enabled is True
    assert cfg.telemetry.port > 0


def test_jetson_config_has_llm_section() -> None:
    """Production config includes LLM configuration."""
    cfg = _load_jetson_cfg()
    assert cfg.llm.enabled is True
    assert cfg.llm.model_path is not None


# ---------------------------------------------------------------------------
# 2. Factory builds all components
# ---------------------------------------------------------------------------


def test_factory_builds_all_sensors() -> None:
    """All sensor factory functions succeed with Jetson config."""
    cfg = _load_jetson_cfg()

    camera = build_camera(cfg)
    assert camera is not None

    distance = build_distance_sensor(cfg)
    assert distance is not None

    esp32 = build_esp32_driver(cfg)
    assert esp32 is not None

    microphone = build_microphone(cfg)
    # Microphone is optional — None if not configured in YAML
    if cfg.microphone is not None:
        assert microphone is not None


def test_factory_builds_world_model() -> None:
    """World model builds from Jetson config."""
    cfg = _load_jetson_cfg()
    wm = build_world_model(cfg)
    assert wm is not None


def test_factory_builds_safety_monitor() -> None:
    """Safety monitor builds from Jetson config."""
    cfg = _load_jetson_cfg()
    monitor = build_safety_monitor(cfg)
    assert monitor is not None


def test_factory_builds_telemetry_publisher() -> None:
    """Telemetry publisher builds when enabled."""
    cfg = _load_jetson_cfg()
    publisher = build_telemetry_publisher(cfg)
    assert publisher is not None


def test_factory_builds_health_monitor() -> None:
    """Health monitor builds from Jetson config."""
    cfg = _load_jetson_cfg()
    hm = build_health_monitor(cfg)
    assert hm is not None


# ---------------------------------------------------------------------------
# 3. Orchestrator lifecycle
# ---------------------------------------------------------------------------


async def test_orchestrator_builds_from_jetson_config() -> None:
    """Full orchestrator builds from Jetson production config."""
    cfg = _load_jetson_cfg()
    orch = build_orchestrator(cfg)
    assert orch is not None


async def test_orchestrator_start_stop() -> None:
    """Orchestrator starts and stops cleanly."""
    cfg = _load_jetson_cfg()
    orch = build_orchestrator(cfg)
    await orch.start()
    try:
        assert orch._running is True
    finally:
        await orch.stop()
    assert orch._running is False


async def test_orchestrator_30_ticks_no_exception() -> None:
    """Orchestrator runs 30 ticks without raising."""
    cfg = _load_jetson_cfg()
    orch = build_orchestrator(cfg)
    orch._world_model.imagine_step = _patch_imagine_step(cfg)

    await orch.start()
    try:
        for _ in range(30):
            await orch.tick()
    finally:
        await orch.stop()

    assert orch._tick_count == 30


async def test_health_check_returns_ok() -> None:
    """Health check succeeds after orchestrator starts."""
    cfg = _load_jetson_cfg()
    orch = build_orchestrator(cfg)
    await orch.start()
    try:
        health = await orch.health_check()
        assert health["status"] == "ok"
        assert health["mock_hardware"] is True
        assert "mouse_droid_navigator" in health["agents"]
    finally:
        await orch.stop()


async def test_telemetry_publisher_receives_frames() -> None:
    """Telemetry publisher queue receives frames during tick loop."""
    cfg = _load_jetson_cfg()
    orch = build_orchestrator(cfg)
    orch._world_model.imagine_step = _patch_imagine_step(cfg)

    await orch.start()
    try:
        for _ in range(5):
            await orch.tick()
    finally:
        await orch.stop()

    # The publisher should have received at least one frame
    publisher = orch._telemetry_publisher
    if publisher is not None:
        stats = publisher.stats
        assert stats["frames_published"] >= 1


async def test_graceful_shutdown_idempotent() -> None:
    """Calling stop() twice does not raise."""
    cfg = _load_jetson_cfg()
    orch = build_orchestrator(cfg)
    await orch.start()
    await orch.stop()
    # Second stop should be safe
    await orch.stop()
