"""Shared fixtures for functional tier tests."""

from __future__ import annotations

import pytest
import torch

from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator


def _patch_imagine_step(cfg: Settings):
    def _fixed_imagine_step(action, h, z):
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return (
            torch.zeros(1, cfg.model.hidden_dim),
            torch.zeros(1, cfg.model.latent_dim),
            torch.tensor([[0.0]]),
        )

    return _fixed_imagine_step


@pytest.fixture
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Create a Settings instance with mock hardware enabled."""
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")
    return Settings(mock_hardware=True)


@pytest.fixture
def functional_orchestrator(mock_settings: Settings):
    """Build a complete orchestrator using the factory for functional testing."""
    mock_settings.llm.enabled = True
    mock_settings.mission.replan_enabled = True
    mock_settings.telemetry.enabled = True
    mock_settings.voice.enabled = True

    orch = build_orchestrator(mock_settings)
    orch._world_model.imagine_step = _patch_imagine_step(mock_settings)
    return orch
