"""Unit tests for ``build_growth_coordinator`` factory dispatch."""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.schema import Settings
from mousedroid.factory import build_growth_coordinator, build_vla_policy, build_world_model
from mousedroid.growth.coordinator import GrowthDistillationCoordinator


def _base(tmp_path: Path) -> dict[str, object]:
    return {
        "mock_hardware": True,
        "experience": {"path": str(tmp_path / "exp"), "map_size_gb": 0.01},
    }


def test_returns_none_when_block_absent(tmp_path: Path) -> None:
    """No ``growth`` block → ``None`` (orchestrator byte-identical)."""
    cfg = Settings.model_validate(_base(tmp_path))
    assert build_growth_coordinator(cfg) is None


def test_returns_none_when_disabled(tmp_path: Path) -> None:
    """``growth.enabled=false`` → ``None``."""
    cfg = Settings.model_validate({**_base(tmp_path), "growth": {"enabled": False}})
    assert build_growth_coordinator(cfg) is None


def test_returns_none_when_no_vla_teacher(tmp_path: Path) -> None:
    """Enabled but no VLA teacher wired → ``None`` (nothing to distil)."""
    cfg = Settings.model_validate(
        {**_base(tmp_path), "growth": {"enabled": True, "student_hidden_dim": 16}}
    )
    # vla.backend defaults to "none" → build_vla_policy returns None.
    assert build_growth_coordinator(cfg, vla_policy=None) is None


def test_returns_coordinator_when_enabled_with_vla(tmp_path: Path) -> None:
    """Enabled + a VLA teacher → a real ``GrowthDistillationCoordinator``."""
    cfg = Settings.model_validate(
        {
            **_base(tmp_path),
            "vla": {"backend": "mock"},
            "growth": {
                "enabled": True,
                "batch_size": 4,
                "distill_steps": 2,
                "student_hidden_dim": 16,
            },
        }
    )
    vla = build_vla_policy(cfg)
    wm = build_world_model(cfg)
    coord = build_growth_coordinator(cfg, vla_policy=vla, world_model=wm)
    assert isinstance(coord, GrowthDistillationCoordinator)


def test_student_dims_match_model_config(tmp_path: Path) -> None:
    """The student is built from ``model`` dims (h=hidden, z=latent, action)."""
    cfg = Settings.model_validate(
        {
            **_base(tmp_path),
            "vla": {"backend": "mock"},
            "growth": {
                "enabled": True,
                "batch_size": 2,
                "distill_steps": 1,
                "student_hidden_dim": 8,
            },
        }
    )
    coord = build_growth_coordinator(cfg, vla_policy=build_vla_policy(cfg))
    assert isinstance(coord, GrowthDistillationCoordinator)
    student = coord._student  # type: ignore[attr-defined]
    assert student.obs_dim == cfg.model.hidden_dim + cfg.model.latent_dim
    assert student.action_dim == cfg.model.action_dim
