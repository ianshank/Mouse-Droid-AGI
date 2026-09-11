"""AQA — F-048 Isaac workstation harness docs; no Prometheus family."""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.schema import MetricsConfig, Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_harness_spec_has_isaac_section() -> None:
    text = (_REPO_ROOT / "HARNESS_SPEC.md").read_text(encoding="utf-8")
    assert "Isaac Lab workstation" in text
    assert "F-043" in text
    assert "MLflow" in text


def test_metrics_config_has_no_isaac_family() -> None:
    assert "track_isaac_sim" not in MetricsConfig.model_fields


def test_c4_mentions_isaac_lab_backend() -> None:
    text = (_REPO_ROOT / "docs" / "architecture" / "c4-rssm-sim-pretraining.md").read_text(
        encoding="utf-8"
    )
    assert "isaac_lab" in text
    assert "MuJoCo-only" not in text or "no longer MuJoCo-only" in text


def test_adr009_has_f043_addendum() -> None:
    text = (_REPO_ROOT / "docs" / "architecture" / "ADR-009-isaac-lab-phase-b.md").read_text(
        encoding="utf-8"
    )
    assert "F-043" in text
    assert "to_body_action" in text


def test_settings_harness_default_none() -> None:
    assert Settings(mock_hardware=True).harness is None
