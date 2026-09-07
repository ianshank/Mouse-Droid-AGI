"""F-041 backwards-compat: runtime BDI init scale and YAML stay unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.schema import Settings
from mousedroid.constants import WEIGHT_INIT_SCALE

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_weight_init_scale_unchanged() -> None:
    assert WEIGHT_INIT_SCALE == 0.01


def test_default_settings_keep_bdi_repo_and_ota_policy_v2() -> None:
    cfg = Settings(mock_hardware=True)
    assert cfg.cognitive.huggingface_repo == "ianshank/mousedroid-weights"
    assert cfg.cloud.weight_update.policy_repo_id == "ianshank/mousedroid-policy-v2"


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in (_REPO_ROOT / "config").glob("*.yaml")),
)
def test_shipped_overlays_still_load(overlay: str) -> None:
    path = _REPO_ROOT / "config" / overlay
    text = path.read_text(encoding="utf-8")
    if "# config-validator: skip" in text:
        pytest.skip(f"{overlay} carries skip marker (not a Settings overlay)")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        return
    Settings.model_validate({**raw, "mock_hardware": True})
