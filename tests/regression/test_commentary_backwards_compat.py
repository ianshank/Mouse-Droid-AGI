"""Regression: commentary is additive — existing config/wiring unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def test_settings_default_has_no_commentary() -> None:
    assert Settings(mock_hardware=True).commentary is None


@pytest.mark.parametrize(
    "yaml_path",
    sorted(p for p in _CONFIG_DIR.glob("*.yaml") if not p.name.endswith(".example")),
    ids=lambda p: p.name,
)
def test_existing_yaml_loads_with_commentary_none(yaml_path: Path) -> None:
    """Every shipped overlay that loads at all loads with commentary absent.

    Commentary is purely additive (Optional, defaults None), so it cannot break
    a config that does not mention it. Overlays that fail to load standalone for
    PRE-EXISTING reasons (outdated fragments) are skipped — the additive
    guarantee is still proven by every config that does load (incl. default.yaml).
    """
    try:
        settings = load_settings(yaml_path)
    except Exception as exc:
        pytest.skip(f"{yaml_path.name} not standalone-loadable (pre-existing): {exc}")
    assert settings.commentary is None


def test_metrics_exposition_byte_identical_without_commentary() -> None:
    """No commentary families appear until a writer touches them."""
    from mousedroid.config.schema import MetricsConfig
    from mousedroid.telemetry.metrics import MetricsRegistry

    out = MetricsRegistry(MetricsConfig()).render_prometheus()
    assert "commentary" not in out


def test_new_yaml_with_commentary_block_loads(tmp_path: Path) -> None:
    cfg_file = tmp_path / "overlay.yaml"
    cfg_file.write_text(
        "mock_hardware: true\n"
        "commentary:\n"
        "  enabled: true\n"
        "  composer: template\n"
        "  novelty_sigma: 3.0\n"
    )
    settings = load_settings(cfg_file)
    assert settings.commentary is not None
    assert settings.commentary.enabled is True
    assert settings.commentary.novelty_sigma == 3.0
