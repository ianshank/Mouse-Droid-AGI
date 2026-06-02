"""PR #108 backwards-compatibility regression tests.

PR #108 introduced the optional ``GreetingConfig`` block. Two
invariants must hold so older deployments keep loading byte-identically:

* ``Settings.greeting`` defaults to ``None`` — every YAML overlay that
  predates PR #108 must construct cleanly without a ``greeting`` block.
* ``GreetingConfig.enabled`` defaults to ``False`` — supplying just an
  empty greeting block does not silently activate the subsystem.

A third invariant guards the operator-facing example overlay:

* ``config/greeting_pilot.yaml.example`` must load through
  :func:`mousedroid.config.loader.load_settings` and yield an
  enabled :class:`GreetingConfig`. Because the validator script
  (``scripts/validate_configs.py``) excludes ``*.yaml.example`` by
  the ``*.yaml`` glob, schema drift would otherwise silently invalidate
  the example until a live operator tried it. This regression test
  closes that gap.

A failure here means PR #108 broke the "new config fields MUST have
defaults" rule from CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import GreetingConfig, Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_settings_loads_without_greeting_block() -> None:
    """Existing YAMLs without ``greeting:`` still parse cleanly."""
    cfg = Settings(mock_hardware=True)
    assert cfg.greeting is None


def test_empty_greeting_block_defaults_to_disabled() -> None:
    """``greeting: {}`` is treated as an explicit no-op, not activation."""
    cfg = Settings(mock_hardware=True, greeting=GreetingConfig())
    assert cfg.greeting is not None
    assert cfg.greeting.enabled is False


def test_greeting_pilot_example_overlay_loads_with_enabled_greeting() -> None:
    """``config/greeting_pilot.yaml.example`` is operator documentation.

    Schema drift would invalidate it silently because
    ``scripts/validate_configs.py`` only globs ``*.yaml`` (the
    ``.example`` suffix is filtered out by accident). This test asserts
    the overlay still parses + produces an enabled
    :class:`GreetingConfig` with the documented name list.
    """
    overlay = _REPO_ROOT / "config" / "greeting_pilot.yaml.example"
    if not overlay.exists():  # pragma: no cover — exists in repo at HEAD
        pytest.skip("greeting_pilot.yaml.example missing from working tree")

    settings = load_settings(overlay)
    assert settings.greeting is not None
    assert settings.greeting.enabled is True
    # Names are operator-edited; assert non-empty rather than a specific
    # list so an example update doesn't break the test.
    assert len(settings.greeting.names) >= 1
    # The template must always carry the {names} placeholder — the
    # validator enforces this when enabled=True, so failure here means
    # the example slipped past the validator.
    assert "{names}" in settings.greeting.message_template
