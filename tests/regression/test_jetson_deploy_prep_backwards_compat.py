"""Regression pins for the Jetson deploy-prep sprint.

Locks the contracts that make the deployment surface *truthful* — each one
encodes a defect that shipped silently before:

* ``config/jetson_production.yaml`` ships ``esp32.enabled: false`` so a dead
  ESP32 cannot crash-loop the container, while the schema default stays
  ``True`` (backwards compatibility, CLAUDE.md invariant #9).
* ``Dockerfile.jetson`` actually sets ``PYTHONOPTIMIZE=1`` — the runtime
  contract that a dozen documents already assert — and the in-container
  ``ci.sh`` steps override it back to ``0`` so the pytest suite keeps
  ``assert`` semantics.
* The dev-tools build arg agrees between Dockerfile and compose, so a lean
  image can't silently void the Pattern-B pillar checks.
* The Phase-2 pillar run is gated by an env-overridable strict-skips knob.
* ``config/docker.env.example`` keeps ``MOUSEDROID_TELEMETRY_TOKEN`` as a
  parseable key (commenting it out would disable the ``host_env_keys``
  preflight drift check and drop it from ``host_bootstrap.sh`` seeding).

Source-text pins follow the precedent of
``tests/regression/test_jetson_phase1_oom_guard.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import ESP32Config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROD_CONFIG = _REPO_ROOT / "config" / "jetson_production.yaml"
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "default.yaml"
_DOCKERFILE = _REPO_ROOT / "Dockerfile.jetson"
_COMPOSE = _REPO_ROOT / "docker-compose.jetson.yml"
_VALIDATION_SCRIPT = _REPO_ROOT / "scripts" / "jetson_full_validation.sh"
_ENV_EXAMPLE = _REPO_ROOT / "config" / "docker.env.example"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"{path} not present in this checkout")
    return path.read_text()


# ---------------------------------------------------------------------------
# ESP32 safe-by-default (F-008 crash-loop guard)
# ---------------------------------------------------------------------------


def test_production_overlay_disables_esp32() -> None:
    """The production overlay must ship ``esp32.enabled: false``.

    A dead board with ``enabled=True`` makes ``orchestrator.start()`` ->
    ``esp32.connect()`` retry-then-raise, and the container crash-loops.
    """
    cfg = load_settings(_PROD_CONFIG)
    assert cfg.esp32.enabled is False, (
        "config/jetson_production.yaml must keep esp32.enabled: false while the "
        "board is dead (F-008); lift it per-host via MOUSEDROID_ESP32__ENABLED"
    )


def test_esp32_schema_default_stays_enabled() -> None:
    """The *schema* default must remain True — only the prod overlay opts out.

    Invariant #9: existing YAML files load unchanged.
    """
    assert ESP32Config().enabled is True


def test_production_overlay_keeps_serial_port_and_baud() -> None:
    """Disabling the driver must not drop the by-id port / baud wiring.

    The values are still needed for USB-C discovery and for the post-repair
    env-flip to resolve the real device.
    """
    cfg = load_settings(_PROD_CONFIG)
    assert cfg.esp32.serial_port
    assert "by-id" in cfg.esp32.serial_port
    assert cfg.esp32.serial_baud > 0


# ---------------------------------------------------------------------------
# PYTHONOPTIMIZE runtime contract + test-suite carve-out
# ---------------------------------------------------------------------------


def test_dockerfile_sets_pythonoptimize() -> None:
    """The image must set the documented ``PYTHONOPTIMIZE=1`` runtime contract."""
    text = _read(_DOCKERFILE)
    assert re.search(r"^ENV PYTHONOPTIMIZE=1\s*$", text, re.MULTILINE), (
        "Dockerfile.jetson must set ENV PYTHONOPTIMIZE=1 — CLAUDE.md, AGENTS.md "
        "and the -O-safety code contracts all assume it"
    )


def test_container_ci_steps_override_pythonoptimize_to_zero() -> None:
    """Every in-container ``ci.sh`` docker-exec must pass ``-e PYTHONOPTIMIZE=0``.

    The pytest suite has only ever run with ``assert`` semantics intact;
    inheriting the image's ``-O`` would silently strip asserts in
    non-rewritten helper modules.
    """
    text = _read(_VALIDATION_SCRIPT)
    ci_exec_lines = [
        line
        for line in text.splitlines()
        if "docker exec" in line and "MOUSEDROID_MOCK_HARDWARE=true" in line
    ]
    assert ci_exec_lines, "no in-container ci.sh docker-exec line found"
    for line in ci_exec_lines:
        assert (
            "-e PYTHONOPTIMIZE=0" in line
        ), f"in-container CI exec must neutralise -O for pytest, got: {line.strip()!r}"


# ---------------------------------------------------------------------------
# Dev-tools build-arg agreement (lean image cannot void Pattern-B pillars)
# ---------------------------------------------------------------------------


def test_dockerfile_and_compose_dev_tools_defaults_agree() -> None:
    """``INSTALL_DEV_TOOLS`` must default the same way in both files.

    Compose defaults to ``true``; a Dockerfile default of ``false`` meant a
    bare ``docker build`` produced an image whose Pattern-B pillar checks
    silently SKIP.
    """
    dockerfile_match = re.search(
        r"^ARG INSTALL_DEV_TOOLS=(\w+)\s*$",
        _read(_DOCKERFILE),
        re.MULTILINE,
    )
    assert dockerfile_match, "ARG INSTALL_DEV_TOOLS not found in Dockerfile.jetson"

    compose_match = re.search(
        r"INSTALL_DEV_TOOLS:\s*\$\{MOUSEDROID_INSTALL_DEV_TOOLS:-(\w+)\}",
        _read(_COMPOSE),
    )
    assert compose_match, "INSTALL_DEV_TOOLS build arg not found in docker-compose.jetson.yml"

    assert dockerfile_match.group(1) == compose_match.group(1), (
        "Dockerfile ARG default and compose build-arg default must agree "
        f"(got {dockerfile_match.group(1)!r} vs {compose_match.group(1)!r})"
    )


# ---------------------------------------------------------------------------
# Strict pillar-skip knob (env-overridable — no hardcoded tunables)
# ---------------------------------------------------------------------------


def test_validation_script_exposes_strict_skips_knob() -> None:
    """The strict-skips gate must be env-overridable and default to on."""
    text = _read(_VALIDATION_SCRIPT)
    assert (
        'PILLARS_STRICT_SKIPS="${MOUSEDROID_VALIDATION_PILLARS_STRICT_SKIPS:-1}"' in text
    ), "strict-skips must be a documented env-overridable knob defaulting to 1"
    assert (
        "MOUSEDROID_VALIDATION_PILLARS_STRICT_SKIPS" in text.split("set -uo pipefail")[0]
    ), "the knob must be documented in the script header env-override catalog"
    assert "--strict-skips" in text, "Phase-2 pillar step must be able to pass --strict-skips"


def test_phase1_dry_run_pillars_never_get_strict_skips() -> None:
    """Phase-1's dry-run pillar steps must not pass the incompatible flag.

    ``--strict-skips`` with ``--dry-run`` is an argparse usage error; wiring
    it into the dry-run step would break Phase 1 outright.
    """
    text = _read(_VALIDATION_SCRIPT)
    for line in text.splitlines():
        if "--dry-run" in line and "validate_pillars" in line:
            assert (
                "--strict-skips" not in line
            ), f"dry-run pillar step must not pass --strict-skips: {line.strip()!r}"


# ---------------------------------------------------------------------------
# Secret-surface template hygiene
# ---------------------------------------------------------------------------


def test_telemetry_token_key_is_parseable_from_env_template() -> None:
    """The token key must stay uncommented so preflight/bootstrap still see it.

    ``_parse_env_keys`` skips ``#`` lines, so commenting the key would (a)
    silently disable the ``host_env_keys`` drift WARN on hosts missing the
    token and (b) drop it from ``host_bootstrap.sh`` seeding.
    """
    from mousedroid.validation.preflight import _parse_env_keys

    keys = _parse_env_keys(_read(_ENV_EXAMPLE))
    assert "MOUSEDROID_TELEMETRY_TOKEN" in keys


def test_env_template_ships_no_placeholder_token_value() -> None:
    """The template must not ship a usable-looking default token."""
    text = _read(_ENV_EXAMPLE)
    assert (
        "MOUSEDROID_TELEMETRY_TOKEN=changeme" not in text
    ), "a placeholder token invites deploying it verbatim; ship an empty value"


def test_env_template_documents_esp32_repair_slot() -> None:
    """The ESP32 repair lever must be discoverable from the env template."""
    assert "MOUSEDROID_ESP32__ENABLED" in _read(_ENV_EXAMPLE)


# ---------------------------------------------------------------------------
# Launcher config-stack sanity (protects the nightly's two-overlay mode)
# ---------------------------------------------------------------------------


def test_default_plus_production_overlay_stack_loads() -> None:
    """The nightly's ``default.yaml,jetson_production.yaml`` stack must load.

    ``jetson-nightly.yml`` stacks both overlays, while compose and the
    full-validation script load the production file alone — both resolution
    modes must stay valid.
    """
    stacked = load_settings(_DEFAULT_CONFIG, _PROD_CONFIG)
    base_only = load_settings(_DEFAULT_CONFIG)

    assert stacked.platform == "mouse_droid"
    # The production overlay must WIN on every key it declares — including the
    # new esp32 opt-out — rather than being shadowed by the earlier overlay.
    assert stacked.esp32.enabled is False
    assert stacked.safety.battery_critical_v == 0.0
    assert stacked.safety.min_valid_sensors == 1
    # ...and those really are overrides, not coincidences.
    assert base_only.safety.battery_critical_v != 0.0
    assert base_only.safety.min_valid_sensors != 1
    # Deep merge, not wholesale replacement: blocks the production overlay
    # does not declare at all (model, mcts, memory, …) keep their base values.
    assert stacked.model.latent_dim == base_only.model.latent_dim
    assert stacked.mcts.n_simulations_base == base_only.mcts.n_simulations_base
