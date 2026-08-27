"""Backwards-compat pins for F-034 — who the mlflow tracking_uri flip does and does not affect.

Mirrors ``tests/regression/test_gcp_egress_defaults_backwards_compat.py``'s
pattern for F-029 (an existing-field default *mutation*, not a new-field
addition, needs proof of exactly whose behaviour it changes).

**No shipped config file is affected**: not one references
``observability:``/``experiment_logger:`` at all, so every file-configured
deployment resolves ``Settings.observability`` to ``None``, which
``build_experiment_logger`` maps to ``NoOpExperimentLogger`` regardless of
what ``tracking_uri`` defaults to.

**The env-var path IS affected, and that is pinned here rather than
glossed.** ``MOUSEDROID_OBSERVABILITY__EXPERIMENT_LOGGER__BACKEND=mlflow``
— the opt-in ``docs/runbooks/mlflow-local-ui.md`` documents — materializes
the whole config block from defaults, so such an operator picks up the new
sqlite backend on upgrade and their existing ``mlruns/`` history stops
appearing in the UI. A file-only scan would report this change as fully
inert, which is not true; the last test below states the real contract so
the claim cannot quietly drift back to the rosier one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.loader import load_settings
from mousedroid.config.schema.root import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"


def _shipped_config_paths() -> list[Path]:
    """Every shipped config file, including the ones a naive glob misses.

    Deliberately broader than ``glob("*.yaml")``: this repo also ships
    ``*.yml`` files, config subdirectories (e.g. ``config/prometheus/``),
    and ``*.yaml.example`` operator templates, and ``load_settings`` accepts
    an arbitrary overlay path — so a narrow top-level glob would let a
    future overlay introduce an ``observability:`` block without tripping
    this pin.
    """
    paths = sorted(
        p
        for pattern in ("**/*.yaml", "**/*.yml", "**/*.yaml.example", "**/*.yml.example")
        for p in _CONFIG_DIR.glob(pattern)
        if p.is_file()
    )
    assert paths, f"no shipped config files found under {_CONFIG_DIR}"
    return paths


def test_absent_observability_block_still_yields_noop_logger() -> None:
    """No observability: block means no experiment-logger config at all."""
    assert Settings(mock_hardware=True).observability is None


def test_no_shipped_config_references_the_experiment_logger_block() -> None:
    """Every config/*.yaml is unaffected by the tracking_uri default flip.

    If a future edit adds an ``observability:`` block to a shipped overlay
    without an explicit ``tracking_uri``, this test turns red rather than
    the rover quietly switching its experiment-tracking backend.
    """
    for path in _shipped_config_paths():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert "observability" not in data, (
            f"{path.name} declares an observability: block -- this test needs "
            "updating to check its tracking_uri is set explicitly, per the "
            "F-029 twin-overlay precedent, since it can no longer rely on "
            "the schema default being inert for every shipped config"
        )


def test_no_shipped_config_sets_mlflow_backend_without_this_check_knowing() -> None:
    """Belt-and-braces: scan the raw YAML text, not just the parsed block key.

    Guards against a hypothetical future overlay setting
    ``experiment_logger.backend: mlflow`` somewhere other than directly
    under a top-level ``observability:`` key (e.g. via a merge/anchor).
    """
    for path in _shipped_config_paths():
        text = path.read_text(encoding="utf-8")
        assert "experiment_logger" not in text, (
            f"{path.name} references experiment_logger -- re-validate this "
            "config's effective tracking_uri explicitly rather than trusting "
            "the schema default stays inert for it"
        )


def test_env_var_optin_materializes_the_new_sqlite_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented env-var opt-in DOES adopt the new backend -- pinned, not hidden.

    ``docs/runbooks/mlflow-local-ui.md`` documents enabling mlflow with the
    backend env var alone. Setting it materializes the entire
    ``observability.experiment_logger`` block from schema defaults, so such
    an operator moves from the legacy ``mlruns/`` directory store to the new
    sqlite database on upgrade, and their old runs stop showing up.

    This is the change's real (narrow) blast radius. Asserting it here keeps
    the file-scan tests above from being mistaken for proof that the flip is
    inert for *everyone* -- it is inert for every shipped config file, which
    is a strictly weaker claim. The runbook carries the operator-facing
    migration note; this pin makes the behaviour itself reviewable.
    """
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")
    monkeypatch.setenv("MOUSEDROID_OBSERVABILITY__EXPERIMENT_LOGGER__BACKEND", "mlflow")

    settings = load_settings()

    assert settings.observability is not None, (
        "the documented env-var opt-in no longer materializes the "
        "observability block -- the runbook's instructions are now wrong"
    )
    logger_cfg = settings.observability.experiment_logger
    assert logger_cfg.backend == "mlflow"
    assert logger_cfg.tracking_uri == "sqlite:///mlflow.db"
