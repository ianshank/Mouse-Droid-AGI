"""Backwards-compat pins for F-034 — flipping the mlflow tracking_uri default is inert.

Mirrors ``tests/regression/test_gcp_egress_defaults_backwards_compat.py``'s
pattern for F-029 (an existing-field default *mutation*, not a new-field
addition, needs proof it changes behaviour for **no shipped config**) —
except every shipped ``config/*.yaml`` overlay is unaffected for a stronger
reason than F-029's twin-overlay case: not one of them even references
``observability:``/``experiment_logger:`` at all, let alone sets
``backend: mlflow`` or an explicit ``tracking_uri``. Every shipped
deployment resolves ``Settings.observability`` to its own ``None`` default,
which ``build_experiment_logger`` maps to ``NoOpExperimentLogger`` regardless
of what ``ExperimentLoggerConfig.tracking_uri`` itself defaults to -- so the
tracking_uri default is unreachable from any shipped config today.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mousedroid.config.schema.root import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"


def _shipped_config_paths() -> list[Path]:
    paths = sorted(_CONFIG_DIR.glob("*.yaml"))
    assert paths, f"no shipped config yaml files found under {_CONFIG_DIR}"
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
