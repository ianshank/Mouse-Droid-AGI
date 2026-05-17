"""Regression tests for per-engine cache subdirs (Copilot MED, PR #98).

The Tier C1.2 ``build_weight_update_pollers`` factory now isolates each
poller's cache directory under a per-engine subdir so the policy and
world-model pollers can't overwrite each other's ``sha256.txt`` manifest
during concurrent poll cycles.

Also pins the new ``cache_dir_override`` constructor kwarg semantics on
``HuggingFaceWeightUpdatePoller`` so external callers building pollers
directly retain back-compat (omit the kwarg → legacy ``cfg.cache_dir``
behaviour).
"""

from __future__ import annotations

from pathlib import Path

from mousedroid.cloud.weight_update_poller import HuggingFaceWeightUpdatePoller
from mousedroid.config.schema import Settings


def test_constructor_uses_cfg_cache_dir_when_override_omitted(tmp_path: Path) -> None:
    """Legacy path: no ``cache_dir_override`` → poller uses ``cfg.cache_dir``."""
    cfg = Settings(mock_hardware=True)
    cfg.cloud.weight_update.cache_dir = str(tmp_path / "legacy_cache")

    poller = HuggingFaceWeightUpdatePoller(
        cfg.cloud.weight_update,
        repo_id="x/policy",
        filename="policy.onnx",
        engine_type="policy",
    )
    assert poller._cache_dir == (tmp_path / "legacy_cache").resolve()


def test_constructor_override_takes_precedence(tmp_path: Path) -> None:
    """``cache_dir_override`` wins over ``cfg.cache_dir``."""
    cfg = Settings(mock_hardware=True)
    cfg.cloud.weight_update.cache_dir = str(tmp_path / "legacy_cache")
    override = tmp_path / "per_engine" / "policy"

    poller = HuggingFaceWeightUpdatePoller(
        cfg.cloud.weight_update,
        repo_id="x/policy",
        filename="policy.onnx",
        engine_type="policy",
        cache_dir_override=override,
    )
    assert poller._cache_dir == override.resolve()


def test_factory_assigns_per_engine_subdirs(tmp_path: Path) -> None:
    """``build_weight_update_pollers`` isolates policy / world_model subdirs.

    Without this isolation the two pollers race on ``sha256.txt`` writes in
    the shared root and produce spurious SHA-mismatch dead-letters.
    """
    from mousedroid.factory import build_weight_update_pollers

    cfg = Settings(mock_hardware=True)
    cfg.cloud.weight_update.cache_dir = str(tmp_path / "shared_root")
    cfg.cloud.weight_update.poll_interval_s = 30.0  # enable pollers
    cfg.cloud.weight_update.world_model_enabled = True
    cfg.cloud.weight_update.world_model_repo_id = "operator/wm-fleet"

    pollers = build_weight_update_pollers(cfg)
    assert set(pollers.keys()) == {"policy", "world_model"}

    policy_cache = pollers["policy"]._cache_dir  # type: ignore[attr-defined]
    world_cache = pollers["world_model"]._cache_dir  # type: ignore[attr-defined]

    expected_root = (tmp_path / "shared_root").resolve()
    assert policy_cache == expected_root / "policy"
    assert world_cache == expected_root / "world_model"
    # Sanity: distinct dirs — manifest collisions impossible.
    assert policy_cache != world_cache


def test_factory_returns_empty_when_polling_disabled(tmp_path: Path) -> None:
    """Default ``poll_interval_s == 0.0`` → empty mapping (back-compat preserved)."""
    from mousedroid.factory import build_weight_update_pollers

    cfg = Settings(mock_hardware=True)
    cfg.cloud.weight_update.cache_dir = str(tmp_path / "ignored")
    # Don't enable polling.
    assert cfg.cloud.weight_update.poll_interval_s == 0.0
    pollers = build_weight_update_pollers(cfg)
    assert pollers == {}


def test_legacy_singular_factory_does_not_use_subdirs(tmp_path: Path) -> None:
    """``build_weight_update_poller`` (deprecated singular) keeps legacy layout.

    The legacy factory only builds one poller, so there's no manifest
    collision to fix — preserving the legacy cache-dir layout avoids
    moving operator weight files on a back-compat upgrade.
    """
    from mousedroid.factory import build_weight_update_poller

    cfg = Settings(mock_hardware=True)
    cfg.cloud.weight_update.cache_dir = str(tmp_path / "legacy")
    cfg.cloud.weight_update.poll_interval_s = 30.0

    poller = build_weight_update_poller(cfg)
    assert poller is not None
    # No engine subdir — exact legacy ``cache_dir`` path.
    assert poller._cache_dir == (tmp_path / "legacy").resolve()  # type: ignore[attr-defined]
