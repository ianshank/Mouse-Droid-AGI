"""C1.2: build_weight_update_pollers returns a mapping keyed by engine_type."""

from __future__ import annotations

from mousedroid.cloud.protocol import WeightUpdatePollerProtocol
from mousedroid.config.schema import Settings
from mousedroid.factory import build_weight_update_pollers


def test_returns_empty_dict_when_disabled() -> None:
    """Empty mapping when polling is disabled (poll_interval_s == 0.0)."""
    cfg = Settings(mock_hardware=True)  # poll_interval_s == 0.0 (default)
    pollers = build_weight_update_pollers(cfg)
    assert pollers == {}


def test_returns_policy_only_by_default_when_enabled() -> None:
    """Only the policy poller is built when world_model_enabled defaults False."""
    cfg = Settings(mock_hardware=True)
    cfg.cloud.weight_update.poll_interval_s = 1.0
    pollers = build_weight_update_pollers(cfg)
    assert set(pollers.keys()) == {"policy"}
    assert isinstance(pollers["policy"], WeightUpdatePollerProtocol)


def test_returns_both_pollers_when_world_model_enabled() -> None:
    """Both pollers are built and iteration order is policy-before-world-model."""
    cfg = Settings(mock_hardware=True)
    cfg.cloud.weight_update.poll_interval_s = 1.0
    cfg.cloud.weight_update.world_model_enabled = True
    pollers = build_weight_update_pollers(cfg)
    assert set(pollers.keys()) == {"policy", "world_model"}
    assert isinstance(pollers["policy"], WeightUpdatePollerProtocol)
    assert isinstance(pollers["world_model"], WeightUpdatePollerProtocol)
    # Insertion order is policy-before-world-model so the orchestrator
    # consumes pending updates in a deterministic order.
    assert list(pollers.keys()) == ["policy", "world_model"]
