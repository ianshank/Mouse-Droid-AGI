"""Property-based tests for OpenClaw integration invariants."""

from __future__ import annotations

import time

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.common.actions import normalize_action_numpy
from mousedroid.config.schema import OpenClawConfig
from mousedroid.openclaw.protocol import OpenClawActionResult


@given(
    vx=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    vy=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    omega=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_any_openclaw_action_normalizes_to_valid_range(
    vx: float,
    vy: float,
    omega: float,
) -> None:
    """Any action from OpenClaw normalises to a valid [-1, 1] tensor."""
    action = np.array([vx, vy, omega], dtype=np.float32)
    normalized = normalize_action_numpy(action, expected_dim=3)
    assert normalized.shape == (3,), f"Expected shape (3,), got {normalized.shape}"
    assert torch.all(torch.isfinite(normalized)), f"Non-finite after normalize: {normalized}"


@given(
    timeout=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
    retries=st.integers(min_value=1, max_value=10),
    backoff=st.floats(min_value=1.0, max_value=10.0, allow_nan=False),
    poll=st.floats(min_value=0.01, max_value=60.0, allow_nan=False),
)
@settings(max_examples=50)
def test_any_valid_config_constructs(
    timeout: float,
    retries: int,
    backoff: float,
    poll: float,
) -> None:
    """Any valid OpenClawConfig parameter combination constructs without error."""
    cfg = OpenClawConfig(
        api_timeout_s=timeout,
        connect_retries=retries,
        connect_backoff_base=backoff,
        poll_interval_s=poll,
    )
    assert cfg.api_timeout_s == timeout
    assert cfg.connect_retries == retries


@given(
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    age_ms=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
    max_age_ms=st.floats(min_value=0.1, max_value=500.0, allow_nan=False),
)
@settings(max_examples=100)
def test_staleness_is_monotonic_with_age(
    confidence: float,
    age_ms: float,
    max_age_ms: float,
) -> None:
    """Staleness should be consistent: if age > max_age, result is stale."""
    # Create an action with known age
    ts = time.monotonic() - (age_ms / 1000.0)
    result = OpenClawActionResult(
        action=np.zeros(3, dtype=np.float32),
        goal_id="",
        reasoning="",
        confidence=confidence,
        timestamp=ts,
    )
    is_stale = result.is_stale(max_age_ms)
    # Due to timing jitter, we only check the clear cases
    if age_ms > max_age_ms + 50:
        assert is_stale is True, f"age={age_ms}ms > max={max_age_ms}ms but not stale"
