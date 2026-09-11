"""Smoke: isaaclab rover env module imports without the [isaac] extra."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


def test_isaaclab_rover_env_imports_without_isaaclab() -> None:
    from mousedroid.sim.isaaclab import rover_env

    assert rover_env.RoverIsaacLabEnv is not None
    assert rover_env._isaaclab_available() in (True, False)
