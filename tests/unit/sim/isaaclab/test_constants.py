"""URDF-parity tests for ``mousedroid.sim.isaaclab.constants``.

The constants module is the single source of truth for joint/link names
used by Phase B wiring. These tests parse the actual URDF and assert the
strings match — preventing the failure mode where the URDF is renamed
but the constants module isn't updated (or vice versa).

Runs entirely in pytest unit context; requires no Isaac Lab install.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mousedroid.sim.isaaclab.constants import (
    ROVER_CHASSIS_POSE_DIM,
    ROVER_IMU_DIM,
    ROVER_NUM_WHEELS,
    ROVER_OBSERVATION_KEYS,
    ROVER_SENSOR_LINK_NAMES,
    ROVER_WHEEL_JOINT_NAMES,
)
from mousedroid.sim.mock_rover_env import MockRoverEnv

# Resolve the URDF path relative to the repo root. We walk up from this
# file to find ``assets/rover/`` so the test is workspace-agnostic.
_URDF_PATH = Path(__file__).resolve()
while not (_URDF_PATH / "assets" / "rover" / "mse6_4wd.urdf").exists():
    if _URDF_PATH.parent == _URDF_PATH:
        break  # reached filesystem root without finding it
    _URDF_PATH = _URDF_PATH.parent
_URDF_PATH = _URDF_PATH / "assets" / "rover" / "mse6_4wd.urdf"


@pytest.fixture(scope="module")
def urdf_root() -> ET.Element:
    """Parse the rover URDF once per test module."""
    if not _URDF_PATH.exists():
        pytest.skip(f"URDF not found at {_URDF_PATH}")

    # version control; not untrusted input. stdlib ElementTree is the
    # right tool here. defusedxml would add a runtime dep for no gain.
    tree = ET.parse(_URDF_PATH)  # noqa: S314
    return tree.getroot()


def _urdf_joint_names(root: ET.Element) -> list[str]:
    """All ``<joint name=...>`` entries in the URDF."""
    return [j.attrib["name"] for j in root.findall("joint") if "name" in j.attrib]


def _urdf_link_names(root: ET.Element) -> list[str]:
    """All ``<link name=...>`` entries in the URDF."""
    return [link.attrib["name"] for link in root.findall("link") if "name" in link.attrib]


# ---------------------------------------------------------------------------
# URDF parity
# ---------------------------------------------------------------------------


def test_wheel_joint_names_exist_in_urdf(urdf_root: ET.Element) -> None:
    """Every ROVER_WHEEL_JOINT_NAMES entry must appear in the URDF."""
    urdf_joints = set(_urdf_joint_names(urdf_root))
    for joint_name in ROVER_WHEEL_JOINT_NAMES:
        assert joint_name in urdf_joints, (
            f"ROVER_WHEEL_JOINT_NAMES references {joint_name!r} but the URDF "
            f"only declares: {sorted(urdf_joints)}"
        )


def test_sensor_link_names_exist_in_urdf(urdf_root: ET.Element) -> None:
    """Every ROVER_SENSOR_LINK_NAMES entry must appear in the URDF."""
    urdf_links = set(_urdf_link_names(urdf_root))
    for link_name in ROVER_SENSOR_LINK_NAMES:
        assert link_name in urdf_links, (
            f"ROVER_SENSOR_LINK_NAMES references {link_name!r} but the URDF "
            f"only declares: {sorted(urdf_links)}"
        )


def test_urdf_has_exactly_n_wheel_joints(urdf_root: ET.Element) -> None:
    """The URDF must declare exactly ROVER_NUM_WHEELS continuous wheel joints.

    Catches the failure mode where a wheel is added or removed in the
    URDF without updating the constants module.
    """
    continuous_joints = [
        j.attrib["name"] for j in urdf_root.findall("joint") if j.attrib.get("type") == "continuous"
    ]
    assert len(continuous_joints) == ROVER_NUM_WHEELS, (
        f"URDF declares {len(continuous_joints)} continuous joints "
        f"({continuous_joints!r}); constants expect {ROVER_NUM_WHEELS}"
    )


# ---------------------------------------------------------------------------
# Constants-module internal invariants
# ---------------------------------------------------------------------------


def test_wheel_joint_count_matches_num_wheels() -> None:
    """ROVER_WHEEL_JOINT_NAMES length must equal ROVER_NUM_WHEELS."""
    assert len(ROVER_WHEEL_JOINT_NAMES) == ROVER_NUM_WHEELS


def test_policy_action_dim_decoupled_from_wheel_count() -> None:
    """Action dim is mode-dependent and lives on RoverActionConfig.

    The constants module deliberately does NOT export a numeric action
    dim. The 4-wheel chassis (``ROVER_NUM_WHEELS``) is a physical
    property of the URDF; the policy action vector is a mode-dependent
    value (2 for the default differential / body_velocity modes, could
    be 4 for a future independent-wheel mode). Conflating the two
    would lock the policy into 4-D commands and silently break the
    cross-backend contract with ``MockRoverEnv``.

    Both backends must read action dim from ``cfg.action.action_dim`` —
    this test pins that decoupling so a future PR can't re-add a
    misleading ``ROVER_ACTION_DIM`` constant.
    """
    from mousedroid.config.schema import RoverActionConfig
    from mousedroid.sim.isaaclab import constants as isaaclab_constants

    # The constants module exposes wheel count for physical chassis use,
    # not for action-vector sizing.
    assert ROVER_NUM_WHEELS == 4
    # The action dim lives on the config (mode-dependent) and is always
    # less than or equal to the wheel count.
    differential_dim = RoverActionConfig(mode="differential").action_dim
    body_velocity_dim = RoverActionConfig(mode="body_velocity").action_dim
    assert differential_dim == 2
    assert body_velocity_dim == 2
    # No ``ROVER_ACTION_DIM`` export — readers should never reach for it.
    assert not hasattr(isaaclab_constants, "ROVER_ACTION_DIM")


def test_observation_keys_exclude_ultrasonic() -> None:
    """Active rover baseline forbids ultrasonic / HC-SR04 observation channels.

    Explicit regression net so a future PR can't accidentally re-introduce
    the parked sensor modality. The arm platform is also parked per
    project CLAUDE.md — observation keys must remain rover-only.
    """
    for forbidden in ("ultrasonic", "hc_sr04", "sonar", "arm", "gripper"):
        assert forbidden not in ROVER_OBSERVATION_KEYS, (
            f"ROVER_OBSERVATION_KEYS must not include {forbidden!r} — "
            f"the active rover production baseline excludes this modality."
        )


# ---------------------------------------------------------------------------
# Cross-backend contract — Isaac Lab constants must match MockRoverEnv
# ---------------------------------------------------------------------------


def test_observation_keys_match_mock_rover_env() -> None:
    """The Isaac Lab observation contract must mirror MockRoverEnv **exactly**.

    Both backends conform to :class:`RoverEnvProtocol` — they're drop-in
    replacements. If their observation keys drift in either direction
    (mock adds one, Isaac Lab skips one, ordering changes) the
    orchestrator silently breaks when switching backends.

    Exact-equality assertion (not subset) is deliberate — a subset
    check would let the mock backend add a new default key the Isaac
    Lab backend silently drops, breaking the cross-backend contract
    while the test stays green.
    """
    from mousedroid.config.schema import RoverConfig

    # MockRoverEnv requires a RoverConfig + wheel geometry. Default values
    # exercise the rover production baseline (IMU + chassis pose +
    # wheel velocity + LiDAR, no ultrasonic, no camera).
    mock = MockRoverEnv(cfg=RoverConfig(), wheel_radius_m=0.1, track_width_m=0.3)
    mock_keys = mock.observation_keys

    assert mock_keys == ROVER_OBSERVATION_KEYS, (
        f"ROVER_OBSERVATION_KEYS {ROVER_OBSERVATION_KEYS!r} does not match "
        f"MockRoverEnv keys {mock_keys!r} exactly. Update both backends "
        f"in lockstep, or one will diverge from the orchestrator contract."
    )


def test_imu_and_chassis_dims_imported_from_protocols() -> None:
    """ROVER_IMU_DIM and ROVER_CHASSIS_POSE_DIM are re-exported, not redefined."""
    from mousedroid.sim import protocols as sim_protocols

    assert ROVER_IMU_DIM is sim_protocols.ROVER_IMU_DIM
    assert ROVER_CHASSIS_POSE_DIM is sim_protocols.ROVER_CHASSIS_POSE_DIM
    assert ROVER_NUM_WHEELS is sim_protocols.ROVER_NUM_WHEELS
