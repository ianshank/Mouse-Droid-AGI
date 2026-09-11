"""Duck-typed IMU / pose / LiDAR readers for the Isaac Lab rover env.

Helpers operate on ``SimpleNamespace`` fakes in CI (the same pattern as the
contact-force tests). Live Isaac Lab types are never imported here. Shapes
come from :mod:`mousedroid.sim.protocols` and
:class:`RoverObservationConfig` — never from hardware lidar schema fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.sim.isaaclab.constants import ROVER_SENSOR_LINK_NAMES
from mousedroid.sim.kinematics import chassis_pose_xy_yaw, yaw_from_wxyz
from mousedroid.sim.protocols import ROVER_CHASSIS_POSE_DIM, ROVER_IMU_DIM

_IMU_LINK = ROVER_SENSOR_LINK_NAMES[0]
_LIDAR_LINK = ROVER_SENSOR_LINK_NAMES[1]


def to_numpy(value: Any) -> NDArray[np.float32] | None:
    """Coerce a tensor / array / list into float32, or ``None`` if empty.

    Args:
        value: Torch tensor, numpy array, nested list, or ``None``.

    Returns:
        A float32 ndarray, or ``None`` when ``value`` is missing/empty.
    """
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        value = numpy_fn()
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def first_env_row(value: Any) -> NDArray[np.float32] | None:
    """Take the first parallel-env row from a batched Isaac buffer.

    Args:
        value: Array-like of shape ``(num_envs, ...)`` or a 1-D vector.

    Returns:
        The first-env slice as float32, or ``None``.
    """
    arr = to_numpy(value)
    if arr is None:
        return None
    if arr.ndim <= 1:
        return arr
    return np.asarray(arr[0], dtype=np.float32)


def fit_vector(raw: NDArray[np.float32], dim: int) -> NDArray[np.float32]:
    """Copy ``raw`` into a length-``dim`` vector (pad with zeros / truncate).

    Args:
        raw: Source values.
        dim: Target length.

    Returns:
        Shape ``(dim,)`` float32 vector.
    """
    out = np.zeros(dim, dtype=np.float32)
    n = min(int(raw.size), dim)
    if n > 0:
        out[:n] = np.asarray(raw, dtype=np.float32).reshape(-1)[:n]
    return out


def identity_chassis_pose() -> NDArray[np.float32]:
    """Return ``[x=0, y=0, cos(0), sin(0)]`` — URDF home pose."""
    return chassis_pose_xy_yaw(0.0, 0.0, 0.0)


def resample_lidar(raw: NDArray[np.float32], n_sectors: int) -> NDArray[np.float32]:
    """Resample a 1-D range scan onto ``n_sectors`` bins by linear interpolation.

    Args:
        raw: Flat range readings (metres).
        n_sectors: Target sector count from
            :attr:`RoverObservationConfig.lidar_num_sectors`.

    Returns:
        Shape ``(n_sectors,)`` float32 vector.
    """
    flat = np.asarray(raw, dtype=np.float32).reshape(-1)
    n = int(flat.size)
    if n == n_sectors:
        return flat
    if n == 0:
        return np.zeros(n_sectors, dtype=np.float32)
    idx = np.linspace(0.0, float(n - 1), n_sectors)
    xp = np.arange(n, dtype=np.float32)
    return np.interp(idx, xp, flat).astype(np.float32)


def _sensor_data(sensors: Mapping[str, Any], *names: str) -> Any:
    """Return the first matching sensor handle's ``.data`` (or the handle)."""
    for name in names:
        handle = sensors.get(name)
        if handle is None:
            continue
        data = getattr(handle, "data", handle)
        if data is not None:
            return data
    return None


def _attr_row(owner: Any, *names: str) -> NDArray[np.float32] | None:
    """Return the first present named attribute as a first-env row."""
    if owner is None:
        return None
    for name in names:
        row = first_env_row(getattr(owner, name, None))
        if row is not None:
            return row
    return None


def read_rover_imu(*, sensors: Mapping[str, Any], articulation: Any) -> NDArray[np.float32]:
    """Read a 6-DoF IMU vector (lin-acc + ang-vel) from duck-typed handles.

    This fills the *observation* IMU channel (``ROVER_IMU_DIM``). It does
    **not** map into the RSSM 3-float ``imu_dim`` fusion slot — that slot stays
    mask-zero in :class:`RoverObsAdapter`.

    Args:
        sensors: Env ``_sensors`` mapping (link name → handle).
        articulation: Articulation handle or ``None``.

    Returns:
        Shape ``(ROVER_IMU_DIM,)`` float32 vector (zeros when unread).
    """
    data = _sensor_data(sensors, _IMU_LINK, "imu")
    art_data = getattr(articulation, "data", None)
    lin = _attr_row(data, "lin_acc_b", "lin_acc_w")
    if lin is None:
        lin = _attr_row(art_data, "lin_acc_b", "lin_acc_w")
    ang = _attr_row(data, "ang_vel_b", "ang_vel_w")
    if ang is None:
        ang = _attr_row(art_data, "ang_vel_b", "ang_vel_w")
    parts: list[NDArray[np.float32]] = []
    if lin is not None:
        parts.append(np.asarray(lin, dtype=np.float32).reshape(-1))
    if ang is not None:
        parts.append(np.asarray(ang, dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros(ROVER_IMU_DIM, dtype=np.float32)
    return fit_vector(np.concatenate(parts), ROVER_IMU_DIM)


def read_rover_pose(*, sensors: Mapping[str, Any], articulation: Any) -> NDArray[np.float32]:
    """Read chassis pose ``[x, y, cos(theta), sin(theta)]``.

    Prefers articulation ``root_pos_w`` / ``root_quat_w`` (wxyz). Falls back
    to an IMU-link pose buffer, then the identity heading.

    Args:
        sensors: Env ``_sensors`` mapping.
        articulation: Articulation handle or ``None``.

    Returns:
        Shape ``(ROVER_CHASSIS_POSE_DIM,)`` float32 vector.
    """
    art_data = getattr(articulation, "data", None)
    pos = _attr_row(art_data, "root_pos_w", "pos_w")
    quat = _attr_row(art_data, "root_quat_w", "quat_w")
    if pos is None or quat is None:
        data = _sensor_data(sensors, _IMU_LINK, "imu")
        if pos is None:
            pos = _attr_row(data, "root_pos_w", "pos_w")
        if quat is None:
            quat = _attr_row(data, "root_quat_w", "quat_w")
    if pos is None or quat is None:
        return identity_chassis_pose()
    x = float(pos[0])
    y = float(pos[1])
    fitted = fit_vector(quat, ROVER_CHASSIS_POSE_DIM)
    w, qx, qy, qz = (float(v) for v in fitted)
    return chassis_pose_xy_yaw(x, y, yaw_from_wxyz(w, qx, qy, qz))


def _lidar_ranges_from_data(data: Any) -> NDArray[np.float32] | None:
    """Extract a 1-D range scan from a duck-typed ray-caster data object."""
    direct = _attr_row(data, "ray_distance", "distances", "ray_distances")
    if direct is not None:
        return np.asarray(direct, dtype=np.float32).reshape(-1)
    hits = first_env_row(getattr(data, "ray_hits_w", None))
    if hits is None:
        return None
    arr = np.asarray(hits, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    origin = first_env_row(getattr(data, "pos_w", None))
    if origin is None:
        origin = np.zeros(arr.shape[-1], dtype=np.float32)
    delta = arr - origin.reshape(1, -1)
    ranges = np.linalg.norm(delta, axis=-1)
    return np.asarray(ranges, dtype=np.float32)


def read_rover_lidar(
    *,
    sensors: Mapping[str, Any],
    n_sectors: int,
    max_range_m: float,
) -> NDArray[np.float32]:
    """Read sector-binned LiDAR clearance in ``[0, 1]`` of ``max_range_m``.

    Args:
        sensors: Env ``_sensors`` mapping.
        n_sectors: ``RoverObservationConfig.lidar_num_sectors``.
        max_range_m: ``RoverObservationConfig.lidar_max_range_m`` (sim clip,
            independent of the hardware LD19 range).

    Returns:
        Shape ``(n_sectors,)`` float32, zeros when no sensor is wired.
    """
    data = _sensor_data(sensors, _LIDAR_LINK, "lidar")
    raw = _lidar_ranges_from_data(data)
    if raw is None:
        return np.zeros(n_sectors, dtype=np.float32)
    resampled = resample_lidar(raw, n_sectors)
    if max_range_m <= 0.0:
        return np.zeros(n_sectors, dtype=np.float32)
    clipped = np.clip(resampled, 0.0, max_range_m)
    return (clipped / max_range_m).astype(np.float32)
