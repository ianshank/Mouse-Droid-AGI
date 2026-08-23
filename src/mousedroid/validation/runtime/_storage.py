"""PCIe/NVMe SSD layout runtime validation helpers (Task 2 domain)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from mousedroid.validation.runtime._shared import (
    _collect_configured_runtime_paths,
    _subprocess_timeout_s,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


@dataclass(frozen=True)
class PcieSsdDiagnostics:
    """Diagnostics record for one ``verify_pcie_ssd_layout`` probe (Task 2).

    Mirrors the read-only nature of the probe — every field defaults so a
    SKIP path (no NVMe device on bus, etc.) can return a partially-populated
    instance without raising.
    """

    pcie_devices: tuple[str, ...] = ()
    block_devices: tuple[str, ...] = ()
    mount_target: Path | None = None
    free_gb: float = 0.0
    total_gb: float = 0.0
    required_gb: float = 0.0
    configured_paths: dict[str, str] = field(default_factory=dict)
    smartctl_health: str | None = None


def verify_pcie_ssd_layout(cfg: Settings) -> PcieSsdDiagnostics:
    """Probe the NVMe SSD on PCIe and assert capacity for the configured paths.

    Read-only probe — never writes, never mutates kernel state. All probes
    use stdlib (``subprocess`` + ``shutil``) so the helper has no extra
    dependencies. SKIP paths (missing tools, no device) populate the
    returned dataclass with safe defaults so the CLI can convert any field
    to PASS / SKIP / FAIL deterministically.

    Resolution chain for the mount target:
      1. ``$MOUSEDROID_SSD_MOUNT`` env override if set + exists.
      2. ``findmnt -no TARGET /dev/nvme0n1p1`` when the block device exists.
      3. The parent dir of ``cfg.experience.path`` if it exists on disk.
      4. ``None`` -> the SKIP branch in the CLI.

    ``required_gb`` is derived from ``cfg.experience.map_size_gb`` (the
    LMDB preallocation) — that's the largest contiguous on-disk
    allocation the runtime makes, so it's the right capacity gate.

    Args:
        cfg: Fully resolved settings — read-only.

    Returns:
        Populated :class:`PcieSsdDiagnostics` instance.
    """
    import shutil

    timeout_s = _subprocess_timeout_s(cfg)

    # 1. PCIe device enumeration (best-effort; missing lspci -> empty list).
    pcie_devices: tuple[str, ...] = ()
    if shutil.which("lspci"):
        try:
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
            )
            if result.returncode == 0:
                pcie_devices = tuple(
                    line.strip()
                    for line in result.stdout.splitlines()
                    if "nvme" in line.lower() or "non-volatile memory" in line.lower()
                )
        except (subprocess.TimeoutExpired, OSError):
            pass

    # 2. Block-device enumeration via lsblk.
    block_devices: tuple[str, ...] = ()
    if shutil.which("lsblk"):
        try:
            result = subprocess.run(
                ["lsblk", "-d", "-o", "NAME,SIZE,TYPE,TRAN", "-n"],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
            )
            if result.returncode == 0:
                block_devices = tuple(
                    line.strip() for line in result.stdout.splitlines() if "nvme" in line.lower()
                )
        except (subprocess.TimeoutExpired, OSError):
            pass

    # 3. Mount-target resolution chain.
    mount_target = _resolve_pcie_ssd_mount(cfg)

    # 4. Capacity probe via shutil.disk_usage (stdlib, cross-platform).
    free_gb = 0.0
    total_gb = 0.0
    if mount_target is not None and mount_target.exists():
        try:
            usage = shutil.disk_usage(mount_target)
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
        except OSError:
            pass

    # 5. SMART health (optional — missing smartctl is a SKIP, not a FAIL).
    smartctl_health: str | None = None
    if shutil.which("smartctl"):
        try:
            result = subprocess.run(
                ["smartctl", "-H", _nvme_device_for(cfg)],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "overall-health" in line.lower():
                        smartctl_health = line.split(":")[-1].strip()
                        break
        except (subprocess.TimeoutExpired, OSError):
            pass

    # 6. Configured-path inventory.
    configured_paths = _collect_configured_runtime_paths(cfg)

    return PcieSsdDiagnostics(
        pcie_devices=pcie_devices,
        block_devices=block_devices,
        mount_target=mount_target,
        free_gb=free_gb,
        total_gb=total_gb,
        required_gb=float(cfg.experience.map_size_gb),
        configured_paths=configured_paths,
        smartctl_health=smartctl_health,
    )


def _resolve_pcie_ssd_mount(cfg: Settings) -> Path | None:
    """Resolution chain for the NVMe mount target (see public helper docstring).

    Returns ``None`` when neither the env override nor ``findmnt`` can pin
    the mount — the CLI consumer then emits a SKIP. The previous
    ``cfg.experience.path.parent`` fallback was deliberately removed
    (PR #104 follow-up): on a freshly imaged Orin Nano with no NVMe at
    all, the parent of ``/home/jetson/mousedroid_experience`` is the
    rootfs ``/home/jetson`` — accepting that as the "SSD mount" produced
    a FALSE PASS on the "is the LMDB actually on the SSD?" check, which
    is the entire reason this smoke exists.
    """
    env_mount = os.environ.get(cfg.experience.ssd_mount_override_env_var, "").strip()
    if env_mount:
        candidate = Path(env_mount)
        if candidate.exists():
            return candidate

    # findmnt against the configured NVMe partition path.
    import shutil

    if shutil.which("findmnt"):
        try:
            result = subprocess.run(
                ["findmnt", "-no", "TARGET", _nvme_partition_for(cfg)],
                capture_output=True,
                text=True,
                check=False,
                timeout=_subprocess_timeout_s(cfg),
            )
            if result.returncode == 0:
                target = result.stdout.strip()
                if target:
                    candidate = Path(target)
                    if candidate.exists():
                        return candidate
        except (subprocess.TimeoutExpired, OSError):
            pass

    return None


def _nvme_partition_for(cfg: Settings) -> str:
    """Return the NVMe partition path to feed ``findmnt``.

    Schema-driven via ``cfg.experience.nvme_partition`` (added in the
    PR #104 hardening pass); falls back to the canonical first-partition
    string for tests that build minimal ``Settings`` instances without
    overriding the new field.
    """
    return str(getattr(cfg.experience, "nvme_partition", "/dev/nvme0n1p1"))


def _nvme_device_for(cfg: Settings) -> str:
    """Return the NVMe block device path to feed ``smartctl``."""
    return str(getattr(cfg.experience, "nvme_device", "/dev/nvme0n1"))
