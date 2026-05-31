"""USB-C device enumeration helper for Jetson smoke gates.

Pure helper — imports nothing hardware-specific. Resolves
``USBCDiscoveryConfig.required_endpoints`` against the configured by-id
root and emits structured logging so operator triage is grep-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from mousedroid.config.schema import USBCDiscoveryConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class EndpointStatus(str, Enum):
    """Status of a single required endpoint."""

    PRESENT = "present"
    MISSING = "missing"
    WARN = "warn"


@dataclass(frozen=True)
class EndpointResult:
    """Resolution outcome for a single endpoint."""

    name: str
    glob: str
    required: bool
    resolved_path: Path | None
    status: EndpointStatus


def resolve_endpoint(cfg: USBCDiscoveryConfig, name: str) -> Path | None:
    """Resolve a single endpoint name to its current by-id path, or None.

    Returns None when discovery is disabled, the named endpoint is not
    declared, or no by-id file matches the configured glob. Callers can
    use this to override a stale literal ``serial_port`` config field
    (e.g. ``esp32.serial_port``) with the live USB-C device path.
    """
    if not cfg.enabled:
        return None
    for spec in cfg.required_endpoints:
        if spec.name != name:
            continue
        matches = sorted(cfg.by_id_root.glob(spec.by_id_glob))
        if matches:
            return matches[0]
        return None
    return None


def enumerate_usbc_devices(
    cfg: USBCDiscoveryConfig,
) -> dict[str, EndpointResult]:
    """Resolve every required endpoint against ``cfg.by_id_root``.

    Returns an empty dict when discovery is disabled so callers can short-
    circuit without per-call enabled checks.
    """
    if not cfg.enabled:
        _log.debug("usbc_enumerate_skipped", reason="discovery_disabled")
        return {}

    results: dict[str, EndpointResult] = {}
    for spec in cfg.required_endpoints:
        matches = sorted(cfg.by_id_root.glob(spec.by_id_glob))
        if matches:
            results[spec.name] = EndpointResult(
                name=spec.name,
                glob=spec.by_id_glob,
                required=spec.required,
                resolved_path=matches[0],
                status=EndpointStatus.PRESENT,
            )
            _log.info(
                "usbc_endpoint_present",
                name=spec.name,
                path=str(matches[0]),
            )
        else:
            status = EndpointStatus.MISSING if spec.required else EndpointStatus.WARN
            results[spec.name] = EndpointResult(
                name=spec.name,
                glob=spec.by_id_glob,
                required=spec.required,
                resolved_path=None,
                status=status,
            )
            _log.warning(
                "usbc_endpoint_missing",
                name=spec.name,
                glob=spec.by_id_glob,
                status=status.value,
            )
    return results
