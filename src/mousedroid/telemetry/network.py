"""Network interface discovery utilities.

Provides functions to enumerate active network interfaces, resolve IP
addresses, and classify interface types (WiFi, Ethernet, loopback).
Used by the telemetry server for the ``/network`` endpoint and mDNS
registration.

All blocking I/O is wrapped with ``asyncio.to_thread`` for async safety.
"""

from __future__ import annotations

import asyncio
import socket
from dataclasses import asdict, dataclass
from typing import Any

from mousedroid.constants import (
    CONNECTIVITY_CHECK_HOST,
    CONNECTIVITY_CHECK_PORT,
    LOOPBACK_IP,
)
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

_WIFI_PREFIXES = ("wlan", "wlp", "wlx")
_ETHERNET_PREFIXES = ("eth", "enp", "ens", "eno")
_LOOPBACK_PREFIXES = ("lo",)


@dataclass(frozen=True)
class NetworkInterface:
    """Discovered network interface information.

    Attributes:
        name: Interface name (e.g. ``wlan0``, ``eth0``).
        ip: IPv4 address string, or empty if not assigned.
        interface_type: One of ``wifi``, ``ethernet``, ``loopback``, ``other``.
        up: Whether the interface is up and has an IP.
    """

    name: str
    ip: str
    interface_type: str
    up: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict.

        Returns:
            Dictionary representation.
        """
        return asdict(self)


def _classify_interface(name: str) -> str:
    """Classify a network interface by its name prefix.

    Args:
        name: Interface name.

    Returns:
        One of ``wifi``, ``ethernet``, ``loopback``, ``other``.
    """
    lower = name.lower()
    if any(lower.startswith(p) for p in _WIFI_PREFIXES):
        return "wifi"
    if any(lower.startswith(p) for p in _ETHERNET_PREFIXES):
        return "ethernet"
    if any(lower.startswith(p) for p in _LOOPBACK_PREFIXES):
        return "loopback"
    return "other"


def _get_interfaces_sync() -> list[NetworkInterface]:
    """Discover active network interfaces (blocking).

    Uses ``socket.getaddrinfo`` and ``socket.if_nameindex`` from stdlib
    to avoid external dependencies.

    Returns:
        List of ``NetworkInterface`` instances.
    """
    interfaces: list[NetworkInterface] = []

    try:
        if_names = socket.if_nameindex()
    except OSError:
        _log.debug("network_if_nameindex_failed")
        return interfaces

    for _idx, name in if_names:
        ip = ""
        up = False
        try:
            infos = socket.getaddrinfo(
                name,
                None,
                socket.AF_INET,
                socket.SOCK_DGRAM,
            )
            for info in infos:
                addr = info[4][0]
                if addr:
                    ip = str(addr)
                    up = True
                    break
        except (socket.gaierror, OSError):
            # Try binding approach as fallback
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    s.connect((CONNECTIVITY_CHECK_HOST, CONNECTIVITY_CHECK_PORT))
                    candidate_ip = s.getsockname()[0]
                    # This gives the default route IP, not per-interface
                    # Only use if this is likely the right interface
                    if not ip:
                        ip = candidate_ip
                        up = bool(ip)
                finally:
                    s.close()
            except OSError:
                pass

        interfaces.append(
            NetworkInterface(
                name=name,
                ip=ip,
                interface_type=_classify_interface(name),
                up=up,
            )
        )

    return interfaces


async def get_network_interfaces() -> list[NetworkInterface]:
    """Discover active network interfaces (async).

    Returns:
        List of ``NetworkInterface`` instances.
    """
    return await asyncio.to_thread(_get_interfaces_sync)


def _get_interface_ip_sync(name: str) -> str:
    """Get the IPv4 address of a specific interface (blocking).

    Args:
        name: Interface name (e.g. ``wlan0``).

    Returns:
        IPv4 address string, or empty string if not found.
    """
    try:
        infos = socket.getaddrinfo(name, None, socket.AF_INET, socket.SOCK_DGRAM)
        for info in infos:
            addr = info[4][0]
            if addr:
                return str(addr)
    except (socket.gaierror, OSError):
        pass
    return ""


async def get_interface_ip(name: str) -> str:
    """Get the IPv4 address of a specific interface (async).

    Args:
        name: Interface name (e.g. ``wlan0``).

    Returns:
        IPv4 address string, or empty string if not found.
    """
    return await asyncio.to_thread(_get_interface_ip_sync, name)


def get_default_ip() -> str:
    """Get the default outbound IP address.

    Creates a UDP socket to determine which interface would be used
    for outbound traffic.

    Returns:
        Default IP address string, or ``127.0.0.1`` if unavailable.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((CONNECTIVITY_CHECK_HOST, CONNECTIVITY_CHECK_PORT))
            return str(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        return LOOPBACK_IP
