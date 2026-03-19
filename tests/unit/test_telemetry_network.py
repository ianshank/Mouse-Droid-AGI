"""Tests for network interface discovery utilities."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

from mousedroid.constants import LOOPBACK_IP
from mousedroid.telemetry.network import (
    NetworkInterface,
    _classify_interface,
    _get_interface_ip_sync,
    _get_interfaces_sync,
    get_default_ip,
)


def test_classify_wifi_interfaces():
    assert _classify_interface("wlan0") == "wifi"
    assert _classify_interface("wlp2s0") == "wifi"
    assert _classify_interface("wlx001122334455") == "wifi"


def test_classify_ethernet_interfaces():
    assert _classify_interface("eth0") == "ethernet"
    assert _classify_interface("enp0s3") == "ethernet"
    assert _classify_interface("ens33") == "ethernet"
    assert _classify_interface("eno1") == "ethernet"


def test_classify_loopback():
    assert _classify_interface("lo") == "loopback"


def test_classify_unknown():
    assert _classify_interface("docker0") == "other"
    assert _classify_interface("br-abc123") == "other"


def test_network_interface_dataclass():
    iface = NetworkInterface(
        name="wlan0",
        ip="192.168.1.42",
        interface_type="wifi",
        up=True,
    )
    assert iface.name == "wlan0"
    assert iface.ip == "192.168.1.42"
    assert iface.interface_type == "wifi"
    assert iface.up is True


def test_network_interface_to_dict():
    iface = NetworkInterface(name="eth0", ip="10.0.0.5", interface_type="ethernet", up=True)
    d = iface.to_dict()
    assert d["name"] == "eth0"
    assert d["ip"] == "10.0.0.5"
    assert d["interface_type"] == "ethernet"
    assert d["up"] is True


def test_get_default_ip_returns_string():
    ip = get_default_ip()
    assert isinstance(ip, str)
    assert len(ip) > 0


def test_get_default_ip_fallback_on_error():
    with patch("socket.socket") as mock_sock:
        mock_sock.side_effect = OSError("no network")
        ip = get_default_ip()
    assert ip == LOOPBACK_IP


async def test_get_network_interfaces_returns_list():
    from mousedroid.telemetry.network import get_network_interfaces

    interfaces = await get_network_interfaces()
    assert isinstance(interfaces, list)
    # Should at least have loopback in most environments
    for iface in interfaces:
        assert isinstance(iface, NetworkInterface)


async def test_get_interface_ip_returns_string():
    from mousedroid.telemetry.network import get_interface_ip

    ip = await get_interface_ip("lo")
    assert isinstance(ip, str)


# ---------------------------------------------------------------------------
# Coverage gap tests for _get_interfaces_sync fallback paths
# ---------------------------------------------------------------------------


def test_get_interfaces_sync_if_nameindex_oserror():
    """Cover line 84-86: if_nameindex raises OSError."""
    with patch("socket.if_nameindex", side_effect=OSError("no interfaces")):
        result = _get_interfaces_sync()
    assert result == []


def test_get_interfaces_sync_getaddrinfo_gaierror_with_socket_fallback():
    """Cover lines 98-103, 118: getaddrinfo fails, socket.connect fallback is used."""
    with patch("socket.if_nameindex", return_value=[(1, "eth0")]), patch(
        "socket.getaddrinfo",
        side_effect=socket.gaierror("lookup failed"),
    ):
        mock_sock_instance = MagicMock()
        mock_sock_instance.getsockname.return_value = ("10.0.0.1", 0)
        with patch("socket.socket", return_value=mock_sock_instance):
            result = _get_interfaces_sync()

    assert len(result) == 1
    assert result[0].ip == "10.0.0.1"
    assert result[0].up is True


def test_get_interfaces_sync_getaddrinfo_and_socket_both_fail():
    """Cover line 118: both getaddrinfo and socket.connect fail."""
    with patch("socket.if_nameindex", return_value=[(1, "eth0")]), patch(
        "socket.getaddrinfo",
        side_effect=socket.gaierror("lookup failed"),
    ):
        mock_sock_instance = MagicMock()
        mock_sock_instance.connect.side_effect = OSError("no route")
        with patch("socket.socket", return_value=mock_sock_instance):
            result = _get_interfaces_sync()

    assert len(result) == 1
    assert result[0].ip == ""
    assert result[0].up is False


def test_get_interface_ip_sync_gaierror_returns_empty():
    """Cover lines 153-156: _get_interface_ip_sync with gaierror."""
    with patch(
        "socket.getaddrinfo",
        side_effect=socket.gaierror("lookup failed"),
    ):
        result = _get_interface_ip_sync("nonexistent0")
    assert result == ""


def test_get_interface_ip_sync_oserror_returns_empty():
    """Cover lines 153-156: _get_interface_ip_sync with OSError."""
    with patch("socket.getaddrinfo", side_effect=OSError("no such interface")):
        result = _get_interface_ip_sync("bad0")
    assert result == ""


def test_get_interface_ip_sync_empty_addr():
    """Cover case where getaddrinfo returns entries with empty addresses."""
    fake_info = [(socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("", 0))]
    with patch("socket.getaddrinfo", return_value=fake_info):
        result = _get_interface_ip_sync("eth0")
    assert result == ""


def test_get_interfaces_sync_getaddrinfo_returns_empty_addr():
    """Cover line where addr is empty in the loop."""
    fake_info = [(socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("", 0))]
    with (
        patch("socket.if_nameindex", return_value=[(1, "eth0")]),
        patch("socket.getaddrinfo", return_value=fake_info),
    ):
        result = _get_interfaces_sync()

    assert len(result) == 1
    assert result[0].ip == ""
    assert result[0].up is False


def test_get_interfaces_sync_getaddrinfo_returns_valid_addr():
    """Cover lines 106-108: getaddrinfo returns a valid address."""
    fake_info = [(socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("192.168.1.10", 0))]
    with (
        patch("socket.if_nameindex", return_value=[(1, "wlan0")]),
        patch("socket.getaddrinfo", return_value=fake_info),
    ):
        result = _get_interfaces_sync()

    assert len(result) == 1
    assert result[0].ip == "192.168.1.10"
    assert result[0].up is True
    assert result[0].interface_type == "wifi"


def test_get_interface_ip_sync_returns_valid_addr():
    """Cover line 161: _get_interface_ip_sync returns a valid address."""
    fake_info = [(socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("10.0.0.5", 0))]
    with patch("socket.getaddrinfo", return_value=fake_info):
        result = _get_interface_ip_sync("eth0")
    assert result == "10.0.0.5"
