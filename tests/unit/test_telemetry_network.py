"""Tests for network interface discovery utilities."""

from __future__ import annotations

from unittest.mock import patch

from mousedroid.telemetry.network import (
    NetworkInterface,
    _classify_interface,
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
    assert ip == "127.0.0.1"


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
