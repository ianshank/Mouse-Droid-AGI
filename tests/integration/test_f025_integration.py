"""Multi-module wiring for the F-025 command-set seam.

The unit tier pins each piece in isolation and the regression tier pins the
config invariants; this tier exercises the wiring *through the factory* — the
path CLAUDE.md's test-tier mirror reserves for integration.

It exists because the stock command set had zero non-hardware coverage below
the driver: ``config/jetson_production.yaml`` ships ``esp32.enabled: false``,
so a factory build returns ``MockESP32Driver``, which implements the protocol
directly and never touches a codec. Every "does stock actually work end to
end" question was therefore answered only by tests that stubbed the transport.
Here the real ``SerialESP32Driver`` is built by the factory and driven over a
fake serial port, so the codec, the driver, the resilience wrapper and the
config validator all participate.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import pytest

from mousedroid.config.schema import Settings

if TYPE_CHECKING:
    from mousedroid.comms.serial_driver import SerialESP32Driver

_ULTRASONIC_CFG = {"trigger_pin": 17, "echo_pin": 27}

#: Held inside the fake port's ``readline`` so two gathered reads genuinely
#: overlap when unserialised. Large enough to dwarf event-loop scheduling
#: jitter, small enough to keep the integration tier fast.
_READ_OVERLAP_DELAY_S = 0.05


class FakeSerialPort:
    """Minimal in-memory stand-in for a ``pyserial`` handle.

    Records every written line and serves queued responses, so a test can
    assert on the exact bytes the driver put on the wire — the level at
    which a firmware-protocol mismatch actually manifests.
    """

    def __init__(self, responses: list[str] | None = None, *, read_delay_s: float = 0.0) -> None:
        self.written: list[str] = []
        self._responses: list[str] = list(responses or [])
        self.closed = False
        self.timeout: float = 0.0
        # Concurrency witness: pyserial handles are not thread-safe, so two
        # overlapping readline() calls are the defect the driver's _io_lock
        # exists to prevent.
        #
        # ``read_delay_s`` is what makes the witness meaningful. Without it
        # the fake returns so fast that each gathered task finishes its
        # to_thread hop before the next one starts, so the overlap never
        # occurs and the assertion below holds *whether or not the lock is
        # there* — a test that cannot fail. A delay wider than the scheduling
        # gap forces a genuine window, and a real port's read latency is
        # exactly what opens that window in production.
        self._read_delay_s = read_delay_s
        self.in_readline = 0
        self.max_concurrent_readline = 0

    def write(self, payload: bytes) -> None:
        self.written.append(payload.decode().strip())

    def readline(self) -> bytes:
        self.in_readline += 1
        self.max_concurrent_readline = max(self.max_concurrent_readline, self.in_readline)
        try:
            if self._read_delay_s:
                time.sleep(self._read_delay_s)
            if not self._responses:
                return b""
            return (self._responses.pop(0) + "\n").encode()
        finally:
            self.in_readline -= 1

    def close(self) -> None:
        self.closed = True

    @property
    def written_commands(self) -> list[dict[str, Any]]:
        """Written lines parsed back into command dicts."""
        return [json.loads(line) for line in self.written]


def _settings(**esp32_overrides: Any) -> Settings:
    """Real-hardware settings so the factory builds a serial driver."""
    return Settings.model_validate(
        {
            "mock_hardware": False,
            "ultrasonic": _ULTRASONIC_CFG,
            "esp32": {"protocol": "serial", "enabled": True, **esp32_overrides},
        }
    )


def _build_with_fake_port(
    monkeypatch: pytest.MonkeyPatch,
    cfg: Settings,
    responses: list[str] | None = None,
    *,
    read_delay_s: float = 0.0,
) -> tuple[Any, FakeSerialPort]:
    """Build the real driver through the factory over a fake serial port.

    ``_serial_mod`` is patched by ``setattr`` on the specific symbol rather
    than via ``patch.dict(sys.modules) + reload`` — the reload form evicts
    real modules from the import cache and poisons later tests in the same
    process (PR #112).
    """
    from mousedroid.comms import serial_driver as serial_driver_mod
    from mousedroid.comms.serial_driver import SerialESP32Driver
    from mousedroid.factory import build_esp32_driver

    port = FakeSerialPort(responses, read_delay_s=read_delay_s)
    # The driver refuses to connect when pyserial is absent; the fake port
    # supplies the behaviour under test, so only the presence check matters.
    monkeypatch.setattr(serial_driver_mod, "_serial_mod", object(), raising=False)
    monkeypatch.setattr(SerialESP32Driver, "_open_serial", lambda _self: port)
    driver = build_esp32_driver(cfg)
    return driver, port


class TestStockCommandSetThroughFactory:
    """A stock-configured rover puts stock bytes on the wire."""

    async def test_connect_arms_heartbeat_then_velocity_is_ros_ctrl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _settings(command_set="waveshare_stock")
        driver, port = _build_with_fake_port(monkeypatch, cfg)

        await driver.connect()
        await driver.send_velocity(0.25, 0.0, 1.0)
        await driver.emergency_stop()

        commands = port.written_commands
        # Heartbeat armed first, then physical-unit velocity, then the
        # zero-velocity stop (stock defines no dedicated e-stop command).
        assert commands[0]["T"] == 136
        assert commands[0]["cmd"] == 3000
        assert commands[1] == {"T": 13, "X": 0.25, "Z": 1.0}
        assert commands[2] == {"T": 13, "X": 0.0, "Z": 0.0}

    async def test_stock_baud_reaches_the_opened_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The derived baud must survive config -> factory -> driver."""
        cfg = _settings(command_set="waveshare_stock")
        driver, _ = _build_with_fake_port(monkeypatch, cfg)
        inner: SerialESP32Driver = driver.inner
        assert inner._baud == 115200

    async def test_battery_read_polls_feedback_and_never_writes_motor_pid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The power-chain battery step must not poke the motor controller.

        Legacy ``{"T":2}`` is stock ``CMD_SET_MOTOR_PID`` — a write issued
        immediately before commanding motion.
        """
        cfg = _settings(command_set="waveshare_stock")
        frame = json.dumps({"T": 1001, "L": 0.1, "R": 0.1, "v": 11.8})
        driver, port = _build_with_fake_port(monkeypatch, cfg, responses=[frame])

        await driver.connect()
        voltage = await driver.get_battery_voltage()

        assert voltage == pytest.approx(11.8)
        polls = [c for c in port.written_commands if c.get("T") == 130]
        assert len(polls) == 1
        assert all(c.get("T") != 2 for c in port.written_commands)

    async def test_unreadable_frame_yields_zero_not_a_fabricated_low_voltage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wrong-typed frame reports 0.0 V, which safety screens as missing.

        Paired with ``SafetyConfig.battery_implausible_below_v`` this is what
        stops a comms fault from latching a permanent emergency stop.
        """
        cfg = _settings(command_set="waveshare_stock")
        driver, _ = _build_with_fake_port(
            monkeypatch, cfg, responses=[json.dumps({"T": 1003, "v": 11.8})]
        )

        await driver.connect()
        assert await driver.get_battery_voltage() == 0.0


class TestLegacyPathThroughFactoryUnchanged:
    """The default selector still emits the pre-F-025 wire bytes."""

    async def test_connect_writes_nothing_and_velocity_is_pwm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _settings()
        assert cfg.esp32.command_set == "legacy"
        driver, port = _build_with_fake_port(monkeypatch, cfg)

        await driver.connect()
        assert port.written == []  # no arming commands under legacy

        await driver.send_velocity(0.5, -0.5, 1.0)
        await driver.emergency_stop()
        # Golden wire bytes, PWM-scaled against the schema defaults
        # (max_velocity_mps 0.5 -> full scale; max_omega_rads 2.0 -> half).
        assert port.written == [
            '{"T": 1, "vx": 255, "vy": -255, "omega": 127}',
            '{"T": 0}',
        ]

    async def test_legacy_keeps_the_one_mbaud_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _settings()
        driver, _ = _build_with_fake_port(monkeypatch, cfg)
        assert driver.inner._baud == 1_000_000


class TestConnectRollback:
    """A failed arming sequence must not leak the opened port."""

    async def test_arm_failure_closes_port_and_clears_connected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without rollback, ResilientESP32Driver's retries leak one fd each."""
        cfg = _settings(command_set="waveshare_stock")
        driver, port = _build_with_fake_port(monkeypatch, cfg)
        inner: SerialESP32Driver = driver.inner

        def boom(_payload: bytes) -> None:
            raise OSError("write failed")

        monkeypatch.setattr(port, "write", boom)

        with pytest.raises(OSError, match="write failed"):
            await inner.connect()

        assert port.closed is True
        assert inner._connected is False
        assert inner._serial is None


class TestConcurrentReadsAreSerialised:
    """The send-then-read pair must be atomic across concurrent tasks."""

    async def test_gathered_reads_do_not_interleave(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``SensorManager`` gathers encoders + battery; replies must pair.

        Without the lock, two ``readline()`` calls run in different threads
        on one handle and a reply can pair with the wrong request.

        The frames are deliberately DISTINGUISHABLE. With identical frames
        every assertion below is satisfied even by a mis-paired reply, so the
        test would pass with the lock removed — coverage in name only.
        """
        import asyncio

        cfg = _settings(command_set="waveshare_stock")
        frames = [
            json.dumps({"T": 1001, "L": 0.4, "R": 0.4, "v": 12.0}),
            json.dumps({"T": 1001, "L": 0.9, "R": 0.9, "v": 7.5}),
        ]
        # The delay is load-bearing: it holds the first readline() open long
        # enough for the second gathered task to reach its own, which is the
        # only way an unlocked driver can be caught overlapping.
        driver, port = _build_with_fake_port(
            monkeypatch, cfg, responses=frames, read_delay_s=_READ_OVERLAP_DELAY_S
        )
        await driver.connect()
        port.written.clear()

        encoders, voltage = await asyncio.gather(
            driver.read_encoders(),
            driver.get_battery_voltage(),
        )

        # Both legs polled, both got a well-formed answer.
        assert [c["T"] for c in port.written_commands] == [130, 130]
        # No two readline() calls overlapped — the lock held each
        # send-then-read pair atomic across the gathered tasks.
        assert port.max_concurrent_readline == 1
        # Each leg consumed exactly one frame, in gather order: the encoder
        # read got the first, the battery read the second.
        assert encoders.left_velocity_mps == pytest.approx(0.4)
        assert voltage == pytest.approx(7.5)
