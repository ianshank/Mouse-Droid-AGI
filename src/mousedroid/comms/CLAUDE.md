# Comms Subsystem — Surface Contract

> ESP32 Wave Rover communication drivers over serial or WiFi
> (``ESP32CommProtocol``; ``SerialESP32Driver``, ``WiFiESP32Driver``, ``MockESP32Driver``).

## Invariants & Comms Rules

1. **Protocol Surface**: Drivers implement ``ESP32CommProtocol`` (``connect``,
   ``send_velocity``, ``read_encoders``, ``get_battery_voltage``, ``emergency_stop``,
   ``disconnect``). Factory selects the concrete driver from ``cfg.esp32.protocol``.
2. **Command Codecs**: Wire framing is behind ``ESP32CommandCodec`` in ``command_set.py``
   (stock Waveshare vs legacy encodings).
3. **Mock Path**: ``MockESP32Driver`` satisfies the same protocol for CI and mock-hardware
   runs — no alternate call sites.

## Key Files

- `protocol.py::ESP32CommProtocol` / `EncoderReading` — driver interface and encoder payload.
- `base_driver.py::BaseESP32Driver` — shared driver base.
- `serial_driver.py::SerialESP32Driver` / `wifi_driver.py::WiFiESP32Driver` — concrete transports.
- `mock_driver.py::MockESP32Driver` — protocol-compatible mock.
- `command_set.py` — command codecs and heartbeat framing helpers.
- `tests/unit/comms/` — subsystem unit tests.
