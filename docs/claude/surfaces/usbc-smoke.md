# USB-C Device Discovery & Smoke Gate Surface

> Dynamic endpoint discovery and enumeration protocol for rover hardware peripherals.

## Adding a New USB-C Endpoint

When adding a new hardware peripheral (e.g. secondary LiDAR, IMU bridge):

1. **Config Spec Registration**: Add a `USBCEndpointSpec(name="...", by_id_glob="...", required=...)`
   entry to `usbc_discovery.required_endpoints` in `config/jetson_production.yaml`.
2. **Driver Resolution Helper**: If driver serial port overrides are required, add a resolution
   helper in `src/mousedroid/factory.py` (e.g. `_resolve_esp32_serial_via_usbc_discovery`).
3. **Unit Status Transitions**: Add unit tests in `tests/unit/diagnostics/test_usbc.py` covering
   `PRESENT`, `MISSING`, and `WARN` status transitions.
4. **Hardware Enumeration Test**: Add a hardware test under `tests/hardware/test_usbc_enumeration.py`
   (gated by `tests._jetson_hardware.is_jetson_host`).
5. **Config Overlay Assertion**: Add a regression check in `tests/unit/test_jetson_production_overlay.py`
   to ensure CI `usbc-config-gate` catches schema-to-driver drift.
