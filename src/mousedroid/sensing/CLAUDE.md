# Sensing Subsystem — Surface Contract

> Concurrent sensor reads fused into observation bundles with per-modality validity
> (``ObservationProtocol``; ``SensorManager``, ``MouseDroidObservationBundle``,
> ``HumanPresenceProtocol``).

## Invariants & Sensing Rules

1. **Protocol Surface**: Bundles expose ``ObservationProtocol`` fields (timestamp, vision,
   distance, motor_state, audio, lidar, valid_mask, n_modalities).
2. **Concurrent Fusion**: ``SensorManager`` reads vision, ultrasonic, ESP32 motor, audio
   (and LiDAR when present) concurrently; failed reads mark modalities invalid rather than
   raising into the control loop.
3. **Human Presence Seam**: ``HumanPresenceProtocol`` / ``NullHumanPresenceDetector`` supply
   an injectable presence sample for safety consumers.

## Key Files

- `protocol.py::ObservationProtocol` — common observation bundle interface.
- `bundle.py::MouseDroidObservationBundle` — concrete fused observation.
- `manager.py::SensorManager` — concurrent sensor orchestration and ring buffers.
- `human_presence.py::HumanPresenceProtocol` — injectable human-presence detector.
- `lidar_scan.py` — LiDAR scan helpers used by the manager.
- `tests/unit/sensing/` — subsystem unit tests.
