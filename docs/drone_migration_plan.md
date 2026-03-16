# Software Migration Plan: 4WD Mouse Droid to Drone

## Executive Summary

This document identifies the software modifications required to migrate the MouseDroidAGI system from the current Wave Rover mecanum-wheel chassis to a drone (quadcopter/hexacopter) platform. The existing architecture — protocol-based dependency injection, async orchestrator, RSSM world model, and configurable YAML schemas — is remarkably well-suited for this transition. Most high-level intelligence (cognition, memory, learning, safety) can be reused directly. The changes concentrate in **hardware abstraction**, **physics modelling**, **control loops**, and **safety enforcement**.

---

## 1. Motor Control & Flight Controller Integration

### Current State
- `ESP32CommProtocol` sends 3-DOF velocity commands `(vx, vy, omega)` to mecanum wheels via JSON-over-serial or WiFi HTTP.
- `SerialESP32Driver` / `WiFiESP32Driver` implement this protocol.

### Required Changes

| Area | Modification |
|------|-------------|
| **Flight controller comm** | Replace ESP32 drivers with a **MAVLink** or **DJI OSDK** driver implementing a new `FlightControllerProtocol`. Libraries: `pymavlink`, `mavsdk-python` (async-native), or DJI's SDK. |
| **Command space** | Expand from 3-DOF `(vx, vy, omega)` to **6-DOF**: `(vx, vy, vz, roll_rate, pitch_rate, yaw_rate)` or use MAVLink `SET_POSITION_TARGET_LOCAL_NED` for position-mode control. A thrust + attitude quaternion interface is also common. |
| **Action vector** | Increase action dimensionality from 3 to at minimum 4 `(vx, vy, vz, yaw_rate)` — this ripples into the RSSM action encoder, MCTS candidate generation, policy MLP output layer, and reward model inputs. |
| **Arming/disarming** | Drones require explicit arm/disarm sequences, pre-flight checks, and flight-mode management (STABILIZE, GUIDED, LOITER, LAND, RTL). The orchestrator needs a **flight state machine** layer. |
| **Failsafe handoff** | The flight controller (Pixhawk/Betaflight) runs its own PID loops at 400+ Hz for attitude stabilization. The Jetson sends *high-level* velocity or position setpoints, not raw motor PWM. This is architecturally simpler than the current direct-motor approach. |
| **Keepalive** | MAVLink heartbeat replaces the current 10 Hz keepalive. Loss of heartbeat triggers RTL (Return to Launch) on the flight controller side. |

### New Protocol Interface (Conceptual)

```python
@runtime_checkable
class FlightControllerProtocol(Protocol):
    async def connect(self) -> None: ...
    async def arm(self) -> None: ...
    async def disarm(self) -> None: ...
    async def takeoff(self, altitude_m: float) -> None: ...
    async def land(self) -> None: ...
    async def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None: ...
    async def send_position(self, x: float, y: float, z: float, yaw: float) -> None: ...
    async def get_telemetry(self) -> DroneTelemetry: ...
    async def set_flight_mode(self, mode: FlightMode) -> None: ...
    async def emergency_stop(self) -> None: ...  # Kill motors / forced land
    async def return_to_launch(self) -> None: ...
```

---

## 2. Sensor Suite Expansion

### Current Sensors
- **Vision**: Jetson CSI / RPi AI Camera (IMX500) — 640x480 @ 30 Hz
- **Distance**: Single HC-SR04 ultrasonic (forward-facing, 2cm–4m)
- **Motor encoders**: Wheel velocities + odometry
- **Audio**: Optional USB microphone

### Required Changes

| Sensor | Modification |
|--------|-------------|
| **IMU** | Drones require high-rate IMU data (accelerometer + gyroscope). The flight controller provides this via MAVLink `ATTITUDE`, `RAW_IMU`, and `LOCAL_POSITION_NED` messages. Add an `ImuProtocol` and parse MAVLink telemetry. |
| **Barometer / Altimeter** | Altitude estimation is critical. The flight controller fuses barometer + IMU internally, but expose altitude in telemetry. Optionally add a downward-facing **TF-Luna LiDAR** (0.1–12m) for precise low-altitude hold. |
| **GPS** | Outdoor drones need GPS. The flight controller handles GPS fusion (EKF), but GPS coordinates should flow into the world model for mapping. Add `GpsProtocol` with lat/lon/alt/fix_type. |
| **Downward-facing camera or optical flow** | Replace wheel odometry (which no longer exists) with visual odometry. A downward-facing camera + PX4Flow or software optical flow provides ground-relative velocity. |
| **Ultrasonic** | Keep for obstacle avoidance, but add multi-directional coverage (down for altitude, forward/left/right for collision avoidance) or replace with a **360-degree LiDAR** (e.g., RPLiDAR A1). |
| **Battery monitoring** | Drone battery monitoring is higher-stakes (crash vs. stop). Pull voltage/current/remaining% from MAVLink `BATTERY_STATUS`. Integrate aggressive RTL thresholds. |
| **Magnetometer** | Provided by flight controller. Important for heading estimation, especially when GPS is available. |

### Sensor Manager Updates
- `ObservationBundle` needs new fields: `imu`, `altitude`, `gps`, `optical_flow`.
- Ring buffers need new modalities.
- Validity mask expands to cover new sensors.
- The multimodal encoder in the RSSM needs new input branches for these modalities.

---

## 3. State Estimation & Localization

### Current State
- 2D odometry from wheel encoders.
- Single ultrasonic distance reading.
- Vision features for world model latent state.

### Required Changes

| Area | Modification |
|------|-------------|
| **3D state estimation** | Move from 2D `(x, y, theta)` to full 6-DOF pose `(x, y, z, roll, pitch, yaw)`. The flight controller's EKF provides this, but the world model's latent state should also capture 3D structure. |
| **Coordinate frames** | Introduce proper frame management: `body`, `local_NED`, `global_LLA`. Use a library like `transforms3d` or implement quaternion utilities. |
| **RSSM observation space** | The multimodal encoder input grows significantly. Vision (256-dim) stays, but motor state (4-dim) becomes drone telemetry (~12-dim: vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, alt, battery, gps_fix). |
| **Map representation** | Consider adding an occupancy grid or elevation map for 3D navigation, especially for indoor flight. |

---

## 4. World Model (RSSM) Modifications

### Current Architecture
- Encoder: vision (256→128) + ultrasonic (1→32) + motor (4→32) = 192-dim fused
- Action: 3-dim
- Hidden state: 256-dim
- Latent state: 64-dim

### Required Changes

| Component | Modification |
|-----------|-------------|
| **Multimodal encoder** | Add branches for IMU (9→32), altitude (1→16), GPS (3→16), optical flow (2→16). Fused dimension increases from 192 to ~300+. |
| **Action encoder** | Input grows from 3-dim to 4-6 dim (adding vz, possibly roll/pitch rates). |
| **Hidden state** | May need to increase from 256 to 512 to capture richer 3D dynamics. |
| **Physics prior** | The RSSM's learned dynamics now need to capture aerial physics: gravity, drag, wind disturbances, momentum in 3D. The training data distribution changes dramatically. |
| **Imagination rollouts** | MCTS imagination must be physically plausible in 3D. A drone hovering at 10m that "imagines" flying forward must also imagine altitude maintenance. Gravity is always acting. |
| **Retraining** | The RSSM and all downstream heads (reward, decoder) need retraining on aerial data. Existing weights are not transferable for dynamics prediction. |

---

## 5. MCTS Planner Modifications

### Current Configuration
- 9 candidate actions per node (3-DOF discretization)
- 5-step rollouts
- 50–200 simulations

### Required Changes

| Area | Modification |
|------|-------------|
| **Action candidates** | Expand to 4-6 DOF. With 3 discrete values per dimension and 4 dimensions, candidates grow from 9 to 81. Need smarter sampling: progressive widening, CEM (Cross-Entropy Method), or learned action priors. |
| **Rollout horizon** | Aerial dynamics are faster — a 5-step rollout at 30 Hz covers only 167ms. May need longer horizons (10-15 steps) for meaningful planning, especially for altitude changes. |
| **Gravity constraint** | Candidate actions must always include sufficient thrust to maintain altitude. The planner should bias toward hover-maintaining actions as a prior. |
| **3D collision checking** | During imagination, check clearance in all directions (floor, ceiling, walls), not just forward distance. |
| **Compute budget** | More dimensions + longer horizons = more compute. May need to increase MCTS budget or switch to a more efficient planner (MPPI — Model Predictive Path Integral). |

---

## 6. Safety Monitor — Critical Overhaul

### Current Safety Checks
1. Forward clearance ≥ 0.2m
2. Battery ≥ 9.5V (emergency) / 10.5V (warning)
3. Sensor staleness < 0.5s
4. Loop timing < 200ms
5. GPU temperature ≤ 90°C
6. ≥ 2 valid sensor modalities

### Required Changes — Drone Safety Is Life-Critical

| Check | Modification |
|-------|-------------|
| **Multi-directional clearance** | Check distances in all 6 directions (forward, back, left, right, up, down). Minimum ground clearance is now critical. |
| **Altitude bounds** | Add hard min/max altitude limits. Geofencing (lat/lon/alt bounding box). |
| **Battery management** | Much more aggressive. Compute remaining flight time based on current draw. Trigger RTL with enough margin to reach home. Voltage sag under load must be accounted for. |
| **Flight controller health** | Monitor FC heartbeat, EKF status, GPS fix quality, vibration levels, motor health (via ESC telemetry). |
| **Geofencing** | Hard geographic boundaries. If GPS fix is lost, hold position or land immediately. |
| **Wind estimation** | Monitor groundspeed vs. airspeed discrepancy. If wind exceeds safe threshold, land. |
| **Propeller/motor failure** | Detect via current draw asymmetry or IMU vibration analysis. Emergency land on detection. |
| **Regulatory compliance** | Max altitude (120m/400ft in most jurisdictions), no-fly zones, visual line of sight constraints. |
| **Emergency actions** | Replace "stop" with graduated responses: hold position → RTL → forced landing → motor kill (last resort only). A ground robot stops safely; a drone that stops its motors crashes. |

### Three Laws Updates

| Law | Drone-Specific Change |
|-----|----------------------|
| **Law 1 (No harm)** | Human detection must work in 3D. Downwash from propellers can injure at close range. Safety radius increases significantly (2-5m vs current 0.5m). Falling debris risk if motors cut. |
| **Law 2 (Obey commands)** | Add regulatory authority as implicit "commands" — no-fly zones, altitude limits. |
| **Law 3 (Self-preservation)** | Crashes are catastrophic, not just inconvenient. Battery preservation directly prevents crashes. Thermal management affects flight stability. |

---

## 7. Cognitive Core & BDI Updates

### Current State
- 10 intention classes
- Dual-cadence: fast (30 Hz) + slow (~1 Hz)
- Policy MLP outputs 3-dim action

### Required Changes

| Component | Modification |
|-----------|-------------|
| **Intention classes** | Add aerial intentions: `hover`, `ascend`, `descend`, `orbit`, `survey`, `return_to_launch`, `land`, `avoid_3d`. Remove ground-specific ones like `wall_follow`. |
| **Policy output** | Expand to 4-6 dim action space. |
| **Belief encoder** | Must incorporate altitude awareness, 3D spatial understanding. |
| **Affect model** | "Anxiety" (arousal) should increase at low battery, high altitude, or strong wind. |
| **Fast path** | The constitutional checker must evaluate 3D safety constraints within <1ms. |
| **Slow path BDI** | Desire generation should factor in mission objectives (survey area, patrol path) rather than just reactive navigation. |

---

## 8. Configuration Schema Changes

### Current Key Constants
- Wheelbase: 0.20m, Track width: 0.20m, Wheel radius: 0.042m
- Max speed: 0.5 m/s, Max angular velocity: 2.0 rad/s

### Required Changes

```yaml
# New drone-specific configuration
drone:
  frame_type: quadcopter  # quadcopter | hexacopter | octocopter
  mass_kg: 1.5
  max_thrust_N: 30.0
  arm_length_m: 0.25
  prop_diameter_m: 0.254  # 10-inch props

flight_envelope:
  max_speed_ms: 5.0       # horizontal
  max_vertical_speed_ms: 2.0
  max_altitude_m: 30.0
  min_altitude_m: 1.0
  max_tilt_deg: 35.0
  geofence_radius_m: 100.0
  geofence_center_lat: 0.0
  geofence_center_lon: 0.0

flight_controller:
  protocol: mavlink       # mavlink | dji_osdk
  connection: serial      # serial | udp
  serial_port: /dev/ttyTHS1
  baud_rate: 921600
  udp_port: 14540
  heartbeat_hz: 1
  telemetry_hz: 50

battery:
  cell_count: 4           # 4S LiPo
  capacity_mah: 5000
  voltage_full: 16.8
  voltage_nominal: 14.8
  voltage_warning: 14.0
  voltage_critical: 13.2   # trigger RTL
  voltage_emergency: 12.8  # trigger forced land

safety:
  human_safety_radius_m: 3.0     # increased from 0.5m
  emergency_stop_distance_m: 1.0  # increased from 0.15m
  min_ground_clearance_m: 0.5
  max_wind_speed_ms: 8.0
  rtl_battery_margin_percent: 20
```

### Remove
- All mecanum wheel parameters (wheelbase, track width, wheel radius)
- Encoder-related configuration
- 2D-only navigation parameters

---

## 9. Orchestrator Loop Changes

### Current Flow
```
Sense → Fuse → Safety Check → RSSM Update → Plan (MCTS/Cognitive) → Act (ESP32 velocity)
```

### New Flow
```
Sense → Fuse → Safety Check → Flight State Machine → RSSM Update → Plan → Act (FC setpoint)
         ↑                           ↑
    [+ IMU, GPS,               [DISARMED → ARMED →
     altitude, flow]            TAKEOFF → FLYING →
                                LANDING → LANDED →
                                RTL → EMERGENCY]
```

| Change | Detail |
|--------|--------|
| **Flight state machine** | New top-level state machine governing what the planner is allowed to do. During takeoff, only altitude commands. During landing, only descent. During emergency, only the FC acts. |
| **Pre-flight checklist** | Before arming: GPS fix quality, battery level, sensor health, IMU calibration, compass calibration, geofence loaded. |
| **Telemetry ingestion** | The FC sends telemetry at 50-100 Hz. The orchestrator must ingest and fuse this alongside vision and other sensors. |
| **Watchdog** | If the Jetson loop stalls (>500ms), the FC must autonomously hold position or RTL. This is handled FC-side but must be configured. |

---

## 10. Training Pipeline Changes

### Required Retraining
1. **RSSM**: Entirely retrained on aerial dynamics data (can start with simulation: AirSim, Gazebo, jMAVSim).
2. **Policy/MCTS**: Action space changes require full retraining.
3. **BDI model**: New intention classes, new observation features.
4. **Reward model**: New objectives (altitude maintenance, smooth flight, energy efficiency).
5. **ICM (Curiosity)**: Forward/inverse models need retraining for new dynamics.

### Simulation-First Approach
- Train in simulation before real flight (AirSim, Gazebo + PX4 SITL).
- Use domain randomization (wind, sensor noise, latency) for sim-to-real transfer.
- The `MockESP32Driver` pattern extends naturally to a `SimulatorDriver` that wraps a physics engine.

---

## 11. What Can Be Reused As-Is

The following components require **zero or minimal changes**:

| Component | Reason |
|-----------|--------|
| **Memory systems** (episodic, semantic, working) | Domain-agnostic storage and retrieval |
| **EWC / Progressive networks** | Continual learning is task-agnostic |
| **MAML meta-learning** | Algorithm is domain-agnostic |
| **Knowledge distillation** | Teacher-student transfer is model-agnostic |
| **Sparse MoE routing** | Architecture pattern is domain-agnostic |
| **Circuit breaker / retry logic** | Communication resilience is protocol-agnostic |
| **Structured logging** | Logging framework is domain-agnostic |
| **Configuration system** | Pydantic + YAML is extensible by design |
| **Protocol-based DI / factory** | Just register new implementations |
| **Test infrastructure** | Pytest + mock pattern extends to new drivers |
| **Docker / deployment** | Same Jetson hardware, same container approach |
| **TensorRT compilation** | Same optimization pipeline, new models |
| **Vision pipeline** | Camera and feature extraction are reusable (add downward camera as second instance) |
| **Three Laws framework** | Logic structure stays, thresholds change |

---

## 12. Migration Priority Order

### Phase 1: Foundation (Minimum Viable Flight)
1. Implement `FlightControllerProtocol` + MAVLink driver
2. Add flight state machine to orchestrator
3. Update safety monitor with altitude/battery/geofence checks
4. Expand action space to 4-DOF
5. Create `config/drone.yaml` with flight parameters
6. Test with PX4 SITL (Software-In-The-Loop) simulator

### Phase 2: Perception
7. Add IMU/GPS/altitude telemetry parsing
8. Implement downward-facing optical flow
9. Update `ObservationBundle` and sensor manager
10. Add multi-directional distance sensing

### Phase 3: Intelligence
11. Retrain RSSM on simulated aerial data
12. Update MCTS for 4-DOF action space + gravity prior
13. Retrain BDI with aerial intention classes
14. Update reward model for flight objectives

### Phase 4: Safety Hardening
15. Implement geofencing with GPS
16. Add wind estimation and motor health monitoring
17. Implement graduated emergency responses (hold → RTL → land → kill)
18. Regulatory compliance checks (altitude limits, no-fly zones)
19. Extensive simulation testing + careful real-world validation

---

## 13. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Crash on software fault | **Critical** | Flight controller runs independent stabilization. Jetson failure = FC holds position or RTL. |
| Battery exhaustion mid-flight | **Critical** | Conservative RTL margins (20%+ remaining). Continuous flight-time estimation. |
| GPS loss | **High** | Optical flow + barometer for position hold without GPS. Auto-land if prolonged. |
| Latency spike in Jetson loop | **High** | FC-side watchdog. If no setpoint received in 500ms, FC holds autonomously. |
| RSSM predicts physically impossible states | **Medium** | Clamp outputs to flight envelope. FC rejects commands outside safe limits. |
| Regulatory violation | **Medium** | Hard-coded geofence + altitude limits that cannot be overridden by learned policy. |

---

## Summary

The existing MouseDroidAGI architecture is **exceptionally well-prepared** for this migration due to:
- **Protocol-based abstraction**: Swap `ESP32CommProtocol` for `FlightControllerProtocol`
- **Configurable everything**: YAML schemas extend naturally to flight parameters
- **Factory pattern**: New hardware drivers register without touching core logic
- **Domain-agnostic intelligence**: Memory, learning, distillation, and MoE are reusable

The **hardest parts** are:
1. **Safety** — a drone that fails is dangerous, not just inconvenient
2. **3D state estimation** — fundamentally more complex than 2D ground navigation
3. **RSSM retraining** — aerial dynamics are entirely different from ground dynamics
4. **MCTS in higher dimensions** — combinatorial explosion in action space

Estimated scope: ~40% of the codebase needs modification, ~30% is new code, ~30% is reused unchanged.
