# MouseDroidAGI — C4 Architecture

This document uses the [C4 model](https://c4model.com/) (Context → Container → Component → Code) to describe the system architecture at four levels of abstraction.

---

## Level 1 — System Context

```
┌──────────────────────────────────────────────────────────────────────┐
│                        System Context                                │
└──────────────────────────────────────────────────────────────────────┘

        ┌─────────────────┐
        │   Human Operator │
        │  (natural lang   │
        │   commands,      │
        │   monitoring)    │
        └────────┬────────┘
                 │ NL commands / health dashboards
                 │
        ┌────────▼────────────────────────────┐
        │                                     │
        │         MouseDroidAGI System        │
        │                                     │
        │  An autonomous Star Wars MSE-6      │
        │  robot powered by an Agentic World  │
        │  Model running on Jetson Orin Nano  │
        │                                     │
        └──────┬──────────────────────┬───────┘
               │                      │
     ┌─────────▼──────┐    ┌──────────▼────────┐
     │  Physical World │    │  Remote Monitoring │
     │  (corridors,    │    │  (Prometheus /     │
     │   obstacles,    │    │   Grafana metrics  │
     │   people)       │    │   over WiFi)       │
     └─────────────────┘    └────────────────────┘
```

**Actors:**
- **Human Operator** — Sends natural language mission commands via LLM gateway; monitors telemetry
- **Physical World** — The environment the robot navigates through
- **Remote Monitoring** — Optional Prometheus/Grafana dashboard for production deployments

---

## Level 2 — Container Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                      MouseDroidAGI Containers                        │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────── NVIDIA Jetson Orin Nano ──────────────────────────┐
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  mousedroid Python process                  │    │
│  │                   (asyncio event loop)                      │    │
│  │                                                             │    │
│  │  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐  │    │
│  │  │  Orchestrator │  │  LLM Gateway │  │  Health Monitor  │  │    │
│  │  │  (30 Hz loop) │  │  (Llama GGUF)│  │  (sysfs polling) │  │    │
│  │  └───────┬───────┘  └──────────────┘  └──────────────────┘  │    │
│  │          │                                                   │    │
│  │  ┌───────▼────────────────────────────────────────────────┐  │    │
│  │  │                  Core AI Pipeline                      │  │    │
│  │  │  World Model (RSSM) ─► MCTS ─► Navigation Agent       │  │    │
│  │  │  Cognitive Core (BDI + Metacognitive)                  │  │    │
│  │  │  Memory Systems (Working / Episodic / Semantic)        │  │    │
│  │  │  Safety Monitor ─► Constitutional Checker              │  │    │
│  │  └───────────────────────────────────────────────────────-┘  │    │
│  │                                                             │    │
│  │  ┌───────────────────┐  ┌──────────────────────────────┐   │    │
│  │  │  Sensor Manager   │  │  Experience Logger (LMDB)    │   │    │
│  │  │  (concurrent I/O) │  │  /home/jetson/experience_db  │   │    │
│  │  └───────────────────┘  └──────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │              Hardware Interface Layer                         │   │
│  │  Camera (IMX500)   Ultrasonic (HC-SR04)   GPIO (Jetson.GPIO)  │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ UART 1 Mbps / HTTP
                    ┌──────────────▼──────────────┐
                    │   ESP32 (Wave Rover)         │
                    │   Motor control              │
                    │   Encoder feedback           │
                    │   Battery ADC                │
                    └─────────────────────────────┘
```

**Containers:**
| Container | Technology | Responsibility |
|-----------|-----------|----------------|
| `mousedroid` process | Python 3.11 asyncio | All AI reasoning + I/O orchestration |
| LMDB experience store | LMDB on-disk | Persistent experience replay buffer |
| Llama GGUF model | llama-cpp-python | Local LLM for NL → velocity |
| ESP32 firmware | C++ (Wave Rover SDK) | Motor PWM control, encoder polling |
| IMX500 camera | picamera2 | Vision capture + onboard neural inference |

---

## Level 3 — Component Diagram (mousedroid process)

```
┌──────────────────────────────────────────────────────────────────────┐
│                   mousedroid Python Process                          │
│                   (asyncio event loop, 30 Hz)                        │
└──────────────────────────────────────────────────────────────────────┘

                         ┌─────────────┐
                         │  CLI Entry  │
                         │  main.py    │
                         └──────┬──────┘
                                │ argparse + load_settings()
                         ┌──────▼──────┐
                         │   Factory   │
                         │  factory.py │  ◄── Only file that imports
                         └──────┬──────┘      concrete types
                                │ injects all components
                         ┌──────▼──────────────────────┐
                         │       Orchestrator           │
                         │   orchestrator/              │
                         │   • tick() at 30 Hz          │
                         │   • sense → plan → act       │
                         └──────┬──────────────┬────────┘
                                │              │
                   ┌────────────▼──┐    ┌──────▼─────────────┐
                   │ Sensor Manager │    │  Safety Monitor    │
                   │ sensing/       │    │  safety/           │
                   │ • Camera       │    │  • Clearance check │
                   │ • Ultrasonic   │    │  • Battery check   │
                   │ • ESP32 enc.   │    │  • Sensor staleness│
                   └────────┬──────┘    └──────┬─────────────┘
                            │                  │
                   ┌────────▼──────────────────▼──────────────┐
                   │           World Model                     │
                   │           world_model/                    │
                   │                                           │
                   │  ┌────────────┐    ┌────────────────┐    │
                   │  │  Encoder   │    │      RSSM      │    │
                   │  │  (vision + │───►│  (latent h,z)  │    │
                   │  │  motor +   │    │  observe_step  │    │
                   │  │  ultrasonic│    │  imagine_step  │    │
                   │  └────────────┘    └───────┬────────┘    │
                   │                            │             │
                   │                   ┌────────▼───────┐     │
                   │                   │      MCTS      │     │
                   │                   │  (50-200 sims) │     │
                   │                   └────────────────┘     │
                   └─────────────────────────────────────────-┘
                                        │
                              ┌─────────▼──────────┐
                              │  Navigation Agent  │
                              │  agents/           │
                              │  act(h, z, safety) │
                              └─────────┬──────────┘
                                        │ action tensor
                              ┌─────────▼──────────┐
                              │   ESP32 Driver     │
                              │   comms/           │
                              │   send_velocity()  │
                              └────────────────────┘

   PARALLEL SUBSYSTEMS (run alongside main loop):

   ┌──────────────────────────────────────────────────────────┐
   │  Cognitive Core (background async task)                  │
   │  cognitive/                                              │
   │                                                          │
   │  Fast tick (30 Hz):  PolicyMLP → ConstitutionalChecker   │
   │  Slow loop (~1 Hz):  BDI inference → Metacog update      │
   └──────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────┐
   │  Memory Systems                                          │
   │  memory/                                                 │
   │                                                          │
   │  Working Memory    — sliding context window (8192 tok)   │
   │  Episodic Memory   — FAISS similarity index (50k cap)    │
   │  Semantic Memory   — concept graph + FAISS search        │
   │  Consolidation     — async episodic → semantic pipeline  │
   └──────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────┐
   │  Continual Learning                                      │
   │  learning/                                               │
   │                                                          │
   │  EWC   — Fisher-information regularisation               │
   │  PNN   — progressive nets with lateral connections       │
   └──────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────┐
   │  LLM Gateway (optional)                                  │
   │  llm_gateway/                                            │
   │                                                          │
   │  NL command → GoalVector via local Llama GGUF            │
   │  Runs in asyncio.to_thread to avoid blocking             │
   └──────────────────────────────────────────────────────────┘
```

---

## Level 4 — Code: Dependency Injection Pattern

Every interface is a `@runtime_checkable Protocol`. Factory functions are the only place that branch on platform:

```python
# src/mousedroid/comms/protocol.py
@runtime_checkable
class ESP32CommProtocol(Protocol):
    async def connect(self) -> None: ...
    async def send_velocity(self, vx: float, vy: float, omega: float) -> None: ...
    async def read_encoders(self) -> EncoderReading: ...
    async def get_battery_voltage(self) -> float: ...
    async def emergency_stop(self) -> None: ...
    async def disconnect(self) -> None: ...

# src/mousedroid/factory.py
def build_esp32_driver(cfg: Settings) -> ESP32CommProtocol:
    if cfg.mock_hardware:
        from mousedroid.comms.mock_driver import MockESP32Driver
        return MockESP32Driver(cfg.esp32)
    if cfg.esp32.protocol == "serial":
        from mousedroid.comms.serial_driver import SerialESP32Driver
        return SerialESP32Driver(cfg.esp32)
    from mousedroid.comms.wifi_driver import WiFiESP32Driver
    return WiFiESP32Driver(cfg.esp32)
```

This means:
- **Tests** inject `MockESP32Driver` via `Settings(mock_hardware=True)` — no GPIO needed
- **CI** runs the full test suite with zero hardware
- **Production** seamlessly switches to `SerialESP32Driver` via config

---

## Data Flows

### Sense-Plan-Act (30 Hz)

```
Camera.capture_features()  ┐
HC-SR04.read_distance_m()  ├──► SensorManager.read_all()
ESP32.read_encoders()      │         │
ESP32.get_battery_voltage()┘         │
                                     ▼
                             ObservationBundle
                                     │
                              ┌──────▼───────┐
                              │  SafetyMonitor│
                              │  .evaluate()  │
                              └──────┬────────┘
                                     │ SafetyContext
                              ┌──────▼────────┐
                              │  WorldModel   │
                              │  .observe_step│
                              └──────┬────────┘
                                     │ (h, z, surprise)
                              ┌──────▼────────┐
                              │  Agent.act()  │
                              └──────┬────────┘
                                     │ action tensor
                              ┌──────▼────────┐
                              │ ESP32         │
                              │ .send_velocity│
                              └───────────────┘
```

### Experience Pipeline (async)

```
ObservationBundle ──► ExperienceRecord ──► ExperienceLogger (LMDB)
                                                  │
                                    ┌─────────────▼──────────────┐
                                    │   Memory Consolidation     │
                                    │   Episodic ──► Semantic    │
                                    └────────────────────────────┘
```

### Learning Pipeline (offline/async)

```
LMDB replay buffer ──► EWC regularisation ──► model parameter update
                   └──► Progressive network lateral connection training
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Protocol-based DI everywhere | Testable without hardware; clean separation of concerns |
| `asyncio` throughout, `to_thread` for I/O | No GIL contention; predictable latency |
| Pydantic v2 for all config | Validation at startup; no silent misconfiguration |
| `structlog` JSON logging | Machine-readable in production; grep-friendly in dev |
| `deque(maxlen=N)` ring buffers | Fixed memory footprint for sensor history |
| LMDB for experience storage | Zero-copy reads; crash-safe; high write throughput |
| FAISS for memory retrieval | Sub-millisecond similarity search at 50k scale |
| numpy-only cognitive inference | No CUDA dependency for BDI/metacog; runs on CPU |
| `torch.no_grad()` for all RSSM/MCTS inference | Prevents accidental gradient accumulation |
| Module-level constants for all magic numbers | Grep-able; documented; not scattered in logic |
