# MouseDroidAGI — C4 Architecture

This document uses the [C4 model](https://c4model.com/) (Context → Container → Component → Code) to describe the system architecture at four levels of abstraction.

---

## Level 1 — System Context

```mermaid
graph TD
    HumanOp["Human Operator\n(NL commands, monitoring)"]
    System["MouseDroidAGI System\nAutonomous Star Wars MSE-6 robot\npowered by an Agentic World Model\nrunning on Jetson Orin Nano"]
    PhysWorld["Physical World\n(corridors, obstacles, people)"]
    RemoteMonitor["Remote Monitoring\nPrometheus / Grafana\nmetrics over WiFi"]

    HumanOp -- "NL commands / health dashboards" --> System
    System -- "navigates" --> PhysWorld
    System -- "publishes metrics" --> RemoteMonitor
```

**Actors:**

- **Human Operator** — Sends natural language mission commands via LLM gateway; monitors telemetry
- **Physical World** — The environment the robot navigates through
- **Remote Monitoring** — Optional Prometheus/Grafana dashboard for production deployments

---

## Level 2 — Container Diagram

```mermaid
graph TD
    subgraph Jetson["NVIDIA Jetson Orin Nano"]
        subgraph DockerContainer["Docker: mousedroid:jetson\nL4T PyTorch r36.4.0\nCUDA 12.6 / TensorRT 10.4"]
            subgraph AppProcess["mousedroid Python process (asyncio event loop)"]
                Orchestrator["Orchestrator\n30 Hz loop"]
                LLMGateway["LLM Gateway\nLlama GGUF"]
                HealthMonitor["Health Monitor\nsysfs polling"]
                CoreAI["Core AI Pipeline\nRSSM + MCTS + Navigation Agent\nCognitive Core BDI + Metacognitive\nMemory: Working / Episodic / Semantic\nSafety Monitor + Constitutional Checker"]
                SensorMgr["Sensor Manager\nconcurrent I/O"]
                ExperienceDB[("Experience Logger\nLMDB\n/home/jetson/experience_db")]
            end
        end
        subgraph HWLayer["Hardware Interface Layer"]
            Camera["Camera\nJetson CSI / IMX500"]
            Ultrasonic["Ultrasonic\nHC-SR04"]
            GPIO["GPIO\nJetson.GPIO"]
        end
        subgraph Storage["NVMe SSD (500 GB)"]
            DockerData["Docker Data Root\n/mnt/ssd/docker"]
            SwapFile["Swap File\n16 GB"]
            Containerd["containerd\nsnapshotter"]
        end
    end
    ESP32["ESP32 Wave Rover\nMotor control\nEncoder feedback\nBattery ADC"]

    Orchestrator --> CoreAI
    Orchestrator --> LLMGateway
    Orchestrator --> HealthMonitor
    CoreAI --> SensorMgr
    CoreAI --> ExperienceDB
    SensorMgr --> Camera
    SensorMgr --> Ultrasonic
    SensorMgr --> GPIO
    Orchestrator -- "UART 1 Mbps / HTTP" --> ESP32
    DockerContainer -.-> DockerData
    DockerContainer -.-> Containerd
```

**Containers:**

| Container | Technology | Responsibility |
|-----------|-----------|----------------|
| Docker `mousedroid:jetson` | L4T PyTorch r36.4.0 | GPU-accelerated container (CUDA 12.6 + TensorRT 10.4) |
| `mousedroid` process | Python 3.10 asyncio | All AI reasoning + I/O orchestration |
| LMDB experience store | LMDB on-disk | Persistent experience replay buffer |
| Llama GGUF model | llama-cpp-python | Local LLM for NL to velocity |
| ESP32 firmware | C++ (Wave Rover SDK) | Motor PWM control, encoder polling |
| Jetson CSI / IMX500 camera | jetson_utils / picamera2 | Vision capture + onboard neural inference |
| NVMe SSD 500 GB | ext4 `/mnt/ssd` | Docker data-root, containerd, 16 GB swap |

---

## Level 3 — Component Diagram (mousedroid process)

```mermaid
graph TD
    CLI["CLI Entry\nmain.py"]
    Factory["Factory\nfactory.py\nOnly file that imports concrete types"]
    Orchestrator["Orchestrator\norchestrator/\ntick at 30 Hz\nsense - plan - act"]
    SensorMgr["Sensor Manager\nsensing/\nCamera - Ultrasonic - ESP32 encoders"]
    SafetyMon["Safety Monitor\nsafety/\nClearance - Battery - Sensor staleness"]
    Encoder["Encoder\nvision + motor + ultrasonic"]
    RSSM["RSSM\nlatent h and z\nobserve step / imagine step"]
    MCTS["MCTS\n50 to 200 sims"]
    NavAgent["Navigation Agent\nagents/\nact with h, z, safety"]
    ESP32Driver["ESP32 Driver\ncomms/\nsend velocity"]
    FastTick["Cognitive Fast tick 30 Hz\nPolicyMLP - ConstitutionalChecker"]
    SlowLoop["Cognitive Slow loop 1 Hz\nBDI inference - Metacog update"]
    WorkingMem["Working Memory\nsliding 8192 token window"]
    EpisodicMem["Episodic Memory\nFAISS 50k cap"]
    SemanticMem["Semantic Memory\nconcept graph + FAISS"]
    Consolidation["Consolidation\nasync episodic to semantic"]
    EWC["EWC\nFisher-information regularisation"]
    PNN["PNN\nprogressive nets + lateral connections"]
    LLM["LLM Gateway optional\nNL to GoalVector\nLocal Llama GGUF"]

    CLI --> Factory
    Factory --> Orchestrator
    Orchestrator --> SensorMgr
    Orchestrator --> SafetyMon
    SensorMgr --> Encoder
    SafetyMon --> RSSM
    Encoder --> RSSM
    RSSM --> MCTS
    MCTS --> NavAgent
    NavAgent -- "action tensor" --> ESP32Driver
    Orchestrator -.-> FastTick
    Orchestrator -.-> SlowLoop
    Orchestrator -.-> WorkingMem
    Orchestrator -.-> LLM
    EpisodicMem --> Consolidation
    Consolidation --> SemanticMem
    Orchestrator -.-> EWC
    Orchestrator -.-> PNN
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

```mermaid
classDiagram
    class ESP32CommProtocol {
        <<Protocol>>
        +connect() None
        +send_velocity(vx, vy, omega) None
        +read_encoders() EncoderReading
        +get_battery_voltage() float
        +emergency_stop() None
        +disconnect() None
    }
    class MockESP32Driver {
        +cfg: ESP32Config
    }
    class SerialESP32Driver {
        +cfg: ESP32Config
    }
    class WiFiESP32Driver {
        +cfg: ESP32Config
    }
    class Factory {
        +build_esp32_driver(cfg) ESP32CommProtocol
        +build_orchestrator(cfg) Orchestrator
        +build_world_model(cfg) RSSM
        +build_safety_monitor(cfg) SafetyMonitor
        +build_agent(cfg) NavigationAgent
    }

    ESP32CommProtocol <|.. MockESP32Driver : implements
    ESP32CommProtocol <|.. SerialESP32Driver : implements
    ESP32CommProtocol <|.. WiFiESP32Driver : implements
    Factory --> ESP32CommProtocol : creates
```

This means:

- **Tests** inject `MockESP32Driver` via `Settings(mock_hardware=True)` — no GPIO needed
- **CI** runs the full test suite with zero hardware
- **Production** seamlessly switches to `SerialESP32Driver` via config

---

## Data Flows

### Sense-Plan-Act (30 Hz)

```mermaid
sequenceDiagram
    participant Camera as Camera IMX500
    participant Sonic as HC-SR04
    participant ESP32 as ESP32 Encoders
    participant SM as SensorManager
    participant Safety as SafetyMonitor
    participant WM as WorldModel RSSM
    participant Agent as NavigationAgent
    participant Motor as ESP32 Motor

    par Concurrent sensor reads
        Camera->>SM: capture_features()
        Sonic->>SM: read_distance_m()
        ESP32->>SM: read_encoders() + get_battery_voltage()
    end

    SM->>Safety: ObservationBundle
    Safety->>WM: SafetyContext + ObservationBundle
    WM->>Agent: latent state h, z, surprise
    Agent->>Motor: send_velocity(vx, vy, omega)
```

### Experience Pipeline (async)

```mermaid
graph LR
    OB["ObservationBundle"]
    ER["ExperienceRecord"]
    EL[("ExperienceLogger\nLMDB")]
    MC["Memory Consolidation"]
    Epi["Episodic Memory\nFAISS"]
    Sem["Semantic Memory\nconcept graph"]

    OB --> ER --> EL --> MC
    MC --> Epi
    Epi --> Sem
```

### Learning Pipeline (offline / async)

```mermaid
graph LR
    LMDB[("LMDB\nReplay Buffer")]
    EWC["EWC Regularisation\nFisher-information"]
    PNN["Progressive Nets\nlateral connections"]
    Params["Model Parameter Update"]

    LMDB --> EWC --> Params
    LMDB --> PNN --> Params
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
| L4T Docker container | GPU-accelerated deployment with consistent environment |
| NVMe SSD for Docker + swap | 500 GB fast storage avoids SD card wear and OOM during builds |
| `systemd-run` for long builds | Persistent processes survive SSH drops |
