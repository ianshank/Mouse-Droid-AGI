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
                TelemetryPub["Telemetry Publisher\nasync queue\n≤60 Hz non-blocking"]
                MetricsRegistry["Metrics Registry\nPrometheus text format\nconfig-driven namespace"]
                TelemetryServer["Telemetry Server\naiohttp REST + WebSocket\nport 8080"]
                ExperienceDB[("Experience Logger\nLMDB\n/home/jetson/experience_db")]
            end
        end
        subgraph HWLayer["Hardware Interface Layer"]
            Camera["Camera\nJetson CSI / IMX708\nFeatureExtractorProtocol"]
            Ultrasonic["Ultrasonic\nHC-SR04"]
            Microphone["USB Microphone\nWonrabai USB Sound Card\nAudioProtocol"]
            Speaker["USB Speaker\nWonrabai USB Sound Card\nSpeakerProtocol"]
            LiDAR["LiDAR\nFHL-LD19\nLidarProtocol"]
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
    Orchestrator --> TelemetryPub
    TelemetryPub --> MetricsRegistry
    HealthMonitor --> MetricsRegistry
    TelemetryPub --> TelemetryServer
    MetricsRegistry --> TelemetryServer
    HealthMonitor --> TelemetryPub
    CoreAI --> SensorMgr
    CoreAI --> ExperienceDB
    SensorMgr --> Camera
    SensorMgr --> Ultrasonic
    SensorMgr --> Microphone
    SensorMgr --> Speaker
    SensorMgr --> LiDAR
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
| Jetson CSI / IMX708 camera | jetson_utils / picamera2 | Vision capture + pluggable feature extraction (MeanPool / TensorRT) |
| Wonrabai USB Sound Card | PyAudio | Combo mic + speaker: audio capture for world model + TTS output |
| FHL-LD19 LiDAR | pyserial UART | 360° 2D distance scanning for obstacle detection + clearance |
| NVMe SSD 500 GB | ext4 `/mnt/ssd` | Docker data-root, containerd, 16 GB swap |
| Telemetry Publisher | Python asyncio queue | Non-blocking sensor-frame fan-out at ≤60 Hz |
| Metrics Registry | Python stdlib exporter | Prometheus metric families and text rendering |
| Telemetry Server | aiohttp 3.x REST + WebSocket | Remote WiFi/Ethernet monitoring (configurable port, default 8080) |
---

## Level 3 — Component Diagram (mousedroid process)

```mermaid
graph TD
    CLI["CLI Entry\nmain.py"]
    Factory["Factory\nfactory.py\nOnly file that imports concrete types"]
    Orchestrator["Orchestrator\norchestrator/\ntick at 30 Hz\nsense - plan - act"]
    SensorMgr["Sensor Manager\nsensing/\nCamera - Ultrasonic - LiDAR - Audio - ESP32 encoders"]
    SafetyMon["Safety Monitor\nsafety/\nClearance - Battery - Sensor staleness"]
    Encoder["Encoder\nvision + motor + ultrasonic + audio + lidar"]
    RSSM["World Model\nRSSM or DualStreamRSSM\nGRU 256 + CfC 64 = 320 combined\nobserve step / imagine step"]
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

    TelemetryPub2["Telemetry Publisher\ntelemetry/publisher.py\nasync queue bridge"]
    MetricsReg2["Metrics Registry\ntelemetry/metrics.py\nPrometheus text rendering"]
    TelemetryServer2["Telemetry Server\ntelemetry/server.py\nREST /api/v1/* + WebSocket /ws"]

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
    Orchestrator --> TelemetryPub2
    Orchestrator --> MetricsReg2
    TelemetryPub2 --> TelemetryServer2
    MetricsReg2 --> TelemetryServer2
    Orchestrator --> Watchdog
    Orchestrator --> MemoryTier
    Orchestrator --> VoiceEngine
```

---

## Level 3b — Component Diagram: Production Hardening (v0.3.0)

New production components added in the v0.3.0 release:

```mermaid
graph TD
    Orchestrator["Orchestrator\nasyncio.wait_for(tick, timeout=tick_timeout_s)"]

    subgraph Watchdog["Watchdog Layer\nhealth/watchdog.py"]
        WatchdogProt["WatchdogProtocol\n@runtime_checkable"]
        SystemdNotif["SystemdNotifier\nWATCHDOG=1 via sdnotify or subprocess"]
        FileHB["FileHeartbeatNotifier\ntimestamp to /tmp/mousedroid_heartbeat"]
        NullNotif["NullNotifier\nmock/dev mode"]
        WatchdogProt <|.. SystemdNotif
        WatchdogProt <|.. FileHB
        WatchdogProt <|.. NullNotif
    end

    subgraph MemoryTierGroup["Memory Tier\nmemory/tier.py"]
        MemTier["MemoryTier\nepisodic + semantic + working + consolidation"]
        EpiRep["EpisodicReplay\nFAISS 50k"]
        SemIdx["SemanticIndex\nconcept graph"]
        WorkMem["WorkingMemory\n8192 token window"]
        Consol["MemoryConsolidation\nasync background task"]
        MemTier --> EpiRep
        MemTier --> SemIdx
        MemTier --> WorkMem
        MemTier --> Consol
    end

    subgraph VoiceLayer["Voice Engine\nvoice/engine.py"]
        Rocky["Rocky Personality\nphrase_bank.py"]
        PiperTTS["Piper TTS\nlocal inference"]
        Speaker["USB Speaker\nSpeakerProtocol"]
        Rocky --> PiperTTS --> Speaker
    end

    subgraph PreFlight["Pre-flight Check\nscripts/preflight_check.sh"]
        PF_ESP32["ESP32 device check"]
        PF_Camera["Camera device check"]
        PF_GPIO["GPIO device check"]
        PF_Disk["Disk space check"]
        PF_Config["Config YAML check"]
        PF_Weights["Model weights check"]
    end

    Orchestrator -- "notify() after successful tick" --> Watchdog
    Orchestrator -- "push ExperienceRecord each tick" --> MemoryTierGroup
    Orchestrator -- "startup/shutdown/error/obstacle events" --> VoiceLayer
    Orchestrator -- "asyncio.TimeoutError → emergency_stop()" --> ESP32Driver["ESP32 emergency_stop()"]
    PreFlight -- "ExecStartPre (systemd)" --> Orchestrator
```

**Tick Safety Loop (v0.3.0):**

```mermaid
flowchart TD
    Start(["run() loop iteration"])
    WaitFor["asyncio.wait_for(tick(), tick_timeout_s)"]
    Success["Tick completed OK"]
    Timeout["TimeoutError"]
    Exception["Unhandled Exception"]
    EStop["esp32.emergency_stop()"]
    VoiceErr["voice_event('error')"]
    WDNotify["watchdog.notify()"]
    MemPush["memory_tier.push(ExperienceRecord)"]
    RateLimit["rate-limit sleep to next tick"]

    Start --> WaitFor
    WaitFor --> Success
    WaitFor --> Timeout
    WaitFor --> Exception
    Timeout --> EStop
    Timeout --> VoiceErr
    Exception --> EStop
    Exception --> VoiceErr
    Success --> WDNotify
    Success --> MemPush
    WDNotify --> RateLimit
    MemPush --> RateLimit
    VoiceErr --> RateLimit
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
        +build_world_model(cfg) RSSM | DualStreamRSSM
        +build_safety_monitor(cfg) SafetyMonitor
        +build_agent(cfg) NavigationAgent
        +build_cognitive_core(cfg) CognitiveCore
        +build_lidar(cfg) LidarProtocol
        +build_lidar_feature_extractor(cfg) LidarFeatureExtractor
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
    participant Camera as Camera IMX708
    participant Sonic as HC-SR04
    participant Mic as Wonrabai USB Mic
    participant LiDAR as FHL-LD19 LiDAR
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
        Mic->>SM: read_chunk()
        LiDAR->>SM: read_scan()
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

### Telemetry Broadcast (WiFi / Ethernet)

```mermaid
sequenceDiagram
    participant Orch as Orchestrator (30 Hz)
    participant TelePub as TelemetryPublisher
    participant TeleServ as TelemetryServer (aiohttp)
    participant WSClient as WebSocket Client
    participant RESTClient as REST Client
    participant PromClient as Prometheus Scraper

    Note over Orch,TelePub: Each control tick
    Orch->>TelePub: publish(TelemetryFrame)
    TelePub->>TelePub: put_nowait — drops if queue full
    TelePub->>TeleServ: _latest_frame updated

    Note over TeleServ,WSClient: Background broadcast loop
    TeleServ->>WSClient: JSON frame over WebSocket

    Note over RESTClient,TeleServ: On-demand REST endpoints
    RESTClient->>TeleServ: GET /api/v1/sensors
    TeleServ-->>RESTClient: latest TelemetryFrame JSON
    RESTClient->>TeleServ: GET /api/v1/health
    TeleServ-->>RESTClient: GPU temp, load, battery
    RESTClient->>TeleServ: GET /api/v1/logs
    TeleServ-->>RESTClient: last-N structured log entries
    RESTClient->>TeleServ: GET /api/v1/network
    TeleServ-->>RESTClient: interfaces + server URL
    Note over PromClient,TeleServ: Prometheus scrape path
    PromClient->>TeleServ: GET /metrics
    TeleServ-->>PromClient: text/plain; version=0.0.4
```

---

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

### Cognitive Core Data Flow (dual-cadence)

```mermaid
sequenceDiagram
    participant Orch as Orchestrator (30 Hz)
    participant Fast as CognitiveCore.tick_fast()
    participant Policy as PolicyMLP
    participant Const as ConstitutionalChecker
    participant BDI as NeuralBDI (slow ~1 Hz)
    participant Meta as MetacognitiveLoop
    participant WM as WeightsManager
    participant HF as HuggingFace Hub

    Orch->>Fast: observation + safety_context
    Fast->>Policy: forward(latent_state)
    Fast->>Const: check(action)
    Const-->>Fast: safe / override
    Fast-->>Orch: action_tensor (or fallback to MCTS)

    Note over BDI,Meta: Background asyncio.Task (~1 Hz)
    BDI->>Meta: belief, desire, intentions, affect
    Meta->>Meta: update approach_rate, surprise

    Note over WM,HF: Startup weight loading
    WM->>WM: check local weights
    alt Weights missing
        WM->>HF: hf_hub_download(retry=3, backoff=2^n)
        HF-->>WM: .npz files
    end
    WM-->>BDI: loaded parameters
```

---

## Training Pipeline (Offline GPU)

```mermaid
graph TD
    subgraph Pipeline["GPU Pretraining Pipeline (training/)"]
        DG["data_generator.py\nSynthetic ObservationBundle sequences"]
        RSSM_T["train_rssm.py\nPhase 2.1: RSSM encoder + dynamics\ncheckpoints every N epochs"]
        WS["warmstart_policy.py\nPhase 2.2: MCTS warm-start weights\nUCB calibration (tune_ucb)"]
        CA["collect_annotations.py\nPhase 2.3a: BDI intention labels"]
        BDI_T["train_bdi.py\nPhase 2.3b: NeuralBDI on annotations"]
        CRL["train_constitutional_rl.py\nPhase 2.4: PPO + Constitutional RL"]
        RP["run_pipeline.py\nOrchestrator: phase1→2→3→4\n--resume / resume_from for checkpoints"]
        UP["upload_weights.py\nHuggingFace Hub push (28 files)\nianshank/mousedroid-weights"]
    end
    subgraph HF["HuggingFace Hub"]
        HFRepo["ianshank/mousedroid-weights\nbdi/ mcts/ rssm/ constitutional_rl/"]
    end
    subgraph Deploy["Jetson Runtime"]
        WM["WeightsManager\nhf_hub_download(subfolder, local_dir)\nretry=3 backoff=2^n"]
        NeuralBDI["NeuralBDI\nbeliefs / desires / intentions"]
    end

    DG --> RSSM_T
    RSSM_T --> WS
    WS --> CA
    CA --> BDI_T
    BDI_T --> CRL
    RSSM_T & WS & BDI_T & CRL --> UP
    UP --> HFRepo
    RP -.-> RSSM_T & WS & BDI_T & CRL
    HFRepo --> WM
    WM --> NeuralBDI
```

**Key training facts:**

| Property | Value |
|----------|-------|
| Pipeline entry | `python training/run_pipeline.py [--resume path/to/checkpoint.pt]` |
| Resume support | `--resume` / `resume_from` in config; forwards checkpoint to RSSM training only |
| Weight storage | `ianshank/mousedroid-weights` HuggingFace repo; `bdi/`, `mcts/`, `rssm/`, `constitutional_rl/` subfolders |
| Runtime download | `WeightsManager.download_weights_from_huggingface(subfolder='bdi', local_dir=weights_dir.parent)` |
| Type safety | All `training/` modules pass `mypy --strict` (NDArray[Any] annotations) |

---

## Dual-Stream CfC/GRU World Model

The world model supports two modes selected via `model.cfc_hidden_dim`:

- **Classic RSSM** (`cfc_hidden_dim=0`): Single GRU stream, 256-dim hidden state
- **Dual-Stream RSSM** (`cfc_hidden_dim>0`): GRU + CfC parallel streams with concat fusion

```mermaid
graph TD
    subgraph DualStream["Dual-Stream CfC/GRU RSSM"]
        Input["Recurrent Input\n[z_prev | action_prev]"]
        GRU["GRU Stream\n256-dim hidden\nSlow planning"]
        CfC["CfC Stream\n64-dim hidden\nFast adaptive reflexes\n(ncps liquid neural network)"]
        Fusion["StreamFusion\nconcat: [h_gru | h_cfc] = 320-dim"]
        Posterior["Posterior Net\n[h_combined | obs_embed] → z"]
        Prior["Prior Net\nh_combined → z_prior"]
        Decoder["Decoder\n[h_combined | z] → obs_recon"]
        SafetyTrace["Safety Trace\nExtracts CfC state for inspection\nget_safety_trace()"]
    end

    Input --> GRU
    Input --> CfC
    GRU --> Fusion
    CfC --> Fusion
    Fusion --> Posterior
    Fusion --> Prior
    Fusion --> Decoder
    Fusion --> SafetyTrace
```

**Training:** Dual optimizers with separate learning rates and gradient clipping per stream.
CfC loss weight ramps linearly from 0.1→1.0 over 10k steps.

**Human activation gate:** CfC disabled by default in production (`cfc_hidden_dim=0`).
Activate via: `MOUSEDROID_MODEL__CFC_HIDDEN_DIM=64 docker compose up -d`

**HuggingFace:** Trained weights at `ianshank/mousedroid-dual-stream-rssm` (experimental).

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
| Dual-cadence cognitive loop | Fast 30 Hz reaction + slow 1 Hz deliberation minimises compute |
| HuggingFace Hub weight loading | `hf_hub_download(subfolder, local_dir)` routes files to exact path; retry=3, backoff=2^n |
| `torch.no_grad()` for all RSSM/MCTS inference | Prevents accidental gradient accumulation |
| Optional audio projection in encoder | `audio_dim=0` disables audio entirely; existing 3-modality checkpoints load unchanged |
| `FeatureExtractorProtocol` for camera | Pluggable backends (MeanPool, TensorRT, ONNX); eliminates duplicate code across camera drivers |
| Module-level constants for all magic numbers | Grep-able; documented; not scattered in logic |
| Dual-stream CfC/GRU RSSM | CfC provides sub-100ms adaptive time constants for reflexes; GRU handles slow planning; concat fusion preserves both information streams; safety trace exposes CfC state for independent monitoring |
| Human activation gate for CfC | `cfc_hidden_dim=0` default ensures classic RSSM in production; explicit env var override required — prevents accidental deployment of experimental architecture |
| Separate dual optimizers | GRU (lr=3e-4, clip=10.0) and CfC (lr=1e-4, clip=1.0) trained independently; CfC loss warmup prevents destabilising GRU early in training |
| L4T Docker container | GPU-accelerated deployment with consistent environment |
| Multi-stage Docker builds | Extract pre-compiled binaries from upstream images to bypass OOM |
| NVMe SSD for Docker + swap | 500 GB fast storage avoids SD card wear and OOM during builds |
| `systemd-run` for long builds | Persistent processes survive SSH drops |
| aiohttp over FastAPI for telemetry | Lightweight asyncio-native HTTP; no ASGI wrapper; fits inside the existing event loop |
| Non-blocking drop-on-full queue | Telemetry must never stall the 30 Hz control loop; frame drops are preferable to back-pressure |
| Stdlib-only network discovery | `socket` + `netifaces` avoids extra system calls; works inside Docker without root |
| Immutable `TelemetryFrame` dataclass | Thread-safe snapshot; safe to pass across asyncio tasks without copying |
