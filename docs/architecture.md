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
    MCPClient["MCP Clients (optional)\nClaude Code / Claude Desktop /\nmcp.client SDK\nauthenticated bearer token"]
    GCPCloud["GCP Digital Twin (optional)\nPub/Sub, Cloud Storage,\nVertex AI, Cloud Monitoring"]
    HFHub["HuggingFace Hub\nModel weight registry"]

    HumanOp -- "NL commands / health dashboards" --> System
    HumanOp -. "tool calls / state reads (chat)" .-> MCPClient
    MCPClient -. "stdio / SSE / streamable_http" .-> System
    System -- "navigates" --> PhysWorld
    System -- "publishes metrics" --> RemoteMonitor
    System -. "telemetry + experience\n(optional, CircuitBreaker)" .-> GCPCloud
    GCPCloud -. "trained weights" .-> HFHub
    HFHub -. "weight pull at startup" .-> System
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
                Orchestrator["Orchestrator\nconfig-driven loop cadence"]
                LLMGateway["LLM Gateway\nLlama GGUF"]
                HealthMonitor["Health Monitor\nsysfs polling"]
                CoreAI["Core AI Pipeline\nRSSM + MCTS + Navigation Agent\nCognitive Core BDI + Metacognitive\nMemory: Working / Episodic / Semantic\nSafety Monitor + Constitutional Checker"]
                SensorMgr["Sensor Manager\nconcurrent I/O"]
                TelemetryPub["Telemetry Publisher\nasync queue\n≤60 Hz non-blocking"]
                MetricsRegistry["Metrics Registry\nPrometheus text format\nconfig-driven namespace"]
                TelemetryServer["Telemetry Server\naiohttp REST + WebSocket\nconfig-driven port"]
                MCPServer["MCP Server (optional)\nstdio / SSE / streamable_http\nbridges ToolRegistry + telemetry +\nlogs + redacted config + memory"]
                ExperienceDB[("Experience Logger\nLMDB\n/home/jetson/experience_db")]
            end
        end
        subgraph OpsLayer["Validation / Operations Layer"]
            RuntimeValidation["Runtime Validation\nvalidation/runtime.py\nshared overlay resolution + factory-backed checks"]
            SmokeHarness["Smoke Harness\njetson_validate.sh + jetson_smoke_test.sh + verify_sensors.py"]
        end
        subgraph HWLayer["Hardware Interface Layer"]
            Camera["Camera\nIMX500 ribbon camera\nJetson / GStreamer / V4L2 fallback"]
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
    Orchestrator -.-> MCPServer
    MCPServer -.-> MetricsRegistry
    MCPServer -.-> TelemetryPub
    CoreAI --> SensorMgr
    CoreAI --> ExperienceDB
    SensorMgr --> Camera
    SensorMgr --> Microphone
    SensorMgr --> Speaker
    SensorMgr --> LiDAR
    SensorMgr --> GPIO
    SmokeHarness --> RuntimeValidation
    RuntimeValidation --> Camera
    RuntimeValidation --> Microphone
    RuntimeValidation --> Speaker
    RuntimeValidation --> LiDAR
    TenPillars["Ten Pillars Campaign\nvalidate_pillar.sh\npytest + factory probe × 10"] --> SmokeHarness
    Orchestrator -- "UART 1 Mbps / HTTP" --> ESP32
    DockerContainer -.-> DockerData
    DockerContainer -.-> Containerd
```

**Containers:**

| Container | Technology | Responsibility |
| --------- | ---------- | -------------- |
| Docker `mousedroid:jetson` | L4T PyTorch r36.4.0 | GPU-accelerated container (CUDA 12.6 + TensorRT 10.4) |
| `mousedroid` process | Python 3.10 asyncio | All AI reasoning + I/O orchestration |
| Runtime validation layer | Python utilities + shell harnesses | Shared config-backed smoke, verification, and host-driven Jetson validation |
| Ten Pillars campaign | `validate_pillar.sh` | Headless dispatcher — runs pytest + factory probe for each of the 10 AGI pillars |
| LMDB experience store | LMDB on-disk | Persistent experience replay buffer |
| Llama GGUF model | llama-cpp-python | Local LLM for NL to velocity |
| ESP32 firmware | C++ (Wave Rover SDK) | Motor PWM control, encoder polling |
| IMX500 ribbon camera | jetson_utils / GStreamer / V4L2 | Vision capture with config-driven fallback and pluggable feature extraction |
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
    Orchestrator["Orchestrator\norchestrator/\nconfig-driven tick cadence\nsense - plan - act"]
    SensorMgr["Sensor Manager\nsensing/\nCamera - LiDAR - Audio - ESP32 encoders"]
    SafetyMon["Safety Monitor\nsafety/\nClearance - Battery - Sensor staleness"]
    Encoder["Encoder\nvision + motor + audio + lidar"]
    RSSM["World Model\nRSSM or DualStreamRSSM\nconfig-driven hidden dims\nobserve step / imagine step"]
    MCTS["MCTS\nconfig-driven simulation budget"]
    NavAgent["Navigation Agent\nagents/\nact with h, z, safety"]
    ESP32Driver["ESP32 Driver\ncomms/\nsend velocity"]
    FastTick["Cognitive Fast tick 30 Hz\nPolicyMLP - ConstitutionalChecker"]
    SlowLoop["Cognitive Slow loop 1 Hz\nBDI inference - Metacog update"]
    WorkingMem["Working Memory\nconfig-driven context window"]
    EpisodicMem["Episodic Memory\nconfig-driven replay capacity"]
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

## Level 3c — Runtime Validation and Smoke Alignment

The Jetson validation surface is deliberately wired through the same config and factory layer as
the application so host-side smoke checks do not drift from the deployed runtime.

```mermaid
graph TD
    HostRunner["Host Runner\nmanual validation / CI smoke"]
    JetsonValidate["scripts/jetson_validate.sh\nremote verify / pytest / smoke orchestration"]
    JetsonFullSmoke["scripts/jetson_full_smoke_run.sh\nfull hardware smoke harness\n13 stages + SUMMARY.md enricher"]
    JetsonSmoke["scripts/jetson_smoke_test.sh\nhost-side smoke harness"]
    VerifySensors["scripts/verify_sensors.py\nJSON and human-readable sensor checks"]
    TenPillars["scripts/validate_pillar.sh\nTen Pillars campaign dispatcher\npytest + factory probe × 10\nwrites ten_pillars.log"]
    RuntimeValidation["validation/runtime.py\nresolve_runtime_config_paths()\nload_runtime_settings()\ncapture_* helpers\nplay_rocky_voice_phrase()"]
    SettingsLoader["config.loader.load_settings\nYAML + env overlay resolution"]
    Factory["factory.py\nprotocol-based DI"]
    Camera["JetsonCSICamera\nJetson / GStreamer / V4L2 fallback"]
    Microphone["Microphone / Speaker"]
    Lidar["LD19 driver\nconfig-driven coverage + timeout"]
    VoiceEngine["Voice Engine\nPiperTTS + Rocky phrase_bank\nUSB Speaker"]

    HostRunner --> JetsonValidate
    HostRunner --> JetsonFullSmoke
    HostRunner --> TenPillars
    JetsonFullSmoke --> RuntimeValidation
    JetsonFullSmoke --> TenPillars
    JetsonValidate --> JetsonSmoke
    JetsonValidate --> VerifySensors
    JetsonSmoke --> RuntimeValidation
    VerifySensors --> RuntimeValidation
    TenPillars --> RuntimeValidation
    TenPillars --> Factory
    RuntimeValidation --> SettingsLoader
    RuntimeValidation --> Factory
    Factory --> Camera
    Factory --> Microphone
    Factory --> Lidar
    Factory --> VoiceEngine
```

All runtime validation paths load overlays through `resolve_runtime_config_paths()` and
`load_runtime_settings()`, so values such as `camera.device_path`,
`lidar.scan_acquisition_timeout_s`, `lidar.min_scan_coverage_deg`, and
`voice.tts_model_path`, `voice.personality_to_model_map`, `voice.event_intensity_thresholds`, and
`voice.output_volume` remain config-driven.

`play_rocky_voice_phrase()` provides a factory-backed end-to-end TTS + speaker smoke check
used by `jetson_full_smoke_run.sh`. Voice smoke status: **PASS** (39,424 samples,
`en_US-lessac-medium`, `20260425T192408Z`).

**Ten Pillars campaign** (`scripts/validate_pillar.sh all`): runs 20 checks (10 pytest stages +
10 factory probes) across all AGI pillars. Last result: **Overall: PASS — 20/20**
(`2026-04-26T23:55:42Z`, Jetson Orin Nano, CUDA 12.6, TensorRT 10.4.0).

| Pillar | pytest marker | Factory probe class |
| ------ | ------------- | ------------------- |
| world_model | `unit/world_model/` | `build_world_model(cfg)` |
| cognitive | `unit/cognitive/` | `build_cognitive_core(cfg)` |
| memory | `unit/memory/` | `build_memory_tier(cfg)` |
| continual | `unit/learning/` | `EWCAgent(cfg.learning, nn.Linear(...))` |
| meta | `unit/meta/` | `MAMLAdapter(nn.Linear(...), ...)` |
| curiosity | `unit/curiosity/` | `build_curiosity_module(cfg)` |
| growth | `unit/growth/` | `KnowledgeDistiller(teacher, student, ...)` |
| reward | `unit/reward/` | `MultiObjectiveRewardModel(cfg.model, cfg.reward)` |
| scaling | `unit/scaling/` | `AdaptiveCompute(input_dim=..., max_steps=8)` |
| safety | `unit/safety/` | `build_safety_monitor(cfg)` |

> **Overlay sync note**: when managed by `scripts/mousedroid-docker.service`, the production
> overlay is synced automatically via `scripts/sync_jetson_overlay.sh` before
> `preflight_check.sh` executes.

---

## Level 3d — Component Diagram: GCP Digital Twin (Optional)

Cloud features are **fully optional** — the droid operates identically with `gcp: null` in config.
All cloud calls are protected by `CircuitBreaker` + `retry_async` patterns so cloud failures
never block the 30 Hz control loop.

```mermaid
graph TD
    subgraph Jetson["Jetson Orin Nano"]
        Orchestrator["Orchestrator\n30 Hz loop"]
        ExperienceDB[("LMDB\nExperience Logger")]
        TelemetryPub["TelemetryPublisher\nasync queue"]
        MetricsReg["MetricsRegistry\nPrometheus"]
    end

    subgraph CloudModule["cloud/ module (optional)"]
        PubSubSink["CloudTelemetrySink\npubsub_sink.py\nmsgpack + CircuitBreaker"]
        GCSExporter["CloudExperienceExporter\nexperience_exporter.py\nLMDB → GCS shards\nhigh-water mark cursor"]
        LogSink["CloudLoggingSink\nlogging_sink.py\nstructlog processor"]
        MonExporter["CloudMetricsExporter\nmonitoring_exporter.py\ngauge → TimeSeries"]
        FireSync["CloudFirestoreSync\nfirestore_sync.py\nepisodic → Firestore"]
        Auth["_auth.py\nADC / service account"]
    end

    subgraph GCP["Google Cloud Platform"]
        PubSub["Cloud Pub/Sub\ntelemetry + experience topics"]
        GCS["Cloud Storage\nexperience/v1/{robot_id}/{date}/{hour}/"]
        CloudLog["Cloud Logging"]
        CloudMon["Cloud Monitoring\ncustom metrics"]
        Firestore["Firestore\nepisodic memory collection"]
        VertexAI["Vertex AI Pipelines\n(Phase 2 — future)"]
        HFHub["HuggingFace Hub\nianshank/mousedroid-weights"]
    end

    Orchestrator --> PubSubSink
    Orchestrator --> GCSExporter
    ExperienceDB --> GCSExporter
    TelemetryPub --> PubSubSink
    MetricsReg --> MonExporter

    PubSubSink --> PubSub
    GCSExporter --> GCS
    LogSink --> CloudLog
    MonExporter --> CloudMon
    FireSync --> Firestore
    GCS --> VertexAI
    VertexAI --> HFHub

    PubSubSink --> Auth
    GCSExporter --> Auth
    LogSink --> Auth
    MonExporter --> Auth
    FireSync --> Auth
```

**GCP config hierarchy:** `Settings.gcp: GCPConfig | None = None`. When `None`, all
`build_cloud_*()` factory functions return `None` and the orchestrator skips cloud calls.

| Component | File | GCP Service | Resilience |
| --------- | ---- | ----------- | ---------- |
| Telemetry sink | `cloud/pubsub_sink.py` | Pub/Sub | CircuitBreaker (60s recovery) |
| Experience exporter | `cloud/experience_exporter.py` | Cloud Storage | CircuitBreaker + HWM cursor |
| Logging sink | `cloud/logging_sink.py` | Cloud Logging | Fire-and-forget (silent drop) |
| Metrics exporter | `cloud/monitoring_exporter.py` | Cloud Monitoring | Async executor |
| Firestore sync | `cloud/firestore_sync.py` | Firestore | Per-entry exception handling |
| Auth | `cloud/_auth.py` | ADC / SA key | Fail-fast at startup |

---

## Level 3e — Configuration Compatibility and CI Quality Gates

Configuration and CI are intentionally layered so legacy YAML schemas remain loadable while
quality gates fail early and deterministically.

```mermaid
graph TD
    LegacyYaml["Legacy YAML / overlays"]
    Loader["config.loader.load_settings()"]
    SettingsBefore["Settings model_validator(before)\nmigrate_legacy_fields()"]
    Migration["config/migration.py\napply_aliases()\nmigrate_section_*()\nmigrate_group_sections()"]
    SettingsAfter["Pydantic validation\ncanonical Settings object"]
    App["factory.py + orchestrator runtime"]

    CI["scripts/ci.sh"]
    Identity["check_settings_identity.py\ncanonical import identity guard"]
    Hardcoded["check_no_hardcoded_values.py\nchanged-line hardcoded gate"]
    Tests["pytest (importlib mode)\nunit/property/integration/perf/regression/e2e"]
    BranchCov["check_branch_coverage.py\nchanged-file threshold"]

    LegacyYaml --> Loader --> SettingsBefore --> Migration --> SettingsAfter --> App
    CI --> Identity --> Hardcoded --> Tests --> BranchCov
```

This model prevents class-identity drift under coverage/import instrumentation and keeps config
migration logic regression-tested as schema aliases evolve.

---

## Level 3f — Component Diagram: MCP Server (Optional)

The MCP (Model Context Protocol) server is an opt-in module that exposes the existing
`ToolRegistry`, telemetry pipeline, log buffer, redacted `Settings`, and (optionally)
episodic memory snapshots to any MCP-compliant client (Claude Code, Claude Desktop, the
`mcp.client` SDK, or future tooling). It is fully config-driven via `MCPConfig`, lazy-imports
the optional `mcp` SDK, and reuses every existing safety / resilience primitive — no
parallel infrastructure is introduced.

```mermaid
graph TD
    subgraph MCPModule["src/mousedroid/mcp/ — Optional MCP Server"]
        ServerCls["MouseDroidMCPServer\nserver.py\n@runtime_checkable MCPServerProtocol\nspawn_tracked + cancel_and_drain lifecycle"]
        Auth["BearerTokenValidator\nauth.py\nenv-only secret\nhmac.compare_digest"]
        Bridge["MCPToolBridge\ntool_bridge.py\ndeny → allow → actuation gate →\nsafety_monitor.evaluate() →\ntoken-bucket rate limit →\nCircuitBreaker(BREAKER_NAME) →\nrequest_timeout_s"]
        TelRes["TelemetryResourceProvider\nresources.py\nlocal deque sampler\n(get_nowait, never blocks loop)"]
        LogRes["LogResourceProvider\nresources.py\nLogRingBuffer.get_recent\n+ regex-driven redaction"]
        ConfRes["ConfigResourceProvider\nresources.py\nSettings.model_dump\nTTL cache + redaction"]
        MemRes["MemoryResourceProvider\nresources.py\nepisodic.sample\n+ ndarray summarisation"]
        Prompts["default_prompts()\nprompts.py\ndiagnose-last-failure /\nsummarise-recent-telemetry /\narm-task-status"]
        MetricsHelpers["metrics.py\nrecord_request / record_tool_call →\nMetricsRegistry"]
    end

    subgraph Reused["Reused infrastructure (no parallel impl)"]
        ToolReg["ToolRegistry\ncommon/tools/registry.py"]
        SafetyMon["SafetyMonitorProtocol\nsafety/monitor.py"]
        Pub["TelemetryPublisherProtocol\ntelemetry/publisher.py"]
        LogBuf["LogRingBuffer\ntelemetry/log_buffer.py"]
        Settings["Settings\nconfig/schema.py\nMCPConfig + MCPResourcesConfig"]
        Mem["MemoryTier\nmemory/tier.py"]
        Metrics["MetricsRegistry\ntelemetry/metrics.py\n+ mcp_requests_total /\n  mcp_tool_calls_total{tool,result} /\n  mcp_request_latency_ms"]
        Resilience["CircuitBreaker / spawn_tracked /\ncancel_and_drain"]
    end

    subgraph Clients["MCP-compliant clients (off-device)"]
        ClaudeCode["Claude Code CLI"]
        ClaudeDesktop["Claude Desktop"]
        MCPCli["mcp.client SDK"]
    end

    Clients -. "stdio / SSE / streamable_http" .-> ServerCls
    ServerCls --> Auth
    ServerCls --> Bridge
    ServerCls --> TelRes
    ServerCls --> LogRes
    ServerCls --> ConfRes
    ServerCls --> MemRes
    ServerCls --> Prompts
    Bridge --> ToolReg
    Bridge --> SafetyMon
    Bridge --> Resilience
    Bridge --> MetricsHelpers
    TelRes --> Pub
    LogRes --> LogBuf
    ConfRes --> Settings
    MemRes --> Mem
    MetricsHelpers --> Metrics
```

**Lifecycle:**

- Built by `factory.build_mcp_server(...)` only when `cfg.mcp.enabled`. Returns `None`
  otherwise; orchestrator boots normally without it.
- `factory.build_mcp_server` raises `ValueError` if `cfg.mcp.port == cfg.telemetry.port`
  for a non-stdio transport (configurable, no hardcoded port).
- Started after `TelemetryServer` and stopped just before it inside
  `MouseDroidOrchestrator.start()` / `stop()`. Background tasks (sampler + serve loop)
  are tracked in a private `set[asyncio.Task]` and drained via
  `cancel_and_drain` — same pattern as the telemetry server.
- The 30 Hz control loop is **never** gated on MCP I/O: the bridge polls
  `publisher.get_queue()` with `get_nowait()` from a low-cadence sampler and writes into
  a `deque(maxlen=resources.recent_frames_max)`.

**Security gates (all config-driven):**

| Gate | Source of truth | Behaviour |
|------|-----------------|-----------|
| Deny-list | `MCPConfig.tools_denylist` | Highest precedence; `health_check` cannot be denied |
| Allow-list | `MCPConfig.tools_allowlist` | When set, only listed tools are exposed |
| Actuation toggle | `MCPConfig.expose_actuation_tools` + `actuation_tools` list | Side-effecting tools hidden / refused unless explicitly enabled |
| Safety monitor | `SafetyMonitorProtocol.evaluate(...)` | Emergency state refuses every actuation tool |
| Rate limit | `MCPConfig.rate_limit_rps` | Per-session token bucket |
| Circuit breaker | `MCPConfig.circuit_breaker` ∨ root `cfg.circuit_breaker` | Opens on repeated failure |
| Per-call timeout | `MCPConfig.request_timeout_s` | `asyncio.wait_for` around handler |
| Auth | `MCPConfig.auth_token_env_var` | Required for non-loopback transports (rejected at config load) |
| Redaction | `MCPConfig.redact_key_pattern` (regex) | Applied to logs, memory, and config snapshots |

**Coverage:** 99.40% across the module; auth / metrics / prompts / protocol / resources /
tool_bridge at 100%; `server.py` at 97% (the 3 uncovered lines are behind the optional
`mcp` SDK import).

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
    inner: ESP32CommProtocol
    if cfg.mock_hardware:
        from mousedroid.comms.mock_driver import MockESP32Driver
        inner = MockESP32Driver(cfg.esp32)
    elif cfg.esp32.protocol == "serial":
        from mousedroid.comms.serial_driver import SerialESP32Driver
        inner = SerialESP32Driver(cfg.esp32)
    else:
        from mousedroid.comms.wifi_driver import WiFiESP32Driver
        inner = WiFiESP32Driver(cfg.esp32)

    from mousedroid.resilience.resilient_driver import ResilientESP32Driver
    return ResilientESP32Driver(inner, cfg.retry, cfg.circuit_breaker)
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
        +build_cloud_telemetry_sink(cfg) CloudTelemetrySinkProtocol
        +build_cloud_experience_exporter(cfg) CloudExperienceExporterProtocol
        +build_cloud_metrics_exporter(cfg) CloudMetricsExporterProtocol
        +build_cloud_logging_sink(cfg) CloudLoggingSinkProtocol
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
        DR["domain_randomization.py\nPhase 1: per-episode RangeF sampler\nvisual / range / chassis / feature noise"]
        DG["data_generator.py\nSynthetic ObservationBundle sequences\n+ EpisodeParams (when DR enabled)"]
        RSSM_T["train_rssm.py\nPhase 2.1: RSSM encoder + dynamics\ncheckpoints every N epochs"]
        WS["warmstart_policy.py\nPhase 2.2: MCTS warm-start weights\nUCB calibration (tune_ucb)"]
        CA["collect_annotations.py\nPhase 2.3a: BDI intention labels"]
        BDI_T["train_bdi.py\nPhase 2.3b: NeuralBDI on annotations"]
        CRL["train_constitutional_rl.py\nPhase 2.4: PPO + Constitutional RL"]
        RP["run_pipeline.py\nOrchestrator: phase1→2→3→4\n--resume / resume_from for checkpoints"]
        UP["upload_weights.py\nHuggingFace Hub push (28 files)\nianshank/mousedroid-weights"]
    end
    DR -.-> DG
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
| -------- | ----- |
| Pipeline entry | `python training/run_pipeline.py [--resume path/to/checkpoint.pt]` |
| Resume support | `--resume` / `resume_from` in config; forwards checkpoint to RSSM training only |
| Weight storage | `ianshank/mousedroid-weights` HuggingFace repo; `bdi/`, `mcts/`, `rssm/`, `constitutional_rl/` subfolders |
| Runtime download | `WeightsManager.download_weights_from_huggingface(subfolder='bdi', local_dir=weights_dir.parent)` |
| Type safety | All `training/` modules pass `mypy --strict` (NDArray[Any] annotations) |

---

## Physical AI Roadmap (Phase 1 — Domain Randomization)

Following the four-gap analysis from Martin Keen's "What is Physical AI?" (IBM
Technology, 2026-04-13), the offline pretraining pipeline is being extended to
shrink the sim-to-real gap. **Phase 1 is in this branch; Phases 2 → 6 are
roadmap items tracked in [`NEXT_STEPS.md`](../NEXT_STEPS.md).**

### Phase 1 — Domain Randomization (this branch)

```mermaid
graph LR
    subgraph DR["mousedroid.training.domain_randomization"]
        DRConfig["DomainRandomizationConfig\nRangeF[low, high] per parameter\n(visual / camera / range / chassis / comms / feature)"]
        Sampler["DomainRandomizer\nstateless; numpy.random.Generator injected"]
        EP["EpisodeParams\nfrozen dataclass\n(empty when DR disabled)"]
        VT["apply_visual_randomization\nbrightness · contrast · noise\n(uint8/float32 dtype-preserving)"]
        RT["apply_range_sensor_randomization\nadditive noise + dropout (NaN)"]
        FT["apply_feature_noise\npost-CNN feature-vector noise"]
    end
    subgraph Pipeline["training/data_generator.py"]
        SSG["SyntheticSequenceGenerator\nseed-driven master RNG\n→ per-episode rng → ep_params"]
        AppEpRand["_apply_episode_randomization\nidentity when ep_params empty\n(byte-identical legacy contract)"]
        Tensors["torch tensors:\nvision (256-d) + ultrasonic (1-d)\n+ motor_state + valid_mask + action"]
    end

    DRConfig --> Sampler
    Sampler --> EP
    EP --> AppEpRand
    AppEpRand --> Tensors
    SSG --> AppEpRand
    AppEpRand -.-> FT
    AppEpRand -.-> RT
    VT -.-> AppEpRand

    classDef rt fill:#dff,stroke:#36a;
    class DR,Pipeline rt;
```

**Contract guarantees (regression-tested in
`tests/regression/test_domain_randomization_backcompat.py`):**

- `DomainRandomizationConfig.enabled=False` returns empty `EpisodeParams`;
  `_apply_episode_randomization` returns the input dict by **identity**, not
  just by equality. Existing artifacts and golden hashes remain byte-identical.
- All randomization ranges are `RangeF` Pydantic models — zero hardcoded values
  in module bodies; YAMLs override per-environment (production tightens; mock
  widens for stress testing).
- Reproducibility: every public function takes an explicit
  `numpy.random.Generator`, so identical seeds produce identical outputs. The
  generator's master RNG is re-seeded per episode via
  `np.iinfo(np.int64).max` (no magic literals).
- 100% line + branch coverage on the new module; 100% changed-line coverage
  across all touched source files; global coverage holds at 91.68% (gate 85%).

### Roadmap (Phases 2 → 6, tracked in `NEXT_STEPS.md`)

| Phase | Goal | Module(s) |
| ----- | ---- | --------- |
| 2 | Real-episode replay loop — LMDB reader + sim:real mixer feeds back into RSSM and Constitutional-RL training | `training/replay/`, schema-versioned `experience/` records |
| 3a | VLA Protocol + `MockVLA` + factory + orchestrator branch | `mousedroid.vla` |
| 3b | `DistilledVLAOnnx` adapter — TensorRT/ONNX with HF Hub weights pull | `mousedroid.vla.policy`, `[vla]` extra in `pyproject.toml` |
| 4 | VLM-derived dense rewards (VLAC pattern) — pluggable into `MultiObjectiveRewardModel` | `mousedroid.reward.vlm_progress` |
| 5 (stretch) | Real physics simulator (MuJoCo MJX / Isaac Sim Lite) — replace synthetic-sequence generator with ground-truth dynamics | `training/sim/` (new) |
| 6 (stretch) | On-device LoRA-adapter fine-tuning for the VLA from continuously-logged real episodes | `mousedroid.vla.adapters` (new) |

Each phase ships in an isolated PR off the default branch; dependency direction
is strictly Phase 1 → 2 → 3 → 4. Phases 5 and 6 are deferred until Phase 3b has
been in production for ≥30 days.

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
| -------- | --------- |
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
| GCP Digital Twin as optional sidecar | `gcp: null` = fully offline; CircuitBreaker on all cloud calls; droid never depends on cloud for safety |
| msgpack for Pub/Sub payloads | Reuses existing serialisation (ExperienceRecord + TelemetryFrame); 30-60% smaller than JSON for numpy arrays |
| LMDB-to-GCS high-water mark cursor | Exactly-once export without needing Pub/Sub ordering; survives power cycles |
| Cloud config as `GCPConfig \| None` | Single Optional field on Settings; all existing YAML loads unchanged; no migration needed |
