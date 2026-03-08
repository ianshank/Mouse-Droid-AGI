# MouseDroidAGI 🤖

**A Star Wars MSE-6 "Mouse Droid" autonomous navigation system powered by an Agentic World Model on NVIDIA Jetson Orin Nano.**

[![Tests](https://img.shields.io/badge/tests-530%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-98.6%25-brightgreen)]()
[![Ruff](https://img.shields.io/badge/lint-ruff%20clean-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()

---

## Overview

MouseDroidAGI implements the **10 Pillars of the Ideal Neural Network** as a cohesive agentic system that enables a physical MSE-6 droid replica to navigate autonomously, avoid obstacles, follow natural language commands, and continuously improve from experience.

The robot is built on a Wave Rover mecanum-wheel chassis, controlled by an ESP32 microcontroller, and powered by a Raspberry Pi AI Camera (IMX500) with onboard neural inference. All high-level reasoning runs on a Jetson Orin Nano.

### The 10 Pillars

| Pillar | Module | Description |
|--------|--------|-------------|
| 1. World Model | `world_model/` | RSSM latent dynamics + MCTS planning |
| 2. Cognitive Architecture | `cognitive/` | Dual-cadence BDI + metacognitive loop |
| 3. Memory Systems | `memory/` | Working, episodic, semantic, consolidation |
| 4. Continual Learning | `learning/` | EWC + progressive neural networks |
| 5. Meta-Learning | `meta/` | MAML + in-context adaptation |
| 6. Curiosity & Exploration | `curiosity/` | ICM intrinsic curiosity |
| 7. Growth & Distillation | `growth/` | Knowledge distillation to smaller models |
| 8. Reward Modelling | `reward/` | Constitutional multi-objective reward |
| 9. Scaling | `scaling/` | Mixture-of-Experts + adaptive compute |
| 10. Safety & Alignment | `safety/` | Constitutional RL + runtime safety monitor |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Jetson Orin Nano                   │
│                                                     │
│  ┌───────────┐  ┌────────────┐  ┌────────────────┐  │
│  │  IMX500   │  │  HC-SR04   │  │  Orchestrator  │  │
│  │  Camera   │  │ Ultrasonic │  │  (30 Hz loop)  │  │
│  └─────┬─────┘  └─────┬──────┘  └───────┬────────┘  │
│        │              │                  │           │
│        └──────────────┴──────────────────┘           │
│                       │                              │
│              ┌────────▼───────┐                      │
│              │  World Model   │                      │
│              │  (RSSM+MCTS)   │                      │
│              └────────┬───────┘                      │
│                       │                              │
│         ┌─────────────┼────────────┐                 │
│    ┌────▼────┐  ┌─────▼────┐  ┌───▼──────┐          │
│    │ Safety  │  │Cognitive │  │  Memory  │          │
│    │Monitor  │  │  Core    │  │ Systems  │          │
│    └─────────┘  └──────────┘  └──────────┘          │
└─────────────────────────┬───────────────────────────┘
                          │ UART / WiFi
                    ┌─────▼──────┐
                    │   ESP32    │
                    │ (Wave Rover│
                    │  motors)   │
                    └────────────┘
```

See [docs/architecture.md](docs/architecture.md) for full C4 diagrams.

---

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA Jetson Orin Nano (or any Linux machine for mock mode)
- Wave Rover chassis with ESP32 controller
- Raspberry Pi AI Camera IMX500 (optional — mock available)

### Installation

```bash
# Base install
pip install -e .

# With hardware drivers
pip install -e ".[hardware]"

# With TensorRT acceleration
pip install -e ".[hardware,jetson]"

# With local LLM
pip install -e ".[llm]"

# Development
pip install -e ".[dev]"
```

### Run in Mock Mode (no hardware required)

```bash
mousedroid --mock-hardware
```

### Health Check

```bash
mousedroid --health-check --config config/default.yaml
```

### Run with Custom Config

```bash
mousedroid --config config/default.yaml config/jetson_production.yaml
```

---

## Configuration

All settings are defined in `config/default.yaml` and validated by Pydantic. Override with additional YAML files:

| File | Purpose |
|------|---------|
| `config/default.yaml` | All defaults |
| `config/jetson_production.yaml` | Jetson Orin Nano production overrides |
| `config/mock_hardware.yaml` | Mock hardware for CI/development |

No values are hardcoded — every threshold, dimension, pin, and rate is configurable.

---

## Project Structure

```
src/mousedroid/
├── agents/           # Navigation agents (protocol + navigation impl)
├── cognitive/        # BDI model, metacognition, constitutional RL
├── comms/            # ESP32 drivers: serial, WiFi, mock
├── config/           # Pydantic settings schema + YAML loader
├── curiosity/        # Intrinsic Curiosity Module (ICM)
├── efficiency/       # TensorRT compilation, runtime profiler
├── experience/       # LMDB experience logger + record format
├── factory.py        # Dependency injection factory functions
├── growth/           # Knowledge distillation
├── hardware/         # Camera, ultrasonic, motor drivers + protocols
├── health/           # Jetson health monitor (GPU temp/load)
├── learning/         # EWC + progressive neural networks
├── llm_gateway/      # Natural language → velocity via local LLM
├── logging/          # structlog setup
├── main.py           # CLI entry point
├── memory/           # Working, episodic, semantic memory + consolidation
├── meta/             # MAML + in-context meta-learning
├── orchestrator/     # Main sense-plan-act loop
├── reward/           # Multi-objective reward model
├── safety/           # Safety context + constitutional monitor
├── scaling/          # MoE routing + adaptive compute
├── sensing/          # Sensor manager + observation bundle
├── tools/            # Tool registry for agentic tool dispatch
└── world_model/      # RSSM encoder, MCTS planner
```

---

## Design Principles

### Protocol-Based Dependency Injection

All components are defined as `@runtime_checkable Protocol` interfaces. Factory functions in `factory.py` are the only place that imports concrete types:

```python
# Protocols define the contract
class ESP32CommProtocol(Protocol):
    async def send_velocity(self, vx: float, vy: float, omega: float) -> None: ...

# Factory selects the implementation
def build_esp32_driver(cfg: Settings) -> ESP32CommProtocol:
    if cfg.mock_hardware:
        return MockESP32Driver(cfg.esp32)
    if cfg.esp32.protocol == "serial":
        return SerialESP32Driver(cfg.esp32)
    return WiFiESP32Driver(cfg.esp32)
```

### Asyncio Throughout

All I/O is async. Blocking calls (serial, file, HTTP) run via `asyncio.to_thread()`. No threading primitives are used.

### Structured Logging

All logging uses `structlog` with machine-readable JSON output in production and coloured console output in development.

```python
_log = get_logger(__name__)
_log.info("orchestrator_started", platform=cfg.platform)
```

### Zero Hardcoded Values

Every number — GPIO pins, network ports, model dimensions, safety thresholds — comes from the Pydantic config schema. Module-level constants document internal algorithm parameters.

---

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src/mousedroid --cov-report=term-missing

# Fast parallel run
pytest tests/ -n auto

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/
```

### Test Statistics

| Category | Count |
|----------|-------|
| Unit tests | ~500 |
| Integration tests | 17 |
| **Total** | **530** |
| **Coverage** | **98.6%** |

Hardware tests (requiring real GPIO/camera) are marked `@pytest.mark.hardware` and skipped in CI.

---

## Linting & Type Checking

```bash
# Lint
ruff check src/ tests/

# Format check
ruff format --check src/ tests/

# Type check
mypy src/ --strict --ignore-missing-imports
```

---

## Deployment

### Flash ESP32

```bash
scripts/flash_esp32.sh
```

### Deploy to Jetson

```bash
scripts/deploy_jetson.sh <jetson-ip>
```

### Systemd Service

```bash
cp scripts/mousedroid.service /etc/systemd/system/
systemctl enable mousedroid
systemctl start mousedroid
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Follow the coding standards (ruff, mypy strict, Google docstrings)
4. Write tests — coverage must remain ≥85%
5. Open a pull request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Hardware BOM

| Component | Part | Notes |
|-----------|------|-------|
| SBC | NVIDIA Jetson Orin Nano 8GB | Primary compute |
| Chassis | Waveshare Wave Rover | Mecanum wheel, ESP32 onboard |
| Camera | Raspberry Pi AI Camera (IMX500) | Onboard ML inference |
| Distance | HC-SR04 Ultrasonic | GPIO pins 23/24 |
| Battery | 3S LiPo 11.1V | Min cutoff 9.5V |
