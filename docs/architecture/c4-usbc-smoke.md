# C4 Component — USB-C Smoke Validation Gate (PR #106)

> The Jetson-side validation chain that confirms the Wave Rover is wired
> right and reachable before the orchestrator opens the serial port.
> Added in PR #106 to handle two operational realities: (1) rovers are
> swapped between bench units (each CP2102N chip has a unique by-id
> serial, so a literal `esp32.serial_port` path breaks the moment a
> different rover is connected); (2) udev sometimes hasn't mounted
> `/dev/serial/by-id/*` when an early-boot probe fires.
>
> Companion to `docs/architecture/c4-overview.md` (Levels 1–2) and
> `docs/runbooks/jetson-rover-smoke.md` (operator workflow).

## Component Diagram

```mermaid
flowchart TB
    subgraph External["External actors"]
        Operator(["Operator / CI"])
        Filesystem[("/dev/serial/by-id\n(udev symlink forest)")]
    end

    subgraph CI["CI surface (.github/workflows/ci.yml)"]
        UsbcGate["usbc-config-gate job\nruns test_jetson_production_overlay.py"]
        Regression["tests/unit/test_jetson_production_overlay.py\n• YAML declares rover_esp32 + lidar_ld19\n• glob matches esp32.serial_port chip family\n• default.yaml stays disabled"]
    end

    subgraph SmokeWrapper["Smoke wrapper (bash)"]
        FullRun["scripts/jetson_full_smoke_run.sh"]
        SmokeStage["scripts/jetson_smoke_test.sh\nstages: system / usbc / gpio /\nserial / motor / power / lidar / ..."]
    end

    subgraph StandaloneProbe["Standalone operator probe"]
        CheckScript["scripts/check_usbc_devices.py\n--config <yaml>  [--json]"]
    end

    subgraph CLI["validation CLIs (src/mousedroid/cli/)"]
        PreflightCLI["preflight.py\nasyncio.run(run_preflight(cfg))"]
        PillarsCLI["validate_pillars.py\nasyncio.run(validate_all_pillars(cfg))"]
    end

    subgraph Dispatcher["validation/ dispatchers"]
        Preflight["validation/preflight.py\n_CHECK_DISPATCH\n{camera, microphone, speaker,\n lidar, esp32, config}"]
        Pillars["validation/pillars.py\n_PILLAR_DISPATCH\n+ _PYTEST_DELEGATION_PATHS"]
    end

    subgraph Factory["src/mousedroid/factory.py"]
        BuildEsp32["build_esp32_driver(cfg)"]
        Resolver["_resolve_esp32_serial_via_usbc_discovery(cfg)"]
    end

    subgraph Diagnostics["src/mousedroid/diagnostics/"]
        UsbcHelper["usbc.py\nresolve_endpoint(cfg, name) -> Path | None\nenumerate_usbc_devices(cfg) -> dict\n(boot-race guard: is_dir() before glob)"]
        PowerChain["power_chain.py\nassert_power_chain(driver, cfg, allow_motion)\n-> PowerChainResult"]
    end

    subgraph Config["src/mousedroid/config/schema.py"]
        UsbcCfg["USBCDiscoveryConfig\nenabled / by_id_root /\nrequired_endpoints"]
        EsP32Cfg["ESP32Config\nserial_port / serial_baud /\ncommand_set (legacy | waveshare_stock) /\nsmoke_test_velocity_mps (ge=0) /\nemergency_stop_budget_ms"]
        SettingsRoot["Settings.usbc_discovery: Optional\n(None default → backwards compat)"]
    end

    subgraph Drivers["src/mousedroid/comms/"]
        SerialDriver["SerialESP32Driver\n(opens overridden serial_port)"]
        ResilientWrapper["ResilientESP32Driver\ncircuit breaker + retry"]
    end

    %% Operator paths
    Operator -- "python -m mousedroid.cli.preflight" --> PreflightCLI
    Operator -- "bash scripts/jetson_full_smoke_run.sh" --> FullRun
    Operator -- "python scripts/check_usbc_devices.py" --> CheckScript

    %% Smoke wrapper internal flow
    FullRun --> SmokeStage
    SmokeStage -- "stage: usbc" --> CheckScript
    SmokeStage -- "stage: power" --> PowerChain
    SmokeStage -- "stage: app_health" --> PreflightCLI

    %% CLI -> dispatcher
    PreflightCLI --> Preflight
    PillarsCLI --> Pillars

    %% Dispatcher -> esp32 check -> factory
    Preflight -- "_CHECK_DISPATCH['esp32']\n_check_esp32(cfg)" --> BuildEsp32

    %% Standalone probe path
    CheckScript --> UsbcHelper

    %% Factory wiring chain
    BuildEsp32 --> Resolver
    Resolver -- "resolve_endpoint(cfg, 'rover_esp32')" --> UsbcHelper
    UsbcHelper -- "Path.glob(spec.by_id_glob)" --> Filesystem
    UsbcHelper -. "Path | None" .-> Resolver
    Resolver -. "ESP32Config\n(possibly model_copy'd with new serial_port)" .-> BuildEsp32
    BuildEsp32 -- "SerialESP32Driver(esp32_cfg)" --> SerialDriver
    BuildEsp32 -- "ResilientESP32Driver(inner, retry, breaker)" --> ResilientWrapper
    SerialDriver --> ResilientWrapper

    %% Power chain
    PowerChain -- "send_velocity / emergency_stop / get_battery_voltage" --> ResilientWrapper

    %% Config feeds
    SettingsRoot --> UsbcCfg
    UsbcCfg -. "cfg.usbc_discovery" .-> Resolver
    UsbcCfg -. "cfg.usbc_discovery" .-> UsbcHelper
    EsP32Cfg -. "cfg.esp32" .-> BuildEsp32
    EsP32Cfg -. "esp32_cfg" .-> PowerChain

    %% CI regression-gate side path
    UsbcGate --> Regression
    Regression -. "load YAML + assert\nglob vs. serial_port alignment" .-> SettingsRoot

    classDef external fill:#fef3c7,stroke:#f59e0b,color:#000
    classDef ci fill:#dbeafe,stroke:#3b82f6,color:#000
    classDef internal fill:#e0f2fe,stroke:#0284c7,color:#000
    classDef config fill:#f3e8ff,stroke:#9333ea,color:#000
    classDef driver fill:#dcfce7,stroke:#16a34a,color:#000

    class Operator,Filesystem external
    class UsbcGate,Regression ci
    class UsbcHelper,PowerChain internal
    class UsbcCfg,EsP32Cfg,SettingsRoot config
    class SerialDriver,ResilientWrapper driver
```

## Resolution chain — `ESP32Config.serial_port`

Walked by `_resolve_esp32_serial_via_usbc_discovery` in
`src/mousedroid/factory.py`. Falls through each guard; the first
condition that holds wins.

| # | Condition | Outcome |
|---|-----------|---------|
| 1 | `cfg.usbc_discovery is None` OR `not cfg.usbc_discovery.enabled` | Return `cfg.esp32` unchanged. Pre-PR YAMLs hit this branch. |
| 2 | `Path(cfg.esp32.serial_port).exists()` | Return `cfg.esp32` unchanged. A pinned operator override is never silently shadowed. |
| 3 | `resolve_endpoint(cfg.usbc_discovery, "rover_esp32") is None` | Log `esp32_serial_port_unresolved` WARN, return `cfg.esp32` unchanged. Driver opens the (stale) literal path and surfaces a clean errno. |
| 4 | Glob matches | `cfg.esp32.model_copy(update={"serial_port": str(resolved)})`. Log `esp32_serial_port_overridden` (with `literal` + `resolved` keys). `cfg.esp32` is NOT mutated (Pydantic model_copy). |

## Boot-race guard — `Path.glob` on missing root

`Path.glob()` on a directory that does not exist raises
`FileNotFoundError` in Python 3.10/3.11 (not an empty iterator). On a
fresh Jetson before `udevd` mounts the `/dev/serial/by-id/*` symlink
forest, a pre-PR call would crash the smoke harness with an uncaught
`OSError`. Both `enumerate_usbc_devices` and `resolve_endpoint` guard
with `if not cfg.by_id_root.is_dir()` and return structured `MISSING` /
`None` so the harness sees a clean FAIL list.

## Failure-mode matrix

| Symptom | Where it shows up | Resolver branch | Operator fix |
|---|---|---|---|
| `usbc` stage FAIL during smoke | `usbc.log` has `usbc_endpoint_missing` | (probe runs before factory) | Reseat the USB-C cable; confirm the rover-side port is the data port, not power-only. |
| Container starts but rover doesn't move | Orchestrator log has `esp32_serial_port_unresolved` | Branch 3 | Set `MOUSEDROID_USBC_DISCOVERY__REQUIRED_ENDPOINTS__0__BY_ID_GLOB=...` to a glob that matches the live chip, OR update YAML literal. |
| Rover-swap broke comms; literal path is stale | Orchestrator log has `esp32_serial_port_overridden` | Branch 4 (override fired) | Nothing to fix — the override re-pointed at the live chip. Confirm via the structured `resolved=` log field. |
| Pre-udev boot; no symlinks yet | Smoke log has `usbc_by_id_root_missing` | (probe sees structured MISSING) | Wait for udev settle; smoke wrapper retries on the next blocking-stage pass. Long-term: add a systemd `Wants=udev-settle.service` to the smoke runner unit. |
| YAML drift between glob and `esp32.serial_port` chip family | CI `usbc-config-gate` fails | (regression test, pre-merge) | Update either the glob or the literal so both reference the same CP210x family. |

## Related diagrams

- `docs/architecture/c4-overview.md` — Levels 1 (Context) and 2
  (Container) for the whole system.
- `docs/architecture/c4-orchestrator.md` — the 30 Hz sense-plan-act
  loop that consumes the (possibly overridden) `SerialESP32Driver`.
- `docs/architecture/c4-dashboard-proxy.md` — the workstation-side
  reverse proxy used to verify dashboard endpoints once the rover is
  up and the orchestrator is serving telemetry.
