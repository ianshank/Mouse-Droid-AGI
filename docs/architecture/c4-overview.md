# C4 Architecture — MouseDroid

> **This is the canonical C4 diagram index.** For a single-page prose walkthrough of every level, see
> [`../architecture.md`](../architecture.md). Full documentation index: [`../README.md`](../README.md).
>
> Four-level C4 model for the MouseDroid codebase. Diagrams use Mermaid;
> render on GitHub or via VS Code's Markdown preview.
>
> - **Level 1 (Context):** systems, people, external dependencies.
> - **Level 2 (Container):** deployable units (Jetson container, workstation
>   proxy, ESP32 firmware, dashboards).
> - **Level 3 (Component):** internal modules + the protocol interfaces
>   between them (this file links to per-area diagrams).
> - **Level 4 (Code):** class-level — covered by Sphinx-style docstrings in
>   each module; not duplicated here.
>
> See also: `c4-dashboard-proxy.md`, `c4-orchestrator.md`, `c4-arm-platform.md`.

---

## Level 1 — System Context

```mermaid
C4Context
title MouseDroid — System Context

Person(operator, "Operator", "Solo developer — pushes code, opens dashboard,\nteaches the rover via voice + mission YAML.")
Person(passenger, "Passenger / Observer", "Reads camera + LiDAR through the dashboard.")

System(mousedroid, "MouseDroid", "Star Wars MSE-6 autonomous navigation.\nParked: hierarchical robot-arm training platform.")

System_Ext(jetson, "NVIDIA Jetson Orin Nano", "Edge inference + sensor I/O.\nRuns the dockerised orchestrator.")
System_Ext(esp32, "ESP32 motor controller", "Wave Rover differential drive\n+ encoder feedback over serial.")
System_Ext(hailo, "Hailo-8 NPU", "M.2 accelerator for YOLO\nfeature extraction.")
System_Ext(cloud, "Cloud weights bucket", "Periodic OTA model updates.")
System_Ext(hf, "HuggingFace Hub", "Model + dataset registry\n(ianshank/* repos).")
System_Ext(wandb, "Weights & Biases", "Experiment tracking for\ntraining runs.")
System_Ext(anthropic, "Anthropic Claude API", "Cloud deliberative brain —\ntranslates NL missions to a GoalVector.\nLocal Phi-3 fallback when off-network.")

Rel(operator, mousedroid, "Edits YAML, dispatches missions,\nbrowses dashboard")
Rel(passenger, mousedroid, "Watches live camera + LiDAR")
Rel(mousedroid, jetson, "Runs on")
Rel(jetson, esp32, "Serial / UART")
Rel(jetson, hailo, "PCIe M.2")
Rel(mousedroid, cloud, "Weight OTA")
Rel(mousedroid, hf, "Model pull / push")
Rel(mousedroid, wandb, "Metrics + run telemetry")
Rel(mousedroid, anthropic, "NL mission translation\n(HTTPS, post sanitize, OUT of 30 Hz loop)")
```

> The **Anthropic Claude API** is the *deliberative* tier only: it
> turns a natural-language mission into a normalised `GoalVector`. It is
> deliberately outside the 30 Hz reactive control loop, which stays
> LLM-free and deterministic. When the rover is off-network, a local
> Phi-3-mini (llama_cpp) fallback serves the same translation. See
> Level 2 and [`c4-llm-gateway.md`](./c4-llm-gateway.md) for detail.

---

## Level 2 — Container

```mermaid
C4Container
title MouseDroid — Container Diagram

Person(operator, "Operator", "Solo developer")

System_Boundary(workstation, "Workstation (Windows / macOS)") {
    Container(browser, "Browser", "Chrome / Edge", "Renders the dashboards.")
    Container(proxy, "Dashboard Proxy", "Python 3.11 + aiohttp", "Bridges the auth-gated Jetson telemetry\nto the browser. Injects bearer token,\nstrips RFC-9110 hop-by-hop headers,\nforwards HTTP + WebSocket + MJPEG.")
    Container(claude, "Claude Code", "CLI", "Edits source, runs tests, commits.")
}

System_Boundary(jetson, "Jetson Orin Nano (Docker)") {
    Container(orchestrator, "Orchestrator", "Python 3.11 + asyncio", "30 Hz sense-plan-act loop.\nFactory-wired modules behind protocols.\nLLM-FREE hot path: sensors -> RSSM -> MCTS -> ESP32.")
    Container(llmgw, "Deliberative LLM Gateway", "FallbackLLMGateway (asyncio)", "OUTSIDE the 30 Hz loop.\nNL mission -> normalised GoalVector\n(vx,vy,omega in [-1,1]) -> process_mission.\nPrimary: cloud Claude. Fallback: local Phi-3 (llama_cpp).")
    Container(telemetry, "Telemetry Server", "aiohttp REST + WS", "Bearer-token-auth API:\n/api/v1/*, /camera/*, /lidar, /ws.")
    Container(grafana, "Grafana", "OSS dashboard", "Prometheus-backed time-series.")
    Container(prom, "Prometheus", "TSDB", "Scrapes orchestrator metrics.")
}

System_Ext(esp32, "ESP32 firmware", "Embedded C")
System_Ext(hailo, "Hailo HailoRT", "NPU runtime")
System_Ext(anthropic, "Anthropic Claude API", "Cloud Messages API\napi.anthropic.com (HTTPS)")

Rel(operator, browser, "Browses dashboards")
Rel(operator, claude, "Edits code")
Rel(browser, proxy, "HTTP/1.1 + WS\n(127.0.0.1:8081)", "loopback")
Rel(proxy, telemetry, "HTTP/1.1 + WS + Bearer", "192.168.55.1:8080")
Rel(browser, grafana, "via proxy\n127.0.0.1:8082")
Rel(browser, prom, "via proxy\n127.0.0.1:8083")
Rel(operator, llmgw, "Dispatches NL mission\n(via process_mission)")
Rel(llmgw, orchestrator, "GoalVector\n(out-of-loop, then loop consumes it)")
Rel(llmgw, anthropic, "messages.create()\nHTTPS, post prompt-injection sanitize")
Rel(orchestrator, telemetry, "in-process")
Rel(orchestrator, esp32, "Serial 1 Mbps")
Rel(orchestrator, hailo, "PCIe")
Rel(prom, orchestrator, "scrape /metrics")
Rel(grafana, prom, "PromQL")
```

> **Loop boundary.** The Deliberative LLM Gateway runs *beside* the
> orchestrator, not inside its 30 Hz tick. NL → `GoalVector` translation
> happens once per mission (or on cloud→local failover) and hands the
> goal to `process_mission`; the deterministic hot path (sensors → RSSM
> world model → MCTS → ESP32 velocity command) never calls an LLM.
> Component-level wiring — the `build_llm_gateway` dispatch chain,
> `FallbackLLMGateway` failover state machine, and the cloud-egress
> security boundary — lives in [`c4-llm-gateway.md`](./c4-llm-gateway.md).

---

## Level 3 — Component (high level — see per-area files for detail)

Each subsystem owns a Component diagram in its own file:

| Subsystem | Component diagram | Owner test surface |
|-----------|-------------------|--------------------|
| Dashboard proxy + telemetry bridge | [`c4-dashboard-proxy.md`](./c4-dashboard-proxy.md) | `tests/unit/tools/test_dashboard_proxy.py`, `tests/e2e/test_pr104_dashboard_e2e.py` |
| 30 Hz orchestrator loop | [`c4-orchestrator.md`](./c4-orchestrator.md) | `tests/integration/test_sense_plan_act.py` |
| Robot-arm training platform | [`c4-arm-platform.md`](./c4-arm-platform.md) | `tests/unit/arm/*` |
| USB-C smoke validation gate (PR #106) | [`c4-usbc-smoke.md`](./c4-usbc-smoke.md) | `tests/unit/diagnostics/test_usbc.py`, `tests/unit/diagnostics/test_power_chain.py`, `tests/unit/test_factory_esp32_discovery.py`, `tests/hardware/test_usbc_enumeration.py`, `tests/hardware/test_power_chain_smoke.py` |
| LLM gateway + cloud/local failover (PR #107) | [`c4-llm-gateway.md`](./c4-llm-gateway.md) | `tests/unit/llm_gateway/test_anthropic_gateway.py`, `tests/unit/llm_gateway/test_fallback_gateway.py`, `tests/unit/config/test_llm_config_anthropic_fallback.py`, `tests/unit/factory/test_build_llm_gateway_dispatch.py`, `tests/integration/test_anthropic_gateway_wiring.py` |
| Validation efficiency: latency stats · trend store · summary renderer · trend timer · phase caching (PR #126 + F-018) | [`c4-validation-efficiency.md`](./c4-validation-efficiency.md) | `tests/unit/validation/test_latency_stats.py`, `tests/unit/validation/test_report_store.py`, `tests/unit/validation/test_summary.py`, `tests/unit/scripts/test_render_validation_summary.py`, `tests/regression/test_trend_timer_units.py`, `tests/unit/validation/test_init_lazy_exports.py`, `tests/unit/cli/test_preflight_cli.py`, `tests/integration/test_validation_report_store_integration.py`, `tests/regression/test_validation_import_decoupling.py`, `tests/smoke/test_jetson_full_validation_sanity.py` |
| Spec-driven harness: feature DAG + validate.py runner (ADR-012) | [`c4-spec-harness.md`](./c4-spec-harness.md) | `tests/unit/harness/test_spec.py`, `tests/regression/test_harness_spec_aqa.py`, `tests/regression/test_harness_cli_contract.py` |
| Claude Code workforce governance: edit-time secret scan · capability freeze gate · post-edit checks (F-024) | [`c4-claude-workforce.md`](./c4-claude-workforce.md) | `tests/unit/tools/claude_hooks/*`, `tests/regression/test_claude_workforce_aqa.py` |

The component-level diagrams are the right entry point when you need to
understand *which protocol object talks to which* — they map every
`build_*` factory in `src/mousedroid/factory.py` to its protocol +
implementations.

---

## Mapping the diagrams to source

| C4 element | Source path |
|------------|-------------|
| Orchestrator container | `src/mousedroid/orchestrator/` |
| Deliberative LLM Gateway | `src/mousedroid/llm_gateway/` (composite: `fallback_gateway.py`) |
| Telemetry Server | `src/mousedroid/telemetry/` |
| Dashboard Proxy | `tools/dashboard_proxy.py` |
| Claude workforce hooks + config | `tools/claude_hooks/`, `.claude/workforce.yaml`, `.claude/settings.json` |
| Factory wiring | `src/mousedroid/factory.py` |
| Configuration schema | `src/mousedroid/config/schema.py` |
| Hardware protocols | `src/mousedroid/hardware/protocols.py`, `src/mousedroid/comms/protocol.py` |
| ESP32 firmware | (out of repo — vendored Wave Rover bootloader) |
| Hailo NPU runtime | `src/mousedroid/hardware/accelerators/hailo_runtime.py` |

---

## Deployment + CI gates

The Jetson container is the only deployed runtime. Its provenance and the
guard rails that keep config + workflows from drifting away from it:

| Concern | Where | Notes |
|---------|-------|-------|
| Deployed image record | `deployments/jetson-image.json` | Source-of-truth SHA for what the rover runs. The PR #107 LLM tier (anthropic SDK + `LLMConfig`) is baked here; the rover bind-mounts editable source over it but the baked SDK survives `--force-recreate`. Bump this whenever the image is rebuilt or the tracked source SHA changes. |
| `config-compat` schema-drift gate | `.github/workflows/config-compat.yml`, `scripts/check_config_compat.py` | On any `config/*.yaml` (or `deployments/*.json`) change, worktrees out the deployed SHA and validates the changed YAML against *that* schema — catches the "yaml-only PR merges, rover crash-loops with `Extra inputs are not permitted`" class. |
| Edit-time workforce hooks (F-024) | `.claude/settings.json` hooks block, `tools/claude_hooks/` | PreToolUse secret scan + capability freeze gate (blocking), PostToolUse checks (advisory). Config: `.claude/workforce.yaml`. See [`c4-claude-workforce.md`](./c4-claude-workforce.md) |
| `actionlint` workflow lint | `.github/workflows/ci.yml` (Stage 0) | Lints every workflow file so an invalid workflow (e.g. an empty `${{ }}` expression) can't silently startup-fail and disable a gate — the exact failure that killed `config-compat` repo-wide before PR #113. |

The deliberative LLM tier is deployed via this image record; per-host
secrets (`ANTHROPIC_API_KEY`) and the CPU-fallback toggle
(`MOUSEDROID_LLM__N_GPU_LAYERS=0`) live in the rover's uncommitted
`docker.env`, never in the image or repo.

---

## Update discipline

When any of these change:

- Add or remove a top-level subsystem under `src/mousedroid/`.
- Add or remove a protocol interface.
- Add or remove an external dependency (HuggingFace, W&B, Hailo,
  Anthropic Claude API, etc.).
- Change the deployment topology (e.g. split telemetry out of the
  orchestrator container).
- Rebuild the Jetson image or repoint the tracked source SHA (keep
  `deployments/jetson-image.json` and the Deployment + CI gates table
  in sync).

…update the relevant C4 diagram in the same PR. Reviewers should bounce
PRs that introduce a new container without a matching diagram update.
