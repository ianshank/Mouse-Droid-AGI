# SKILLS.md — Skill index for MouseDroid

> Maps high-level capability to the *files an agent should read* + *commands
> an agent should run* to exercise that capability. Companion to
> `AGENTS.md` (rules) and `CLAUDE.md` (project facts); all three are
> subordinate to `docs/CHARTER.md` (the project constitution).

Skills are organised by **trigger**: when you see the trigger, invoke the
skill. If no project-local skill matches, fall back to the
`plugin:superpowers:*` family or the Anthropic skill index.

---

## Operational skills (operator-facing)

### large-artifact-handling

**Trigger:** "the clone is huge / bloated", "regenerate `bdi_annotations.npz`",
"get the CAD / STL files", "purge the big files from git history", "why is
`.git` so large".

**Read:**
- `training/data/README.md` + `docs/3D_printing_files/README.md` — where the blobs live now
- `scripts/fetch_data.sh` — regenerate the `.npz` (or `--from-hf` mirror)
- `scripts/purge_history.sh` + `docs/runbooks/history-purge.md` — the operator-run history purge
- `docs/architecture/c4-artifact-storage.md` — C4 diagram

**Run:**
```bash
# Regenerate the training annotations (default), or pull the HF mirror
bash scripts/fetch_data.sh
bash scripts/fetch_data.sh --from-hf

# History purge — DRY RUN first (clone -> purge -> verify; NO push), then --push
bash scripts/purge_history.sh
bash scripts/purge_history.sh --push
```

**Guardrails:** the purge is destructive + irreversible (rewrites every commit
SHA). Preserve the blobs externally first (HF dataset + `hardware-v6` Release),
and never commit the demo video or a regenerated `.npz` — they are gitignored
for a reason. Contracts pinned by `tests/regression/test_portfolio_reframe_aqa.py`.

### dashboard-proxy

**Trigger:** "open the dashboard", "browse the rover from my workstation",
"the Jetson telemetry server is auth-gated".

**Read:**
- `tools/dashboard_proxy.py` — the proxy itself
- `scripts/launch_dashboard.ps1` — Windows / PowerShell launcher
- `config/dev_dashboard.yaml.example` — overlay template
- `docs/architecture/c4-dashboard-proxy.md` — C4 diagrams

**Run:**
```bash
# CLI form
python tools/dashboard_proxy.py 8081 http://192.168.55.1:8080 "$JETSON_TOKEN"
# Env form
PROXY_PORT=8081 JETSON_HTTP=http://192.168.55.1:8080 JETSON_TOKEN=... \
    python tools/dashboard_proxy.py
```

Then open `http://127.0.0.1:8081/lidar` in a browser; the proxy forwards
the auth header + handles MJPEG + WebSockets transparently.

### rover-dashboard

**Trigger:** "show me the rover dashboard", "camera + lidar + sensor fusion in
one view", "access the rover from my phone", "what's the `/dashboard` page".

**Read:**
- `src/mousedroid/telemetry/static/dashboard.html` — the unified page (single
  `/ws` feed → camera MJPEG + lidar polar + fusion panel + status).
- `src/mousedroid/telemetry/server/` — `_handle_root` (`/`→302) +
  `_handle_dashboard_page` (`/dashboard`).
- `src/mousedroid/telemetry/frame_builder.py` — `_build_fused_summary` (the
  `fused` field; handles the length-4 / length-5 `valid_mask`).
- `docs/runbooks/jetson-full-bringup.md` — deploy + WiFi access flow.

**Run:**
```bash
# From any device on the WiFi (token-gated):
open "http://<rover-ip>:8080/?token=$MOUSEDROID_TELEMETRY_TOKEN"
# or via mDNS: http://mousedroid-telemetry.local:8080/?token=...
# Workstation, through the proxy:
python tools/dashboard_proxy.py 8081 http://<rover-ip>:8080 "$JETSON_TOKEN"
# then open http://127.0.0.1:8081/
```

The fusion panel reads `TelemetryFrame.fused` (`n_valid`/`n_modalities`/
`lidar_present`/`modalities`/`fused_norm`) + the three-state `sensor_liveness`.
`fused` is a pure summary of the existing observation — not a new fusion model.

### live-camera-verification

**Trigger:** "the camera feed is solid green", "no JPEG appearing on
`/camera/frame.jpg`", "IMX708 / Bayer / RG10".

**Read:**
- `src/mousedroid/hardware/camera/jetson_csi.py` — focus on
  `capture_raw_jpeg` + `_frame_to_rgb_for_snapshot`
- `src/mousedroid/config/schema/` — `CameraConfig.v4l2_grayscale_extract`
  field docstring (background on the IMX708 Bayer-misinterpretation)
- `tests/unit/test_jetson_csi.py` — the test surface that pins the per-
  backend colour-conversion paths.

**Run:**
```bash
# Live JPEG fetch through the dashboard proxy (workstation)
curl -o /tmp/snap.jpg http://127.0.0.1:8081/camera/frame.jpg
# Offline snapshot via the verify script (Jetson or mock workstation)
MOUSEDROID_MOCK_HARDWARE=true python scripts/verify_sensors.py \
    --config config/default.yaml --sensor camera --frames 3 \
    --save-frame /tmp/snap.jpg
```

### esp32-disconnected-mode

**Trigger:** "the rover won't boot — orchestrator crashes on connect",
"no ESP32 plugged in", "running camera + LiDAR without motors".

**Read:**
- `src/mousedroid/factory.py:90-127` — the `build_esp32_driver` branch.
- `src/mousedroid/config/schema/` — `ESP32Config.enabled` docstring.
- `tests/integration/test_pr104_esp32_disabled_integration.py` —
  reference tests.

**Run:**
```bash
# YAML override
echo "esp32:\n  enabled: false" >> /etc/mousedroid/jetson_production.yaml
# Or env override (docker.env)
MOUSEDROID_ESP32__ENABLED=false python -m mousedroid.main \
    --config /etc/mousedroid/jetson_production.yaml
```

### preflight-validation

**Trigger:** "smoke test the rover", "is hardware healthy before mission?".

**Read:**
- `src/mousedroid/validation/preflight.py` — hardware probe dispatcher.
- `src/mousedroid/validation/pillars.py` — 10-pillar validator.
- `src/mousedroid/cli/preflight.py` + `cli/validate_pillars.py` — argparse
  wrappers (exit-code contract: 0=OK/DEGRADED, 1=FAIL).

**Run:**
```bash
python -m mousedroid.cli.preflight --config config/default.yaml
python -m mousedroid.cli.validate_pillars --config config/default.yaml
```

### usbc-smoke-validation

**Trigger:** "validate USB-C wiring", "rover swap broke the serial path",
"check by-id endpoints", "smoke gate failing at usbc stage".

**Read:**
- `src/mousedroid/diagnostics/usbc.py` — pure helper:
  `enumerate_usbc_devices` + `resolve_endpoint`.
- `src/mousedroid/config/schema/` — `USBCDiscoveryConfig`,
  `USBCEndpointSpec`, and the `Settings.usbc_discovery` field.
- `src/mousedroid/factory.py:_resolve_esp32_serial_via_usbc_discovery` —
  two-condition override (only fires when discovery enabled AND literal
  path missing).
- `scripts/check_usbc_devices.py` — standalone operator probe.
- `tests/unit/diagnostics/test_usbc.py` + `tests/unit/test_factory_esp32_discovery.py`
  — unit coverage including boot-race missing-`by_id_root` guard.
- `tests/unit/test_jetson_production_overlay.py` — CI regression
  invariant: YAML glob must match `esp32.serial_port` chip family.
- `docs/runbooks/jetson-rover-smoke.md` — operator workflow.
- `docs/architecture/c4-usbc-smoke.md` — C4 component diagram.

**Run:**
```bash
# Standalone enumeration gate (no orchestrator, no Docker):
python scripts/check_usbc_devices.py --config config/jetson_production.yaml
# JSON output for machine-readable triage:
python scripts/check_usbc_devices.py --config config/jetson_production.yaml --json
# Full smoke flow (writes timestamped reports/jetson_smoke/<UTC>/):
bash scripts/jetson_full_smoke_run.sh
```

### power-chain-smoke

**Trigger:** "smoke the e-stop budget", "rover battery + motion check",
"why is power stage failing on smoke?".

**Read:**
- `src/mousedroid/diagnostics/power_chain.py` — `assert_power_chain`
  three-step probe (battery → send_velocity → emergency_stop timing).
- `src/mousedroid/config/schema/` — `ESP32Config.smoke_test_velocity_mps`
  (`ge=0`, so `0.0` permanently locks to zero-motion) +
  `emergency_stop_budget_ms`.
- `tests/hardware/test_power_chain_smoke.py` — `@pytest.mark.hardware`-
  gated rover-side test (asyncio `auto` mode in `pyproject.toml`).

**Run:**
```bash
# Default zero-velocity probe (untethered rover safe):
python -m pytest tests/hardware/test_power_chain_smoke.py -v
# Override only when rover is on rollers / tethered:
MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true \
    python -m pytest tests/hardware/test_power_chain_smoke.py -v
```

### rover-firmware-diagnosis

**Trigger:** "rover won't respond", "ESP32 silent on UART", "rover doesn't
move when commanded", "no boot banner from ESP32", "is the rover dead?",
"check what wave rover canonical baud is".

**Read:**
- `docs/runbooks/jetson-rover-smoke.md` — triage matrix (warm-vs-cold
  smoke, rover-swap by-id drift).
- `src/mousedroid/comms/serial_driver.py` — note: `_read_line` decodes
  with `errors="replace"` so a garbled byte never raises
  `UnicodeDecodeError` past the adaptive-timeout state machine.
- (External reference) Wave Rover stock firmware repo:
  `https://github.com/waveshareteam/ugv_base_ros`. Canonical baud is
  **115200** (`ROS_Driver/ROS_Driver.ino:96` `Serial.begin(115200)`). If
  `cfg.esp32.serial_baud` is not 115200, stock firmware will not respond.

**Run:**
```bash
# Inside the Jetson container, against the canonical ESP32 port:
docker stop mousedroid && cd /opt/mousedroid && \
  docker compose -f docker-compose.jetson.yml run --rm -T --entrypoint bash \
    mousedroid -c 'pip install esptool && \
      esptool --port /dev/serial/by-id/usb-Silicon_Labs_CP2102N*-if00-port0 \
              --before no-reset --connect-attempts 3 chip-id'
# Raw 10s listen for any unsolicited bytes from the chip:
sudo timeout 10 cat /dev/ttyUSB1 | xxd | head -40
```

If `{esptool sync, raw listen, WiFi AP scan for WAVE_ROVER_*}` are ALL
silent, the chip is physically dead — escalate to hardware repair /
module replacement. **Don't waste hours on cable swaps without first
confirming `esp32.enabled: true` in the live config** — the PR #104
escape hatch silently falls back to `MockESP32Driver` when set `false`,
masking every hardware diagnosis path.

### claude-llm-gateway

**Trigger:** "switch to Claude for missions", "enable Anthropic backend",
"hook up cloud LLM", "deploy `jetson_claude_pilot.yaml`", "wire the
Tier C deliberative brain".

**Read:**
- `src/mousedroid/llm_gateway/anthropic_gateway.py` — async Claude
  backend (lazy SDK import, prompt-injection pre-egress, SecretStr key
  handling, markdown-fence JSON resilience, self-heal on success,
  CancelledError propagation).
- `src/mousedroid/llm_gateway/fallback_gateway.py` — primary/secondary
  composite (cooldown-based primary retry, concurrent start, safe stop
  fan-out, secondary unexpected-exception guard).
- `src/mousedroid/config/schema/` — `LLMConfig.backend`,
  `fallback_backend`, `fallback_model_name`,
  `fallback_retry_cooldown_s`, `api_key` (`SecretStr`).
- `src/mousedroid/factory.py:_build_single_llm_gateway` +
  `build_llm_gateway` — dispatch + composite wrap.
- `config/jetson_claude_pilot.yaml` — canonical anthropic-primary +
  llama_cpp-fallback overlay.
- `docs/architecture/c4-llm-gateway.md` — C4 component diagram.

**Run:**
```bash
# Install optional deps (anthropic SDK + local llama_cpp model loader)
pip install -e ".[anthropic,llm]"

# Set the API key (never commit; SDK reads ANTHROPIC_API_KEY natively,
# or use the schema-mapped MOUSEDROID_LLM__API_KEY for SecretStr wrap)
export ANTHROPIC_API_KEY=sk-ant-...

# Deploy with the canonical pilot overlay
MOUSEDROID_JETSON_CONFIGS=config/jetson_claude_pilot.yaml \
    python -m mousedroid.main
```

**Diagnose:**
```bash
# Grep structlog stream for the gateway lifecycle events:
jq -c 'select(.event|test("anthropic_gateway|fallback_"))' run.log
# Key signal events:
#   anthropic_gateway_started       — SDK + client + API key OK
#   anthropic_gateway_degraded_*    — start-time degrade (no SDK / blank model)
#   anthropic_gateway_request_failed — runtime failure; degraded latched
#   anthropic_gateway_recovered     — self-heal (DEBUG; degrade reset on success)
#   fallback_primary_to_secondary   — failover happened
#   fallback_primary_retry_attempt  — cooldown elapsed; re-probing cloud
#   fallback_served (served_by=...) — which tier handled the command
```

### translate-mission

**Trigger:** "translate a mission", "test the LLM gateway", "dry-run the
deliberative path", "does NL→GoalVector work?", "validate Claude failover
without driving the rover".

**Read:**
- `scripts/translate_mission.py` — operator dry-run probe: builds the
  gateway via `build_llm_gateway` + `resolve_runtime_config_paths`,
  translates an NL command to a `GoalVector`, engages NO motors (safe with
  the dead ESP32).
- `config/jetson_production.yaml` — the live `llm:` block (Claude-haiku
  primary + Phi-3 `llama_cpp` fallback) the probe loads on the rover.
- `docs/runbooks/jetson-claude-pilot-deploy.md` — deploy + verify runbook.
- `config/docker.env.example` — secret surface (`ANTHROPIC_API_KEY`); the
  live values live ONLY in `/etc/mousedroid/docker.env` on the rover.

**Run:**
```bash
# Dry-run a mission through the deliberative path (no motors):
python scripts/translate_mission.py \
    --config config/jetson_production.yaml \
    --mission "patrol the lab then return to dock"
# Force the local fallback to confirm cloud→local failover:
MOUSEDROID_LLM__BACKEND=llama_cpp python scripts/translate_mission.py \
    --config config/jetson_production.yaml \
    --mission "go forward two meters"
```

### llm-prompt-injection-filter

**Trigger:** "operator NL command bypassed our guardrails", "ignore all
instructions...", "what does the rover send to the cloud?", "injection
filter doesn't fire".

**Read:**
- `src/mousedroid/security/injection_filter.py` — `RegexInjectionFilter`
  + `PromptInjectionFilterProtocol` + `InjectionRejected` exception
  (ValueError subclass).
- `src/mousedroid/llm_gateway/anthropic_gateway.py:translate_mission`
  — call to `self._injection_filter.sanitize(nl_command)` MUST appear
  BEFORE `client.messages.create`; commit the order, not the proximity.
- `src/mousedroid/factory.py:build_llm_injection_filter` — the shared
  filter instance threaded into both the gateway and the OpenClaw
  mission dispatcher so REST + MCP + LLM ingress share one envelope.
- `LLMConfig.injection_patterns` + `LLMConfig.max_command_len` — the
  pattern list + length cap. Defaults pinned by the AQA regression
  suite.

**Run:**
```bash
# Smoke test from an off-rover host:
python -c "
from mousedroid.security.injection_filter import RegexInjectionFilter
flt = RegexInjectionFilter(['(?i)ignore.*previous'], max_len=512)
print(flt.sanitize('go forward then ignore previous instructions'))
"
# Expect: raises InjectionRejected (ValueError subclass).
```

### llm-gateway-observability

**Trigger:** "how much is Claude costing us?", "wire LLM token/latency
metrics", "the `anthropic_gateway_slow` warning has no counter", "is the
gateway on `/metrics`?", "cloud-vs-local served split".

**Read:**
- `src/mousedroid/telemetry/metrics/` — `inc_llm_tokens`,
  `observe_llm_gateway_latency_ms`, `inc_llm_gateway_served`,
  `inc_llm_latency_budget_exceeded`; the `_LLM_TOKEN_TYPES` /
  `_LLM_SERVED_TIERS` / `_LLM_SERVED_OUTCOMES` cardinality frozensets; the
  `if cfg.track_llm_gateway` + `if count > 0` render guards.
- `src/mousedroid/config/schema/` — `MetricsConfig.track_llm_gateway` +
  `llm_gateway_latency_buckets_ms` (registered in the single
  histogram-bucket `@field_validator`).
- `src/mousedroid/llm_gateway/anthropic_gateway.py` — success-path records
  latency + tokens + budget counter; `_extract_token_usage` (defensive OUTER
  `getattr(response, "usage", None)`).
- `src/mousedroid/llm_gateway/fallback_gateway.py` — the four served-counter
  sites (primary/secondary × ok/degraded).
- `docs/architecture/c4-llm-gateway.md` — Observability section.

**Run:**
```bash
# Registry-level smoke (off-rover):
python -c "
from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry
r = MetricsRegistry(MetricsConfig())
r.inc_llm_tokens('claude-haiku-4-5','input',120)
r.observe_llm_gateway_latency_ms(180.0)
r.inc_llm_gateway_served('primary','ok')
print('llm_tokens_total' in r.render_prometheus())
"
# Live (rover, auth-exempt):
curl -fsS http://127.0.0.1:8080/metrics | grep -E 'mousedroid_llm_(tokens|gateway_latency|gateway_served|latency_budget)'
```

The four families are pure-add (absent until first write). Population on the
LIVE server requires a translation through the orchestrator's gateway — proven
in-process by `tests/hardware/test_llm_gateway_metrics_live_jetson.py` (prod has
no HTTP mission ingress; see `jetson-full-validation`).

### jetson-full-validation

**Trigger:** "validate everything on the rover", "full e2e + smoke on the
Jetson", "did the merged work actually land on-device?", "one-command
validation pass".

**Read:**
- `scripts/jetson_full_validation.sh` — the wrapper (Phase 0-4; composes
  `ci.sh`, `verify_sensors.py`, `jetson_smoke_test.sh`, `translate_mission.py`,
  `lidar_telemetry_probe.py`, the `preflight`/`validate_pillars` CLIs).
- `docs/runbooks/jetson-full-validation.md` — operator flow, cold-then-warm,
  validate-around-ESP32, the #115 `/metrics` grep recipe, and the F-018 trend
  surface (Phase-2 journal append, Phase-4 SUMMARY Trend section, hourly
  `mousedroid-trend.timer` install via `host_bootstrap.sh --with-trend-timer`).
- `tests/hardware/test_llm_gateway_metrics_live_jetson.py` — the live #115
  metric assertions (Test A live scrape; Test B in-process population).

**Run:**
```bash
# Off-rover sanity (no hardware touched):
bash scripts/jetson_full_validation.sh --help
bash scripts/jetson_full_validation.sh --dry-run
# On the rover (repo at /opt/mousedroid):
bash scripts/jetson_full_validation.sh                 # all phases -> reports/jetson_full_validation/<UTC>/SUMMARY.md
bash scripts/jetson_full_validation.sh --phase 1       # static CI only
bash scripts/jetson_full_validation.sh --pytest-only   # hardware tier only
```

Exit non-zero iff a BLOCKING step failed; the dead-ESP32 serial/motor/power
steps are non-blocking WARNs. Every tunable is env-overridable
(`MOUSEDROID_VALIDATION_*`, `MOUSEDROID_METRICS__NAMESPACE`); secrets are
presence-checked only.

---

## Engineering skills (developer-facing)

### add-schema-field

**Trigger:** "add a new config knob", "expose this threshold to YAML".

**Process:**

1. Add the `Field(default=..., ge=..., le=..., description=...)` to the
   appropriate Pydantic model in `src/mousedroid/config/schema/`.
2. The `description` MUST be ≥ 20 chars + explain *why* the operator
   would change it.
3. Default MUST preserve legacy behaviour (an existing YAML without the
   field must load identically).
4. Add a regression test in `tests/regression/` pinning the default.
5. Add an AQA test in `tests/regression/test_pr*_aqa.py` confirming the
   description + default + range are reachable via `FieldInfo`. For the
   full paired shape (AQA + backwards-compat, not just the AQA half), see
   `regression-pair-scaffold` below.
6. If the field gates a code branch, add an integration test in
   `tests/integration/` exercising the wiring through `factory.py`.

Reference: PR #104 added 3 fields this way — see commit `1b7a12e` for the
shape.

### add-hardware-driver

**Trigger:** "implement the IMX500 / VL53L0X / new sensor driver".

**Process:**

1. Define `@runtime_checkable Protocol` in
   `src/mousedroid/hardware/protocols.py` (or the appropriate
   subsystem-local protocol module). Keep the interface narrow.
2. Implement the real driver in `src/mousedroid/hardware/<subsystem>/`.
3. Implement a `MockX` mirror — same protocol, no real I/O. Mock must
   produce sensible synthetic data (`np.random.default_rng(seed).normal()`
   shaped to match the real driver's output).
4. Add a `build_<driver>(cfg, ...)` builder in `factory.py`. Honour
   `mock_hardware` AND any per-subsystem `enabled: bool`.
4a. If the driver talks to real hardware and can transiently fail (serial,
    USB, CSI/I2C), wrap the real backend in a `ResilientX` class under
    `src/mousedroid/resilience/` — `CircuitBreaker` + `retry_async` around
    the same Protocol the driver implements, reusing the existing top-level
    `cfg.retry`/`cfg.circuit_breaker` config (no new config field). The mock
    branch stays unwrapped. This is an established, repeated pattern —
    `ResilientESP32Driver` → `ResilientLidarDriver` → `ResilientCamera`, each
    docstring naming the prior one as its template — not a one-off;
    `resilient_camera.py` is the cleanest reference shape (it additionally
    shows how to transparently delegate an optional, non-Protocol driver
    capability via `cast()` rather than `getattr()`, so a new suppression
    isn't needed).
5. Tests: unit (mock-driven), integration (factory wiring), hardware
   (rover-only, `@pytest.mark.hardware`). If 4a applies, also add a
   dedicated `tests/unit/resilience/test_resilient_<driver>.py` mirroring
   `test_resilient_camera.py`.
6. Update `docs/architecture/` if you added a new external boundary.

### test-tier-mirror

> Project skill: `.claude/skills/test-tier-mirror/SKILL.md` — invoke it,
> don't paraphrase it.

**Trigger:** "where should this test go?", "which tier?", "is this tested at
the right level?", "add coverage for X".

**Read:**
- `.claude/skills/test-tier-mirror/SKILL.md` — the nine-tier table, the
  tier-selection heuristics, skip-gate conventions
- `tests/integration/test_f025_integration.py` — reference shape for
  "the mock bypasses the code under test, so fake at the transport boundary"
- `tests/regression/test_optional_extra_import_gates.py` — the gate that
  enforces `pytest.importorskip` for optional extras

**Guardrails:** a test that cannot fail reads as coverage and is worse than no
test. Three patterns to avoid — asserting a flag that is set once and never
cleared, `isinstance` against a `runtime_checkable` Protocol (presence only),
and pinning YAML key-absence instead of the value. Under `PYTHONOPTIMIZE=1`
(the Jetson entrypoint) `assert` is stripped — use explicit
`if not ...: raise` in `src/` and in inline shell one-liners.

### regression-pair-scaffold

> Project skill: `.claude/skills/regression-pair-scaffold/SKILL.md` — invoke
> it, don't paraphrase it. Complements `test-tier-mirror`: that skill answers
> *where* a test goes, this one answers *what shape* the AQA +
> backwards-compat pair takes once you know it belongs in `tests/regression/`.

**Trigger:** "add a regression test for this field", "does this feature have
its AQA/backwards-compat pair yet?", after `add-schema-field` or
`add-hardware-driver` says a config field or driver needs an AQA test.

**Read:**
- `.claude/skills/regression-pair-scaffold/SKILL.md` — the two literal file
  skeletons (angle-bracket placeholders), naming discipline, and the
  loader-path-vs-direct-construction call
- `tests/regression/test_pr106_aqa.py` +
  `tests/regression/test_pr106_backwards_compat.py` — a real, current example
  pair safe to read verbatim

**Guardrails:** check field hygiene on `model_fields` (`FieldInfo`), never via
instantiation. Name the backwards-compat file `_backwards_compat.py` in full
— several existing files drifted to `_backcompat.py`/`_compat.py`; don't
extend that. Confirm both files actually go red against a temporary revert of
the change before trusting them green.

### feature-closeout

> Project skill: `.claude/skills/feature-closeout/SKILL.md`.

**Trigger:** "mark F-nnn done", "close out the feature", "is F-nnn actually
closed?", "the nightly harness went red on implemented_in".

**Read:**
- `.claude/skills/feature-closeout/SKILL.md` — the full chain + a detector
  script for the whole catalog
- `features.yaml` — the catalog; `scripts/validate.py` — the gate

**Run:**
```bash
python scripts/validate.py --check F-025          # one feature, any tier
python scripts/validate.py --tier fast            # what every push runs
python scripts/validate.py --tier fast,slow --strict-git   # what nightly runs
```

**Guardrails:** `implemented_in` must be a hex commit SHA on every `done`
feature. A branch name resolves while the branch is alive and stops resolving
the moment it is deleted post-merge — reddening the nightly `--strict-git` job
the morning after, far from the change that caused it. Land the code, then
amend `features.yaml` with the real SHA; never leave a placeholder.

### run-pre-pr-validation

> The ordered local ladder lives in `.claude/skills/gate-ladder/SKILL.md`
> (invoke it for the full sequence + failure-triage table). This section
> keeps the Windows-interpreter form and the subagent delegation step.

**Trigger:** "I'm about to push a PR", "is this ready to merge?"

**Install first — this is step 0, not a footnote:**

```bash
pip install -e ".[dev,telemetry,mcp]"
```

matching `.github/workflows/ci.yml`'s `test` job. A bare `[dev]` environment
lacks Pillow (`[telemetry]`) and the MCP SDK (`[mcp]`) and produces dozens of
failures plus several `mypy` errors that do not exist on CI and are not
defects in your change.

**Process:**

```bash
# Local — must all be green
"/c/Program Files/Python311/python.exe" -m ruff check src/ tests/ tools/ scripts/
"/c/Program Files/Python311/python.exe" -m ruff format --check src/ tests/ tools/ scripts/
"/c/Program Files/Python311/python.exe" -m mypy --strict --no-incremental src/mousedroid/<touched>.py
"/c/Program Files/Python311/python.exe" -m pytest tests/unit tests/integration tests/property \
    -m "not hardware" --import-mode=importlib --timeout=180 -q
ALLOW_PYTEST_COLLECTION_SKIP=1 \
    "/c/Program Files/Python311/python.exe" scripts/check_branch_coverage.py \
        --min 90 --tests tests/unit/<your-test>.py

# If you touched .claude/ or tools/claude_hooks/ (workforce governance, F-024):
"/c/Program Files/Python311/python.exe" -m pytest \
    tests/regression/test_claude_workforce_aqa.py tests/unit/tools/claude_hooks \
    --import-mode=importlib -q
MYPYPATH=. "/c/Program Files/Python311/python.exe" -m mypy tools/claude_hooks/ \
    --strict --ignore-missing-imports --explicit-package-bases
```

Then delegate to subagents in parallel:
- `security-auditor` — focused on the diff's files.
- `code-quality` — ruff + mypy + branch-coverage gate.
- `feature-dev:code-reviewer` — independent reviewer for high-confidence
  issues.

When all three return green, push + open the PR with the body template
from `.github/pull_request_template.md` (or PR #104 as a worked example).

### ci-deploy-gates

**Trigger:** "config-compat gate failing", "actionlint error", "invalid
workflow file", "I changed a workflow / a config YAML — what CI runs?",
"bump the deployed image SHA".

**Read:**
- `.github/workflows/config-compat.yml` — gate that `git worktree`s the
  SHA in `deployments/jetson-image.json` and validates changed
  `config/*.yaml` against that historical schema.
- `scripts/check_config_compat.py` — the validator (`_validation_env`
  builds a cross-platform child env for the worktree subprocess).
- `deployments/jetson-image.json` — the deployed-image record; its `sha`
  MUST be a reachable trunk commit carrying the schema the image has.
- `.github/workflows/ci.yml` (Stage 0 `actionlint` job, pinned
  `docker://rhysd/actionlint:1.7.12`) + `.github/actionlint.yaml` (custom
  `jetson` runner label).

**Run:**
```bash
# Lint all workflows the way CI Stage 0 does (pinned actionlint):
docker run --rm -v "$PWD:/repo" -w /repo rhysd/actionlint:1.7.12 -color
# Reproduce the config-compat gate for a changed YAML:
python scripts/check_config_compat.py --platform jetson \
    --changed-files config/jetson_production.yaml
```

**Gotcha:** never put a literal `${{ ... }}` token in a workflow `run:`
block, even in a comment — GitHub evaluates it and an empty one is an
"invalid workflow file" startup failure (PR #113). If a YAML change needs
a schema field the deployed image lacks, bump `deployments/jetson-image.json`
to a reachable commit that carries it.

### validate-skills

**Trigger:** "validate skills", "skill drift", "did I break a skill?",
"lint the .claude/skills", "does this skill reference a real path?"

**What it covers:** the two skill families are validated independently —
neither can silently drift from reality.

- **`.claude/skills/<name>/SKILL.md` project skills** (migrated from the
  legacy `.claude/commands/*.md` layout, WS-F7a) are linted for a
  non-empty front-matter `description`, referenced-path existence
  (every backtick-wrapped repo path must resolve on disk), the absence
  of any hardcoded host/IP, and front-matter `name`/directory agreement.
  The reusable rule lives in exactly one place —
  `tools/validate_skill_commands.py` — and is consumed by both the CLI
  (layout auto-discovery) and the AQA test.
- **Builtin `SkillSpec` ↔ publishable doc pairing** is pinned so every
  spec in `src/mousedroid/skills/builtin/` has a matching
  `docs/openclaw_skills/<name>/SKILL.md` (H1 == skill name) and no doc
  is orphaned.

**Run:**

```bash
# Fast standalone CLI signal (also wired into scripts/ci.sh):
"/c/Program Files/Python311/python.exe" tools/validate_skill_commands.py

# The PR gate (AQA regression) + the builtin pairing test:
"/c/Program Files/Python311/python.exe" -m pytest \
    tests/regression/test_skill_commands_aqa.py \
    tests/unit/skills/builtin/test_skill_specs_match_docs.py \
    --import-mode=importlib -q
```

**Gotcha:** the validator *discovers* referenced paths from the body —
it never enumerates skill names or paths — so illustrative patterns with
format/glob tokens (`weights/arm/{task}_final.pt`, `*`, `$`, `<>`) are
intentionally skipped. If you mean a literal repo path, write it without
those metacharacters or the validator will (correctly) not check it.

---

### workforce-hooks

**Trigger:** "why was my edit blocked?", "freeze gate", "capability freeze",
"secret scan blocked my write", "workforce config", "hook denied",
"MOUSEDROID_WORKFORCE_ALLOW_FROZEN"

**What it covers:** the edit-time governance that runs *inside* a Claude Code
session (F-024) — an edit-time secret scan and a capability freeze gate (both
can block), plus advisory post-edit `ruff`/`mypy`. Everything tunable lives in
`.claude/workforce.yaml`, validated by `WorkforceConfig` with `extra="forbid"`,
so a misspelled key fails loudly instead of silently disabling a gate.

- **Blocked by the freeze gate?** The path matched `freeze.frozen_paths` while
  F-008 is not `done` — the same "hardware readiness preempts all in-flight
  software streams" rule the three frozen skills carry. It self-disables when
  F-008 lands.
- **Blocked by the secret scan?** Add the placeholder's literal **regex** to
  `.gitleaks.toml`; never allowlist by path.
- **Nothing happening at all?** The hooks need the repo's deps installed
  (`pip install -e .`) and shell out via `python3`.

**Run:**

```bash
# Drive a hook by hand with a synthetic payload (empty stdout == allow):
echo '{"tool_name":"Write","tool_input":{"file_path":"src/mousedroid/arm/x.py"}}' \
    | python3 -m tools.claude_hooks.freeze_gate

# Verbose diagnostics (logs go to stderr — stdout is the decision channel):
MOUSEDROID_WORKFORCE_DEBUG=1 python3 -m tools.claude_hooks.secret_scan

# Gates:
python3 -m pytest tests/regression/test_claude_workforce_aqa.py \
    tests/unit/tools/claude_hooks --import-mode=importlib -q
```

**Gotcha:** hook commands must be
`cd "$CLAUDE_PROJECT_DIR" && python3 -m tools.claude_hooks.<module>`. Running the
module file by path leaves the repo root off `sys.path`, so every hook dies with
`ModuleNotFoundError` — and a PreToolUse crash is a non-blocking warning, so the
gates would be silently inactive rather than obviously broken. Details:
`docs/runbooks/claude-workforce-hooks.md`.

---

## Subagent skills (delegation-facing)

| Skill | Subagent | Use when |
|-------|----------|----------|
| Security audit | `security-auditor` | Before any PR that touches auth, networking, or shells out |
| Code quality | `code-quality` | Before any PR — runs ruff + mypy + branch-coverage gates |
| Code review | `feature-dev:code-reviewer` | Independent second-opinion on a diff |
| ML review | `ai-ml-toolkit:ml-engineer` | Training-loop changes, model deployment, A/B tests |
| Architecture | `feature-dev:code-architect` | Designing a new feature with multi-file scope |
| Test engineering | `testing-suite:test-engineer` | When the test pyramid needs a structural change |
| Documentation | `documentation-generator:technical-writer` | README rewrites, ADRs, user guides |

When dispatching, ALWAYS:

- Provide the worktree path explicitly.
- Provide the Python interpreter path (`"/c/Program Files/Python311/python.exe"`)
  on Windows.
- Provide the exact files / paths to focus on.
- Ask for a length-capped report ("under 250 words inline").
- Use `run_in_background=true` for any check that doesn't block your next
  step — then continue work in parallel and respond to the completion
  notification.

---

## Discovery

To find more skills, in this order:
1. Grep this file: `grep -i "<keyword>" SKILLS.md`.
2. Check `~/.claude/plugins/cache/claude-plugins-official/superpowers/*/skills/`
   for general-purpose Anthropic skills.
3. Check `docs/operations/` for operator runbooks (which often map to
   skills here).
