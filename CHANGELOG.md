# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added — Physical-AI Phase 5: MuJoCo skid-steer sim → RSSM dynamics pretraining + vision-on fine-tune

Replaces the NumPy kinematic rover sim with a MuJoCo (classic) skid-steer physics
simulator and pretrains the RSSM world-model dynamics core on its episodes,
end-to-end through the training pipeline orchestrator. A follow-on phase renders an
RGB camera, extracts vision features, and fine-tunes the pretrained (vision-OFF)
RSSM with vision turned ON. All opt-in and backwards-compatible — existing YAML,
checkpoints, and the deployed world model are byte-identical.

**MuJoCo skid-steer rover env** (`sim/mujoco_rover_env.py`, `assets/rover/mse6_4wd.xml`)

- `RoverMuJoCoEnv` fills the reserved `rover.sim.backend == "mujoco"` factory slot
  with the SAME observation-dict contract as `MockRoverEnv` (imu / chassis_pose /
  wheel_vel / lidar, FL/FR/RL/RR order). IMU from `<accelerometer>`+`<gyro>`;
  config-driven N-sector `<rangefinder>` lidar spliced into the MJCF at load; a
  rest-state finite-`qacc` assertion guards the silent-NaN wheel-grounding footgun.
- Domain-randomization params (`wheel_friction` → `geom_friction`, `chassis_mass_kg`
  → `body_mass`+inertia, `motor_gain` → `actuator_gainprm`; `wheel_slip` as a
  documented observation-noise proxy) are now consumed per-episode via
  `SimEpisodeGenerator` + `DomainRandomizer`.

**RSSM dynamics pretraining** (`world_model/rssm.py`, `world_model/encoder.py`,
`world_model/latent_utils.py`, `training/rssm_pretrainer.py`)

- `RSSM.train_sequence` — a gradient-enabled rollout reconstructing the RAW
  per-modality observations (not the encoder's own embedding — avoids
  representation collapse) with Dreamer-style balanced free-bits KL computed in
  float32. Reconstruction heads live in a pretraining-only `RawModalityDecoders`
  module so the deployment RSSM `state_dict` + seeded init stay byte-identical.
- `MultimodalEncoder` vision branch is now optional (`vision_dim=0`), mirroring the
  audio/lidar gating; default `vision_dim=256` is byte-identical.

**Vision-on fine-tune** (`factory.build_rssm_vision_finetune`,
`checkpoint_migration`, `pipeline_orchestrator`)

- Renders RGB via `mujoco.Renderer` → the deployed (non-learned) `MeanPoolExtractor`
  → 256-d `vision_features` (sim/deploy distributions match by construction — no CNN
  trained). `build_rssm_vision_finetune` migrates a vision-OFF checkpoint to vision-ON
  via the existing `checkpoint_migration` machinery (extended to handle the vision
  modality) — dynamics core copied verbatim, vision fusion columns + `vision_proj`
  Kaiming-initialised.
- Opt-in orchestrator phases: `training.rssm_pretrain_enabled` and
  `training.rssm_vision_finetune_enabled` (both default OFF). The blocking torch loop
  runs in `asyncio.to_thread` so the thermal-pause safety check is never starved.

**Config** (`config/schema.py`): additive `MujocoSimConfig` (mjcf path, arena, lidar
sectors/range, render fields, DR defaults) under `rover.sim.mujoco`; `ModelConfig`
KL knobs; `TrainingConfig` `rssm_*` pretrain/fine-tune knobs — all defaulted so
pre-feature YAML loads unchanged.

Architecture: [`docs/architecture/c4-rssm-sim-pretraining.md`](docs/architecture/c4-rssm-sim-pretraining.md).

### Added — Validation efficiency: latency percentiles, trend store, phase-1 caching

Runtime/resource-efficiency layer on the existing Jetson validation harness. All
three surfaces are additive and opt-in — defaults preserve byte-identical legacy
behaviour.

**Latency-percentile probes** (`src/mousedroid/validation/latency_stats.py`)

- New pure, dependency-free `summarize(samples_ms) -> LatencySummary`
  (min/mean/p50/p95/p99/max) + `intervals_ms(timestamps_s)` (arrival-timestamp →
  inter-arrival gaps). No I/O, no clock reads, no verdict — the caller gates
  against its config-supplied target. `mypy --strict` clean, 100 % covered.
- `tools/llm_latency_probe.py --iterations N` — `1` (default) is the legacy
  single-shot gate verbatim; `>1` emits `llm_latency_summary` and gates on **p95**
  to absorb cloud/GPU tail variance.
- `tools/lidar_telemetry_probe.py` emits `lidar_frame_interval_summary`
  (inter-arrival jitter — high p95/p99 vs p50 = dropped/bunched dashboard frames).

**Run-over-run trend store** (`src/mousedroid/validation/report_store.py`)

- Persists each `PreflightReport` to the **existing** harness journal (no parallel
  store) as a `preflight_report` event; `detect_regressions(history)` compares the
  two newest runs for status downgrade / new FAIL / latency creep (gated by both a
  `slow_ratio` and an absolute `slow_floor_s` so sub-50 ms checks don't false-fire).
  `recorded_at_ns` is wall-clock `time.time_ns()` (stable across reboots).
- Wired via `mousedroid.cli.preflight --journal-path PATH` (opt-in record) +
  `--trend` (print regressions; exit 1 on regression) + operator-tunable
  `--trend-slow-ratio` / `--trend-slow-floor-s` (no hardcoded call site).

**Phase-1 caching** (`scripts/jetson_full_validation.sh`)

- Phase 1 (static CI) is a pure function of the committed source: a clean tree
  unchanged since the last green run SKIPs it (`PASS "static CI (cached)"`); a
  dirty tree forces a miss (never masks an uncommitted edit). `--no-cache` forces a
  re-run; `--phases 0,1,3` runs an ordered subset (`--phase` kept as the
  single-phase alias). Hardware (Phase 2) + live (Phase 3) are never cached.

**Modularity** (`src/mousedroid/validation/__init__.py`)

- The heavy `runtime` sensor helpers (numpy/cv2/pyaudio) are now re-exported
  **lazily** via PEP 562 `__getattr__`, so importing the pure `latency_stats` /
  `report_store` modules no longer drags the sensor stack into the process.
  Backwards compatible — the re-exported names still resolve on access.

**Tests** — unit (`latency_stats`, `report_store`, lazy `__init__`, CLI flags),
integration (`report_store` through factory `build_journal` for JSONL **and** LMDB +
NullJournal default), regression (subprocess import-decoupling guard), smoke (script
arg surface). Changed source files at 100 % line coverage.

**Docs** — `docs/architecture/c4-validation-efficiency.md` (C4 component diagram),
CLAUDE.md "Validation-efficiency surface" section, README validation block.

### Added — MLflow experiment logger for training observability

Opt-in training-metrics logging via the `mlflow-skinny` backend, threaded through the factory as a NEVER-None `ExperimentLoggerProtocol` (the `NoOpExperimentLogger` default is a byte-identical no-op, so the wiring is free when OFF). `PipelineOrchestrator` emits a parent run per pipeline + a child run per phase and consumes `run_name` (falls back to `"pipeline"`) and `log_artifacts` (gates the resolved-`Settings` snapshot + per-phase checkpoint uploads); `OfflineRLTrainer` (CQL/IQL) logs per-step losses and consumes `log_step_every_n` (`gt=0`) as its `step % n` throttle — fail-fasting on `< 1` so the modulo can never `ZeroDivisionError` (config→trainer wiring in the orchestrator's offline-RL phases is follow-up). All protocol methods are total (never raise on backend failure), and `build_experiment_logger` degrades to NoOp on a missing `[mlflow]` extra **or** a construction failure (bad `tracking_uri`/store/permissions). Defaults OFF; opt in via YAML or `MOUSEDROID_OBSERVABILITY__EXPERIMENT_LOGGER__BACKEND=mlflow`. Runbook: `docs/runbooks/mlflow-local-ui.md`; C4: `docs/architecture/c4-experiment-logger.md`.

### Added — Full rover bring-up: unified dashboard + sensor-fusion summary

Deploy-and-run-everything bring-up plus a single dashboard showing camera + lidar
+ sensor-fusion + status, reachable over WiFi from any device on the network.

**Sensor-fusion summary on telemetry frames** (`telemetry/protocol.py`,
`telemetry/frame_builder.py`)

- New `TelemetryFrame.fused` field (default-factory empty dict, mirrors
  `sensor_liveness` — purely additive, byte-identical when unused). Shape:
  `{n_valid, n_modalities, lidar_present, modalities{vision,ultrasonic,motor,
  audio,lidar}, fused_norm}`.
- Computed in `build_telemetry_frame` from the fused observation's `valid_mask`
  + existing scalar norms — a pure function (no new sensor reads, no hot-loop
  cost). Handles BOTH mask lengths (4 without lidar, 5 with) by zipping the fixed
  modality-name tuple against the actual length — never indexes a fixed slot.

**Unified dashboard** (`telemetry/server.py`, `telemetry/static/dashboard.html`)

- New `/dashboard` page + `/` → `/dashboard` redirect (token-preserving). One
  responsive page renders the live camera MJPEG, the lidar polar plot, a
  sensor-fusion panel (per-modality `sensor_liveness` tiles + the `fused`
  summary), and safety/health/battery/motor status — all from a single `/ws`
  connection. No hardcoded host/port (derives origin from `window.location`);
  token carried via `?token=` (the existing `/camera`+`/lidar` pattern).
- WiFi access uses the existing posture: `0.0.0.0:8080`, bearer token, mDNS
  `mousedroid-telemetry.local`; `/api/v1/network` advertises `server_url`+`mdns_name`.

**Deploy + bring-up** (`docs/runbooks/jetson-full-bringup.md`)

- One-command bring-up composing the container/systemd path + the #116
  validation wrapper. **Real motors are attempted probe-first:** the ESP32 is
  probed before bring-up; only if it responds does the orchestrator boot with
  motors live, otherwise `MOUSEDROID_ESP32__ENABLED=false` keeps the container
  from crash-looping (`start()`→`connect()` retry-then-raise on a dead board) and
  the rest of the rover runs. No motion is armed without lifting the rover.

**Tests:** unit (fused, both mask lengths), integration (publish-path + route
wiring), e2e (real `TelemetryServer`: `/`→302, `/dashboard` 200), regression
(additive/backwards-compat), AQA (modality-order + no-hardcoded-host hygiene),
smoke (`-m smoke`), hardware (double-gated live `/dashboard` + `/ws` `fused`).

### Added — PR #115: LLM-gateway observability (Prometheus token/latency/served/budget metrics)

First observability over the deliberative Claude tier (PRs #107/#111), which
had been burning Anthropic API quota on every NL→`GoalVector` translation with
zero metrics. Purely additive + config-gated — a deployment built with
`metrics=None` is byte-identical.

**Four config-gated metric families** (`src/mousedroid/telemetry/metrics.py`,
namespaced via `cfg.metrics.namespace`)

- `{ns}_llm_tokens_total{model,token_type}` — input/output token usage
  (`_DoubleLabeledCounter`).
- `{ns}_llm_gateway_latency_ms` — round-trip latency histogram (label-free;
  buckets from `MetricsConfig.llm_gateway_latency_buckets_ms`).
- `{ns}_llm_gateway_served_total{tier,outcome}` — the durable cloud-vs-local
  served split (recorded by `FallbackLLMGateway`).
- `{ns}_llm_latency_budget_exceeded_total{model}` — counter on the existing
  `anthropic_gateway_slow` branch (the log event name was **kept** — the metric
  carries the budget semantics).

**Wiring + contracts**

- One flag `MetricsConfig.track_llm_gateway` (default `True`) gates all four;
  families are pure-add (omitted from `/metrics` until first write).
- A shared `MetricsRegistry` is threaded `build_orchestrator → build_llm_gateway
  → AnthropicLLMGateway / FallbackLLMGateway` via a keyword-only `metrics=None`
  param. No hardcoded values: buckets, namespace, and the budget threshold
  (`cfg.llm.latency_target_ms`) all come from config.
- Label values are validated against fixed low-cardinality sets
  (`_LLM_TOKEN_TYPES` / `_LLM_SERVED_TIERS` / `_LLM_SERVED_OUTCOMES`) and
  out-of-set values are dropped with a DEBUG log — a typo can never open a new
  Prometheus time series.
- Records on the success path only; `asyncio.CancelledError` still propagates.

**Tests** — full tier matrix: unit (registry + gateway + factory + fallback
served-counter), integration (factory threading + faked translate), e2e (real
`TelemetryServer` GET `cfg.telemetry.metrics_path`), regression
(defaults/YAML/byte-identical), AQA (field + protocol + label hygiene), smoke
(`-m smoke`), hardware (double-gated `is_jetson_host` + `ANTHROPIC_API_KEY`).

### Added — PR #116: full Jetson on-device validation (one-command pass + live `/metrics` test)

Consolidates the scattered on-device tooling into a single ordered,
artifact-producing pass and adds the missing live confirmation of the PR #115
`/metrics` families on the rover.

- **`scripts/jetson_full_validation.sh`** — composes the existing tooling
  (`ci.sh`, `verify_sensors.py`, `jetson_smoke_test.sh`, `translate_mission.py`,
  `lidar_telemetry_probe.py`, the `preflight`/`validate_pillars` CLIs) into
  static-CI → cold-hardware → warm-live phases with a timestamped report dir +
  PASS/WARN/FAIL gate. Tolerates the functionally-dead ESP32 (validate-around:
  `serial`/`motor`/`power` non-blocking, no motion, orchestrator e2e with
  `MOUSEDROID_ESP32__ENABLED=false`); cold-then-warm discipline with a `trap`
  that always restarts the container. Every tunable (ports, timeouts, retries,
  namespace) is env-overridable — no hardcoded values; secrets presence-only.
  Selectors: `--phase N`, `--pytest-only`, `--dry-run`, `--help`.
- **`tests/hardware/test_llm_gateway_metrics_live_jetson.py`** — Test A scrapes
  the live (auth-exempt) `/metrics` for a healthy Prometheus surface; Test B
  drives the gateway **in-process** via `build_orchestrator → process_mission`
  with a guaranteed-UNKNOWN command and asserts `orch._metrics` renders the four
  families (proving the wiring on live Claude without needing the disabled
  `openclaw` HTTP ingress); Test C (HTTP POST) skips on prod.
- **Docs:** `docs/runbooks/jetson-full-validation.md` (+ cross-link from
  `jetson-rover-smoke.md`); `.gitignore` ignores the new report dir.

### Fixed — PR #113: config-compat CI gate startup-failure + cross-platform validation hardening

Repairs the repo-wide `config-compat` workflow, which was startup-failing
(GitHub "invalid workflow file") on **every** push across all branches,
and hardens the validation subprocess so the gate runs identically on the
Jetson, CI, and a Windows workstation.

**Workflow startup-failure fixed** (`.github/workflows/`)

- The `config-compat` workflow declared a literal empty `${{ }}`
  expression inside a shell **comment** in a `run:` block. GitHub
  evaluates `${{ }}` interpolation even inside `#` comments, so the empty
  expression failed validation and the workflow never started — every
  push showed a red startup-failure. Reworded the comment to remove the
  bare `${{ }}` token.

**Deploy record re-pinned** (`deployments/jetson-image.json`)

- The recorded SHA had been orphaned by an upstream squash-merge, so the
  gate's `git worktree` checkout of the deployed commit failed. Re-pinned
  to a reachable trunk commit so the gate can materialise the deployed
  config tree.

**Validation-subprocess hardening** (`scripts/check_config_compat.py`)

- Extracted `_validation_env()` so the validation subprocess **inherits**
  the base process environment instead of being handed a minimal env. A
  minimal env broke the subprocess on Windows with a spurious
  `No module named yaml` (the interpreter could not locate its
  site-packages). The helper strips `MOUSEDROID_*` overrides (so an
  operator's env cannot skew the compat check) and pins `PYTHONPATH` to
  the deployed-SHA worktree.

**CI surface — invalid-workflow guard** (`.github/workflows/`, `.github/actionlint.yaml`)

- Added a pinned `actionlint` job (`rhysd/actionlint:1.7.12`) so the
  whole `.github/workflows/` tree is lint-gated and this class of
  invalid-workflow regression is caught repo-wide before merge.
- New `.github/actionlint.yaml` declares the custom `jetson` self-hosted
  runner label so actionlint does not flag the rover's hardware jobs as
  unknown-runner errors.

### Fixed — PR #112: cv2-eviction test isolation in the JetsonCSI camera pipeline

Fixes a full-tree test-isolation footgun: running `pytest tests/` in a
single process broke 19 `test_jetson_csi` tests that passed in isolation.

- **Root cause** — `tests/integration/test_camera_pipeline.py::test_jetson_csi_stop_releases_camera`
  used `patch.dict("sys.modules", ...)` + `importlib.reload(jetson_csi)`.
  On `patch.dict` context exit, `cv2` was evicted from `sys.modules`, and
  OpenCV cannot be re-imported within the same process (the
  `cv2.dnn.DictValue` typing bug raises on re-import). Every later
  `test_jetson_csi` test in the same process then failed at import.
- **Fix** — replaced the `sys.modules` patch + reload with
  `patch.object(jcsi_mod, "_jetson_utils", ...)`, which swaps only the
  module-level backend handle with no `sys.modules` churn, so `cv2` is
  never evicted.
- **Regression guard** — added
  `test_jetson_csi_backend_patch_does_not_evict_cv2` pinning that the
  backend patch leaves `cv2` resident in `sys.modules`.
- **Verification** — `pytest tests/` now runs green full-tree
  (**5169 passed, 0 failures**).

### Added — PR #111: Deploy the Anthropic Claude gateway live to the Jetson rover

Lands the PR #107 LLM gateway as a live deployment on the MSE-6 rover:
Claude-haiku as the primary deliberative mission-translation brain with
the already-staged local Phi-3-mini GGUF as the off-network fallback. The
deterministic, LLM-free 30 Hz reactive control loop (RSSM → MCTS → ESP32
velocity command) is untouched — only natural-language → `GoalVector`
translation is deliberative and runs off the hot path.

**Container** (`Dockerfile.jetson`)

- New non-fatal `anthropic>=0.40` install layer (Stage 4b) following the
  graceful-degradation pattern — the build does not fail if the SDK wheel
  cannot be resolved, and the gateway degrades to the local fallback at
  runtime when `anthropic` is absent.

**Production overlay** (`config/jetson_production.yaml`)

- Minimal additive merge into the existing `llm:` block (no removals, so
  the overlay still loads on the prior schema): `backend: "anthropic"`,
  `model_name: "claude-haiku-4-5"`, cloud-calibrated `request_timeout_s`
  and `latency_target_ms: 5000` (the local-GGUF default 500 ms would spam
  `anthropic_gateway_slow` on normal cloud round-trips),
  `fallback_backend: "llama_cpp"`, and `fallback_retry_cooldown_s`. The
  fallback reuses the already-staged Phi-3-mini GGUF via the existing
  `model_path` — no new model download.

**Secret docs** (`config/docker.env.example`)

- New example env file documenting the two credential paths the
  anthropic backend accepts: `ANTHROPIC_API_KEY` (preferred — the SDK
  resolves it natively) and the schema-mapped `MOUSEDROID_LLM__API_KEY`
  override. No real credentials — placeholders + instructions only.

**Operator probe** (`scripts/translate_mission.py`)

- New dry-run CLI: takes a natural-language mission, runs it through the
  full deliberative translation path, and prints the normalised
  `GoalVector` **without** issuing any motor command. Resolves config via
  `resolve_runtime_config_paths` and reports the serving tier (primary
  cloud vs. local fallback) so an operator can confirm which brain
  answered before trusting it on the rover.

**Tests added**

- `tests/regression/test_jetson_claude_pilot_config.py` — pins the
  `jetson_production.yaml` `llm:` block contract (anthropic primary +
  llama_cpp fallback + cloud-calibrated latency/timeout).
- `tests/integration/test_jetson_pilot_gateway_wiring.py` — the
  production overlay wires through `build_llm_gateway` to a
  `FallbackLLMGateway` (faked SDK, no network / no key).
- `tests/unit/test_translate_mission_cli.py` — the operator probe parses
  args, resolves config, prints a `GoalVector`, and emits no motor
  command.

**Docs added**

- `docs/runbooks/jetson-claude-pilot-deploy.md` — operator runbook for
  the live deploy (SDK hot-install ordering, the `ENABLED` gate, CPU
  fallback notes).

**Deploy record** (`deployments/jetson-image.json`)

- Records the rebuilt Jetson image SHA carrying the PR #107 schema + the
  baked anthropic layer.

**Live validation** (on the rover)

- Claude-haiku primary translates a mission in ~1 s; the Phi-3 CPU
  fallback is staged and ready; the 30 Hz reactive loop is unaffected by
  the deliberative path.

### Added — PR #107: Anthropic Claude LLM gateway + cloud/local failover for rover missions

Enables Claude (via the Anthropic Messages API) as the deliberative
mission-translation brain for the MSE-6 rover, with transparent fallback
to a local model when off-network. Natural-language missions are
translated to a normalised `GoalVector` (vx, vy, omega ∈ [-1, 1])
**outside** the 30 Hz reactive control loop — the deterministic,
LLM-free hot path (RSSM → MCTS → ESP32 velocity command) is untouched
by design.

**New modules**

- `src/mousedroid/llm_gateway/anthropic_gateway.py` — `AnthropicLLMGateway`,
  async-native Claude backend conforming structurally to
  `LLMGatewayProtocol`. Lazy SDK import (deferred to `start()`), degrades
  rather than crashing when `anthropic` is missing / blank model id /
  init failure. `SecretStr` API key (read once at client construction,
  never logged). Markdown-fence JSON resilience via `_JSON_OBJECT_RE`
  (claims first `{...}` span before `json.loads`). Dict-block defensive
  text extraction. Self-heals on a successful request (clears
  `_degraded`). `asyncio.CancelledError` propagates cleanly without
  poisoning the degrade flag.
- `src/mousedroid/llm_gateway/fallback_gateway.py` — `FallbackLLMGateway`,
  primary/secondary composite. Failover keyed off `is_degraded` (not the
  returned vector — a legitimate `"stop"` neutral vector does not
  trigger spurious failover). Cooldown-based primary retry
  (operator-tunable `recovery_interval_s` / `LLMConfig.fallback_retry_cooldown_s`)
  with injectable clock for deterministic tests. Concurrent
  `start()`/`stop()` via `asyncio.gather(return_exceptions=True)`.
  Secondary call wrapped symmetrically with the primary's try/except.
  CancelledError propagates without stamping `_last_primary_attempt`.

**Schema additions** (`src/mousedroid/config/schema.py`)

- `LLMConfig.backend: Literal["llama_cpp", "openai_compatible", "anthropic"]`
  — default `"llama_cpp"` keeps existing YAML byte-identical.
- `LLMConfig.fallback_backend: Literal["none", "llama_cpp", "openai_compatible"]`
  — local-only fallback target. `"anthropic"` is rejected at
  YAML-parse time (cloud-to-cloud failover defeats off-network
  autonomy). Default `"none"`.
- `LLMConfig.fallback_model_name: str | None` — secondary `model_name`
  override; `None` reuses primary's.
- `LLMConfig.fallback_retry_cooldown_s: float = 30.0` — cooldown
  before re-probing a degraded primary. Range-gated `gt=0`.
- `LLMConfig.api_key: SecretStr | None` — read via `.get_secret_value()`
  ONCE at client construction, never logged.

**Factory wiring** (`src/mousedroid/factory.py`)

- `_build_single_llm_gateway(llm_cfg, *, injection_filter)` extracted
  for reuse across both tiers.
- `build_llm_gateway(cfg)` wraps primary + secondary in
  `FallbackLLMGateway` when `fallback_backend != "none"` AND
  `fallback_backend != backend`. Same-backend fallback is a logged
  no-op. `cfg.llm.fallback_retry_cooldown_s` threaded through to the
  composite.

**Config example**

- `config/jetson_claude_pilot.yaml` — canonical anthropic-primary +
  llama_cpp-fallback overlay with secret-handling + network-policy
  notes. `latency_target_ms: 5000.0` (default 500 ms is calibrated for
  local GGUF; cloud round-trips are 1-5 s and would spam
  `anthropic_gateway_slow` WARNINGs without the override).
  `fallback_retry_cooldown_s: 30.0` with operator-tuning guidance.

**Tests added** (across the pyramid)

- `tests/unit/llm_gateway/test_anthropic_gateway.py` — start /
  degrade / parse / clamp / injection-rejected / lifecycle / slow-path
  / self-heal / markdown-fence (3 variants) / dict-block extract /
  end-to-end recovery after WAN flap / `stop()` clears degraded /
  CancelledError handling.
- `tests/unit/llm_gateway/test_fallback_gateway.py` — failover
  semantics, value-error propagation, cooldown retry (in-cooldown
  skip + cooldown-elapsed re-probe + primary-recovers-on-reprobe),
  secondary unexpected-exception guard, concurrent `start()` timing,
  safe `stop()` fan-out, CancelledError propagation without poisoning
  the cooldown timer.
- `tests/unit/config/test_llm_config_anthropic_fallback.py` — backend
  Literal extensions + backcompat defaults + env overrides + SecretStr
  hygiene + cooldown range gates.
- `tests/unit/factory/test_build_llm_gateway_dispatch.py` — anthropic
  dispatch + composite wrap branches + same-backend no-op +
  `fallback_model_name` override scope + `fallback_retry_cooldown_s`
  threaded through to composite.
- `tests/integration/test_anthropic_gateway_wiring.py` — full path
  through `build_llm_gateway` + orchestrator `process_mission` with
  faked SDK (no network / no key).

**Security posture** (final audit: PASS)

- API key never appears in repr / logs / exception messages
  (`SecretStr` wraps; `.get_secret_value()` is called ONCE at client
  construction in `factory.py`).
- Prompt-injection filter (`RegexInjectionFilter.sanitize()`) fires
  BEFORE `client.messages.create` so `"ignore all instructions and..."`-
  shaped commands never leave the rover.
- Example YAML overlay contains NO real credentials — placeholders +
  env-var instructions only.
- Exception logs use `f"{type(exc).__name__}:{exc}"` (SDK error type +
  message); no request payload (NL command + system prompt) leaked.
- Failover does NOT bypass filtering — both primary and secondary
  independently apply the shared filter instance.

**Docs added**

- `docs/architecture/c4-llm-gateway.md` — Level-3 C4 component diagram
  for the LLM gateway showing the configuration → dispatch → composite
  → child-gateway chain plus the cross-cutting RegexInjectionFilter +
  the api.anthropic.com cloud boundary.
- `CLAUDE.md` — new "LLM gateway + cloud/local failover" section
  pinning the seven non-negotiable contracts (backend Literal, fallback
  Literal, cooldown, SecretStr, pre-egress filter, CancelledError,
  markdown-fence, concurrent lifecycle).
- `AGENTS.md` — new "LLM gateway — adding a new backend" 10-step
  workflow.
- `SKILLS.md` — two new skill entries: `claude-llm-gateway`,
  `llm-prompt-injection-filter`.
- `NEXT_STEPS.md` — PR #107 follow-ups + a new "Claude Code on Jetson
  — install + configure" operator runbook so engineers can run the
  Claude Code agent natively on the rover.

**Pre-merge resolution history** (three rounds)

- **Round 1** — Gemini Code Assist 5-finding pass folded in by
  upstream commit `7010648` (degrade-reset-on-success, dict-block
  extract, JSON-extraction regex, primary cooldown retry, schema
  cooldown field).
- **Round 2** (commit `a989a8f`) — independent `feature-dev:code-reviewer`
  + `code-explorer` agents raised 5 additional findings outside
  Gemini's scope: `stop()` clears `_degraded`, concurrent `start()`,
  safe `stop()` fan-out, secondary unexpected-exception guard,
  `latency_target_ms` overlay. + the PR #106 subprocess PYTHONPATH
  fix re-applied (rebase artifact, 8 pre-existing failures closed).
- **Round 3** (merge commit) — final review pass: CancelledError
  propagation in both gateways, `_last_primary_attempt` ordering
  (stamp AFTER await), factory wiring assertion test. Security-auditor
  PASS across all 8 critical checks.

### Added — PR #106: USB-C smoke validation gate + rover-swap auto-override + power-chain probe

Closes the rover-bringup gap left after PR #104's dashboard-stability sprint.
Adds a config-driven USB-C enumeration layer so the Jetson smoke pipeline
fails loudly when the rover is wired wrong, and so swapping rovers
between bench units stops breaking the literal `esp32.serial_port` path.

**New modules**

- `src/mousedroid/diagnostics/usbc.py` — pure USB-C endpoint enumeration
  helper. `enumerate_usbc_devices(cfg)` returns a `{name: EndpointResult}`
  dict; `resolve_endpoint(cfg, name)` returns a single live by-id `Path`.
  Both short-circuit when `usbc_discovery.enabled=False` and now both
  guard against `by_id_root` not existing (boot race with udev) so the
  smoke harness sees a structured `MISSING` instead of an uncaught
  `FileNotFoundError`. Status enum: `PRESENT` / `MISSING` / `WARN`.
- `src/mousedroid/diagnostics/power_chain.py` — `assert_power_chain`
  three-probe sequence (battery → send_velocity → emergency_stop timing)
  against an `@runtime_checkable Protocol` slice of the ESP32 driver.
  Returns a frozen `PowerChainResult` for the smoke harness to assert
  the e-stop latency against `ESP32Config.emergency_stop_budget_ms`.
- `scripts/check_usbc_devices.py` — standalone CLI smoke gate. Exits 1
  iff any `required=True` endpoint is `MISSING`. Supports `--json`. Does
  not need the orchestrator running.

**Factory wiring** (`src/mousedroid/factory.py:_resolve_esp32_serial_via_usbc_discovery`)

Two-condition override chain for `ESP32Config.serial_port`:
1. Discovery disabled OR `usbc_discovery is None` → return cfg unchanged.
2. Literal `esp32.serial_port` exists on disk → return cfg unchanged
   (a pinned operator override is never silently shadowed).
3. Resolver returns `None` (no matching glob) → warn + return cfg
   unchanged (driver will surface the failure with a clean errno).
4. Glob matches → `cfg.esp32.model_copy(update={"serial_port": str(resolved)})`,
   log the `esp32_serial_port_overridden` structured event. Original
   `cfg.esp32` is not mutated.

**Schema additions** (`src/mousedroid/config/schema.py`)

- `USBCEndpointSpec(name, by_id_glob, required: bool = True)` —
  declarative endpoint contract for the discovery layer.
- `USBCDiscoveryConfig(enabled: bool = False, by_id_root: Path =
  "/dev/serial/by-id", required_endpoints: list = [])` — master config.
  `_require_endpoints_when_enabled` model validator forbids
  `enabled=True` with an empty list at YAML-load time so a misconfigured
  gate never silently passes.
- `Settings.usbc_discovery: USBCDiscoveryConfig | None = None` —
  backwards-compat default; pre-PR YAML files load unchanged.
- `ESP32Config.smoke_test_velocity_mps` constraint relaxed from `gt=0`
  to `ge=0` so operators can express a permanent zero-motion safe-bench
  config; the runtime `allow_motion` gate in `assert_power_chain`
  remains authoritative regardless of this setpoint.

**Smoke wrapper** (`scripts/jetson_smoke_test.sh`, `scripts/jetson_full_smoke_run.sh`)

- New `usbc` blocking stage runs `python scripts/check_usbc_devices.py`
  before `serial` so a wiring problem fails at the cheapest possible
  test rather than after the serial driver tries to open a non-existent
  port.
- New `power` blocking stage runs `assert_power_chain` against the
  built ESP32 driver (mock or real). Defaults to zero-velocity so an
  untethered rover does not roll while the smoke runs unattended.
- Stage gating env-var matrix documented in `docs/runbooks/jetson-rover-smoke.md`
  (every stage flippable via `MOUSEDROID_SMOKE_BLOCKING_<NAME>={yes,no}`).
- E2E inline script's `assert isinstance(orch, MouseDroidOrchestrator)`
  replaced with an explicit `if not isinstance(...): raise RuntimeError(...)`
  so the Jetson Docker entrypoint's `PYTHONOPTIMIZE=1` cannot silently
  strip the check (CLAUDE.md validation contract).

**Comms hardening** (`src/mousedroid/comms/serial_driver.py`)

- `_read_line` now decodes with `errors="replace"` instead of the strict
  default codec. A garbled byte from firmware churn, brown-out, or UART
  noise no longer propagates `UnicodeDecodeError` past the adaptive-
  timeout state machine — the replacement char flows into `json.loads`
  and surfaces via the existing `esp32_non_json_response` log path.
- New `esp32_raw_line` DEBUG event lets operators grep the structlog
  stream for the literal bytes the ESP32 emitted, useful for triaging
  firmware-version drift without rewiring the driver.

**Security hardening** (`tools/dashboard_proxy.py`)

- Removed hardcoded default token `dev-dashboard-token-1779157616` from
  the env-fallback default. Operators MUST now supply `JETSON_TOKEN`
  (env var) or the third positional CLI arg; no hardcoded fallback means
  a deploy without the env var fails loudly at the upstream's 401, never
  silently reusing a baked-in dev credential. Surfaced by `security-auditor`
  subagent during PR #106 pre-merge review.

**CI surface**

- `.github/workflows/ci.yml` adds a `usbc-config-gate` job that runs
  `pytest tests/unit/test_jetson_production_overlay.py` after the
  config-validate stage. Asserts: `jetson_production.yaml` declares
  `usbc_discovery.enabled=True` with both `rover_esp32` and `lidar_ld19`
  endpoint names present; the `rover_esp32` glob and `esp32.serial_port`
  share the same `"CP2102N"` chip family marker; `default.yaml` does
  not auto-enable discovery. Required job (no `continue-on-error`).
- `scripts/ci.sh` extended to lint `scripts/` so the bash smoke wrappers
  get the same ruff treatment as the Python sources.

**Tests added** (extensions + new files)

- `tests/unit/diagnostics/test_usbc.py` (10 tests, extended) — PRESENT
  / MISSING / WARN transitions, deterministic sort-first-match,
  disabled-discovery short-circuit, and the new boot-race
  `by_id_root`-missing guard.
- `tests/unit/diagnostics/test_power_chain.py` — battery probe, motion
  gate, e-stop timing assertion.
- `tests/unit/test_factory_esp32_discovery.py` — every branch of the
  two-condition override chain (disabled / None / literal-exists /
  glob-match / no-match).
- `tests/unit/scripts/test_check_usbc_devices.py` — CLI smoke (exit
  codes, `--json` payload shape, no-orchestrator path).
- `tests/unit/test_serial_driver.py` (extended) — regression test for
  `_read_line` `UnicodeDecodeError`-resistance (the chip emits `0xff
  0xfe` invalid UTF-8 start bytes; decode must not raise).
- `tests/unit/tools/test_dashboard_proxy.py` (extended) — regression
  for the security-auditor finding (`TOKEN == ""` and `_AUTH_HEADER ==
  {}` when neither CLI positional nor env var is set).
- `tests/hardware/test_usbc_enumeration.py` (`@pytest.mark.hardware`) —
  rover-side: every required endpoint resolves on a live Jetson.
- `tests/hardware/test_power_chain_smoke.py` (`@pytest.mark.hardware`) —
  rover-side: battery + e-stop latency within
  `ESP32Config.emergency_stop_budget_ms`.

**Docs added**

- `docs/runbooks/jetson-rover-smoke.md` — operator runbook for the full
  smoke pass, the warm-vs-cold smoke discipline (orchestrator container
  holds FDs; restart before smoke or trust `/api/v1/health` for warm
  signal), the rover-swap by-id drift symptom + override trigger, the
  structlog grep recipes (`usbc_endpoint_*`,
  `esp32_serial_port_overridden`, `power_chain_probe_complete`,
  `esp32_raw_line`), and the triage matrix.
- `docs/architecture/c4-usbc-smoke.md` — Level-3 C4 component diagram
  for the USB-C smoke gate, including the configuration → resolver →
  driver chain, the CI regression-gate side path, and the standalone
  operator probe path.
- `CLAUDE.md` — new "USB-C smoke validation surface" section pinning the
  four contracts (master switch default, factory override chain,
  zero-motion bound, boot-race guard) + the explicit-raise vs. assert
  rule for Jetson Docker entrypoint code.
- `AGENTS.md` — new "USB-C smoke gate — adding a new endpoint" workflow
  + red-flag entries for `assert isinstance` under `PYTHONOPTIMIZE`,
  hardcoded credentials, missing `Path.glob` guards, and `*.bak.*`
  hygiene.
- `SKILLS.md` — three new skill entries: `usbc-smoke-validation`,
  `power-chain-smoke`, `rover-firmware-diagnosis`.

**.gitignore additions**

- `*.bak`, `*.bak.*` — config-edit backup sidecars from `sed -i.bak` /
  `cp foo.yaml foo.yaml.bak.<timestamp>` drills (the rover-swap /
  baud-change operator runbook emits these on the Jetson).
- `/tmp/wave_rover_*`, `/tmp/*.html` — external-doc research scratch
  files (firecrawl / curl caches from looking up Wave Rover wiki +
  `ugv_base_ros` GitHub repo).

**Pre-merge verification** (workstation, post-fix)

- `ruff check src/ tests/ tools/` — clean across the PR's touched files.
- `ruff format --check` — clean.
- `mypy --strict --no-incremental` on `src/mousedroid/diagnostics/` and
  `src/mousedroid/comms/serial_driver.py` — `Success: no issues found`.
- `pytest tests/unit tests/integration tests/property tests/regression
  tests/smoke -m "not hardware and not slow"` —
  **4993 passed, 22 skipped, 8 pre-existing failures, 95.45% coverage**
  (gate is 85%). The 8 failures are subprocess-import issues in
  `tests/unit/test_scripts.py` (last touched in PR #44) and
  `tests/unit/vla/test_distilled_onnx.py` (PR #89) — neither file is
  modified on this branch; the root cause is a Windows-specific
  PYTHONPATH issue in the test subprocess, tracked separately.
- Independent `feature-dev:code-reviewer` subagent: HOLD on 7 findings
  → all Critical + High addressed in-branch (`UnicodeDecodeError`,
  `by_id_root` guard, explicit-raise, `ge=0`); Medium asyncio-mode
  check confirmed `auto`; Lows folded into `NEXT_STEPS.md` for follow-up.
- Independent `security-auditor` subagent: FAIL → addressed (hardcoded
  `dev-dashboard-token` removed; regression test added).

### Fixed — PR #105b: Tech-debt closure (mypy + coverage-script base-ref autodetect)

First half of a two-PR stack (#105a follows). Tiny, isolated, low-risk. Lands FIRST so its `scripts/check_branch_coverage.py` fix can gate PR #105a's own coverage check correctly. No new features, no schema changes; pure debt closure.

**Mypy `--strict` errors closed** — `mypy --strict --no-incremental src/mousedroid/` now reports zero errors (previously two):

- **`src/mousedroid/factory.py:2216-2261`** — tightened the local `inner: object` annotation to `inner: ApprovalGateProtocol` on `build_approval_gate`. Closes the PR-98-introduced `Argument 1 to "PolicyApprovalGate" has incompatible type "object"` error. Added the protocol to the file's `TYPE_CHECKING` block at line 62 so the annotation resolves at static-analysis time without a runtime-import cost.
- **`src/mousedroid/reward/vlm_progress.py:31`** — added `types-cachetools>=5.3` to the `[dev]` extras. Resolves the `[import-untyped]` error without polluting the runtime install (the stub is dev-only). CI's typecheck job installs `[dev]`.

**Coverage-script footgun fixed** — `scripts/check_branch_coverage.py`:

- New `_local_dev_base_candidates()` adds three local-dev fallbacks to the base-ref autodetect chain: (1) upstream-tracking branch via `git rev-parse --abbrev-ref @{u}`, (2) `origin/HEAD` symbolic-ref target, (3) `origin/main` literal. The chain previously only honoured `--base-ref` CLI flag + `GITHUB_BASE_REF` env, which silently failed during PR #104 local invocations and returned empty per-file coverage data.
- `_first_valid_base_ref` now emits a stderr line naming the resolved candidate (or listing the tried set when none resolve). The silent-resolution behaviour was the PR-104 footgun.

**New tests** (regression-guard surface):

- `tests/unit/factory/test_policy_approval_gate.py` (7 tests) — pin every branch of `build_approval_gate` and the `PolicyApprovalGate` inner-protocol-conformance invariant the mypy fix represents.
- `tests/unit/scripts/test_check_branch_coverage_base_ref.py` (10 tests after harden — 6 base + 4 harden-fix) — tmp-dir git-repo sandbox exercising each fallback leg + CLI/env precedence + stderr-logging invariant + the new `COVERAGE_FALLBACK_BASE_REF` env override + the `_SCRIPT_TAG` constant + the `resolved_base` double-print suppression.
- `tests/regression/test_pr105b_mypy_clean.py` (1 `@pytest.mark.slow` test) — subprocess-runs `mypy --strict --no-incremental` on the two touched files + asserts `Success`. Wall time ~282 s on workstation; timeout sourced from `MYPY_TIMEOUT_S` env (default 300 s, harden-fix #3). CI's typecheck job runs it.

**Harden pass (gap-analysis follow-up)** — addressed 7/8 findings from an independent peer-review scan; defers the 8th (importlib helper consolidation — 6+ call sites, scope creep):

- **#1 double stderr print** — `_first_valid_base_ref` was called by BOTH `_changed_source_files` and `_changed_line_map`, emitting the structured `resolved base ref` line twice per `main()` invocation. Fixed by resolving once in `main()` + threading the result through both consumers via a new `resolved_base` keyword arg. Pinned by `test_resolved_base_threaded_through_avoids_double_print`.
- **#2 hardcoded `"origin/main"` literal** — clones whose remote default branch is `master`/`develop`/`trunk` previously fell off the chain. Replaced with `_fallback_base_ref()` reading the new `COVERAGE_FALLBACK_BASE_REF` env var; default `"origin/main"` preserves prior behaviour exactly. Pinned by `test_fallback_base_ref_env_overrides_default` + `test_fallback_base_ref_defaults_to_origin_main`.
- **#3 hardcoded `timeout=300` in mypy regression test** — slow CI runners (or the Jetson) might need more. Replaced with `_MYPY_TIMEOUT_S = int(os.environ.get("MYPY_TIMEOUT_S", "300"))`; operator overrides via env.
- **#4 hardcoded stderr prefix `"[check_branch_coverage]"`** — appeared in two `print` calls. Centralised in a module-level `_SCRIPT_TAG: Final[str]` constant + both call sites reference it. Pinned by `test_script_tag_constant_used_by_stderr_lines`.
- **#6 `_git` helper naming collision with script's `_run`** — renamed to `_git_checked` to make the `check=True`-raises-on-failure contract visible vs the script's `check=False`-returns-CompletedProcess `_run`.
- **#7 `types-cachetools>=5.3` pin wider than runtime `cachetools>=5.0`** — environments resolving runtime `cachetools<5.3` got mismatched stubs. Lowered to `>=5.0,<6` so stub range matches the runtime contract + caps the major to defend against a hypothetical breaking 6.0 release.
- **#8 tier rationale undocumented** — added a "Tier rationale" paragraph to both new test file docstrings: this is script-/annotation-level utility code, so unit + slow-regression are the canonical tiers; integration / e2e / property / hardware tiers are N/A — formally considered + declined.
- **#5 importlib-helper consolidation DEFERRED** — same `spec_from_file_location` pattern appears in 6+ test files across the repo. The right fix is a shared `tests/conftest.py` helper; out of scope for this PR. Tracked as future tech-debt.

**Verification** — all green on workstation (post-harden):

- `ruff check src/ tests/` (pinned `ruff==0.8.0`, matches CI) — clean
- `ruff format --check src/ tests/` — 762 files already formatted
- `mypy --strict --no-incremental src/mousedroid/` — Success: no issues found in 289 source files
- PR-105b suite (17 tests: 7 factory + 10 script-helper) — all passing
- Full non-hardware-non-slow suite: **4,965 passed, 22 skipped, 10 deselected** in 169 s
- Branch coverage gate: `factory.py` 100% on the 1 changed line (gate ≥85% held)
- Mypy regression guard (`test_pr105b_mypy_clean.py`) — passes in 282 s, env-overridable via `MYPY_TIMEOUT_S`
- Coverage script self-validation: `scripts/check_branch_coverage.py --base-ref HEAD --min 0` → `resolved base ref: origin/HEAD`, gate passed; no double-emit of the resolved-ref line
- Security audit (security-auditor subagent): PASS — no hardcoded credentials, list-form subprocess argv, read-only git ops
- Independent code reviewer (feature-dev:code-reviewer subagent): APPROVE on the pre-harden surface; gap-analysis pass found 8 findings (7 addressed inline, 1 deferred as scope creep)

### Added — PR #104 harden-3: Test pyramid expansion + project-wide doc hardening + reviewer-follow-up fixes

Final pre-PR pass that closes the dashboard-stability sprint. Builds on the PR #104 harden-1 (smoke hardening) and harden-2 (live-dashboard enablement) blocks below.

**Test pyramid expansion — 6 new files, 58 tests + 3 hardware-gated**

- `tests/integration/test_pr104_esp32_disabled_integration.py` (8) — `build_esp32_driver` end-to-end through `ResilientESP32Driver` + `MockESP32Driver`; concurrent `send_velocity` fan-out.
- `tests/e2e/test_pr104_dashboard_e2e.py` (5) — `MockCamera` → in-process upstream → `dashboard_proxy` → aiohttp client; bearer-token injection + JPEG round-trip + 503 propagation.
- `tests/regression/test_pr104_backwards_compat.py` (9) — CLAUDE.md invariant #9; defaults pinned (`esp32.enabled=True`, `v4l2_grayscale_extract=True`, `snapshot_jpeg_quality=90`); standalone YAML roots parse; Pydantic ge/le range guards.
- `tests/regression/test_pr104_aqa.py` (21) — Automated QA on schema-field hygiene (description ≥20 chars + documented default reachable via `FieldInfo`); protocol conformance for `MockCamera` + `JetsonCSICamera` (`RawFrameSourceProtocol`) and `MockESP32Driver` (`ESP32CommProtocol`); RFC-9110 §7.6.1 hop-by-hop blocklist parametrized over `dashboard_proxy._HOP_BY_HOP`; env-override surface (`MOUSEDROID_ESP32__ENABLED=false`).
- `tests/smoke/test_pr104_sanity.py` (13) — sub-second module-import smoke; YAML round-trip preserves PR-104 fields; standalone YAML root validation.
- `tests/hardware/test_pr104_jetson_dashboard.py` (3 hw-gated) — rover-side mirror: live JetsonCSI JPEG decode via Pillow, factory wires `MockESP32Driver` on the Jetson, orchestrator boots + stops cleanly with `esp32.enabled=False`.

**C4 architecture documentation**

- `docs/architecture/c4-overview.md` — Level 1 (Context) + Level 2 (Container) for the whole system, workstation ↔ Jetson topology with the dashboard proxy.
- `docs/architecture/c4-dashboard-proxy.md` — Level 3 (Component) for the proxy with HTTP + WebSocket sequence diagrams + configuration precedence + failure-mode matrix.
- `docs/architecture/c4-orchestrator.md` — Level 3 for the 30 Hz sense-plan-act loop with the factory-wiring branch diagram (PR #104 `esp32.enabled` branch emphasised) + lifecycle sequence.
- `docs/architecture/c4-arm-platform.md` — Level 3 for the four-layer hierarchical arm reasoning architecture + curriculum state diagram + reused-modules matrix.

**Agentic-worker contract surface**

- Top-level `AGENTS.md` — behavioural rules for Claude Code + subagents + MCP clients (factory-first DI, schema-driven config, structured logging, asyncio, strict typing, backwards-compat, `torch.no_grad()`, test-pyramid discipline, commit-message tone, red flags).
- Top-level `SKILLS.md` — capability index keyed by trigger phrase. Maps operator skills (`dashboard-proxy`, `live-camera-verification`, `esp32-disconnected-mode`, `preflight-validation`) + engineering skills (`add-schema-field`, `add-hardware-driver`, `run-pre-pr-validation`) + subagent dispatch patterns to the files + commands needed.

**Docs updates**

- `CLAUDE.md` — new "Dashboard live-verification surface" section documenting the three PR-104 schema toggles + dashboard proxy invariants + test-pyramid mirror table.
- `README.md` — new "Workstation Dashboard Verification (PR #104)" section (proxy quickstart + dashboard-mode escape-hatch table) + new "Next Steps / Roadmap" section with 5-item forward roadmap.
- `.gitignore` — added `torch-baseline-*.txt`, `workstation-smoke-*.log`, `coverage-pr104-*.json`, the literal `%SystemDrive%/` Windows-shell stray, mock-smoke snapshot artefacts, `.vscode/launch.local.json`.

**Reviewer-follow-up fixes (harden-3-review-fixes)**

Independent code review surfaced 4 issues; all addressed before push:

- **HIGH — WS pipe pool-slot leak** (`tools/dashboard_proxy.py:_ws_handler`) — replaced `asyncio.gather` of two pipe coroutines with `asyncio.create_task` + `asyncio.wait(..., return_when=FIRST_COMPLETED)` + explicit task cancellation. Without the fix, surviving pipe blocks indefinitely after one-sided close, holding `TCPConnector` pool slots (limit=64).
- **HIGH — misleading bearer-token startup log** (`tools/dashboard_proxy.py:main`) — was emitting `[proxy] auth bearer token: ...` even when TOKEN was empty (auth injection legitimately disabled). Now three faithful states logged.
- **MEDIUM — `_http_handler` upstream not released if `out.prepare` raises** — wrapped the downstream-write block in `try/finally` so the upstream `ClientResponse` is released on client-disconnect-during-prepare.
- **MEDIUM — `_frame_to_rgb_for_snapshot` 2-D luma frame `IndexError`** — added explicit `elif frame.ndim == 2` guard cloning the luma plane to RGB.

Plus 3 new tests covering the fixes:
- `tests/unit/tools/test_dashboard_proxy.py::test_websocket_text_message_round_trips`
- `tests/unit/tools/test_dashboard_proxy.py::test_websocket_upstream_close_propagates_to_client`
- `tests/unit/test_jetson_csi.py::test_frame_to_rgb_2d_luma_frame_cloned_to_rgb_without_crash`

**Verification**

- **Tests**: 130 passing across the combined PR-104 surface; 3 hardware-gated tests skip cleanly on workstation.
- **Ruff**: `check` + `format --check` clean across all touched files.
- **Mypy**: `mypy --strict` clean on touched src files.
- **Branch coverage** (vs PR-104 base commit `8f89186`): `schema.py 100%`, `factory.py 100%`, `jetson_csi.py 100%`, `validation/runtime.py 85.71%`. Gate held.
- **Security audit**: clean — no hardcoded production credentials, RFC-compliant hop-by-hop stripping, intentional loopback-only proxy scope, test sentinels properly `noqa: S105`-tagged.
- **Independent code review**: APPROVE_WITH_FIXES → APPROVE after the 4 review findings fixed + verified.

### Added — Live-dashboard E2E enablement (PR #104 harden-2)

Resolution of the gap-analysis + tech-debt findings discovered while running the live Jetson dashboard end-to-end. All changes backwards-compatible (new schema fields default to legacy behaviour).

- **`ESP32Config.enabled: bool = Field(True)`** + factory wiring — `build_esp32_driver` now returns `MockESP32Driver` whenever `esp32.enabled is False`, regardless of `mock_hardware`. Replaces the prior workaround of monkey-patching `orchestrator.start()` to swallow ESP32 connect failures (see PR #104 harden-2 conversation): operators running the orchestrator on a Jetson WITHOUT the motor controller plugged in (camera + LiDAR + Hailo dashboard verification, hardware bring-up) flip the flag in their YAML overlay and the full orchestrator pipeline runs at real-hardware speeds — no patches, no open circuit breakers dragging the tick rate down.
- **`JetsonCSICamera.capture_raw_jpeg()`** implementing `RawFrameSourceProtocol` — `/camera/frame.jpg` and `/camera/stream` now register (previously HTTP 404 because the driver only implemented `VisionProtocol`). Three backend-specific colour paths: `jetson_utils` (already RGB), `gstreamer` (BGR → RGB swap), `v4l2` (workaround for IMX708-on-RG10-Bayer-via-V4L2; see new schema field below). Encoded via Pillow at `cfg.camera.snapshot_jpeg_quality`.
- **`CameraConfig.v4l2_grayscale_extract: bool = Field(True)`** — workaround toggle for the JetsonCSICamera's V4L2 fallback path. When the container lacks the `nvarguscamerasrc` GStreamer plugin, the IMX708 sensor's RG10 Bayer raw output gets misinterpreted as YUYV by OpenCV → solid green output. With the workaround on (default), the green channel (which carries the actual luma signal) is extracted as grayscale and cloned across R/G/B so operators see the scene (with mosaic artefacts) instead of nothing. Set `False` once the container rebuilds with proper `nvarguscamerasrc` support.
- **`tools/dashboard_proxy.py`** — workstation-side reverse proxy that forwards HTTP + WebSocket traffic from a local port to a configurable upstream (the Jetson telemetry server, Grafana, Prometheus, …) with optional bearer-token injection. Used to make the auth-gated mousedroid telemetry server (port 8080) + the no-auth Grafana (3000) + Prometheus (9090) all browsable from a single Claude Preview session. CLI args + env-var configurable; tests round-trip through an in-process aiohttp upstream so we never need to bind to 192.168.55.1 during CI.
- **`launch_dashboard.ps1`** + **`config/dev_dashboard.yaml.example`** — PowerShell launcher + dev YAML overlay template. The example overlay disables the in-process llama.cpp LLM (operators can wire LM Studio via the existing `openai_compatible` backend), enables telemetry with `force_real_server`, and switches the camera to `mock_source: screen_capture` for desktop content. `dev_dashboard.yaml` is gitignored so operator-personal values (LM Studio model name, etc.) don't leak.

### Added — Hardware smoke hardening (PR #104 follow-up)

Resolution of the gap-analysis + tech-debt findings on the smoke-test PR. All changes are backwards-compatible (new schema fields default to the previously-hardcoded values).

- **Schema-driven thresholds** — four new Pydantic fields replace the previously-hardcoded literals in `mousedroid.validation.runtime`:
  - `camera.snapshot_jpeg_quality: int` (default `90`, range 1-100) — Pillow JPEG quality for the `--save-frame` snapshot encoder
  - `experience.nvme_device: str` (default `/dev/nvme0n1`) — `smartctl` target
  - `experience.nvme_partition: str` (default `/dev/nvme0n1p1`) — `findmnt` target
  - `experience.diagnostics_subprocess_timeout_s: float` (default `10.0`) — per-tool timeout for `lspci` / `lsblk` / `smartctl` / `findmnt`
  - `hailo.synthetic_input_shape: tuple[int, int, int]` (default `(640, 640, 3)`) — zero-tensor shape for the Hailo synthetic-inference round-trip
- **Schema-driven HEF role inventory** — `verify_hailo_accelerator` now derives the HEF role list from `HailoConfig.model_fields` (any field ending in `_hef_path`) instead of a hardcoded `("yolo", "feature_extractor")` tuple. Adding a third HEF role (e.g. `depth_hef_path`, `segmentation_hef_path`) flows into the smoke automatically.

### Fixed — Hardware smoke hardening

- **Logging hygiene** — replaced plain `import logging` + `logging.getLogger` in `verify_hailo_accelerator`'s `finally` with the project-mandatory `mousedroid.logging.setup.get_logger` so smoke stop-failures route through the same structlog processor chain as the rest of the orchestrator (CLAUDE.md invariant 4).
- **`_resolve_pcie_ssd_mount` rootfs-parent FALSE PASS** — removed the `cfg.experience.path.parent` fallback. On a freshly-imaged Orin Nano with no NVMe at all, the previous chain would accept `/home/jetson/` (the rootfs!) as the "SSD mount" and report the LMDB path as "on SSD" — defeating the entire point of the check. The smoke now SKIPs cleanly when neither `$MOUSEDROID_SSD_MOUNT` nor `findmnt /dev/nvme0n1p1` can pin the mount.
- **`infer_sync` event-loop block** — wrapped the Hailo synthetic-inference call in `asyncio.to_thread` so the smoke does not stall the asyncio event loop during the (potentially tens-of-ms) blocking PCIe VStream call. Mirrors how `HailoRuntime.start()` dispatches its own blocking calls.
- **Dead `_device_id`/`_fw_version`/`_arch` reflection** — `HailoRuntime` never assigns these attrs, so the `getattr` loop produced a perpetually-empty `device_info` dict. Replaced with two concrete operator-meaningful signals: `device_path` (resolved from `cfg.hailo.device_path`) and `models_loaded` count (derived from the HEF inventory).
- **Misleading `is_available()` signal** — `runtime.is_available()` returns `False` if HEFs failed to load even when the device was found, producing confusing PASS/FAIL signals. Removed; the new `device_info` keys carry concrete signals operators can interpret.
- **Sync `capture_raw_frame` crash** — `_resolve_raw_frame_capture` now `asyncio.iscoroutinefunction`-checks the method and wraps sync drivers in `asyncio.to_thread`. The previous code would `await` a non-coroutine and produce a confusing `TypeError` deep inside `capture_camera_frame`.
- **`_via_jpeg` Pillow import** — wrapped in `try/except ImportError` so bare `[dev]` CI installs (without `[hardware]` or `[telemetry]` extras) get a clear operator-actionable `RuntimeError` instead of a bare ImportError traceback.
- **Dead-defensive `getattr(cfg.hailo, "fallback_on_failure", True)`** — replaced with direct attribute access. The `HailoConfig` field is guaranteed by the schema; the wrapper would silently swallow a future rename.

### Documentation — Hardware smoke hardening

- `docs/operations/jetson_smoke_runbook.md` — three new common-failure sections:
  - `$MOUSEDROID_SSD_MOUNT` operator override for non-standard mount points
  - YAML override pattern for non-canonical `experience.nvme_device` / `nvme_partition`
  - `frame shape FAIL` interpretation when running with `MOUSEDROID_MOCK_HARDWARE=true` on a dev host (MockCamera 320×240 vs default 640×480)

### Added — Hardware smoke (post-adjust evidence + PCIe SSD + Hailo-8)

Three additive sensor-verification flows that extend the existing `scripts/verify_sensors.py` + `scripts/jetson_smoke_test.sh` harness so the operator can validate the rover end-to-end after a hardware change. Zero new top-level deps; every threshold and path comes from existing Pydantic config or a documented `MOUSEDROID_*` env override.

- **Camera snapshot capture** (`scripts/verify_sensors.py --sensor camera --save-frame PATH --frames N`). Writes a real JPEG snapshot of the LAST captured frame so the operator can visually verify focus/exposure/framing after a ribbon-cable / lens / refocus adjustment. JPEG encoding happens in the validation helper via Pillow (already a project dep via `[telemetry]` / `[hardware]` extras). New `CameraFrameDiagnostics` frozen dataclass carries the frame + per-call timing + saved-JPEG path. `capture_camera_frame()` return shape changed from `tuple[NDArray, str]` to `tuple[CameraFrameDiagnostics, str]` — the 2-tuple form is preserved so the existing destructure in `verify_sensors.py::check_camera` keeps working; the consumer is updated atomically in the same commit. Backend resolution chain widened: `capture_raw_frame()` → `capture_raw_jpeg()` + Pillow decode → private `_capture_frame` legacy, covering both `IMX500Camera` and `MockCamera`.
- **PCIe NVMe SSD smoke** (`scripts/verify_sensors.py --sensor pcie_ssd`). Probes `lspci` / `lsblk` / `findmnt` / `smartctl` (each via `shutil.which`-guarded `subprocess.run`; missing tools are SKIPs not FAILs) and asserts the configured runtime paths (`experience.path`, `jetson.tensorrt_cache_dir`, `cloud.weight_update.cache_dir`, `harness.journal.path`) resolve to a mount with ≥ `cfg.experience.map_size_gb` free capacity. Mount detection uses `$MOUSEDROID_SSD_MOUNT` env override, falling back to `findmnt /dev/nvme0n1p1`, then to the parent dir of `cfg.experience.path`. New `PcieSsdDiagnostics` frozen dataclass.
- **Hailo-8 smoke** (`scripts/verify_sensors.py --sensor hailo`). Runs the same `HailoRuntime` the orchestrator uses (via `build_hailo_runtime()`) — dumps best-effort device info, loads `cfg.hailo.yolo_hef_path` + `cfg.hailo.feature_extractor_hef_path`, and times one synthetic inference against `cfg.hailo.timeout_ms`. SKIP semantics on missing SDK / missing `/dev/hailo0` / `cfg.hailo.enabled=False` (Hailo is opt-in extras and the runtime is documented to fall back to GPU). `try/finally` around `runtime.start()` / `stop()` so the PCIe device lock is ALWAYS released even if inference raises. Input-shape resolution falls back to the YOLO-canonical `(640, 640, 3)` when the runtime API doesn't expose vstream shapes. New `HailoDiagnostics` frozen dataclass.
- **Bash harness dispatch** (`scripts/jetson_smoke_test.sh`). Two new dispatch entries: `bash scripts/jetson_smoke_test.sh pcie_ssd` (`ssd` is an alias) and `bash scripts/jetson_smoke_test.sh hailo`, both delegating to the existing `_run_verify_sensor` aggregator for PASS/SKIP/FAIL accounting. Inclusion in the `all` aggregator after `speaker` and BEFORE `app` so device-lock collisions (PCIe NVMe, Hailo) surface before the app-health step.
- **Operator runbook** (`docs/operations/jetson_smoke_runbook.md`). One-page post-hardware-change playbook: pre-flight (ping + SSH key + venv + systemd service check), rsync with `.gitignore`-aware filtering, three smoke commands with `tee` transcript logging, camera-snapshot SCP-back workflow, PASS/SKIP/FAIL interpretation table, common failure modes (incl. Hailo module-name discovery via `lsmod | grep hailo` rather than hardcoded `modprobe hailo_pci`), and the safe reseat protocol (power off + ground first).

### Fixed — Ops Hardening (F-006 / F-009 / F-013 / F-014 follow-ups from PR #100)

The smoke-stability sprint (PR #100) live-verified the rover at `192.168.55.1` and surfaced five operator-actionable findings. This sprint closes four of them in code (the fifth — Hailo PCIe wiring — is an operator hardware decision):

- **F-006 — LLM CPU-only inference at 0.5 tok/s** (`config/jetson_production.yaml:57` + `src/mousedroid/config/loader.py:91-130`). Flipped the production overlay's `llm.n_gpu_layers: 0` → `-1` (offload every layer to the iGPU; the schema default at `config/schema.py:620` was already `-1`, only the overlay was downgrading it). Extended `load_settings` to honour **nested** env-var precedence (`MOUSEDROID_LLM__N_GPU_LAYERS=0` now beats the overlay yaml) — the prior loader only handled top-level `MOUSEDROID_<KEY>`, silently dropping nested overrides. Three regression tests in `tests/unit/llm_gateway/test_gateway_n_gpu_layers_env_override.py` pin env-beats-overlay, schema-default-without-overlay, and the `=0` CPU fallback.
- **F-009 — TensorRT silent-mock observability** (`src/mousedroid/factory.py:1855-1900`). Consolidated the two log events (`tensorrt_compiler_built` vs `tensorrt_compiler_mock_built`) into one with a `backend=real|mock` label so dashboards don't have to merge event-name strings. Added a `torch2trt_available` field to both branches that reports the truthful library state (previously the mock branch hardcoded `False` even on dev hosts where the library WAS installed but tensorrt was disabled in cfg). Three regression tests in `tests/unit/factory/test_build_tensorrt_compiler.py`.
- **F-013 — stale `/etc/mousedroid/jetson_production.yaml` drift** (`scripts/sync_jetson_overlay.sh`). Strengthened the existing operator-deployed sync script: added a `--verify` flag (read-only hash-compare; non-zero on drift), promoted the previous silent-skip to a `WARN overlay_sync_source_missing` audible event, added `overlay_sync_match` / `overlay_sync_replacing` / `overlay_sync_replaced` / `overlay_sync_drift` structured-log events on every code path. Six integration tests in `tests/integration/test_sync_jetson_overlay.py` (Windows-skipped due to WSL/Git-Bash subprocess unreliability; Linux CI exercises end-to-end). New runbook section in `docs/operator/JETSON_SMOKE_RUNBOOK.md` documents the operator commands.
- **F-014 — compose env-file directive + env-file-sourced hardware/token settings** (`docker-compose.jetson.yml:26-47`). Added long-form `env_file: [{path: /etc/mousedroid/docker.env, required: false}]` directive (the `required: false` is critical — first-time bringup and CI lint runs would crash with "env file not found" otherwise). The compose file deliberately **omits** `MOUSEDROID_MOCK_HARDWARE` and `MOUSEDROID_TELEMETRY_TOKEN` from its inline `environment:` block so `/etc/mousedroid/docker.env` is the single source of truth for both (Compose precedence: inline `environment:` ALWAYS overrides `env_file:`, so any inline default with `${VAR:-}` would silently mask the docker.env value when the host shell var is unset — the very crash-loop F-014 is meant to fix). Five regression tests in `tests/integration/test_compose_jetson_env_file.py` pin `env_file` declared with `required: false`, MOCK_HARDWARE absent from inline, TELEMETRY_TOKEN absent from inline, and the operator template at `config/.env.jetson.example`.

### Notes — PR-A2 status

Per `docs/planning/PHASE_2_1_AND_BEYOND_PLAN.md` PR-A2 (Phase 3b `[vla]` CI matrix + Tier-B1 telemetry counters): planning-time verification found that **both halves were already shipped on the base branch** despite a stale CHANGELOG note:

- Phase 2.1 PR-A1 (BC into PPO): `training/train_offline_rl.py:165-198` has the full bc_update call site + `offline_rl_bc_active` structured log + `losses.update(bc_losses)` aggregation. 8 tests pass in `tests/integration/test_phase21_bc_into_offline_rl.py`.
- PR-A2 `[vla]` CI matrix: a fully-promoted (non-advisory) `vla-extras` job already exists at `.github/workflows/ci.yml:219`. A reviewer-flagged standalone advisory copy was deleted as redundant.
- PR-A2 Tier-B1 telemetry counters (`replay_records_total`, `vla_inference_seconds`, `vla_timeout_total`, `vlm_progress_cache_hits/misses`): all four are declared in `MetricsRegistry` (`src/mousedroid/telemetry/metrics.py:580-586`) and wired at their call sites in `training/replay/lmdb_reader.py`, `vla/policy.py`, `orchestrator/orchestrator.py`, and `reward/vlm_progress.py`. Five existing tests pin them (`tests/integration/test_writer_side_*.py` + `tests/smoke/test_writer_side_metrics_smoke.py`).

### Added — Smoke-Test Stability Pass

- **`run_preflight(cfg) -> PreflightReport`** async API (`src/mousedroid/validation/preflight.py`) — replaces the shell-only `scripts/preflight_check.sh` flow with a Pydantic-typed report. Six built-in checks (camera, microphone, speaker, lidar, esp32, config) reuse `validation/runtime.py` helpers; per-check exceptions are caught and recorded as FAIL entries (never bubble).
- **`validate_all_pillars(cfg)` + `python -m mousedroid.cli.validate_pillars` CLI** (`src/mousedroid/validation/pillars.py` + `src/mousedroid/cli/validate_pillars.py`) — dispatch-table over the 10 pillars from `TEN_PILLARS_VALIDATION.md`. Six pillars use Pattern A (factory-builder smoke); four (continual / meta / scaling / growth) use Pattern B (in-process `pytest.main` delegation) because their `build_*` factories don't yet exist. CI runs the `--dry-run` variant on every commit between typecheck and tests.
- **SSD1306 face display smoke test** (`tests/hardware/test_face_display_smoke.py`) — exercises `build_face_display` + `build_face_controller` + the `fallback_to_mock_on_error` path.
- **Hailo accelerator smoke test** (`tests/hardware/test_hailo_smoke.py`) — gated by `pytest.importorskip("hailort")` + `is_jetson_host()`; covers disabled / mock-fallback / mock-inference branches.
- New structured log events: `preflight_{start,complete,check_exception}` and `pillar_validation_{start,complete}` + `pillar_check_exception` for operator dashboard ingestion.

### Fixed — Pre-existing Windows-host infra failures

- `tests/smoke/test_telemetry_smoke.py::test_publisher_initial_stats_are_zero` — inclusive assertion that auto-inherits new counters (`lidar_raw_published` / `lidar_raw_dropped` added by Tier C1).
- `tests/integration/test_docker_gpu.py::TestContainerEnvironment::test_nvcc_available` — `@pytest.mark.skipif(shutil.which("nvcc") is None, ...)`.
- `tests/unit/test_jetson_smoke_orchestrator.py` (10 tests) — module-level `pytest.mark.skipif` on Windows hosts without `python3` reachable from the bash subprocess. Tests still RUN on Linux / Jetson hosts (operator runbook validates this).

### Docs — Smoke pass

- `docs/operator/JETSON_SMOKE_RUNBOOK.md` — step-by-step rover validation runbook.
- `docs/planning/SMOKE_REPORT_TEMPLATE.md` — empty template the operator fills after running the runbook.

### Backwards compatibility — Smoke pass

- `scripts/preflight_check.sh` retained unchanged as the bash entry point; new programmatic API is an addition, not a replacement.
- Three pre-existing Windows-host failures now SKIP cleanly with documented reasons. Tests still RUN on Linux / Jetson hosts.
- No new runtime dependencies — every new module reuses existing `validation/runtime.py` helpers + `factory.py` builders.

### Added — Live-Jetson verification + diagnostics (smoke pass, second phase)

- **`tools/lidar_telemetry_probe.py`** — standalone non-orchestrator LiDAR → telemetry publisher → telemetry server → `/ws/v1/lidar/raw` WS-client probe. Lets the operator verify the dashboard pipeline end-to-end even when the rover is detached (ESP32 / CSI not present): the probe binds a non-default port (8090) so it never collides with the running orchestrator on 8080.
- **CSI-ribbon-disconnect diagnostic** in `_check_camera` — `_detect_csi_ribbon_disconnect(video_nodes=…, modules_text=…)` distinguishes the operator-actionable "reconnect the ribbon" case from a real driver bug. The detector accepts injected `/proc/modules` text and `/dev/video*` node lists so it's unit-testable without root or Jetson hardware. The check helper surfaces it as `WARN`, not `FAIL`.
- **`python -m mousedroid.cli.preflight`** — argparse wrapper over `run_preflight(cfg)` (mirrors the `validate_pillars` CLI). Flags: `--config`, `--checks` (subset filter), `--mock-hardware`, `--json`. Exit 0 on OK or DEGRADED (WARN-only); exit 1 only on FAIL.
- **Findings F-006 → F-014** documented in `SMOKE_REPORT.md` addenda A+B+C from the live Jetson run at `192.168.55.1`. Notably: F-006 (Phi-3-mini at 0.52 tok/s = 260 s per `translate_mission` — operator fix is `llm.n_gpu_layers: -1`), F-013 (stale `/etc/mousedroid/jetson_production.yaml` was missing `mock_force_real_when_enabled: true` → telemetry server bound nothing), F-014 (compose default `MOUSEDROID_MOCK_HARDWARE=${VAR:-true}` overrides the env_file's `false`).

### Fixed — Reviewer findings (PR-prep)

- `mousedroid/cli/validate_pillars.py` — exit code 0 on `DEGRADED` (WARN-only), not just `OK`. Matches the preflight CLI contract; prevents CI false-fails on warn-only runs. Pinned by `test_cli_exits_0_when_overall_status_is_degraded`.
- `mousedroid/validation/pillars.py` — Pattern-B `_PYTEST_DELEGATION_PATHS` now resolves against a module-level `_REPO_ROOT` (`Path(__file__).resolve().parents[3]`) instead of relying on `os.getcwd()`. The dispatcher works regardless of which directory the operator invokes the CLI from.
- `mousedroid/validation/pillars.py` — replaced `assert x is not None` smoke checks in Pattern-A pillar checks with explicit `if x is None: return _fail(...)` blocks. `assert` is stripped under `python -O` (the standard Jetson Docker entrypoint typically sets `PYTHONOPTIMIZE=1`) so the previous code silently returned `OK` on a `None` factory result. The new pattern is correct under any optimisation level.
- Added missing coverage: `test_check_camera_returns_warn_when_ribbon_disconnect_detected` proves the end-to-end WARN propagation from `_detect_csi_ribbon_disconnect` through `_check_camera`.

### Added — Tier C2.3: Mission Lifecycle Activation

- **`OpenAICompatibleLLMGateway`** (`src/mousedroid/llm_gateway/openai_compatible.py`) — new HTTP backend implementing `LLMGatewayProtocol` against `{base_url}/v1/chat/completions`. Talks to Ollama (default at `http://127.0.0.1:11434`), LM Studio, OpenAI, or any OpenAI-compatible endpoint. Selected via `cfg.llm.backend = "openai_compatible"`. Always returns a neutral `GoalVector` on transport / parse failures so the orchestrator never crashes on a misbehaving LLM. API key stored as `SecretStr` and forwarded as `Authorization: Bearer …` only (never logged).
- **`LLMGatewayMissionReplanner`** (`src/mousedroid/orchestrator/llm_replanner.py`) — adapter implementing `MissionReplannerProtocol` on any `LLMGatewayProtocol`. Built by `build_mission_replanner(cfg, *, llm_gateway, metrics)` when `cfg.mission.llm_replanner_enabled=True` and a gateway is wired. Augments the goal text with `(last_progress=<float>)` (toggle via `replanner.include_progress_in_prompt`) and clips at `replanner.max_prompt_chars`. Counts outcomes (`ok | degraded | exception`) via the new `mission_replan_llm_calls_total` Prometheus counter.
- **`build_vlm_progress(cfg)`** factory — builds a `VLMProgressHead` (reusing `cfg.reward.vlm_progress` sub-block) with the existing `MockVLMProgress` backend whose value comes from `cfg.mission.vlm_mock_progress_value`. Default off; pre-Tier-C2.3 deployments byte-identical.
- **`build_orchestrator`** now threads both new dependencies into `build_mission_lifecycle`, so the orchestrator's POST_TICK lifecycle seam is no longer a permanent no-op once the operator enables the three new flags.
- **Five new fields on `LLMConfig`** (`backend`, `base_url`, `model_name`, `api_key`, `request_timeout_s`) — env-driven via the existing `MOUSEDROID_LLM__*` Pydantic prefix.
- **Four new fields on `MissionConfig`** (`vlm_progress_enabled`, `vlm_mock_progress_value`, `llm_replanner_enabled`, `replanner`) + new nested `MissionReplannerConfig`. All defaults preserve byte-identical pre-Tier-C2.3 behavior.
- New Prometheus counter `mousedroid_mission_replan_llm_calls_total{outcome=ok|degraded|exception}` registered in `MetricsRegistry`.
- Tests: 49 new across `tests/unit/{config,factory,llm_gateway,orchestrator,telemetry}/`, plus the closed-loop integration (`tests/integration/test_mission_lifecycle_closed_loop.py`), boot smoke (`tests/smoke/test_mission_lifecycle_smoke.py`), Hypothesis property pinning terminal-state absorption, and an E2E mission-success path appended to `tests/e2e/test_full_pipeline.py`.

### Backwards compatibility — Tier C2.3

- Existing YAML loads unchanged. `cfg.llm.backend` defaults to `"llama_cpp"`. All three new boolean gates on `MissionConfig` default to `False`. `cfg.llm.api_key` defaults to `None` so anonymous local Ollama works without env vars.
- The `OpenAICompatibleLLMGateway` is opt-in via `cfg.llm.backend="openai_compatible"`; the legacy in-process `LLMGateway` remains the default and is unchanged.
- Env overrides via the established `MOUSEDROID_LLM__*` prefix (e.g. `MOUSEDROID_LLM__BASE_URL=http://192.168.55.1:11434` to point the Jetson at the host-PC LLM over the documented USB-network bridge).

### Added — Tier C hardening (gap + tech-debt closure)

- **Typed `EngineType` discriminator** — `mousedroid.cloud.protocol` now exports `EngineType: TypeAlias = Literal["policy", "world_model"]` plus `ENGINE_TYPE_POLICY` / `ENGINE_TYPE_WORLD_MODEL` constants. Factory, orchestrator, and `HuggingFaceWeightUpdatePoller` all switch from bare string literals to the typed constants; a typo at any call site now fails `mypy --strict` instead of silently dead-lettering as `cloud_weight_update_unknown_engine_type` at runtime.
- **`engine_type` property** exposed via a new opt-in extension protocol `EngineTypedWeightUpdatePollerProtocol` (subclass of `WeightUpdatePollerProtocol`). The base `WeightUpdatePollerProtocol` deliberately does NOT require `engine_type` so external pollers predating Tier C1.2 keep satisfying it structurally; the orchestrator's legacy-kwarg fold-in path queries `getattr(poller, "engine_type", getattr(poller, "_engine_type", ENGINE_TYPE_POLICY))` to support all three shapes (extension protocol, legacy private attribute, missing-default-to-policy).
- **Two schema-driven fields** on `WeightUpdatePollConfig` — `upload_extensions: tuple[str, ...]` (default `(".onnx", ".pt", ".npz", ".json", ".safetensors")`) and `gcs_artifact_prefix: str` (default `"trained/"`). The cloud-trainer leg of the OTA loop resolves both from `Settings` instead of carrying hardcoded literals; operators can extend the filter or change the prefix without code changes.
- **Footgun validator** on `WeightUpdatePollConfig`: when `world_model_enabled=True` and `world_model_repo_id` is still the maintainer's default repo, the validator logs a structured warning at config-load time (`world_model_poller_using_default_repo`) so an operator who enables the WM poller without overriding the repo gets a loud warning instead of silently OTA-deploying from someone else's HF Hub repo.
- New tests across the hardening surface: `tests/unit/cloud/test_engine_type_protocol.py` (Literal + property + protocol conformance), `tests/unit/orchestrator/test_weight_update_kwarg_precedence.py` (kwarg precedence + back-compat fallback chain), 5 extra `tests/unit/cloud/test_weight_update_poll_config.py` cases (new fields + validator branches), 8 extra `tests/unit/training/test_upload_weights_cloud_sync.py` cases (`main()` CLI flow, lazy GCS client, empty filename, custom extensions, model-card metadata branch), and an empty-vision-features test in `tests/unit/orchestrator/test_mission_lifecycle_wiring.py`.

### Fixed — Tier C hardening

- `training/upload_weights.py` now uses `mousedroid.logging.setup.get_logger` (the project's mandatory structlog setup) instead of calling `structlog.get_logger` directly — the prior call bypassed the project's processor chain (JSON renderer, contextvars, bound fields), so cloud log aggregation saw a different format from this file (violated CLAUDE.md invariant 4).
- `training/upload_weights.py::main` replaced bare `assert gcs_bucket is not None` / `assert repo_id is not None` guards with explicit `if … is None: parser.error(...)` checks. The previous asserts would have been stripped under `python -O`, letting `None` slip into `sync_gcs_to_hf` and surface as a confusing downstream `AttributeError` deep in the GCS client.
- `MouseDroidOrchestrator.__init__` precedence: the legacy `weight_update_poller=` kwarg now only folds into the internal mapping when `weight_update_pollers=` was *omitted* (was previously: when the resolved mapping was *empty*). Explicit `weight_update_pollers={}` is now a clean "disable OTA" signal that cannot be silently overridden by a stale legacy kwarg.
- Module-level `_DEFAULT_LEGACY_REPO_ID` + `_DEFAULT_UPLOAD_EXTENSIONS` + `_CLOUD_TRAINER_UPLOAD_EXTENSIONS` constants in `training/upload_weights.py` so the function signature default, the CLI fallback branch, and the help text cannot drift apart.
- Module-level `_WORLD_MODEL_DEFAULT_REPO_ID` constant in `mousedroid.config.schema` so the `world_model_repo_id` field default and the footgun validator's match check share one source of truth.

### Added
- **Tier C2.1**: `MissionLifecycle` now ticks once per orchestrator loop at the POST_TICK seam (was a no-op since PR #95 shipped the class). `process_mission` calls `start_mission()` so the lifecycle actually transitions PENDING→RUNNING in production.
- **Tier C1.1**: `training/upload_weights.py::sync_gcs_to_hf` closes the cloud→HF Hub leg of the OTA loop. New CLI flags `--from-gcs`, `--gcs-bucket`, `--gcs-prefix`.
- **Tier C1.2**: Dual `WeightUpdatePoller` slots — `policy` and `world_model` engines can both receive OTA updates. Gated by `cloud.weight_update.world_model_enabled` (default `False`, backwards-compatible). New `build_weight_update_pollers()` factory returns `Mapping[str, WeightUpdatePollerProtocol]`.
- Integration test (`tests/integration/test_tier_c_closeout_integration.py`) covering MissionLifecycle + dual poller + safety projector on a single tick + via factory.
- Property test (`tests/property/test_mission_lifecycle_property.py`) verifying MissionLifecycle state-machine transitions across 100 Hypothesis-generated score sequences.

### Changed
- `numpy` soft-pinned to `>=1.24,!=2.0.0,!=2.0.1` to lock out the initial NumPy 2.x releases that broke transitive deps. Tree audit confirmed no in-repo usage of removed symbols (`np.float_`, `np.NAN`, `np.in1d`, etc.); the pin is defensive.

### Backwards compatibility
- Existing YAML files load unchanged. The new boolean gate `cloud.weight_update.world_model_enabled` defaults to `False` (preserves the policy-only pre-C1.2 wiring), and the new non-boolean fields are schema-defaulted to safe values: `cloud.weight_update.upload_extensions = (".onnx", ".pt", ".npz", ".json", ".safetensors")` and `cloud.weight_update.gcs_artifact_prefix = "trained/"` (validated non-blank to block a full-bucket enumeration footgun).
- `build_weight_update_poller()` (singular) retained as a deprecated shim for one minor version; new code should call `build_weight_update_pollers()` (plural).
- `MouseDroidOrchestrator` constructor accepts both the legacy `weight_update_poller=` kwarg and the new `weight_update_pollers=` mapping; legacy is folded into the mapping at runtime under the poller's `engine_type` property (falling back to the legacy private `_engine_type` attribute, then defaulting to `"policy"`).
- The Tier C1.2 `engine_type` property lives on the new opt-in `EngineTypedWeightUpdatePollerProtocol` extension protocol — the base `WeightUpdatePollerProtocol` keeps its pre-C1.2 surface so external pollers predating this PR continue to satisfy it structurally without modification.
- `mission_lifecycle=None` is a true no-op — `_maybe_tick_mission_lifecycle` short-circuits.
- `build_mission_lifecycle` returns `None` (and logs `mission_lifecycle_dependencies_missing` at warning level) when `cfg.mission.replan_enabled=True` but either `vlm_progress` or `replanner` is unwired, so the orchestrator's POST_TICK seam stays a no-op until both Tier C2.3 dependencies are supplied — pre-C2.2 deployments remain byte-identical.

### Documentation — Tier C3.2: Sprint closeout — planning docs refreshed to post-Tier-C reality

Closes the Tier C sprint. No source code changes — only the planning + architecture
docs are refreshed to reflect what shipped across PRs #93 / #94 / #95 / #96:

- `docs/planning/IMPLEMENTATION_PLAN.md` — new "Tier C Status (2026-05-16)"
  section near the top with the 4-track table, phase-status table (Phase 1-5
  marked complete; Phase 6 promoted to ACTIVE), and the 5-item operator-follow-up
  checklist for items blocked on hardware/Linux access.
- `docs/planning/NEXT_STEPS.md` — new "Next Major Milestone — Phase 6: On-Device
  Incremental Learning" entry at the top, plus a "Recently Completed — 2026-05-16
  Tier C" block summarising each of the four PRs (#93 C3.1 / #94 C1 / #95 C2 /
  #96 C4) + operator follow-up tasks.
- `docs/architecture.md` — new Level 3g diagram "Tier C Closed-Loop Autonomy"
  showing the cloud→Jetson OTA pull, the orchestrator tick ordering with both
  the C1 atomic swap seam (post-`_select_action`) and the C2 safety-projection
  seam (wraps `_select_action` return value, covers all 4 branches), the
  default-state config-flag table, the safety invariants the tick ordering
  enforces, and the ADR cross-references (ADR-009/010/011).

---

## [v0.4.0] — 2026-05-16 — Tier C: Closed-Loop Autonomy + Cloud Retraining + Isaac Lab

Tier C is the sprint that converts the Tier B foundation (ONNX-via-ORT world
model + Isaac Lab Phase B foundation) into a closed-loop autonomous rover.
After Tier C, the only autonomy gap is on-device incremental learning
(Phase 6, multi-week, deferred).

Four PRs landed in sequence over 2026-05-16 in the documented order:

1. **PR #93** — Tier C3.1: Production hardening + B2 telemetry follow-through
2. **PR #94** — Tier C1: Closed-loop cloud retraining + Jetson OTA puller
3. **PR #95** — Tier C2: Mission lifecycle + geometric safety projection
4. **PR #96** — Tier C4: Isaac Lab env body (B3.2–B3.5) + RoverRewardConfig

Each track was default-disabled (config flags default to `False` / `0.0` /
`None`) so existing deployments produce byte-identical pre-Tier-C behaviour;
operators flip one config flag per track to opt in. See the Tier C track
sections below under `## [v0.4.0]` for the surface diff per PR plus rollback
paths.

### Changed — Tier C3.1: Production hardening (CI matrix promotion + B2 telemetry/dashboard polish)

Lands first in the Tier C sprint to enable real CI gating before the
C1/C2/C4 PRs merge. Four concrete changes:

1. **`vla-extras` CI matrix promoted from advisory to required.** The
   matrix has 6+ consecutive green runs against the integration branch
   (run IDs 25966739992, 25966733947, 25964900365, 25951166960,
   25948534154, 25944439195 — the seventh predates the matrix and
   doesn't apply). `continue-on-error: true` removed; subsequent
   merges that break the `[vla]` extras path will now fail the gate.
2. **`onnx-world-model-extras` CI matrix stays advisory** (intentional)
   because it only has 1 green run since landing with PR #92 on
   2026-05-16. Comment in `.github/workflows/ci.yml` documents the
   pending 7-run gate; promotion follow-up in Tier C2's PR (by which
   time the gate will have been met).
3. **Wired the missing Tier B2 telemetry helper.** PR #92 documented
   `MetricsRegistry.observe_world_model_observe_step_seconds` but
   never wired it — `DualStreamRSSMOnnx` was using a defensive
   `getattr(self._metrics, "observe_world_model_observe_step_seconds",
   None)` lookup at `world_model/dual_stream_rssm_onnx.py:293`. The
   helper now exists unconditionally, the runtime calls it directly
   (still gated on `metrics is None`), and `generate_metrics_sample()`
   exercises it so promtool + Grafana see non-empty series.
   - `MetricsConfig.world_model_observe_step_seconds_buckets` added
     (default `(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, inf)`)
     — covers the 10 ms Orin Nano + TensorRT EP target and the 33 ms
     portable dev gate (30 Hz tick).
   - Grafana panel id=22 `World-Model observe_step Latency p50 / p95 /
     p99` lands in `docs/grafana_dashboard.json`. Test extension
     `test_grafana_dashboard_json.py::TestPrB2PanelsPresent` asserts
     the panel title is present.
   - Prometheus alert rule `WorldModelObserveStepLatencyHigh` lands in
     `config/prometheus/alerts.yml` group `mousedroid_world_model` —
     fires when p95 > 33 ms for 2 minutes. Operators on Jetson tighten
     to 10 ms via their alertmanager overlay or by editing the
     threshold inline.
4. **Tier C dashboard E2E smoke scaffold** —
   `tests/smoke/test_prometheus_format_tier_c.py` covers the B2
   `observe_step` histogram surface end-to-end via
   `generate_metrics_sample()`. The C1 placeholder is replaced with
   real assertions in this PR; the C2 placeholder remains until that
   track lands.

**Operator follow-up (out of PR scope):** confirm branch protection on
the integration branch is configured to require the `vla-extras (3.11)`
status check at <https://github.com/ianshank/Mouse-Droid-AGI/settings/branches>.
Until that UI step is done, the workflow reports red/green but doesn't
block merges.

### Added — Tier C1: Closed-loop cloud retraining + OTA weight updates

Lands the **rover → GCP → Jetson** half of the closed-loop cloud-retraining
flow. Cloud Vertex AI training jobs consume the experience LMDB shards
exported by `cloud/experience_exporter.py` and retrain the policy; the
trained artifact remains in GCS for an operator to publish to HuggingFace
Hub manually via `huggingface-cli upload` (the in-process upload step + the
`--push-to-hf` / `--hf-repo-id` / `--hf-artifact-filename` flags +
`training/upload_weights.py` module are deferred to a follow-up PR — see
ADR-010 §"Out-of-Scope"). A new `WeightUpdatePoller` on the Jetson polls
HF Hub for newer artifacts (once published), downloads with **SHA-256
integrity verification**, and the orchestrator atomically swaps the live
models **after** `_select_action` returns — so the current tick saw one
consistent weight set for both `_update_world_model` and `_select_action`.

- **`src/mousedroid/cloud/protocol.py`** — added `PendingWeightUpdate`
  frozen dataclass and `WeightUpdatePollerProtocol`.
- **`src/mousedroid/cloud/weight_update_poller.py`** — new
  `HuggingFaceWeightUpdatePoller` with lazy `huggingface_hub` import,
  SHA-256 manifest verification, and structured logs for every state
  transition (`cloud_weight_update_poll_started`,
  `cloud_weight_update_new_revision`, `cloud_weight_update_sha256_verified`,
  `cloud_weight_update_sha256_mismatch`, `cloud_weight_update_swap_pending`).
- **`src/mousedroid/utils/weights_manager.py`** — new `verify_sha256()`
  helper, reusable safety-critical integrity check.
- **`src/mousedroid/orchestrator/orchestrator.py`** — new
  `_apply_pending_weight_update()` called once per tick AFTER
  `_select_action`. Resets `(h, z)` to zeros on world-model swap when
  `cfg.cloud.weight_update.reset_state_on_swap = True` (default) to avoid
  one-tick cross-model contamination.
- **`src/mousedroid/factory.py`** — `build_weight_update_poller(cfg, metrics)`
  + `build_weight_update_loader(cfg)` thread the optional poller into
  `MouseDroidOrchestrator`. Default `poll_interval_s = 0.0` keeps the
  pre-Tier-C1 path byte-identical.
- **`src/mousedroid/telemetry/metrics.py`** — four new Prometheus families
  exposed on `/metrics`: `mousedroid_cloud_weight_update_downloads_total`
  (labeled by `repo_id`), `mousedroid_cloud_weight_update_sha256_mismatches_total`
  (`repo_id`), `mousedroid_cloud_weight_update_download_seconds` (Histogram
  with buckets from
  `MetricsConfig.cloud_weight_update_download_seconds_buckets`), and
  `mousedroid_cloud_weight_update_swaps_total` (labeled by `engine_type`).
  All four families are seeded inside `generate_metrics_sample()` so
  `promtool check rules` sees non-empty series from the first scrape.
- **`src/mousedroid/config/schema.py`** — new `WeightUpdatePollConfig`
  nested under `CloudConfig` (mounted on `Settings.cloud`), and new
  `MetricsConfig.cloud_weight_update_download_seconds_buckets`. All fields
  have defaults; existing YAML loads unchanged.
- **`docker/Dockerfile.cloud`** — new x86 GPU image built on
  `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` for Vertex AI workers.
  Companion to `Dockerfile.jetson` which uses ARM L4T-only base.
- **`cloud/vertex_ai_job_spec.yaml`** — Vertex AI custom-training job
  spec template.
- **`scripts/cloud_train.sh`** — container entry point.
- **`training/train_offline_rl.py`** — added `--shard-consumed-marker-uri`
  flag (the only Tier C1 cloud flag actually wired today). Idempotency:
  the trainer checks the GCS marker at startup and writes it on success
  so duplicate Jetson re-uploads do NOT double-train. Per-job semantics
  (single marker checked + written once per run), NOT per-shard. The
  HF-upload flags (`--push-to-hf` / `--hf-repo-id` /
  `--hf-artifact-filename`) and the LMDB-iteration flag
  (`--lmdb-shards-gcs-prefix`) are deferred to the follow-up PR that
  ships `training/upload_weights.py` + a shard-iteration loop.
- **`docs/architecture/ADR-010-cloud-weight-update-ota.md`** — design
  rationale, swap-timing invariant, SHA-256 integrity contract.

Test coverage: 8 poller tests (revision skip, download, SHA mismatch,
log transitions, stop cancellation, ACK clearing, protocol conformance,
disabled no-op), 6 `verify_sha256` tests, 9 orchestrator swap tests
(no-op, no-loader, post-action call order, state reset / preserve, metric
+ log emission, multi-update ordering, loader-exception isolation,
engine-type routing). Tier C dashboard E2E smoke
(`tests/smoke/test_prometheus_format_tier_c.py`) extended: the C3.1
placeholder for the C1 cloud OTA families is now a real assertion that
all 4 families render in `generate_metrics_sample()` + per-label render
contract tests.

### Added — Tier C2: Mission Closed-Loop + Safety Projection

Closes Track C2 of the Tier C sprint (see plan
`.claude/plans/please-create-a-comprehensive-sunny-hennessy.md`).
Adds two stateless, default-disabled seams to the orchestrator tick:

- **`SafetyActionProjector`** — geometric soft-constraint projection of
  the policy's proposed action, applied AFTER `_select_action` returns
  and BEFORE `_execute_action` runs. Pure function of `SafetyContext` +
  action; three independent clamp rules (forward velocity / human
  keepout / tight quarters). All thresholds come from
  `cfg.safety.projector.*`; default `enabled=false` keeps existing
  deployments byte-identical.
  See `src/mousedroid/safety/projector.py`,
  `src/mousedroid/safety/projector_protocol.py`.
- **`MissionLifecycle`** — state machine (PENDING → RUNNING →
  SUCCEEDED | FAILED | REPLANNING → RUNNING) wrapping the existing
  `TaskTrackerProtocol`. Polls `VLMProgressHead` once per tick;
  transitions to REPLANNING on stall and submits an async replan
  request via the LLM gateway. All thresholds come from `cfg.mission.*`;
  default `replan_enabled=false` keeps existing deployments
  byte-identical. **NOTE (post-merge audit, surfaced by Gemini review on
  PR #97)**: this PR ships the class + `build_mission_lifecycle()` factory
  + telemetry families, but the orchestrator constructor does NOT yet
  thread a `mission_lifecycle` kwarg and `tick()` does NOT yet invoke
  `mission_lifecycle.tick()`. The lifecycle is currently exercised
  standalone via its own tests + external orchestrator drivers. A
  follow-up PR (Tier C2.1) will close this gap by wiring the lifecycle
  through `__init__` + invoking it at the POST_TICK seam. See
  `docs/planning/NEXT_STEPS.md` §"Tier C2.1 follow-up".
  See `src/mousedroid/orchestrator/mission_lifecycle.py`.
- **Orchestrator tick seam** — single insertion point at
  `Orchestrator.tick()` around the unified `_select_action` call site
  ensures all four `_select_action` return branches (cognitive / VLA /
  VLA-strict-timeout / nav_agent) hit the projector uniformly. The
  three branch-coverage regression tests parametrise this contract so a
  future refactor that adds a fifth return path fails the suite.
- **Telemetry families** — four new Prometheus families wired into
  `MetricsRegistry`:
  - `mousedroid_safety_action_clamps_total{reason}` (counter)
  - `mousedroid_mission_state_transitions_total{from_state,to_state}` (counter)
  - `mousedroid_mission_replans_total{outcome}` (counter)
  - `mousedroid_mission_active_duration_seconds` (histogram with
    operator-tunable buckets from
    `MetricsConfig.mission_duration_seconds_buckets`).
  All four follow the PR-A2 pure-add pattern — rendered only after a
  writer first touches them, so default-disabled deployments produce
  byte-identical `/metrics` output.
- **Factory wiring** — `build_safety_projector(cfg, metrics=...)` and
  `build_mission_lifecycle(cfg, ...)` both return `None` when their
  feature flag is off; the orchestrator behaves byte-identically.
- **ADR-011** — documents the geometric-over-Lagrangian-over-masking
  rationale, the soft-vs-hard constraint split (projection = soft;
  `emergency_stop` = hard), and the `MissionLifecycle`'s relationship
  to the existing `TaskTrackerProtocol`.
  See `docs/architecture/ADR-011-mission-closed-loop-safety-projection.md`.

**Test coverage:**
- 14 projector unit + branch-coverage tests in
  `tests/unit/safety/test_projector.py` (8 unit + 3 branch-coverage
  required by plan + 3 additional regression-net tests).
- 10 mission lifecycle unit tests in
  `tests/unit/orchestrator/test_mission_lifecycle.py`.
- 3 multi-minute closed-loop integration tests in
  `tests/integration/test_mission_closed_loop.py` (rising progress,
  stall + LLM replan, replan limit exhaustion).
- 4 Tier-C smoke tests in `tests/smoke/test_prometheus_format_tier_c.py`
  (replaces the C3.1 placeholder for C2 with real assertions).

### Changed — Tier B1: Ten-Pillars nightly — workflow-side promotion ready

After the Tier A sprint landed (PRs #85-#89), the `jetson-nightly.yml`
workflow has been running in advisory mode. This PR ships the
**workflow-side** half of the required-check promotion; the
**branch-protection** UI step is a post-merge operator follow-up (full
playbook in `docs/jetson-runner-setup.md`).

- **`.github/workflows/jetson-nightly.yml`** — removed
  `continue-on-error: true` from the `ten-pillars` job block.
- **`.github/workflows/jetson-nightly.yml`** — changed the `Report status`
  step's trailing `exit 0` to `exit "${PILLAR_RC:-1}"` so the workflow's
  overall exit code reflects the captured `validate_pillar.sh` exit
  status. Without this second edit, removing the advisory flag would have
  no effect on branch protection — the workflow stayed green from the
  swallowed `exit 0`.
- **`docs/jetson-runner-setup.md`** — replaced the "Promotion to Required
  Check" runbook with a two-half framing (workflow change in this PR /
  branch-protection UI step post-merge), updated exit-code semantics
  (rc=2 points operators at the workflow console output because
  `ten_pillars.log` is only generated on successful completion),
  rollback path, and a fresh "Promotion Observation Log" table for the
  operator to populate during the 7-night observation window.

**Operator follow-up (out of PR scope, required to make the gate
effective):** configure GitHub branch protection at
<https://github.com/ianshank/Mouse-Droid-AGI/settings/branches> to require
the **Ten Pillars on Jetson** check. Until this UI step is done, the
workflow reports red/green but does not block merges. After it is done,
merges to `main` are blocked when:

- Any blocking pillar (`safety`, `world_model`, `memory`, `cognitive`,
  `reward`) reports FAIL → `PILLAR_RC=1` → exit 1.
- A precondition error fires (Docker container down, etc.) →
  `PILLAR_RC=2` → exit 2 (inspect workflow console output —
  `ten_pillars.log` is not written on this path).

Rollback: revert this PR + disable the required-check setting in
branch-protection UI.

### Added — PR-A2.1: Writer-side instrumentation activating PR-A2 metrics

Closes the loop on Tier A. PR-A2 (PR #87) shipped the registry helpers
and metric definitions; PR-B2 (PR #88) shipped the Grafana panels +
Prometheus alert rules. Until this PR merged, all four PR-A2 metric
families were registered-but-zero in production. After PR-A2.1, the
dashboards populate the first time their respective code paths fire.

- **`src/mousedroid/training/replay/lmdb_reader.py`** — `LMDBReplayReader`
  accepts optional `metrics: MetricsRegistry | None = None`. On each
  successful decode in `_read_chunk_with_env`, emits
  `inc_replay_record("ok")`; on schema mismatch, emits
  `inc_replay_record("schema_mismatch")` alongside the existing
  `replay_schema_mismatch` structured log.
- **`src/mousedroid/vla/policy.py`** — both `MockVLA` and `DistilledVLAOnnx`
  accept optional `metrics`. Each `predict()` brackets inference with
  `time.perf_counter()` and calls `observe_vla_inference_seconds(elapsed)`
  **outside** the `torch.no_grad()` block (matches the existing
  orchestrator timing convention at `orchestrator.py:733`). MockVLA also
  emits (near-zero observation) so end-to-end test runs populate the
  histogram for operator visibility on mock-hardware deployments.
- **`src/mousedroid/orchestrator/orchestrator.py`** —
  `MouseDroidOrchestrator` accepts optional `metrics`. `_try_vla_action()`
  calls `metrics.inc_vla_timeout(cast(VLAActiveBackendLiteral, cfg.vla.backend))`
  on the timeout branch. The mode value is guaranteed in
  `VLAActiveBackendLiteral` because the policy-selector gate
  short-circuits when `backend == "none"` (no `_try_vla_action` call in
  that case). mypy can't see the upstream gate, so the cast is explicit.
- **`src/mousedroid/reward/vlm_progress.py`** — `VLMProgressHead` accepts
  optional `metrics`. Three cache decision branches in `_score_single`
  each emit a guarded metric call alongside the existing `self._hits` /
  `self._misses` counter bumps: identity-cache hit → `inc_vlm_cache_hit()`;
  content-cache hit → `inc_vlm_cache_hit()`; cache miss →
  `inc_vlm_cache_miss()`.
- **`src/mousedroid/factory.py`** — `build_replay_reader`,
  `build_vla_policy`, `_build_distilled_onnx_vla`, and `build_reward_model`
  thread the optional `metrics: MetricsRegistry | None = None` parameter
  through. The `MetricsRegistry` `TYPE_CHECKING` import was already in
  place from the pre-existing `build_metrics_registry` factory; no new
  imports required.

**Test coverage** (16 new tests, all green):

- `tests/unit/training/replay/test_lmdb_reader.py` (extended, +3): ok
  outcome metric, schema-mismatch metric, no-op-when-`metrics=None`
- `tests/unit/vla/test_policy.py` (extended, +2): MockVLA emits + no-op default
- `tests/unit/vla/test_distilled_onnx.py` (extended, +2): DistilledVLAOnnx
  emits + finite-value bound + no-op default
- `tests/unit/orchestrator/test_policy_selector.py` (extended, +4):
  timeout emits with `mode="mock"`, happy path silent, no-op when
  `metrics=None`, timeout emits with `mode="distilled_onnx"`
- `tests/unit/reward/test_vlm_progress.py` (extended, +4): miss emits,
  identity-cache hit emits, content-cache hit emits, no-op default
- `tests/integration/test_writer_side_instrumentation_http.py` (new, +1
  parametrized over 2 cases): full `/metrics` HTTP scrape via aiohttp
  `TestClient(TestServer(app))` against a real `TelemetryServer` +
  `MetricsRegistry`; verifies all 4 PR-A2 families appear when
  instrumentation is wired, AND verifies they're correctly omitted from
  `/metrics` output when no observations fire
- `tests/integration/test_writer_side_e2e_concurrent.py` (new, +2):
  replay (async) + VLA (sync thread) + VLM (sync thread) fired in
  parallel — counter totals exact under contention; mid-burst render
  produces well-formed Prometheus exposition output
- `tests/smoke/test_writer_side_metrics_smoke.py` (new, +1): in-process
  smoke; <1s; `pytest -m smoke` selector picks it up so CI's smoke stage
  catches wiring regressions before the slower integration suite runs
- `tests/performance/test_instrumentation_overhead.py` (new, +2,
  `slow`-marked): per-call overhead budget (default 1.15x;
  `MOUSEDROID_INSTRUMENTATION_OVERHEAD_BUDGET` env override for slower
  hardware)

**Backwards compatibility:** Every new constructor parameter is
`Optional[MetricsRegistry] = None`. Existing call sites that don't pass
`metrics` continue to work unchanged. No schema additions. No log
additions (every instrumentation site piggybacks on an existing
structured event: `replay_schema_mismatch`, `vla_inference_seconds_dropped`,
`vla_timeout`).

### Added — PR-B2: Ten-Pillars nightly regression net + Grafana visibility

- **`pyproject.toml`** — new ``pillar`` pytest marker registered under
  ``[tool.pytest.ini_options].markers``. Selects exactly the Ten-Pillars
  validation suite in [`tests/regression/test_validate_pillar.py`](tests/regression/test_validate_pillar.py)
  (9 tests) via ``pytest -m pillar``. The Jetson nightly workflow can now
  invoke the campaign without coupling to specific test file paths.
- **`tests/regression/test_validate_pillar.py`** — added ``pytestmark =
  pytest.mark.pillar`` at module level so the marker selection works.
- **`docs/grafana_dashboard.json`** — four new panels (ids 18-21) covering
  the PR-A2 metric families:
  - Panel 18: *Replay Records (rate by outcome)* — ``rate(mousedroid_replay_records_total[1m])`` split by ``outcome`` label
  - Panel 19: *VLA Inference Latency p50 / p95 / p99 (seconds)* — three ``histogram_quantile`` queries over ``mousedroid_vla_inference_seconds_bucket``
  - Panel 20: *VLA Timeouts (rate by mode)* — ``rate(mousedroid_vla_timeouts_total[5m])`` split by ``mode``
  - Panel 21: *VLM Progress Cache Hit Rate* — hits / (hits + misses) ratio gauge with a tiny epsilon to prevent ``0/0`` NaN
- **`tests/unit/test_grafana_dashboard_json.py`** (new) — 9 structural tests
  that lock in: dashboard JSON parses, every panel has stable id + title,
  panel ids are unique, every Prometheus expression references a metric
  emitted by ``generate_metrics_sample`` (catches typo-on-rename drift),
  and one panel per PR-A2 metric family exists. The query-vs-sample test
  has an explicit whitelist for MCP metrics that are gated on
  ``track_mcp=True`` and absent from the default sample.
- **`config/prometheus/alerts.yml`** — new ``mousedroid_replay_vla_vlm``
  group with 4 alert rules:
  - ``ReplaySchemaMismatchSpike`` — warning when schema-mismatch rate >0.1/s for 5m
  - ``VLAInferenceLatencyHigh`` — warning when p95 inference >100ms for 2m
  - ``VLATimeoutSpike`` — warning when timeout rate by mode >0.01/s for 5m
  - ``VLMCacheHitRateCollapse`` — info when hit-rate <50% for 10m
  All thresholds are documented as operator-tunable in the rule annotations.
- **`config/loki/promtail.yml`** — comment-block LogQL query examples
  extended with PR-A2 event names: ``offline_rl_bc_active``,
  ``replay_schema_mismatch.*``, ``offline_rl_mixer_active``,
  ``offline_rl_mixer_requested_but_unavailable``,
  ``vla_inference_seconds_dropped`` (with field references), and a
  ``pillar_validation`` placeholder for the Ten-Pillars campaign.
- **`docs/jetson-runner-setup.md`** — new "Promotion to Required Check"
  section documenting the 4-step gate to remove ``continue-on-error: true``
  from the ``ten-pillars`` job: register runner, manual trigger, 7
  consecutive green nights, follow-up PR removing the advisory flag.
  Includes a local-nightly-equivalent command set so operators can mirror
  the workflow on the Jetson host.

> ⚠ **No data in dashboards until writer-side instrumentation lands.** The
> Grafana panels and alert rules above reference PR-A2 metric names. Per
> ADR-006's PR-A2 addendum, the writer-side call-site instrumentation in
> ``replay/lmdb_reader.py``, ``vla/policy.py``, and ``reward/vlm_progress.py``
> is deferred to a follow-up PR. Until that lands, all four panels and all
> four alert rules will match-zero — they're staged in advance so the
> dashboard layout and alert thresholds can be reviewed offline.

### Added — PR-A2: Replay / VLA / VLM Prometheus observability metrics

- **`src/mousedroid/telemetry/metrics.py`** — four new metric families
  available on the existing `/metrics` endpoint:
  - `mousedroid_replay_records_total{outcome="ok"|"schema_mismatch"}` —
    LMDB replay-record deserialization outcomes
  - `mousedroid_vla_inference_seconds` — config-driven histogram
    (`MetricsConfig.vla_inference_seconds_buckets`, default 5 ms..2.5 s)
  - `mousedroid_vla_timeouts_total{mode="mock"|"distilled_onnx"}` —
    VLA fallback events by backend mode
  - `mousedroid_vlm_progress_cache_hits_total` and `..._misses_total` —
    VLM progress-reward cache effectiveness
- Public writer-side helpers (`inc_replay_record`,
  `observe_vla_inference_seconds`, `inc_vla_timeout`, `inc_vlm_cache_hit`,
  `inc_vlm_cache_miss`) follow the project's existing `inc_*` / `observe_*`
  naming convention. All counters accept an `amount: int = 1` kwarg
  and guard against non-positive deltas to preserve Prometheus counter
  monotonicity. `observe_vla_inference_seconds` rejects both negative
  values and NaN, emitting a DEBUG-level `vla_inference_seconds_dropped`
  structured log so operators can correlate missing histogram observations
  with upstream timer bugs.
- Label values are typed via new public `Literal` aliases in
  `mousedroid.config.schema` so mypy `--strict` catches label drift:
  - `ReplayOutcomeLiteral = Literal["ok", "schema_mismatch"]`
  - `VLABackendLiteral = Literal["none", "mock", "distilled_onnx"]` (matches `VLAConfig.backend`)
  - `VLAActiveBackendLiteral = Literal["mock", "distilled_onnx"]` (subset of `VLABackendLiteral` minus `"none"`, used for metrics that only fire from a running backend — prevents accidental `{mode="none"}` cardinality on `mousedroid_vla_timeouts_total`)

  `VLAConfig.backend` now uses `VLABackendLiteral` directly — single source of truth.
- `.github/workflows/ci.yml` adds an advisory `vla-extras` job
  (Python 3.11, `[dev,vla]` extras, `continue-on-error: true`) covering
  `tests/unit/vla/`. Promotion gate documented in
  `docs/planning/PHASE_2_1_AND_BEYOND_PLAN.md` Story 2.5.
- `docs/architecture/ADR-006-telemetry-server.md` gains a PR-A2 addendum
  documenting the new metrics, design invariants, and deferred items.

### Changed — Histogram bucket validation now enforced at schema load

- `src/mousedroid/config/schema.py` adds a shared
  `_validate_histogram_buckets` Pydantic validator applied via
  `@field_validator` to **four** `MetricsConfig` bucket fields:
  - `loop_latency_buckets_ms` (existing)
  - `llm_latency_buckets_ms` (existing)
  - `mcp_latency_buckets_ms` (existing)
  - `vla_inference_seconds_buckets` (new in PR-A2)
- Invariants enforced: monotonically ascending, strictly positive,
  unique, non-empty (a trailing `+Inf` sentinel is permitted but
  optional — the registry appends one at runtime if missing).
- **⚠ Behavior change for downstream operator overlays.** Any
  externally-supplied YAML / env overlay that previously specified a
  bucket tuple containing zeros, negatives, duplicates, descending values,
  or an empty tuple will now fail config load with a Pydantic
  `ValidationError`. Previously such input would silently render
  malformed histograms. All in-repo `config/*.yaml` defaults pass the
  validator unchanged. If you maintain external overlays and hit a
  `ValidationError`, fix the offending bucket tuple to be ascending,
  positive, and unique.

### Added — Phase 2.1: BC supervised loss into offline-RL training loop

- **`training/train_offline_rl.py`** — wires the existing
  `OfflineRLTrainer.bc_update(states, actions, weight)` auxiliary loss
  into the per-batch training loop, gated by
  `cfg.offline_rl.real_supervised_weight` (default `0.0`). Adopts the
  TD3+BC pattern (Fujimoto & Gu 2021): on each batch, after the
  algorithm-specific `update_step`, an auxiliary `weight * MSE(policy(s),
  a_data)` is applied to the actor optimizer. At the schema default
  (`weight=0.0`) `bc_update` short-circuits and performs **no optimizer
  step**, so legacy training paths remain byte-identical (proven by the
  new regression tests below). When `weight > 0`, a one-shot
  `offline_rl_bc_active` structured log is emitted at run start and the
  scalar `bc_loss` is aggregated alongside `q_loss` / `policy_loss` in
  the epoch summary, surfacing as `final_bc_loss` in the returned stats.
- **Optional dedicated BC optimizer** (`OfflineRLConfig.bc_lr`,
  `OfflineRLConfig.bc_batch_size`) — when `bc_lr` is set, the trainer
  builds a separate `bc_optimizer` over policy parameters so the BC
  auxiliary loss can step at a different learning rate from the actor
  PPO step. When `bc_lr is None` (default), `bc_optimizer is
  policy_optimizer` — byte-identical to the pre-Phase-2.1 path. The
  trainer emits a one-shot `offline_rl_bc_optimizer_built` log at
  construction documenting whether the optimizer is shared or dedicated.
  Checkpoint compatibility is preserved: legacy checkpoints (no
  `bc_optimizer` key) load cleanly into trainers with or without a
  dedicated optimizer.
- **Sim/real ReplayMixer integration** (`OfflineRLConfig.use_replay_mixer`)
  — when `True` and `cfg.training.replay.enabled` is `True` with a
  distinct `cfg.training.replay.source_path`, `train_offline_rl` draws
  batches from a deterministic `RealSimMixer` interleaving the sim
  (`cfg.experience.path`) and real (`cfg.training.replay.source_path`)
  LMDB stores. Alpha ramp is driven by `cfg.training.replay_mixer`. If
  the mixer is requested but the real path is missing or identical, a
  `offline_rl_mixer_requested_but_unavailable` warning is logged and
  training proceeds on the single LMDB. Default `False` preserves
  byte-identical legacy behavior.
- **`tests/integration/test_phase21_bc_into_offline_rl.py`** — patches
  `test_bc_loss_recorded_in_stats` to assert the `offline_rl_bc_active`
  activation log fires (closes a prior gap where `caplog` was passed but
  never asserted). Existing 6 integration test classes (byte-identity at
  `weight=0` for CQL + IQL, measurable parameter divergence at
  `weight>0`, finite-weights guarantee, `final_bc_loss` aggregation, and
  the empty-LMDB short-circuit) all continue to pass unchanged.
- **`tests/integration/test_train_offline_rl_mixer.py`** (new) — 4 tests
  covering the new `use_replay_mixer` toggle: default-false safety,
  fallback when no distinct real path is configured, fallback when the
  source path is identical to the experience path, and the mixer-active
  log + training-completes path when a distinct real LMDB is present.
- **`tests/unit/test_offline_rl_bc.py`** (extended) — 8 new tests
  covering the dedicated `bc_optimizer` wiring: alias-when-`bc_lr=None`,
  separation-when-`bc_lr` set, policy_optimizer state isolation,
  legacy-checkpoint-loads-without-bc-state, dedicated-bc-checkpoint
  round-trip, empty-batch no-op, and shape-mismatch raises.
- **`tests/unit/test_config_schema.py`** (extended) — 7 new tests
  validating `bc_lr`, `bc_batch_size`, and `use_replay_mixer` defaults,
  positive-value acceptance, zero/negative rejection, and composite
  field configuration.
- **`tests/performance/test_offline_rl_bc_overhead.py`** (new) — slow
  regression: BC-active training must stay within `2.5×` baseline
  wall-clock (operator-tunable via `MOUSEDROID_BC_OVERHEAD_BUDGET`).
  Adam internal state size is bounded to one entry per policy
  parameter.
- **`docs/planning/PHASE_2_1_AND_BEYOND_PLAN.md`** — plan-of-record for
  the next sprint cycle (PR-A1/A2/B1/B2) with risks and Definition of Done.

#### Rollback path (no code revert required)

- To fully disable Phase 2.1 in production: set
  `offline_rl.real_supervised_weight: 0.0` in YAML and restart training.
  Byte-identity at this default is guaranteed by the existing
  `TestBcByteIdentityAtZeroWeight` integration test.
- To disable only the dedicated BC optimizer: set
  `offline_rl.bc_lr: null` (the default).
- To disable only the sim/real mixer: set
  `offline_rl.use_replay_mixer: false` (the default).

### Added — Production-config validation gate

- **`scripts/validate_configs.py`** — CLI that loads every `config/*.yaml`
  through `mousedroid.config.loader.load_settings()` and reports per-file
  pass/fail. Catches Pydantic schema drift, type drift, and cross-field
  validator regressions before they reach the Jetson. Supports a
  `# config-validator: skip` marker (in the YAML's first 10 lines) to
  exclude deploy-time descriptors that share the `config/` directory but
  are not runtime overlays. Flags: `--config-dir`, `--fail-fast`,
  `--include-default`. Exit codes: 0 / 1 (validation failures) / 2 (usage).
- **`tests/regression/test_config_overlays_load.py`** — 16 tests that
  parametrize over every overlay returned by
  `validate_configs.discover_overlays()`, assert `Settings` construction
  succeeds with a valid `platform`, and end-to-end exercise the CLI via
  `subprocess`. Test set stays in lockstep with the script by importing
  its `discover_overlays` helper.
- **`config/jetson_setup.yaml`** — annotated with `# config-validator: skip`
  (deploy descriptor; reuses the `jetson:` namespace with deploy-only keys
  `host`/`user`/`ssh_port`/`install_dir`/`swap_size_gb` that conflict with
  the runtime `JetsonConfig` schema).
- **`.github/workflows/ci.yml`** — new `config-validate` job (Python 3.11,
  needs `lint`) that runs `python scripts/validate_configs.py
  --include-default` after a runtime-only `pip install -e .`. Fast,
  isolated, no heavy deps.

### Added — Phase 2 acceptance: golden RSSM loss-curve regression

- **`tests/regression/_rssm_golden_helper.py`** — deterministic, CPU-only,
  tiny-dim RSSM training harness. Runs N (default 10) optimizer steps on
  synthetic batches sampled from a seeded `torch.Generator`; mirrors the
  per-step loss formula in `training/train_rssm.py` (recon MSE + kl_beta·KL)
  but strips file I/O, the `DataLoader`, and the AMP path so the curve is
  bit-stable across runs. `ModelConfig` is built via `model_validate` so the
  harness stays backwards-compatible when new schema fields land.
- **`tests/regression/fixtures/phase2_rssm_golden_baseline.json`** — committed
  10-step baseline (schema_version=1, seed=0). Regenerate intentionally via
  `MOUSEDROID_UPDATE_GOLDEN=1 pytest tests/regression/test_phase2_rssm_golden.py`.
- **`tests/regression/test_phase2_rssm_golden.py`** — 8 tests:
  fixture presence, curve length, finite/typed loss keys, monotone-decrease
  smoke, baseline tolerance (±1% on `recon`/`total`, ±5% on `kl` for
  reparameterized-sample noise), and prefix stability at `num_steps ∈
  {1, 3, 10}` (guards against accidental coupling between batch generation
  and step count). Closes the last open Phase 2 acceptance bullet from
  `NEXT_STEPS.md`.
- **Repo hygiene** — dropped 3 unused `# noqa: UP038` directives flagged by
  `RUF100` in `usb_speaker.py` and `mcp/resources.py` (autofix; no behavior
  change).
- All 3,391 unit/regression/integration tests pass; ruff + ruff format +
  `mypy --strict` (on the new files) + hardcoded-value gate clean.

### Added — Phase 2: Real-Episode Replay Loop (`feat/phase2-real-episode-replay`)

- **`src/mousedroid/training/replay/`** — new package
  - `ReplayReaderProtocol` — `@runtime_checkable Protocol` defining
    `stream(chunk_size) -> AsyncIterator[list[MouseDroidExperienceRecord]]`
    plus a `stats` dict, so callers cannot couple to LMDB internals
    (CLAUDE.md invariants 1+2).
  - `LMDBReplayReader` — async, chunked reader. Each chunk is fetched
    via `asyncio.to_thread` so the LMDB cursor never blocks the event
    loop; an empty or missing env logs a single `replay_empty_db`
    warning and yields nothing rather than raising. Schema-mismatched
    records are counted under `stats["skipped_schema_mismatch"]` and
    skipped.
  - `MixerConfig` (Pydantic) + `RealSimMixer` — deterministic sim/real
    interleaver seeded by a single `numpy.random.Generator`. Linear
    `current_alpha = min(target, target * step / ramp_steps)` ramp
    realises the RL-Co two-stage curriculum. Realized fraction is
    exposed via `stats["realized_alpha"]` for tests and telemetry.
- **`training/replay_real_episodes.py`** — new operator CLI exercising
  the reader+mixer end-to-end. Supports `--dry-run`, `--use-real-replay`,
  `--draws`, `--chunk-size`, `--alpha-target`, `--seed`. Empty LMDB is
  a no-op exit-0.
- **`src/mousedroid/factory.py`** — `build_replay_reader(cfg)` returns a
  `ReplayReaderProtocol`; concrete `LMDBReplayReader` import lives
  inside the function per project DI rules.
- **`src/mousedroid/config/schema.py`** — `OfflineRLConfig.real_supervised_weight`
  added with default `0.0`. The new field gates the future BC auxiliary
  loss in `train_constitutional_rl.py` (Phase 2.1 follow-up). Default of
  `0.0` keeps offline-RL training byte-identical to the pre-Phase-2
  path; backwards compat is preserved.
- **Tests** — `tests/unit/training/replay/test_lmdb_reader.py` (7 tests:
  protocol fit, empty/missing path, full round-trip, chunk-size guard,
  schema-mismatch counting, path override) and `test_mixer.py` (11
  tests: parametrized realized-ratio at `{0.1, 0.5, 0.9}` over 10 000
  draws within ±1.5%, seed determinism, monotone ramp, exhaustion
  fallback, validation guards).

### Notes

- The auxiliary BC loss `MSE(policy(s_real), a_real)` weighted by
  `real_supervised_weight` is intentionally deferred to Phase 2.1
  because the existing `train_constitutional_rl.py` is a numpy-MLP with
  custom numerical-gradient updates rather than a torch loop;
  retrofitting BC there warrants its own PR with a dedicated
  numerical-stability test.

### Added — Phase 2.1: BC auxiliary loss + tech-debt sweep

- **`src/mousedroid/learning/offline_rl.py`** — new `bc_update(states,
  actions, weight)` method on the `OfflineRLTrainer` ABC. Computes
  `MSE(policy(s_real), a_real)` and steps **only** the policy
  optimizer (Q-network frozen). When `weight <= 0.0` it is a strict
  no-op returning `{"bc_loss": 0.0}`, so any trainer wired through the
  default `real_supervised_weight=0.0` is byte-identical to the
  pre-Phase-2 path. Lands the BC hook in the torch trainer instead of
  the numpy `train_constitutional_rl.py` MLP — much cleaner integration
  surface.
- **`src/mousedroid/training/replay/mixer.py`** — magic numbers `1000`,
  `500`, `2`, `4` hoisted to named module-level constants
  (`DEFAULT_RAMP_STEPS`, `DEFAULT_LOG_INTERVAL`, `_NUM_SOURCES`,
  `_LOG_ALPHA_PRECISION`). New `MixerConfig.from_settings()` classmethod
  builds a mixer config from a YAML-loaded `ReplayMixerConfig` without
  importing the schema (avoids circular imports).
- **`src/mousedroid/config/schema.py`** — new `ReplayMixerConfig`
  Pydantic model exposed as `TrainingConfig.replay_mixer`. YAML can now
  drive `alpha_target`, `alpha_ramp_steps`, `chunk_size`, `seed`, and
  `log_every_n` without touching code. All fields default to inert
  values so existing YAMLs load unchanged (CLAUDE.md invariant 9).
- **`training/replay_real_episodes.py`** — CLI now reads defaults from
  `cfg.training.replay_mixer`; `--alpha-target` and `--seed` flags
  override per-invocation. `--dry-run` now strictly wins over
  `--use-real-replay` (logs and skips the LMDB open). Argparse defaults
  hoisted to `DEFAULT_DRAWS` / `DEFAULT_CHUNK_SIZE` constants. Fixed
  pre-existing bug where `load_settings(str)` raised because the loader
  needs a `Path`.
- **Tests** — three new test files:
  - `tests/unit/test_offline_rl_bc.py` (6 tests, both CQL and IQL):
    weight=0 is byte-identical, positive weight reduces loss
    monotonically over 20 steps, Q-network never moves.
  - `tests/unit/test_factory_replay_reader.py` (2 tests):
    `build_replay_reader` returns a `ReplayReaderProtocol`; respects
    `training.replay.source_path` override.
  - `tests/unit/training/replay/test_cli_replay_real_episodes.py` (4
    tests): dry-run exit-0, `--dry-run` overrides `--use-real-replay`,
    empty-LMDB exit-0, argparse wiring.
- **Coverage** — 97.21% on Phase 2 + 2.1 modules (gate is 85%).

### Added — Phase 2 acceptance: integration test on synthetic LMDB

- **`tests/integration/test_phase2_replay_pipeline.py`** — 6 new tests
  closing the third Phase 2 acceptance bullet from `NEXT_STEPS.md`. The
  end-to-end test:
  1. Writes 10 deterministic synthetic records via the production
     `ExperienceLogger` (so the same write path used on the Jetson is
     exercised).
  2. Builds the reader through `factory.build_replay_reader` and
     drains it via async `stream(chunk_size=4)`.
  3. Asserts every record round-tripped (`read_records == 10`,
     `skipped_schema_mismatch == 0`).
  4. Tensorizes `vision_features -> states`, `action -> actions`,
     runs 25 BC updates on a `CQLTrainer`, asserts loss strictly
     decreases.
  5. Saves a checkpoint to disk, re-loads it into a fresh trainer,
     and asserts policy outputs match byte-exactly on the training
     batch.
- Companion tests guarantee `weight=0.0` is byte-identical (CLAUDE.md
  invariant 9 backwards compat) and that reader output is invariant
  across `chunk_size ∈ {1, 3, 4, 64}`.

### Added — Phase 4: VLM-Derived Dense Progress Reward (`feat/phase4-vlm-progress-rewards`)

- **`src/mousedroid/reward/vlm_progress.py`** — new module
  - `VLMProgressBackend` — `@runtime_checkable Protocol` with
    `score(prev_obs, curr_obs, instruction) -> float` so the actual VLM
    call site is pluggable without coupling the reward module to a
    specific model SDK
  - `MockVLMProgress` — constant-value backend used for tests and the
    default-off operating mode; raises `ValueError` for out-of-`[0, 1]`
    initial values
  - `VLMProgressHead(nn.Module)` — wraps a backend with a bounded
    **`cachetools.LRUCache`** (NOT `functools.lru_cache` — the project
    requires explicit `maxsize` to keep memory bounded across long
    training runs). Cache key is
    `(sha1(round(prev,d)), sha1(round(curr,d)), sha1(instruction))`
    where the rounding precision `d` is configurable via
    `VLMProgressConfig.hash_decimals`. Inference runs under
    `torch.no_grad()`; out-of-`[0, 1]` backend returns raise
    `ValueError`; cache hits/misses exposed via `cache_info` for
    observability
- **`src/mousedroid/reward/model.py`** — `MultiObjectiveRewardModel`
  extended with optional `vlm_head: VLMProgressHead | None = None` and
  optional `prev_obs` / `curr_obs` / `instruction` kwargs on
  `compute_reward`, `aggregate`, and `forward`. **The Law-1
  multiplicative sigmoid gate is preserved**: when the Three Laws head
  is present, the VLM term is added to the gated bonus (alongside
  laws 2/3) so a contrived high progress score cannot override a harm
  violation. When `vlm_head=None`, behaviour is byte-identical to the
  pre-Phase-4 path
- **`src/mousedroid/config/schema.py`** — new `VLMProgressConfig`
  (`enabled`, `cache_size`, `instruction`, `mock_progress_value`,
  `hash_decimals`) with all defaults set per CLAUDE.md rule 3, plus
  `RewardConfig.weight_vlm_progress: float = 0.0` (off by default for
  safety) and `RewardConfig.vlm_progress: VLMProgressConfig`
- **`src/mousedroid/factory.py`** — new `build_reward_model(cfg)` factory.
  The VLM head is attached only when **both**
  `cfg.reward.vlm_progress.enabled` and
  `cfg.reward.weight_vlm_progress > 0` so a stray flag flip cannot
  silently change reward behaviour
- **`training/train_constitutional_rl.py`** — replaced direct
  `MultiObjectiveRewardModel(...)` construction with
  `build_reward_model(cfg)`; the existing constitutional check
  (`law1 → -1.0`, other violations → `0.0`) is unchanged — Phase 4
  augments the multi-objective signal, it does not weaken the safety
  override
- **`pyproject.toml`** — added `cachetools>=5.0` to core dependencies
- **`tests/unit/reward/test_vlm_progress.py`** — 20 new tests across six
  classes covering: `MockVLMProgress` value validation; cache hit/miss
  + LRU eviction at `cache_size=2`; instruction-keyed determinism;
  `hash_decimals` floating-point grouping; backend out-of-range
  guarding; backwards-compatible aggregator (no head, weight set);
  weight-zero invisibility (byte-identical to no-head path); factory
  default-off + opt-in + zero-weight wiring; **constitutional override
  Hypothesis property test** (`max_examples=50`) verifying that for any
  `(harm_bias, vlm_value, weight)` the contribution equals
  `sigmoid(harm_bias) * weight * vlm_value` to `1e-4` tolerance — i.e.
  Law-1 violations always zero the VLM contribution

### Backwards compatibility

- `RewardConfig` adds new optional fields with defaults — existing YAML
  configs load unchanged
- `MultiObjectiveRewardModel.__init__` adds keyword-only `vlm_head`
  defaulting to `None` — all existing call sites unchanged
- `compute_reward` / `aggregate` / `forward` add keyword-only optional
  args — all existing call sites unchanged
- All 36 pre-existing reward + Three Laws + constitutional RL tests
  pass unmodified

---

## [Released]

### Added — Phase 3b: DistilledVLAOnnx + HF weights pull (`feat/phase3b-distilled-onnx-vla`)

- **`src/mousedroid/vla/policy.py`** — new `DistilledVLAOnnx` class
  - Wraps an ONNX Runtime `InferenceSession` with provider fallback chain
    `TensorrtExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider`
    (configurable via `VLAConfig.providers`); requested providers are
    intersected with `ort.get_available_providers()` preserving order;
    falls back to CPU when the intersection is empty
  - **Lazy** import of `onnxruntime` and `numpy` inside `warmup()` /
    `predict()` so module import keeps the cold-import budget intact and
    no ORT runtime is required just to type-check or unit-test
  - Idempotent `warmup()` (configurable `warmup_iterations`, zero skips
    dummy runs); `predict(obs)` runs under `torch.no_grad()`, surfaces
    shape mismatches as `ValueError`, and emits structlog events
    (`distilled_vla_onnx_warmup_start/_complete`)
  - Module-level `DEFAULT_ORT_PROVIDERS` constant exported from
    `mousedroid.vla`
- **`config/schema.py`** — `VLAConfig` extended with seven Phase 3b fields:
  `model_repo_id`, `model_filename`, `cache_dir`, `providers`,
  `warmup_iterations`, `h_input_name`, `z_input_name`,
  `action_output_name`. `protected_namespaces=()` opts out of pydantic's
  `model_*` warning so `model_filename` / `model_repo_id` are clean
- **`factory.py`** — `build_vla_policy` now implements the
  `"distilled_onnx"` backend via private `_build_distilled_onnx_vla`
  helper. Reuses `mousedroid.utils.weights_manager.download_weights_from_huggingface`
  to pull the model when absent and `model_repo_id` is configured;
  raises a clear `ValueError` when neither a local file nor a repo is
  available, and when the download fails
- **`pyproject.toml`** — new `[vla]` extra:
  `onnxruntime-gpu>=1.18; platform_machine=='aarch64'`,
  `onnxruntime>=1.18; platform_machine!='aarch64'`,
  `transformers>=4.40`, `huggingface-hub>=0.20`
- **Tests (~30 new in `tests/unit/vla/test_distilled_onnx.py`)**
  - Construction validation (`action_dim`, `confidence`,
    `warmup_iterations`); `VLAPolicyProtocol` conformance
  - Pure provider-resolution tests (no ORT required)
  - **Subprocess import-graph isolation**: asserts
    `import mousedroid.vla.policy` does NOT pull `onnxruntime` or
    `transformers` into `sys.modules`
  - Stubbed-ORT warmup / predict tests (provider intersection,
    explicit ordering, configurable warmup count, idempotent warmup,
    shape-mismatch error path, `no_grad` enforcement, h/z named-input
    routing)
  - Factory-level tests (missing-file-and-no-repo, existing local file,
    HuggingFace download invocation, download-failure error path,
    provider/IO-name/confidence propagation)
- Updated `tests/unit/vla/test_policy.py::test_distilled_onnx_reserved`
  → `test_distilled_onnx_requires_model_or_repo` (Phase 3a's
  `NotImplementedError` is gone now that the backend is implemented)

### Added — Phase 3a: VLA Protocol + MockVLA (`feat/phase3a-vla-protocol`)

- **`src/mousedroid/vla/`** — new package
  - `VLAObservation` / `VLAAction` frozen dataclasses; `@runtime_checkable
    VLAPolicyProtocol` with sync `predict(obs) -> VLAAction`
  - `MockVLA` — deterministic, zero-dependency reference implementation;
    optional canned action and configurable confidence; runs inference
    under `torch.no_grad()`; validates `action_dim`, `confidence`, and
    canned-action shape
- **`config/schema.py`** — backwards-compatible additions
  - `LoopConfig.policy_selector: Literal["nav_agent", "vla", "auto"]`
    (default `"nav_agent"` preserves byte-identical legacy behavior)
  - `LoopConfig.inference_timeout_s: float | None` (None ⇒ `1/control_hz`)
  - New `VLAConfig` block with `backend ∈ {"none", "mock", "distilled_onnx"}`
    (default `"none"`), optional `canned_action`, `confidence`, and
    `fallback_on_timeout`. Wired into `Settings.vla` with safe defaults
  - `"distilled_onnx"` backend reserved for Phase 3b — raises
    `NotImplementedError` from the factory
- **`factory.py`** — `build_vla_policy(cfg) -> VLAPolicyProtocol | None`
  next to `build_llm_gateway`; orchestrator construction wires
  `vla_policy=build_vla_policy(cfg)`. Returns `None` for `backend="none"`
- **`orchestrator/orchestrator.py`** — VLA branch in `_select_action`
  - Default selector skips VLA entirely; `"vla"` and `"auto"` route through
    the policy and enforce a per-tick latency budget via `time.monotonic()`
  - `"auto"` mode silently falls back to the nav agent on timeout, predict
    exception, or shape mismatch
  - Strict `"vla"` mode honors `vla.fallback_on_timeout`: when False emits
    a structlog `vla_timeout_safe_stop` event and returns a zero action so
    the safety monitor can escalate; when True falls back like `"auto"`
- **Tests (43 new)**
  - `tests/unit/vla/test_policy.py` — protocol conformance, validation,
    determinism, no-grad inference, factory plumbing
  - `tests/unit/orchestrator/test_policy_selector.py` — default
    backwards-compatibility, all selector modes, timeout / exception /
    shape-mismatch fallback paths, default budget derivation from
    `control_hz`

### Added — Ten Pillars Validation Campaign (`feat/smoke-post-pr55`)

- **`scripts/validate_pillar.sh`** — headless Ten Pillars campaign dispatcher
  - Accepts a pillar name (e.g. `safety`) or `all` as its first positional argument;
    optionally a second argument `yes|no` to override the default blocking mode for that
    pillar; runs `pytest` headlessly then executes a factory-backed in-container Python probe
  - Correct probe implementations for all 10 pillars — uses `build_memory_tier`,
    `build_world_model`, `build_cognitive_core`, `build_curiosity_module`, and
    `build_safety_monitor` from `factory.py`; uses direct class instantiation for
    `MultiObjectiveRewardModel`, `EWCAgent`, `MAMLAdapter`, `AdaptiveCompute`, and
    `KnowledgeDistiller` (no phantom factory functions)
  - Writes a Markdown result table to `ten_pillars.log` alongside the SUMMARY.md
  - Blocking / non-blocking default mode declared per-pillar via `run_pillar_check
    <name> <yes|no> ...` call sites; individual pillars can be overridden via
    `MOUSEDROID_PILLAR_BLOCKING_<PILLAR>=yes|no` environment variable

- **`scripts/jetson_full_smoke_run.sh`** — wired Ten Pillars section into SUMMARY.md
  - Appends `## Ten Pillars Validation` block (from `ten_pillars.log`) to the smoke
    run SUMMARY.md when a pillar campaign was run in the same invocation

- **`docs/planning/TEN_PILLARS_VALIDATION.md`** — operator-grade Ten Pillars validation plan
  - Pre-conditions, per-pillar dependency order, PASS criteria, telemetry, and reporting layout

- **`.github/skills/jetson-hardware-debug/`** — Jetson hardware debug skill for Copilot agent
  - Covers SSH connection, sensor verification (camera, GPIO, LiDAR, speaker, microphone),
    config sync, Docker deployment, smoke testing, and hardware failure troubleshooting

- **`tests/regression/test_validate_pillar.py`** — 9 regression tests for `validate_pillar.sh`
  - Structural invariants: all 10 pillars have `case` branches, blocking defaults are correct,
    summary writes `ten_pillars.log` with correct columns, fallback shim creation,
    `jetson_full_smoke_run.sh` references `ten_pillars.log`

### Validated — Jetson Ten Pillars Campaign (`2026-04-26T23:55:42Z`)

- All **20 checks PASS** (20/20): 10 pytest stages + 10 factory probes
- Pillar results: World Model ✅, Cognitive ✅, Memory ✅, Continual ✅, Meta ✅,
  Curiosity ✅, Growth ✅, Reward ✅, Scaling ✅, Safety ✅
- Platform: Jetson Orin Nano, L4T r36.4.0, CUDA 12.6, TensorRT 10.4.0

### Added — Phase 1: Domain Randomization for Sim-to-Real RSSM Pretraining

First of four Physical AI gaps closed (per Martin Keen, IBM Technology — "What is
Physical AI?"). Per-episode randomization of physical and sensor parameters so
the RSSM world model and downstream policies generalize beyond a single nominal
simulator configuration. **Training-only change; runtime control loop untouched.**

- **`src/mousedroid/training/domain_randomization.py`** — new module
  - `DomainRandomizer` — stateless sampler driven by an injected
    `numpy.random.Generator` for deterministic seeding from the training pipeline
  - `EpisodeParams` — frozen dataclass carrying per-episode visual / camera /
    range-sensor / chassis / comms / disturbance / feature parameters
  - `apply_visual_randomization()` — RGB-frame transform with brightness,
    contrast, and additive Gaussian noise; preserves `uint8`/`float32` dtype
  - `apply_range_sensor_randomization()` — additive Gaussian noise + stochastic
    dropout (returns `nan`) for HC-SR04 readings
  - `apply_feature_noise()` — post-CNN feature-vector noise applied during data
    generation (the actual integration point — raw frames are not yet exposed)
  - 100% line + branch coverage on the new module
- **`src/mousedroid/config/schema.py`** — Pydantic v2 schema additions
  - `RangeF` — inclusive `[low, high]` range with `model_validator` ordering check
  - `DomainRandomizationConfig` — every threshold/probability is configurable;
    `enabled=True` by default; `enabled=False` restores byte-identical legacy behaviour
  - Wired into root `Settings.domain_randomization` via the existing
    `_settings_default_factory` pattern; existing YAML files load unchanged
- **`training/data_generator.py`** — `SyntheticSequenceGenerator` accepts an
  optional `seed: int | None`. When `cfg.domain_randomization.enabled=True`,
  per-episode `EpisodeParams` are sampled from a seeded master RNG and applied
  via `_apply_episode_randomization`; when disabled, the legacy `torch.randn`
  action path is preserved verbatim. The `2**63 - 1` literal for RNG re-seeding
  was replaced with `np.iinfo(np.int64).max` to keep the hardcoded-values gate clean.
- **`training/run_pipeline.py`** — `run_phase_0_data_gen(cfg, *, seed=None)`
  forwards `seed` into the generator and emits a structured
  `rssm_epoch_randomization` log event with the active envelope so audit trails
  can reproduce a run from logs alone.
- **YAMLs**:
  - `config/default.yaml` — explicit `domain_randomization:` block with documented ranges
  - `config/jetson_production.yaml` — tightened envelope for production fine-tuning
  - `config/mock_hardware.yaml` — widened envelope for stress-testing
- **Tests** (66 new tests, 100% changed-line coverage):
  - `tests/unit/training/test_domain_randomization.py` — 26 unit tests
  - `tests/unit/training/test_data_generator_dr.py` — 6 byte-identity regression tests
  - `tests/integration/training/test_data_generator_integration.py` — 6 end-to-end
    tests through the mock orchestrator (DR off / DR on / seed reproducibility)
  - `tests/unit/test_run_pipeline.py::TestPhase0DomainRandomization` — 2 tests
  - `tests/unit/test_config_loader.py` — 2 YAML-overlay round-trip tests
  - `tests/regression/test_domain_randomization_backcompat.py` — 18 pinned-default
    regression tests guarding YAML load hygiene, `RangeF` validation invariants,
    disabled-DR identity, env-var override path, and `Settings` round-trips

### Added — Current-branch voice rollout completion

- `VoiceConfig` gains `output_volume` with a backwards-compatible default of `1.0`
- `config/jetson_production.yaml` now uses `voice.personality_to_model_map`,
  `voice.event_intensity_thresholds`, and `voice.output_volume`
- `src/mousedroid/voice/tts.py` now applies configured output gain with clipping in the
  synthesized float32 path
- Added end-to-end TTS integration coverage, speaker+TTS integration coverage, and smoke-harness
  unit coverage
- Added operator recovery playbooks under `docs/playbooks/` for voice, LiDAR, and camera failures

### Changed

- Documentation is rebased to the current production truth: overlay sync is automatic under
  `mousedroid-docker.service`, the active Jetson baseline is camera + LiDAR + USB audio + ESP32,
  and the HC-SR04 / robot-arm tracks are explicitly deferred from the active roadmap.

### Added — Rocky Voice Engine (Piper TTS) + Full Jetson Smoke Harness

- **Rocky TTS pipeline** — end-to-end Piper voice synthesis, verified `PASS` on Jetson Orin Nano
  - `src/mousedroid/voice/tts.py` — async-safe `PiperTTS` wrapper
    - Prefers `synthesize_wav()` API (Piper ≥ 1.3); gracefully falls back to legacy
      `synthesize()` if `synthesize_wav` is not present on the voice object
    - Normalises int16 WAV output via `INT16_MAX_F` for downstream float32 pipeline
    - `torch.no_grad()` guard on all inference paths
    - 100% changed-line branch coverage (11 unit tests in `tests/unit/test_piper_tts.py`)
  - `VoiceConfig` and `SpeakerConfig` added to `src/mousedroid/config/schema.py`
    - `VoiceConfig`: `enabled` (default `false`), `tts_model_path`, `tts_sample_rate`,
      `cooldown_s`, `queue_size`, `personality`, `phrase_overrides`, `intensity_threshold`
    - `SpeakerConfig`: `device_name`, `sample_rate`, `channels`, `chunk_size`, `format`,
      `write_timeout_s`, `write_poll_interval_s`
    - Both fields are optional with safe defaults — existing YAML configs load unchanged
  - `config/jetson_production.yaml` — `voice` block added: `enabled: true`, model path
    `/opt/voice_models/en_US-lessac-medium.onnx`, `cooldown_s: 5.0`, `queue_size: 16`

- **USB Speaker driver improvements** (`src/mousedroid/hardware/audio/usb_speaker.py`)
  - Buffer-availability polling loop driven by `SpeakerConfig.write_timeout_s` and
    `write_poll_interval_s` — eliminates blocking on slow ALSA `write()` calls
  - Automatic device discovery by name substring (`SpeakerConfig.device_name`)
  - Clamps float32 samples to `[-1.0, 1.0)` before int16 conversion, preventing wrap-around
  - Graceful `OSError` / ImportError handling — falls back cleanly when PyAudio unavailable

- **Jetson full-smoke harness** (`scripts/jetson_full_smoke_run.sh`)
  - Orchestrates all hardware smoke stages inside the Docker container with per-stage
    timeouts, pass/fail tracking, and a colour SUMMARY.md report
  - Voice-failure enricher appends `## Rocky voice prerequisites` remediation block to
    SUMMARY.md when the voice stage fails (model path, overlay sync instructions)
  - All magic numbers are env-overridable: `MOUSEDROID_SMOKE_CONTAINER`, `MOUSEDROID_SMOKE_BUS`,
    `MOUSEDROID_SMOKE_REPORT_ROOT`, `MOUSEDROID_JETSON_CONFIGS`
  - Smoke run `20260425T192408Z`: all stages PASS including `voice` (39,424 audio samples)

- **`validation/runtime.py` voice helper**
  - `play_rocky_voice_phrase()` — factory-backed end-to-end TTS smoke check reused by
    `jetson_full_smoke_run.sh` and standalone validation scripts

- **Piper TTS Docker stage** (`Dockerfile.jetson`)
  - Stage 5b: `pip install piper-tts` + `curl --retry 3` download of
    `en_US-lessac-medium.onnx` + `.onnx.json` from HuggingFace (non-fatal fallback if
    network unavailable during build)

- **Regression tests** (`tests/regression/test_voice_speaker_backcompat.py`)
  - 22 backwards-compatibility tests covering: YAML load hygiene for all committed configs,
    `VoiceConfig` and `SpeakerConfig` default-value invariants, `jetson_production.yaml`
    voice fields, Piper model path format, and partial stanza round-trips

### Fixed

- **Piper ≥ 1.3 API compatibility** — `synthesize(text, wav_file)` no longer writes a WAV
  file in the new API; `_synthesize_sync` now calls `synthesize_wav()` when available
- **`/etc/mousedroid/jetson_production.yaml` overlay sync** — the service contract now uses
  `scripts/sync_jetson_overlay.sh` as a non-fatal `ExecStartPre` step before preflight, so the
  deployment docs no longer describe manual post-`git pull` copying as the standard path
- **`reports/jetson_smoke/` gitignore gap** — smoke run timestamped directories and
  `python3-in-container` shims were being committed; `.gitignore` now excludes
  `reports/jetson_smoke/*/` and the two committed run directories have been removed from tracking

### Added — CI Determinism and Config Compatibility Hardening

- **`scripts/check_settings_identity.py`** — pre-test guard that validates canonical
  `mousedroid.config.schema.Settings` identity before executing pytest stages
- **`tests/unit/test_config_migration.py`** — direct branch-coverage tests for
  `config/migration.py` alias and transform helpers

### Changed — CI Execution Path

- **`scripts/ci.sh`** now resolves Python deterministically (`MOUSEDROID_PYTHON` → workspace virtualenv
  → PATH), exports `PYTHONNOUSERSITE=1`, and runs pytest stages in `importlib` mode for stable import identity
- CI now prints runtime Python/pydantic versions and runs a dedicated Settings identity smoke check before tests

### Changed — Documentation Layout and References

- Planning and analysis documents were moved from repo root to `docs/planning/` and `docs/analysis/`
  to keep runtime code and deployment assets prominent at the top level
- README, changelog references, and planning links were updated to the new structure

### Added — Jetson Runtime Validation Alignment

- **`src/mousedroid/validation/runtime.py`** — shared runtime validation helpers used by smoke,
  verification, and host-driven Jetson validation flows
  - `resolve_runtime_config_paths()` + `load_runtime_settings()` keep CLI utilities aligned with
    the same overlay precedence as the deployed application
  - `capture_camera_frame()`, `capture_microphone_chunk()`, `read_lidar_scan()`, and
    `collect_lidar_diagnostics()` provide factory-backed runtime checks without duplicating device logic
  - `lidar_scan_validation_coverage_deg()` keeps smoke assertions aligned with the driver's own
    coverage semantics

### Changed — Jetson Camera, LiDAR, and Smoke Harnesses

- **`JetsonCSICamera`** now falls back from the Jetson-native path to GStreamer and then the
  configured V4L2 `camera.device_path`, keeping ribbon-camera deployments usable when
  `jetson_utils` or the primary pipeline is unavailable
- **`LidarConfig`** gains `scan_acquisition_timeout_s` and `min_scan_coverage_deg`, so LD19 scan
  completeness is config-driven instead of hardcoded in the driver or validation scripts
- **`scripts/jetson_smoke_test.sh`**, **`scripts/jetson_validate.sh`**, and
  **`scripts/verify_sensors.py`** now reuse the shared runtime validation layer and runtime config
  overlays instead of resolving hardware paths independently

### Fixed — CI / Test Alignment

- **Performance and E2E fixture mismatch** — test paths that previously constructed settings
  directly now use runtime-loaded settings so mock-hardware CI and hardware-targeted validation stay aligned
- **Strict type-check blockers** — NumPy feature-extractor typing and optional cloud SDK boundaries
  are now mypy-clean without broad ignore directives
- **Mock-hardware endurance expectation** — the 30 Hz deadline assertion in the endurance suite now
  only applies to non-mock hardware, while mock runs still validate the rest of the endurance path

### Added — GCP Digital Twin (Phase 1: Telemetry Bridge + Cloud Storage)

- **`src/mousedroid/cloud/` module** — complete GCP cloud integration layer
  - `CloudTelemetrySinkProtocol`, `CloudExperienceExporterProtocol`,
    `CloudLoggingSinkProtocol`, `CloudMetricsExporterProtocol` — 4 `@runtime_checkable` protocols
  - `CloudTelemetrySink` — Pub/Sub publisher with `CircuitBreaker` + msgpack serialization;
    non-blocking fire-and-forget; circuit-open messages silently dropped
  - `CloudExperienceExporter` — LMDB-to-GCS batch exporter with high-water-mark cursor,
    date-hour partitioned shards (`gs://{bucket}/{prefix}/{robot_id}/{date}/{hour}/shard_{uuid}.msgpack.gz`),
    configurable gzip/zstd compression
  - `CloudLoggingSink` — structlog processor forwarding to Cloud Logging (fire-and-forget)
  - `CloudMetricsExporter` — parses Prometheus text exposition output from `MetricsRegistry`,
    writes gauge metrics to Cloud Monitoring custom metrics
  - `CloudFirestoreSync` — episodic memory sync to Firestore collection
  - `_auth.py` — credential resolver (ADC or explicit service account key)

- **8 GCP Pydantic config models** (`src/mousedroid/config/schema.py`)
  - `GCPConfig` umbrella with `GCPPubSubConfig`, `GCPStorageConfig`, `GCPLoggingConfig`,
    `GCPMonitoringConfig`, `GCPFirestoreConfig`, `GCPTrainingConfig`, `GCPSimulationConfig`
  - `Settings.gcp: GCPConfig | None = None` — all GCP features disabled by default;
    existing YAML files load unchanged (full backwards compatibility)

- **4 `build_cloud_*()` factory functions** (`src/mousedroid/factory.py`)
  - All return `None` when `cfg.gcp is None` (offline mode)
  - Graceful ImportError fallback when `google-cloud-*` packages not installed

- **Orchestrator cloud integration** (`src/mousedroid/orchestrator/orchestrator.py`)
  - Optional `cloud_sink` + `cloud_experience_exporter` constructor params
  - Telemetry + experience published to cloud at each tick; lifecycle managed in start/stop

- **`config/gcp_digital_twin.yaml`** — YAML overlay for GCP-enabled deployments

- **`pyproject.toml`** — `gcp`, `gcp-training`, `gcp-simulation` optional dependency groups

- **`Dockerfile.jetson`** — GCP SDK install stage (non-fatal graceful fallback)

- **`docker-compose.jetson.yml`** — GCP credentials volume mount + env vars

- **88 cloud unit tests** (85 passing, 3 skipped when google-auth absent)
  - Config backwards compatibility (10), Pub/Sub sink (14), experience exporter (18),
    logging sink (11), monitoring exporter (18), firestore sync (13), auth (4)
  - Cloud module coverage: **88.77%** (above 85% gate)

### Fixed

- **`LogRingBuffer` NameError in `build_orchestrator()`** — `LogRingBuffer` was imported under
  `TYPE_CHECKING` but used at runtime when `telemetry.log_stream_buffer > 0`; moved to local
  import inside `build_orchestrator()` (fixes ~39 pre-existing test failures across e2e,
  integration, and performance suites)

---

## [0.3.0] — 2026-04-14 — Production Readiness

This release completes the **MouseDroidAGI Production Readiness** milestone across 7 phases,
bringing all cognitive, memory, voice, safety, and deployment subsystems to a production-ready
state on the NVIDIA Jetson Orin Nano. 2505 tests pass; branch-coverage gate ≥ 85%.

### Added — Phase 1: Deployment Hardening

- **Docker device passthrough** (`docker-compose.jetson.yml`)
  - All device mappings now active with env-var overrides: `${MOUSEDROID_ESP32_DEV:-/dev/ttyUSB0}`,
    `${MOUSEDROID_CAMERA_DEV:-/dev/video0}`, `${MOUSEDROID_LIDAR_DEV:-/dev/ttyUSB1}`, GPIO, audio
  - `group_add: [audio, video, dialout, gpio]` for correct device permissions
  - Docker `HEALTHCHECK` directive polling `/api/v1/health` (30s interval, 3 retries)
  - Persistent `promtail_positions` volume to survive restarts

- **Tick timeout + emergency stop** (`src/mousedroid/config/schema.py`, `orchestrator.py`)
  - `LoopConfig.tick_timeout_s` — configurable per-tick timeout (default 1.0 s, `gt=0`)
  - `asyncio.wait_for(self.tick(), timeout=tick_timeout)` wraps every orchestrator tick
  - `asyncio.TimeoutError` → `emergency_stop()` + critical log + voice error event
  - Unhandled exception in `tick()` → `emergency_stop()` + voice error event
  - `LoopConfig.watchdog_enabled`, `watchdog_interval_s` fields added

- **Systemd watchdog integration** (`src/mousedroid/health/watchdog.py`)
  - `WatchdogProtocol` — `@runtime_checkable Protocol` with `notify()` method
  - `SystemdNotifier` — sends `WATCHDOG=1` via `sdnotify` package or `systemd-notify` subprocess fallback
  - `FileHeartbeatNotifier` — writes monotonic timestamp to configurable path for Docker HEALTHCHECK
  - `NullNotifier` — no-op for mock/dev mode
  - `build_watchdog(cfg)` factory function auto-selects notifier based on environment + config
  - Orchestrator calls `watchdog.notify()` after each successful tick

- **systemd service hardening** (`scripts/mousedroid.service`, `scripts/mousedroid-docker.service`)
  - `Type=notify` + `WatchdogSec=30` on both service units
  - `ExecStartPre=/opt/mousedroid/scripts/preflight_check.sh` blocks startup on hardware failure
  - `MOUSEDROID_LOOP__WATCHDOG_ENABLED=true` injected into service environment

- **Pre-flight validation script** (`scripts/preflight_check.sh`)
  - Checks ESP32, camera, GPIO (required); LiDAR, audio (optional warnings)
  - Validates Docker/NVIDIA runtime, disk space (configurable `MOUSEDROID_MIN_DISK_GB`), config YAML syntax
  - Checks model weights presence (LLM + BDI)
  - Coloured PASS/FAIL/WARN output, exits non-zero on any required failure
  - Fully configurable via env vars: `MOUSEDROID_ESP32_DEV`, `MOUSEDROID_CAMERA_DEV`, `MOUSEDROID_LIDAR_DEV`

- **Docker env documentation** (`config/docker.env.example`)
  - Documents all device path env vars with default values and required/optional annotations

- **Pre-commit coverage hook extended** (`scripts/check_branch_coverage.py`)
  - Detects Pydantic Settings + coverage.py class-identity false-failure pattern
  - Falls through gracefully with `ALLOW_PYTEST_COLLECTION_SKIP=1` bypass

- **Tests** — `tests/unit/test_watchdog.py` (12 tests), `tests/unit/test_tick_timeout.py` (7 tests),
  `tests/integration/test_preflight_validation.py` (11 tests)

### Added — Phase 2: Memory & Curiosity Pipeline Wiring

- **`MemoryTier` dataclass** (`src/mousedroid/memory/tier.py`)
  - Groups `episodic`, `semantic`, `working`, and `consolidation` managers into a single injectable unit
  - `build_memory_tier(cfg)` factory function; enabled via `cfg.memory.enabled` (default `False`)

- **Orchestrator memory integration** (`src/mousedroid/orchestrator/orchestrator.py`)
  - Optional `memory_tier: MemoryTier | None` parameter in `MouseDroidOrchestrator.__init__`
  - Each tick: creates `ExperienceRecord` from obs + action + safety context; pushed to episodic + working memory
  - Background `asyncio.Task` runs `MemoryConsolidation.consolidate()` on `consolidation_interval_s` interval

- **Curiosity wiring**
  - ICM intrinsic reward computed from previous/current latent states each tick
  - `"curiosity"` key injected into `obs_dict` with per-channel curiosity scores
  - `SemanticIndex.retrieve()` queried for epistemic novelty when memory enabled

- **Tests** — `tests/unit/test_memory_tier.py` (8 tests), `tests/integration/test_memory_pipeline.py` (6 tests),
  `tests/unit/test_curiosity_wiring.py` (5 tests)

### Added — Phase 3: Voice & Rocky End-to-End

- **Startup/shutdown voice events** (`src/mousedroid/orchestrator/orchestrator.py`)
  - `start()` fires `"startup"` voice event after voice engine initialises
  - `stop()` fires `"shutdown"` voice event before teardown

- **Enriched voice context**
  - Emergency stop paths fire `"error"` voice event with safety context
  - `lidar_min_dist_m` included in obstacle voice events
  - Audio level RMS included in voice context when microphone available

- **Tests** — `tests/integration/test_orchestrator_voice_events.py` (8 tests)

### Added — Phase 4: Sensor Fusion Resilience

- **Sensor recovery protocol** (`src/mousedroid/sensing/manager.py`)
  - `async recovery_attempt() -> int` — tries to reinitialise failed sensors; returns recovered count
  - Orchestrator attempts recovery before triggering emergency stop on sensor degradation

- **Config additions** (`src/mousedroid/config/schema.py`)
  - `SafetyConfig.sensor_recovery_attempts` (default 1)
  - `SafetyConfig.sensor_recovery_delay_s` (default 0.5 s)

- **Self-healing orchestrator tests** — `tests/integration/test_self_healing_orchestrator.py` (9 tests)
- **Cascading sensor failure tests** — `tests/integration/test_cascading_sensor_failure.py` (11 tests)

### Added — Phase 5: LLM Gateway Integration

- **LLM gateway wired into orchestrator** (`src/mousedroid/factory.py`, `orchestrator.py`)
  - `build_llm_gateway(cfg)` called in `build_orchestrator()` when `cfg.llm.enabled`
  - `process_mission(nl_command)` method on orchestrator for NL → `GoalVector` translation
  - Rule-based parser first (< 1 ms for common commands); LLM fallback for complex/unknown commands
  - Prompt injection detection rejects malicious inputs

- **Degraded mode** (`src/mousedroid/llm_gateway/gateway.py`)
  - `start()` enters degraded mode (log warning, `_degraded=True`) instead of raising when
    `llama-cpp-python` or model file is missing — service continues operating safely

- **Tests** — `tests/integration/test_llm_gateway_wiring.py` (6 tests),
  updated `tests/unit/test_llm_gateway.py` (degraded-mode tests)

### Added — Phase 6: Jetson On-Device Validation Suite

- **Hardware E2E tests** — `tests/e2e/test_jetson_hardware_e2e.py` (marked `@pytest.mark.jetson`)
  - Camera, ultrasonic, ESP32, LiDAR, microphone, speaker, full 5-tick orchestrator loop with real sensors

- **Endurance tests** — `tests/performance/test_jetson_endurance.py`
  - 5-minute 30 Hz run; GPU temp < 85 °C; RSS stable within 10 %; loop p95 < 33 ms

- **Sensor verification script** (`scripts/verify_sensors.py`)
  - Updated with LiDAR + speaker checks, `--json` output flag for CI integration

### Added — Phase 7: Production Telemetry & Metrics

- **New Prometheus metrics** (`src/mousedroid/telemetry/metrics.py`)
  - `{ns}_memory_episodic_size`, `{ns}_memory_semantic_size` — episodic and semantic index size gauges
  - `{ns}_memory_working_size` — working memory context window size gauge
  - `{ns}_curiosity_intrinsic_reward` — intrinsic curiosity reward gauge per tick
  - `{ns}_voice_events` — voice event counter labelled by event type
  - `{ns}_llm_requests`, `{ns}_llm_latency_ms` — LLM gateway request counter + latency gauge (ms)
  - `{ns}_sensor_recoveries`, `{ns}_sensor_recovery_failures` — sensor recovery counters
  - All metric names use `{ns}` = `MetricsConfig.namespace` (default: `mousedroid`)

### Fixed

- **LLM gateway RuntimeError regression** — `gateway.py` `start()` no longer raises when
  `llama-cpp-python` is absent; uses degraded mode so tests relying on `build_orchestrator()`
  default config continue to pass
- **Pydantic Settings + coverage.py false failure** — `check_branch_coverage.py` pre-commit hook
  extended to detect `is_instance_of` + `Settings` coverage fingerprint and bypass cleanly
- **`test_file_heartbeat_notify_updates_timestamp` flakiness** — sleep increased to 50 ms
  to avoid race under load

### Changed

- **`config/schema.py`** — `LoopConfig` gains `tick_timeout_s`, `watchdog_enabled`,
  `watchdog_interval_s`; `SafetyConfig` gains `sensor_recovery_attempts`, `sensor_recovery_delay_s`;
  all new fields have defaults preserving full backward compatibility
- **`orchestrator.py`** — `run()` loop restructured around `asyncio.wait_for`; adds optional
  `watchdog` and `memory_tier` constructor parameters; enriches voice event context
- **`factory.py`** — adds `build_watchdog()`, `build_memory_tier()`, `build_llm_gateway()`
  (when `cfg.llm.enabled`) wired into `build_orchestrator()`
- **`docker-compose.jetson.yml`** — all devices uncommented with env-var overrides;
  healthcheck added; `group_add` permissions granted; Promtail positions volume added
- **`.gitignore`** — adds Serena workspace, heartbeat runtime files, LLM model dir,
  pre-flight output, validation output patterns

### Added

- **Dual-Stream CfC/GRU RSSM world model** — liquid neural network hybrid for adaptive reflexes
  - `DualStreamRSSM` — dual-stream architecture: GRU (slow planning, 256-dim) + CfC (fast reflexes, 64-dim) with concat fusion producing 320-dim combined hidden state
  - `CfCWrapper` — Closed-form Continuous-time cell wrapping `ncps.torch.CfC` with configurable backbone (units, layers, sparsity)
  - `StreamFusion` — concatenation-based fusion layer with `fuse()`, `extract_gru_state()`, `extract_cfc_state()` operations
  - `WorldModelProtocol` + `SafetyTraceProtocol` — `@runtime_checkable` protocol interfaces for world model DI
  - `DualStreamTrainingConfig` — Pydantic config for dual optimizers, gradient clipping, CfC loss warmup schedule
  - `ModelConfig` gains CfC fields: `cfc_hidden_dim`, `cfc_backbone_units`, `cfc_backbone_layers`, `cfc_mode`, `cfc_sparsity_level`
  - `build_world_model()` factory dispatch: `cfc_hidden_dim > 0` → `DualStreamRSSM`, else classic `RSSM`
  - `gru_parameters()` / `cfc_parameters()` — separate parameter groups for dual optimizer training
  - `get_safety_trace()` — extracts CfC hidden state from combined state for independent safety monitoring
- **Dual-stream training script** — `training/train_dual_stream_rssm.py` (712 LOC)
  - Dual Adam optimizers: GRU params (lr=3e-4) + CfC params (lr=1e-4)
  - Separate gradient clipping: GRU (max_norm=10.0) + CfC (max_norm=1.0)
  - Linear CfC loss weight warmup from 0.1→1.0 over 10k steps
  - Periodic fallback monitoring: logs CfC contribution quality, warns on >5% degradation
  - Full AMP support, checkpoint resume with dual optimizer states
  - CLI: `--config`, `--data`, `--device`, `--resume`, `--validate-only`
- **Jetson dual-stream config** — `config/jetson_dual_stream.yaml` with CfC activation gate
- **Human activation gate** — CfC disabled by default (`cfc_hidden_dim=0`); requires explicit `MOUSEDROID_MODEL__CFC_HIDDEN_DIM=64` to enable
- **HuggingFace model repo** — `ianshank/mousedroid-dual-stream-rssm` with 5-epoch validation weights + training metadata
- **57 new dual-stream tests**:
  - `test_cfc_cell.py` — CfC wrapper unit tests (initialization, forward, hidden dims)
  - `test_dual_stream_rssm.py` — DualStreamRSSM observe/imagine, protocol conformance, safety trace
  - `test_stream_fusion.py` — fusion layer, extract/fuse roundtrip
  - `test_dual_stream_training.py` — dual optimizer construction, warmup schedule, gradient clipping, checkpoint roundtrip
  - `test_dual_stream_compat.py` — factory dispatch, config backward compatibility, regression suite
  - `test_world_model_property.py` — Hypothesis property tests for rollout stability
  - `test_factory_integration.py` — integration tests for factory dispatch paths
- **ncps dependency** — `ncps>=0.0.7` added to `pyproject.toml` `[cfc]` extra and `Dockerfile.jetson`

### Changed

- **`Dockerfile.jetson`** — added `ncps>=0.0.7` install step (non-fatal graceful fallback)
- **`world_model/__init__.py`** — exports `DualStreamRSSM`, `CfCWrapper`, `StreamFusion`, protocol types
- **`factory.py`** — `build_world_model()` gains dual-stream dispatch branch

- **FHL-LD19 2D LiDAR sensor** — 5th modality integrated end-to-end through the cognitive stack
  - `LD19LidarDriver` — async UART driver with CRC8-validated binary protocol parsing
  - `LD19FrameParser` — LD19 packet parser with angle interpolation (n-1 intervals)
  - `LidarFeatureExtractor` — sector-binned distance features normalised to `[0, 1]`, vectorised via `np.minimum.at`
  - `MockLidar` — configurable mock driver for CI/testing
  - `ResilientLidarDriver` — circuit-breaker + retry wrapper for production reliability
  - `LidarScan` dataclass for typed scan data (angles, distances, confidences)
  - `LidarProtocol` — `@runtime_checkable Protocol` for DI
  - `LidarConfig` — Pydantic config with range validation, sector count, feature dim
  - `build_lidar()` / `build_lidar_feature_extractor()` factory functions
  - `SensorManager` gains LiDAR ring buffer + concurrent `_safe_lidar_read()`
  - `MultimodalEncoder` gains optional `lidar_proj` layer (enabled when `ModelConfig.lidar_dim > 0`)
  - `RSSM.observe_step()` threads LiDAR features through observation pipeline
  - `SafetyMonitor` evaluates LiDAR clearance via `SafetyConfig.lidar_max_range_m`
  - `TelemetryFrame.lidar_min_dist_m` — LiDAR distance surfaced in telemetry
  - `lidar_diagnostics` tool registered in tool registry
  - 12 new test files with 200+ LiDAR-specific tests
- **Wonrabai USB Sound Card** — combo mic + 8Ω 5W speaker on single USB interface
  - Speaker and voice engine enabled in `config/default.yaml` and `config/jetson_production.yaml`
  - Docker ALSA audio passthrough (`/dev/snd` + `group_add: [audio]`) in `docker-compose.jetson.yml`
  - 6 combo audio device tests verifying both mic and speaker discover the same USB device
- **Audio constants** — `POWER_CLIP_MAX` and `LOG_FLOOR` extracted to `hardware/audio/constants.py`

### Changed

- **`UsbMicrophone`** — renamed from "SuziePi" to generic USB; added graceful degradation
  matching `UsbSpeaker` pattern (try/except ImportError + OSError, return silence on failure)
- **`AudioFeatureExtractor`** — magic numbers `1e20` / `1e-10` replaced with named constants
- **`SafetyMonitor`** — `lidar_max_range_m` accessed directly from `SafetyConfig` field
  (was `getattr` with hardcoded `12.0` fallback)
- **`build_telemetry_frame()`** — uses `safety_ctx.lidar_min_dist_m` (actual metres)
  instead of raw normalised feature minimum
- **`MultimodalEncoder`** — missing LiDAR mask slot now treated as invalid (zeroed out)
  instead of silently passing unvalidated projection
- **`SensorManager._safe_lidar_read()`** — returns `ok=False` when feature extractor
  is missing (was `True`, feeding fake all-ones data marked valid)

### Fixed

- **LD19 angle interpolation** — fixed n-1 intervals formula (`step = diff / (n_points - 1)`)
- **3 mypy strict errors** — `torch.jit.save` untyped call, `depth_processor` Any return, stale `cv2` type-ignore
- **CRC test flakiness** — replaced probabilistic different-inputs-differ assertion with deterministic known test vectors (`0x74`, `0x4C`)
- **`usb_microphone.py` coverage** — removed from `pyproject.toml` coverage omit list

### Added (previous)

- **Audio integration into world model** — microphone data now flows end-to-end through the cognitive stack
  - `MultimodalEncoder` gains optional `audio_proj` layer (enabled when `ModelConfig.audio_dim > 0`)
  - `RSSM.observe_step()` extracts `audio_chunk` from observations and passes it to the encoder
  - `ModelConfig` gains `audio_dim` (default 0, backwards-compatible) and `audio_proj_dim` (default 32) fields
  - `config/default.yaml` enables microphone and sets `audio_dim: 1024`
- **Reusable camera feature extraction** — `FeatureExtractorProtocol` with pluggable backends
  - `src/mousedroid/hardware/camera/feature_extractor.py` — new module with `MeanPoolExtractor`, `TensorRTExtractor`, and `build_feature_extractor()` factory
  - `CameraConfig` gains `feature_extractor` (Literal `"mean_pool"` / `"tensorrt"` / `"auto"`) and `l2_normalize` (bool) fields
  - `TensorRTExtractor` loads ONNX models via `onnxruntime` with graceful fallback to mean-pool
- **Audio pipeline tests** — `tests/integration/test_audio_pipeline.py` (3 tests), 10 new encoder tests, 3 new RSSM tests
- **Feature extractor tests** — `tests/unit/test_feature_extractor.py` (13 tests) covering protocol compliance, L2-norm, TRT fallback
- **Config tests** — 9 new tests for `audio_dim`, `audio_proj_dim`, `feature_extractor`, `l2_normalize` fields

### Changed

- **`MicrophoneConfig.device_name`** — default changed from `"SuziePi"` to `"USB"` (matches common USB mics like TI PCM2902)
- **`JetsonCSICamera` / `IMX500Camera`** — feature extraction delegated to `FeatureExtractorProtocol`; 15 lines of duplicate mean-pool code removed from each driver
- **`MultimodalEncoder`** — docstring updated to reflect up-to-4-modality valid mask; `_AUDIO_IDX = 3` constant added
- **`telemetry/server.py`** — `isinstance(gpu_temp, (int, float))` → `isinstance(gpu_temp, int | float)` (UP038 lint fix)

### Fixed

- **Stale `type: ignore` comments** — removed `[no-redef]` on `jetson_csi.py` optional imports and `[untyped-decorator]` on `rssm.py` `@torch.no_grad()`
- **Duplicate feature extraction** — identical 15-line `_extract_features()` in both camera drivers replaced with shared `MeanPoolExtractor`
- **Ruff UP038 violation** — `isinstance(gpu_temp, (int, float))` in `telemetry/server.py` now uses union syntax

### Removed

- Duplicate `_extract_features()` implementations from `jetson_csi.py` and `imx500.py` (replaced by shared `feature_extractor.py`)

---

- **Phase A — Training pipeline resume + type cleanup** (`training/`)
  - `run_pipeline()` and `run_phase_1_rssm()` now accept `resume_from: Path | None`; CLI gains `--resume` flag to resume RSSM training from an existing checkpoint
  - `training/training_utils.py`, `train_bdi.py`, `collect_annotations.py`, `data_generator.py`, `warmstart_policy.py`, `train_constitutional_rl.py` — replaced bare `np.ndarray` annotations with `numpy.typing.NDArray[Any]`; added local `# type: ignore[attr-defined]` on factory-returned `object` access; `mypy training/ --ignore-missing-imports` now reports `Success: no issues found in 12 source files`
  - `warmstart_policy.py` — `tune_ucb()` now correctly passes `ucb_target_ms` when constructing the nested `MCTSConfig` candidate
- **Training test surface extended**
  - `tests/unit/test_run_pipeline.py` — added `test_resume_from_is_forwarded_to_phase_1`, `test_phase_0_and_2_runs_without_prior_rssm_artifact_if_phase_1_skipped`, `test_phases_2_3_4_require_missing_upstream_artifacts`
  - `tests/unit/test_warmstart_policy.py` — added `TestComputeLatentStatistics.test_returns_correct_shapes`, `TestRunWarmstart.test_run_warmstart_creates_artifacts`, `TestRunWarmstart.test_run_warmstart_passes_ucb_target_from_config`
- **Jetson production config activated** (`config/jetson_production.yaml`)
  - Cognitive core enabled: `cognitive.enabled: true`, `weights_dir: /opt/mousedroid/weights/bdi`, HF auto-download with up to 5 retries
  - Telemetry enabled: host `0.0.0.0:8080`, 10 Hz, mDNS broadcast, JSON serialisation
  - Prometheus metrics enabled at `/metrics` under `mousedroid` namespace
  - Jetson safety overrides: `min_valid_sensors: 1`, `battery_critical_v: 9.5`
- **HuggingFace weight publishing** — BDI, constitutional-RL, MCTS warmstart, and RSSM checkpoint weights uploaded to `ianshank/mousedroid-weights` (28 files, ~30 MB)
- **HuggingFace download subfolder fix** (`src/mousedroid/`)
  - `CognitiveConfig.huggingface_subfolder` field added (default `"bdi"`) — determines which subfolder of the HF repo contains the BDI `.npz` files
  - `download_weights_from_huggingface()` / `_download_file_with_retry()` gain `subfolder` and `local_dir` kwargs; `hf_hub_download` is now called with `subfolder=` and `local_dir=weights_dir.parent` so files land exactly at `weights_dir/belief.npz` etc.
  - `config/default.yaml` gains `cognitive.huggingface_subfolder: "bdi"`
  - Local production smoke confirms end-to-end: config overlay loads → BDI weights auto-downloaded from HF → `NeuralBDI` initialised with `weights_source=huggingface` → orchestrator `health_check` returns `status: ok`

- **WiFi/Ethernet Telemetry Server** — `src/mousedroid/telemetry/` — real-time remote monitoring over the local network
  - `TelemetryServer` — aiohttp-based REST + WebSocket server (`/api/v1/status`, `/api/v1/sensors`, `/api/v1/health`, `/api/v1/logs`, `/api/v1/network`, `/metrics`, `/ws`)
  - `TelemetryPublisher` — non-blocking async queue bridge; rate-limiting (≤60 Hz); drop-on-full semantics
  - `TelemetryFrame` — immutable frozen dataclass snapshot (all plain Python types; JSON/msgpack serialisable)
  - `LogRingBuffer` — structlog processor that captures the last *N* log entries for `/api/v1/logs`
  - `FrameBuilder` — converts `ObservationBundle` → `TelemetryFrame` each control-loop cycle
  - `NetworkInterface` discovery — uses stdlib `socket` only; no external dependencies
  - Optional API-key authentication (`X-API-Key` header), CORS middleware, mDNS/Zeroconf registration
  - Optional msgpack serialisation for binary-efficient WebSocket streaming
  - `MockTelemetryServer` — zero-dependency stub that satisfies `TelemetryServerProtocol` for CI/unit tests
- **Telemetry configuration** — `TelemetryConfig` Pydantic model added to `config/schema.py`
  - Fields: `enabled`, `host`, `port`, `publish_hz` (≤60), `queue_size`, `api_key`, `cors_origins`, `max_clients`, `mdns_enabled`, `mdns_service_name`, `serialization`, `log_stream_buffer`
- **`common/actions.py`** — `ActionNormalizer` utility extracted from orchestrator for reuse
- **Prometheus metrics registry** — `src/mousedroid/telemetry/metrics.py`
  - Pure-stdlib Prometheus text-format exporter (no third-party metrics dependency)
  - Config-driven metric namespace and per-metric toggles (no hardcoded metric names)
  - Tracks loop time, battery voltage, websocket clients, frame drops, safety violations, and GPU temperature
- **Telemetry smoke tests** — `tests/smoke/test_telemetry_smoke.py` (43 tests)
  - Covers full stack: `TelemetryFrame` → `LogRingBuffer` → `TelemetryPublisher` → `TelemetryServer` REST + WebSocket → E2E integration
  - All network I/O mocked to avoid DNS hangs on Windows; Windows-only `socket.getaddrinfo` test skipped with `@pytest.mark.skipif`
- **Telemetry unit tests** — 10 config, 14 network, 20+ server unit tests in `tests/unit/`
- **Telemetry integration test** — `tests/integration/test_telemetry_e2e.py`
- **Modular refactor** — `ab6b01c` — hard-coded values eliminated; `constants.py` expanded; dependency injection improved across `orchestrator`, `factory`, `cognitive_core`, `sensing/manager`

### Changed

- **`pyproject.toml`** — added `smoke` pytest marker; `aiohttp` added to `[server]` extras
- **Coverage config** — removed `src/mousedroid/telemetry/server.py` from coverage omit list so telemetry route changes are gated
- **`config/default.yaml`** — `telemetry` section with sensible defaults
- **`factory.py`** — `build_telemetry_server()` wires `TelemetryPublisher` → `TelemetryServer` → `Orchestrator`
- **`orchestrator.py`** — publishes `TelemetryFrame` each tick when telemetry enabled; lifecycle `start()`/`stop()` for server
- **`sensing/manager.py`** — `SensorManager` injects `TelemetryPublisher` for frame forwarding
- **`scripts/ci.sh`** — adds branch changed-line coverage gate (`scripts/check_branch_coverage.py --min 85`)
- **Git pre-commit hook** — local hook runs branch coverage gate automatically before commit when `src/mousedroid` Python files are modified

### Fixed

- **`tests/unit/test_cognitive_core.py`** — fixed I001 import sort
- **`tests/unit/test_telemetry_config.py`** — added `# noqa: S104` for `0.0.0.0`; `PT011` match patterns on all `pytest.raises`
- **`tests/unit/test_telemetry_network.py`** — SIM117 nested `with` blocks combined; Windows DNS-hang test skipped
- **`tests/unit/test_telemetry_server.py`** — E402 noqa after `importorskip`; network endpoints mocked to avoid real socket I/O
- **`tests/integration/test_docker_gpu.py`** — Jetson/container-specific assertions now guarded with `skipif` on non-Jetson hosts or non-L4T containers
- **`scripts/check_branch_coverage.py`** — branch coverage enforcement now based on changed executable lines instead of whole-file percentage
- **17 ruff violations** resolved across 4 PR test files

---

## [0.12.0] — Previous unreleased work

### Added

- **GPU Pre-Training Pipeline** — end-to-end orchestration for running phases natively on Jetson Orin Nano
  - `run_pipeline.py` orchestrator and native AMP support in `train_rssm.py`
  - GPU-accelerated MCTS rollouts in `warmstart_policy.py`
  - Native fallback logic and memory limit checks (6 GB default) via `GPUConfig`
  - Automated HuggingFace Hub artifact uploading via `upload_weights.py`
  - Full CI test-suite coverage (24 new unit tests added)

- **CognitiveCore integration** — dual-cadence BDI + metacognitive + constitutional loops wired into `MouseDroidOrchestrator`
  - Fast path (30 Hz): `PolicyMLP` + `ConstitutionalChecker` via `tick_fast()`
  - Slow path (~1 Hz): `NeuralBDI` inference + metacognitive updates via background `asyncio.Task`
  - Graceful fallback to MCTS agent on cognitive failure
- **`CognitiveConfig`** — Pydantic config in `schema.py` with HuggingFace auto-download, weights dir, fallback settings
- **`build_cognitive_core()`** — factory function with weight loading strategy (local → HuggingFace → random init)
- **`weights_manager.py`** — HuggingFace weight download with exponential backoff retry logic
- **21 new tests** — orchestrator cognitive paths (7), factory cognitive (4), weights manager (10)
- **`docs/analysis/COVERAGE_ANALYSIS.md`** — coverage gap analysis and 85% enforcement plan
- **`docs/analysis/TEST_SUITE_SUMMARY.md`** — detailed breakdown of all 21 cognitive test cases
- **`docs/analysis/VALIDATION_CHECKLIST.md`** — step-by-step validation and CI/CD simulation guide
- **Docker GPU deployment** — `Dockerfile.jetson` using NVIDIA L4T PyTorch base (`dustynv/l4t-pytorch:r36.4.0`) with CUDA 12.6, TensorRT 10.4, and pycuda pre-installed
- **Docker Compose** — `docker-compose.jetson.yml` with NVIDIA runtime, optional hardware passthrough, and volume mounts
- **CI/CD pipeline** — `.github/workflows/ci.yml` with 5-stage pipeline (lint → typecheck → test → security → Docker) across Python 3.10/3.11 matrix
- **Docker GPU integration tests** — `tests/integration/test_docker_gpu.py` with auto-skip outside L4T container
- **Container test runner** — `scripts/jetson_test_runner.sh` for running categorised tests inside the container
- **Docker deploy script** — `scripts/docker_deploy.sh` for automated container deployment
- **Systemd Docker service** — `scripts/mousedroid-docker.service` for automatic container startup on boot
- **`.dockerignore`** — optimised Docker build context (excludes `.git`, caches, docs)
- **L4T container ADR** — `docs/architecture/ADR-l4t-container.md` documenting containerisation decision
- **Pre-built AI container ADR** — `docs/architecture/ADR-004-prebuilt-ai-containers.md` documenting multi-stage Docker build
- **Product requirements** — `docs/prd/prd-l4t-container-deployment.md`, `docs/prd/prd-prebuilt-llm-container.md`
- **Common utilities** — `src/mousedroid/common/math/numpy_ops.py` and `src/mousedroid/common/tools/registry.py` (reusable module extraction)
- **NVMe SSD support** — 500 GB NVMe partition, mount, 16 GB swap, Docker data-root, containerd symlink to SSD
- **4 GB → 16 GB swap** — SSD-backed swap file for memory-intensive builds (replaces zram-only swap)

### Changed

- **Coverage** — 54% → 97.34% (959 tests, 85% gate enforced by `pyproject.toml`)
- **Orchestrator** — cognitive core as primary action source with MCTS fallback; `start()`/`stop()` lifecycle for cognitive core
- **Factory** — `build_orchestrator()` now builds and injects `CognitiveCore` with graceful error handling
- **`bdi_model.py`** — replaced private `_relu`/`_safe_softmax_impl` with shared `numpy_ops` imports
- **`constitutional_rl.py`** — replaced private `_relu`/`_layer_norm` with shared `numpy_ops` imports
- **`tools/registry.py`** — added import from canonical `common.tools.registry` (backward compatible)
- **`tools/__init__.py`** — import from canonical `common.tools.registry`
- **Python compatibility** — ruff target `py311` → `py310`, mypy `python_version` 3.11 → 3.10 (Jetson JetPack 6.x ships Python 3.10)
- **`pyproject.toml`** — added `huggingface-hub` to `[llm]` extras
- **`factory.py`** — explicit `UltrasonicConfig` default values for all fields (mypy strict compliance)
- **`loader.py`** — removed stale `type: ignore[import-untyped]` on yaml import
- **`jetson_csi.py`** — fixed optional import types (`Any` annotation for `_jetson_utils` / `_cv2`)

### Fixed

- **`test_bdi_model.py`** — fixed stale `_relu` import (renamed to `relu` in `numpy_ops`)
- **`common/tools/registry.py`** — added `_mic_diagnostics` handler (9th tool, matching tests)
- **`weights_manager.py`** — fixed mypy `no-redef` via `_hf_hub_download` alias pattern
- **`test_docker_gpu.py`** — `_has_cuda()` moved before `pytestmark` (was undefined F821)
- **`numpy_ops.py`** — removed unused imports (F401), sorted `__all__`
- **`record.py`** — fixed import sort order (I001)
- **`registry.py`** — sorted `__all__` (RUF022)
- **21 lint errors** resolved (20 auto-fixed, 1 manual)
- **7 mypy errors** resolved across 4 source files

### Removed

- Deprecated modules consolidated into `common/` package with backward-compatible shims
