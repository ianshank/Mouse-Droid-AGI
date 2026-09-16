# MouseDroid — Code Quality & Tech-Debt Remediation Plan

> **Date:** 2026-09-16
> **Branch:** `claude/code-quality-tech-debt-plan-36irdi`
> **Baseline:** `dddc16c` (`feat(sim): Isaac Lab workstation harness (F-043–F-046, F-048)`)
> **Status:** Proposal — awaiting review. No code changes in this PR.
> **Scope:** Whole tree. Optimisation, tech-debt reduction, hardening, god-file
> reduction, CI/CD greenness, hardcoded-value elimination, dead/redundant code,
> enterprise organisation, coverage integrity.
> **Method:** Six parallel evidence-gathering audits (config, security, dead code,
> tests, CI, docs) plus direct measurement. Every finding below cites `file:line`.
> Claims that failed independent re-verification were dropped — see §9.

---

## 1. Verdict up front

This is a **well-governed codebase with a narrow band of real debt**, not a
codebase in trouble. The audits confirmed the things that usually go wrong here
are already right:

| Checked | Result |
|---|---|
| Secrets in tree or history | **Clean** — `gitleaks` 0 findings, and still 0 with the allowlist removed |
| Dependency CVEs | **Clean** — `pip-audit` 0 vulnerabilities across 123 resolved packages |
| `SecretStr` coverage on credential fields | **Complete** — no credential field is a plain `str` |
| Pre-egress prompt-injection sanitisation | **No bypass** — every cloud-hitting dispatch path sanitises first |
| Blocking calls inside `async def` (invariant 4) | **Zero** across `src/` (AST scan) |
| `deque(maxlen=…)` schema-sourced (invariant 8) | **13 of 14** — one violation, `hardware/lidar_driver.py:37` |
| Factory-First DI (invariant 1) | **Enforced in CI** by `scripts/check_subsystem_boundaries.py`, whole-tree, no diff carve-out |
| Root `CLAUDE.md` accuracy | **0 drifted claims** — 17-job CI list, Surface Map (14/14 links), Makefile targets all verified |
| `features.yaml` governance | **Clean** — 39/39 `done` entries carry a resolvable `implemented_in` + `validation_command` |
| Test-tier mirror discipline | **41 of 42** packages have a `tests/unit/<pkg>/` mirror |

So this plan is not a rescue. It targets six specific weaknesses:

1. **A production-only correctness hole.** `PYTHONOPTIMIZE=1` is set in the
   shipped Jetson image, and 8 `assert` statements in `src/` — 4 of them in the
   mission-lifecycle / e-stop path — are silently stripped there. The lint rule
   that would catch this is globally disabled.
2. **CI is green but not *reproducibly* green.** 23 matrix legs resolve a fresh
   dependency tree on every run, with no lockfile.
3. **Config has three parallel roots**, and the gate meant to prevent that has a
   blind spot that exempts the codebase's own dominant naming convention.
4. **God-file reduction stopped at the method boundary.** ADR-017 split the
   classes; a 46-parameter, 381-line constructor and a 208-line `tick()` remain.
5. **The delta-coverage gate exists, is tested, and never runs in CI.**
6. **Documentation truth holds at the root and drifts at the edges** — 9 broken
   links, 26 stale path references, and two competing per-directory doc formats.
7. **No trunk and no tags.** 90 remote branches, no `main`, and **zero** tags —
   so `release.yml` (tag-triggered) has never fired, and the 39 feature pins
   have no tag protecting their reachability.

**Governance note:** the three suppression ratchets are all at ceiling —
`# hardcoded-ok` 28/28, `noqa` 19/19, `type: ignore` 8/8
(`.claude/workforce.yaml:100-112`). `tests/regression/test_hardcoded_value_marker_budget.py`
fails any change that adds a marker. **Every wave below must be net-reducing**,
and Wave 0 exists to buy headroom before the work that needs it.

---

## 2. Measured baseline

Recorded so later waves can be judged against it rather than against impressions.

| Dimension | Value | Source |
|---|---|---|
| `src/` Python | 73,759 LOC / 41 packages | `find src -name '*.py' \| wc -l` |
| `tests/` Python | 128,080 LOC / 728 test files across 11 tiers | per-tier `find` |
| Test:source ratio | 1.74:1 | derived |
| Largest `src` file | `sim/isaaclab/rover_env.py` — 804 LOC | `wc -l` sorted |
| Largest class | `RoverIsaacLabEnv` — 678 lines, 24 methods | AST scan |
| Functions at C901 ceiling (15/15) | 4 | `ruff check --select C901 --config mccabe.max-complexity=11` |
| Functions in the 12–15 band | 15 | same |
| `# pragma: no cover` in `src/` | 92 | `grep -c` |
| `noqa` / `type: ignore` in `src/` | 19 / 8 (both at ratchet ceiling) | `grep -c` |
| `TODO`/`FIXME`/`HACK`/`XXX` in `src/` | **1** | `grep -rnE` |
| CI jobs / matrix legs per run | 17 / 23 | `.github/workflows/ci.yml` |
| CI jobs missing `timeout-minutes` | **0** | parsed from ci.yml |
| Advisory (`continue-on-error`) jobs | 6, all tracked | `.github/advisory_stages.yaml` |
| `Protocol` classes defined | 91 across 60 files | `grep -c 'class .*(Protocol)'` |
| Dependency lockfile | **none** | `ls *.lock requirements*.txt constraints*.txt` |
| Remote branches / tags | **90 / 0** | `git ls-remote --heads\|--tags origin` |
| Default branch | `claude/markdown-implementation-plan-aVJ2l` (no `main`) | `git ls-remote --symref origin HEAD` |
| `openspec/changes/` dirs, of which live | 18, **2** | per-dir `tasks.md` checkbox count |

---

## 3. Workstreams

Each workstream states the evidence, the change, and — the part that matters —
**the gate that stops it coming back**. A fix without a gate is a fix with a
half-life.

### WS-1 — Production correctness: `assert` under `PYTHONOPTIMIZE=1`

**Severity: highest. Smallest diff in the plan.**

`Dockerfile.jetson:176` sets `ENV PYTHONOPTIMIZE=1` — the documented Jetson
runtime contract. Under `-O`, every `assert` is removed. `pyproject.toml:264`
globally ignores ruff's `S101`, so nothing flags them:

```toml
ignore = ["D100", "D104", "D107", "D203", "D213", "S101", "ANN401", "TCH", "TC"]
```

With that ignore lifted, `ruff check --isolated --select S101 src/` reports
exactly 8 sites. Four are in the mission lifecycle:

| Site | Form |
|---|---|
| `orchestrator/mission_lifecycle.py:388` (`_handle_stall`) | `assert self._mission is not None` |
| `orchestrator/mission_lifecycle.py:445` (`_transition_to_failed`) | same |
| `orchestrator/mission_lifecycle.py:462` (`_transition`) | same |
| `orchestrator/mission_lifecycle.py:489` (`_record_terminal_duration`) | same |
| `world_model/dual_stream_rssm_onnx.py:225` | post-`warmup()` narrowing |
| `vla/policy.py:334` | post-`warmup()` narrowing |
| `cloud/experience_exporter.py:259` | narrowing in an async closure |
| `cloud/pubsub_sink.py:199` | narrowing in an async closure |

All four lifecycle asserts are the **first statement** in their method, with no
fallback. On the rover the guard is absent and `self._mission.state` /
`self._mission.started_at_s` raises `AttributeError: 'NoneType' object has no
attribute 'state'` mid-transition — inside the path that drives e-stop and
mission-failure telemetry. This is the one finding where CI passing and the
rover behaving are genuinely different things.

The codebase already knows the rule and applies it elsewhere:
`world_model/checkpoint_migration.py:184` carries the comment *"NOT assert:
stripped under PYTHONOPTIMIZE=1"*, and `resilience/retry.py:91` plus
`orchestrator/_background_cadence_mixin.py:145` use explicit raises. This is
drift, not a policy disagreement.

**Change.** Remove `"S101"` from `pyproject.toml:264` (it is already in
`per-file-ignores` for `tests/**` at `:274`, so the global entry only suppresses
`src/`). Convert all 8 to `if x is None: raise RuntimeError(...)`.

**Gate.** `S101` becomes blocking for `src/` via the existing `lint` job. No new
CI surface.

**Effort:** S. **Risk:** very low — 8 mechanical edits, each already
type-narrowing.

---

### WS-2 — Reproducible CI

CI's structure is sound: all 17 jobs carry `timeout-minutes`, pip caching is
present wherever deps are installed, `concurrency` cancels superseded runs, and
`permissions:` is minimal (`contents: read`). The problem is not structure.

**WS-2a — No lockfile (root cause of recurring CI redness).** Core dependencies
are floating lower bounds (`pyproject.toml`), and 23 matrix legs each resolve
independently at run time. The repo's own comments record the resulting damage —
three upper bounds added *reactively* after a transitive release broke CI:

- `numpy>=1.24,!=2.0.0,!=2.0.1,<2.5` — *"numpy 2.5.0 ships PEP 695 `type` stubs
  that mypy rejects"*
- `mcp>=1.0,<2` — *"the old `>=1.0` happily resolved to it (**same class of bug
  as the ruff/coverage pin skews**)"*
- a `types-*`/tool skew the `mcp` comment refers back to

"Same class of bug" naming a third prior incident is the tell: this recurs
because nothing pins the resolution.

**Change.** Add a `uv.lock` (cross-platform universal resolution — correct
choice here over a platform-specific `requirements.txt`, because the matrix
spans Python 3.10–3.12 *and* `windows-latest`). CI installs against it; a
scheduled refresh PR bumps it deliberately. Per Context7's uv documentation,
`uv.lock` is the checked-in, human-readable-but-tool-managed cross-platform
lockfile, and `uv pip compile pyproject.toml --all-extras` covers the extras
surface. Dependabot already runs weekly on `pip` (`.github/dependabot.yml`) but
with `>=` bounds it has almost nothing to act on — the lockfile is what gives it
teeth.

**Gate.** A `uv lock --check` step fails the build when `pyproject.toml` and the
lockfile disagree.

**WS-2b — Promote the delta-coverage gate into CI.** `scripts/check_branch_coverage.py`
(675 LOC, itself covered by a 621-LOC test at
`tests/unit/scripts/test_check_branch_coverage_base_ref.py`) enforces ≥90%
branch coverage **on changed lines**. It runs in `scripts/ci.sh:178` and
`make branch-coverage` — and in **no** GitHub Actions workflow. The only
mention in `ci.yml` is the comment at `:366` explaining it is *"Still local-only
by design (need heavy deps)"*.

That rationale no longer holds. The `test` job already installs
`.[dev,telemetry,mcp]` and already runs exactly the tiers the gate needs
(`tests/unit tests/property tests/integration`, `ci.yml:216-224`). Adding the
gate as a step on the 3.11 leg costs one extra invocation and **no new
installs**.

This is the single highest-value coverage change available: the 90% global floor
resists erosion slowly (one uncovered file barely moves a 22k-statement
denominator), whereas a changed-lines gate refuses new uncovered code outright.

**WS-2c — Advisory ladder: two promotions come due this week.** Computed from
`.github/advisory_stages.yaml` against 2026-09-16. `scripts/check_advisory_promotions.py`
currently reports *"all stages tracked and within window"* — accurate today, and
about to stop being:

| Job | `since` | Window | **Due** |
|---|---|---|---|
| `test-windows` | 2026-08-20 | 30 d | **2026-09-19 — 3 days** |
| `security` (pip-audit) | 2026-07-25 | 60 d | **2026-09-23 — 7 days** |
| `vulture-audit` | 2026-07-03 | 90 d | 2026-10-01 |
| `performance` | 2026-07-25 | 90 d | 2026-10-23 |
| `onnx-world-model-extras` | 2026-05-16 | 180 d | 2026-11-12 |
| `mlflow-extras` | 2026-08-27 | 180 d | 2027-02-23 |

`security` is the easy one — `pip-audit` reports zero vulnerabilities today, so
it can be promoted to blocking now. `test-windows` needs a green-streak check
against the Actions API before flipping. `NEXT_STEPS.md:59-60` already carries
this as an open item; this plan supplies the dates.

**WS-2d — Consolidate the three extras jobs.** `vla-extras`,
`onnx-world-model-extras`, and `mlflow-extras` are three separate jobs, each
`ubuntu-latest` / Python 3.11 / `timeout-minutes: 30`, differing only in which
extra they install. Collapse to one job matrixed over
`[vla, onnx_world_model, mlflow]`: same coverage, one job definition instead of
three, and their advisory states become per-matrix-entry rather than per-job.

**Effort:** M (2a), S (2b, 2c, 2d). **Risk:** 2a is the only one with real
surface — a lockfile can surface a pin conflict the floating resolution was
papering over. That is the gate working, not the gate failing, but budget for it.

---

### WS-3 — Config single-source

`grep` finds ~750 non-trivial numeric literals outside `config/schema/`. About
**60% are legitimate** — HTTP status codes, RFC 6455 close codes, the mel-scale
formula, numerical epsilons, SHA-256's 64 hex chars, torch axis arguments. A
count-driven cleanup here would be mostly wasted motion. Triaged, the real
inventory is **28 findings across 62 call sites**.

**WS-3a — Fix the gate before the findings.** `scripts/check_no_hardcoded_values.py:280`:

```python
return "DEFAULT_" in stripped
```

Any line containing the substring `DEFAULT_` is exempt. Since `DEFAULT_<THING>`
is this codebase's dominant convention for naming a default, **the idiomatic way
to write a hardcoded value is also the way to make it invisible to the gate.**
Compounding blind spots:

| # | Blind spot | Evidence |
|---|---|---|
| B1 | `DEFAULT_` substring exempts the whole line | `:280` |
| B2 | String literals never checked — no paths, URLs, IPs, ports | `_is_numeric_constant` `:283-284` |
| B3 | `src/mousedroid/**` only — `scripts/`, `tools/`, `training/`, `cloud/` unscanned | `TARGET_PREFIX` `:27` |
| B4 | Diff-scoped: only added/modified lines | `:351-354` |
| B5 | 6 permanent directory exemptions, including all of `src/mousedroid/factory/` (5,964 LOC — the entire DI surface) | `ALLOWED_DIR_PREFIXES` `:59-66` |
| B6 | No-op on a clean tree / non-PR push | `:454-456` |
| B7 | `getattr(cfg, "field", default)` schema bypass undetectable | whole file |

Remediating findings under a gate that cannot hold the line guarantees
regression. Fix B1, B2, B3 and B7 first; keep B4/B5 as declared, expiring
exemptions rather than permanent ones.

**WS-3b — Three parallel config roots.** Invariant 2 says schemas in
`config/schema/` are the single source. Three competing roots exist:

1. **`src/mousedroid/constants.py`** — 326 lines, one of only two `ALLOWED_FILES`
   in the gate (`:29`). Its own header says its values *"mirror the defaults in
   `ModelConfig` and related config classes"*, but
   `tests/regression/test_constants_schema_parity_aqa.py:44-54` pins only **9**
   pairs. Unpinned mirrors and orphans include `DEFAULT_CAMERA_WIDTH=640`,
   `DEFAULT_LIDAR_MIN_RANGE_M=0.15`, `DEFAULT_LIDAR_SAMPLES_PER_REV=360` (no
   schema field at all), `DEFAULT_LIDAR_BUFFER_SIZE=100` (no field),
   `DEFAULT_MOTOR_MAX_ANGULAR_VELOCITY=2.0` (no field, and *conflicts* with the
   schema — see below).
2. **`src/mousedroid/llm_gateway/config.py:10`** — `class GatewayConfig(BaseModel)`,
   **not** `StrictBaseModel`, so no `extra="forbid"`. 21 fields duplicating
   `LLMConfig`, including a hardcoded model path (`:15`) and download URL
   (`:19`). `factory/llm_gateway.py:107-128` hand-copies all 21, and no test
   asserts the two field sets agree.
3. **`src/mousedroid/cognitive/constitutional_rl.py:42,45`** —
   `_MCTS_MIN_SIMS_DEFAULT = 16` and `_OBSTACLE_CLEARANCE_DEFAULT = 0.25` are
   unreachable from YAML; `factory/cognitive.py:124-127` wires only two of four
   fields. The clearance floor also disagrees with
   `SafetyConfig.min_forward_clearance_m = 0.2` (`config/schema/reward_safety.py:143`).

**WS-3c — Values with no schema field (5).** `samples_per_rev`, lidar
`buffer_size` (also the sole invariant-8 violation, `hardware/lidar_driver.py:37`),
motor `max_angular_velocity`, constitutional `obstacle_clearance_floor_m`,
constitutional `mcts_min_sims`.

**WS-3d — Divergent safety ceiling (verified, severity corrected).** Three
sources disagree on angular velocity: `constants.py:276` = **2.0**,
`MotorLimitsConfig.max_angular_velocity` (`config/schema/hardware.py:666`) =
**1.5**, `llm_gateway/config.py:40` `max_omega_norm_rads` = **2.0**.
`MockMotorController.__init__` (`hardware/motor_controller.py:112`) defaults to
the 2.0 constant, and `factory/autonomous.py:64` builds it as
`MockMotorController(metrics=metrics)` — passing no limits — so the mock clamps
33% looser than the configured safety limit.

*Severity qualifier, established by tracing every construction site:*
`MockMotorController` is instantiated **only** at `factory/autonomous.py:64`,
inside the parked ADR-016 `AutonomousOrchestrator` path. Production motors go
through `factory/hardware.py::build_esp32_driver` → `MockESP32Driver`. So this
is a real divergence on the parked path, **not** a live defect in the 30 Hz
production loop. It still wants fixing — a constant that contradicts the schema
safety limit and has no schema field is wrong regardless of which path reads it.

**WS-3e — `getattr` schema bypass (9 sites).** Each silently substitutes a
literal instead of raising on a renamed field. Worst cases:

- `hardware/lidar_driver.py:38-47` — doubly nested `getattr`, where the inner
  fallback names `min_distance_m`/`max_distance_m`, which **do not exist** on
  `LidarConfig`. Dead code masking a legacy name.
- `training/replay/mixer.py:99-103` — five `getattr` reads off a
  `ReplayMixerConfig`; a rename of `alpha_target` silently yields `0.0`, which
  **disables sim/real mixing entirely** — a silent training regression.
- `arm/planning/symbolic_planner.py:333` — reads a field that genuinely exists
  (`config/schema/arm.py:156`). *Blocked by the F-008 `freeze_gate.py`.*

**WS-3f — Loop target hardwired to 33 ms.** `cognitive/metacognitive.py:36`
uses `DEFAULT_TARGET_LOOP_MS = 33.0` directly in the capability score at `:182`.
Set `loop.control_hz = 10.0` and the model permanently reports degraded timing.
The same tunable is triple-encoded — `config/schema/misc.py:177` `control_hz=30.0`,
`constants.py:288` `0.033`, `constants.py:98` `33.0` — and the two rounded forms
already disagree with 30 Hz by ~1%. The fix pattern exists:
`config/schema/root.py:532` `derive_max_loop_time_from_control_hz`.

**WS-3g — Three env reads bypassing the schema-named-env-var pattern.**
`config/schema/misc.py:105-118` states the rule explicitly. 52 of 55 `src/` env
reads follow it. The three outliers: `health/watchdog.py:93` and
`factory/health.py:68` (both bare `os.environ.get("NOTIFY_SOCKET")`), and
`training/observability/mlflow_logger.py:44`, which does
`os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")` **at module import
time** — a process-env mutation as an import side effect.

**Gate.** Extend `test_constants_schema_parity_aqa.py::_MIRRORED_PAIRS` to every
mirroring constant; add a `GatewayConfig`↔`LLMConfig` field-set parity test; fix
the four gate blind spots.

**Explicitly de-prioritised:** `telemetry/mock_source.py` (44 literals) and
`hardware/camera/mock_camera.py` (~30) are synthetic-waveform and rendering
maths. High churn, no behavioural value. Leave them.

**Effort:** L. **Risk:** low per change, but touches safety-adjacent defaults —
each needs a paired AQA + backwards-compat regression test
(`regression-pair-scaffold` skill).

---

### WS-4 — God-file reduction, round two

ADR-017 decomposed `factory.py` → `factory/` and split the orchestrator into
7 mixins over a shared `_OrchestratorState` (diamond MRO, deliberately
designed). That work was real and holds up: an AST comparison across
`orchestrator.py`, `_state.py` and all 7 `_*_mixin.py` found **0 MRO-shadowed
duplicate methods** and **0 `_state.py` stubs with no implementor**.

But it split *methods*. What remains in `src/mousedroid/orchestrator/orchestrator.py`:

```
class MouseDroidOrchestrator(_LifecycleMixin, _MissionMixin, _WorldModelStateMixin,
                             _ActionMixin, _TelemetryExperienceMixin,
                             _VoiceFaceMixin, _BackgroundCadenceMixin)
    __init__   381 lines,  46 parameters  (22 positional + 24 keyword-only)
    tick       208 lines   ← the 30 Hz hot loop
```

The constructor's own comments show it accreting one block per feature:
`# Tier C1 / C1.2 — OTA weight-update wiring` (`:374`), `# Tier C2 / C2.1 —
soft-constraint safety projector` (`:408`), `# Phase 6 WS3 — replay-triggered
on-device update coordinator` (`:420`), `# Growth pillar` (`:429`),
`# Issue #109 — one-shot startup greeting` (`:434`), `# F-023`, `# F-025`. Every
new subsystem costs one more parameter. This is also *why*
`factory/orchestrator.py::build_orchestrator` sits at C901 **15/15** — it has to
wire 46 arguments.

`tick()` at 208 lines is not flagged by C901 (its branching is low; it is long
and linear), but it is the most timing- and safety-critical function in the
project sitting in one method.

**The pattern recurs, which is the real finding.** The newest large class in the
tree is `sim/isaaclab/rover_env.py::RoverIsaacLabEnv` — **678 lines, 24
methods**, in the repo's largest file (804 LOC). Hand-decomposition is a
one-time event; without a mechanical ceiling it regresses.

**Change.**
1. Group the 24 keyword-only constructor parameters into cohesive frozen
   wiring bundles (`HarnessWiring`, `CloudWiring`, `LearningWiring`,
   `SafetyWiring`, `VoiceWiring`), each field `Field(default=None, description=…)`.
   Keep the existing kwargs accepted via a deprecation shim for one release so
   `Settings`/YAML and external callers are unaffected (invariant 6).
   `NEXT_STEPS.md:72-74` already gestures at this ("property-test orchestrator
   kwargs vs `_OrchestratorState`").
2. Extract `tick()`'s phases into named private methods on the existing mixins,
   preserving ordering and the `_mark_phase` instrumentation (the comment at
   `:461` notes it runs 8× per tick — keep the resolved-once pattern).
3. Split `RoverIsaacLabEnv` along its own seams: scene wiring
   (`_wire_isaaclab_scene`, `_apply_reset_randomization`), actuation
   (`_apply_wheel_targets`, `_fan_out_action`, `_substep_physics`), and
   observation (`_read_observation`, `_zero_observation`, `_measured_body_velocity`).

**Gate.** Two new ratchets in `.claude/workforce.yaml`, seeded at current values
so nothing regresses and nothing must be fixed at once:
- max constructor parameter count per class
- max class body lines / method count

Plus tighten C901 from 15 → **12**. 4 functions are at exactly 15/15 (zero
headroom) and 15 sit in the 12–15 band, so this is a real ratchet with a
bounded, listed backlog:

```
15  validation/runtime/_hailo.py:40        verify_hailo_accelerator
15  telemetry/server/_rest_handlers.py:193 _handle_mission_post
15  harness/spec.py:112                    check_dag
15  factory/orchestrator.py:103            build_orchestrator
14  validation/runtime/_storage.py:39      verify_pcie_ssd_layout
14  telemetry/server/_ws_handlers.py:436   _lidar_raw_broadcast_loop
14  safety/monitor.py:128                  evaluate            ← safety path
14  hardware/lidar/ld19_driver.py:162      _read_frames_with_stats_blocking
13  telemetry/server/_lifecycle.py:193     start
13  sensing/manager.py:162                 recovery_attempt
13  orchestrator/_lifecycle_mixin.py:22    start
13  mcp/tool_bridge.py:240                 call_tool
13  llm_gateway/fallback_gateway.py:228    _route_with_failover
12  learning/on_device/rssm_refiner.py:59  build_sequence_batch
12  config/loader.py:58                    load_settings
```

Note `hardware/lidar/ld19_driver.py` is *also* one of the six files omitted from
coverage measurement (§WS-7) — complex, unmeasured, hardware-facing code is the
worst combination in the list and should be sequenced first.

**Not debt — explicitly leave alone.** `learning/offline_rl.py` (755 LOC) is six
cohesive classes (`QNetwork`, `DeterministicPolicy`, `OfflineRLTrainer`,
`CQLTrainer`, `ValueNetwork`, `IQLTrainer`), largest 243 lines. `config/schema/hardware.py`
(700 LOC) is 12 domain config classes. Both are large *because the domain is
large*; splitting them is churn. LOC alone is not the signal — methods-per-class
and constructor arity are.

**Effort:** L. **Risk:** M. The constructor change touches every orchestrator
construction site; the deprecation shim plus the existing property tests and
`tests/regression/test_orchestrator_mixin_surface.py` contain it. Use the
`module-split-consistency-sweep` skill before merging — it exists for exactly
this class of change.

---

### WS-5 — Interface coherence

91 `Protocol` classes across 60 files, under **three** competing module naming
conventions — `protocol.py` (singular), `protocols.py` (plural), `_protocol.py`
(private) — plus 41 Protocols defined inline in implementation modules.

The decentralised `<package>/protocol.py` pattern is defensible (co-locate the
interface with its domain). The inconsistency is not, and it has produced two
verified duplications:

**WS-5a — `PromptInjectionFilterProtocol` is defined twice, with divergent
signatures.**

| | |
|---|---|
| `interfaces/protocols.py:129` | `def sanitize(self, command: str) -> str` |
| `security/injection_filter.py:35` | `def sanitize(self, text: str) -> str` |

Both are `@runtime_checkable`. Under PEP 544, positional-or-keyword parameter
*names* participate in Protocol compatibility, so these are not
interchangeable for keyword calls — a latent `mypy --strict` divergence, not
just cosmetic duplication.

Both are live, and the split runs straight through the DI seam:
`factory/llm_gateway.py:9,21` imports and returns the **`security`** one as
`build_injection_filter`'s return type, while `factory/__init__.py:255,308`
re-exports the **`interfaces`** one. The same package exports two different
names for the same builder's return type.

**WS-5b — `LiDARProtocol` vs `LidarProtocol`.** `interfaces/protocols.py:70`
and `hardware/protocols.py:160` — case-differing near-duplicates of the same
concept.

**WS-5c — `interfaces/` is under-adopted.** It holds 7 cross-cutting Protocols
(`MotorControllerProtocol`, `CameraProtocol`, `LiDARProtocol`,
`LLMGatewayProtocol`, `MetricsRegistryProtocol`,
`PromptInjectionFilterProtocol`, plus `GoalVector`) and has **15** import sites
across `src/` + `tests/` — and is the one package of 42 with **no**
`tests/unit/` mirror.

**Change.** Rule: cross-subsystem Protocols live in `interfaces/protocols.py`;
subsystem-internal Protocols live in `<package>/protocols.py` (settle on plural);
none are declared in implementation modules. De-duplicate 5a and 5b to one
definition each, keeping a re-export alias for one release. Add
`tests/unit/interfaces/`.

**Gate.** A regression test asserting no Protocol *name* is defined twice
across `src/mousedroid/**` — cheap, AST-based, and it pins the whole class of
bug rather than these two instances.

**Do not add `import-linter`.** Initial reading suggested the Factory-First DI
invariant lacked mechanical enforcement. It does not:
`scripts/check_subsystem_boundaries.py` runs whole-tree with no diff carve-out,
wired at `.github/workflows/ci.yml:397` and `scripts/ci.sh:98`, and an
independent AST sweep found **zero** violations beyond its 6 pre-audited
allowlist entries. The right move is *tightening* the existing checker — its
`_SHARED_KERNEL_PREFIXES` (`:83-94`) whitelists `mousedroid.common.*` and
`mousedroid.utils.*`, which makes those grab-bag packages a structural exemption
from the invariant (see WS-8c).

**Effort:** M. **Risk:** low.

---

### WS-6 — Dead and redundant code

`scripts/dead_code_audit.py` (the advisory `vulture-audit` job) reports **406
findings** at 60% confidence. `NEXT_STEPS.md:104-111` correctly flags this as
needing a dedicated triage pass rather than a rubber-stamp allowlist. Full
triage of all 406:

| Bucket | Count |
|---|---|
| Pydantic `Field()` / validators / `ConfigDict` in `config/schema/**` | 165 |
| Protocol members / mocks / helpers referenced from `tests/` or `training/` but not `src/` | 142 |
| Instance attributes read within the same class (vulture cannot see `self.x` reads) | 46 |
| Enum member reached by YAML value; intentional `if False: yield` idiom | 2 |
| Write-only attributes read only by tests | 40 |
| **Genuinely dead** | **11** |

So the signal-to-noise is ~2.7%, which is why the job is advisory by design.
The decision at the 2026-10-01 promotion window is "keep advisory + record an
ADR" vs "strict mode on a curated allowlist" — and the triage below is the
input that decision needs.

**WS-6a — Verified latent bug: dangling `__all__`.**
`src/mousedroid/harness/approval/__init__.py` is 8 lines: it declares
`__all__ = ["OpenClawSafetyGate", "SandboxPolicyGate"]` and imports nothing.
Confirmed by execution:

```
$ PYTHONPATH=src python3 -c "from mousedroid.harness.approval import *"
AttributeError: module 'mousedroid.harness.approval' has no attribute 'OpenClawSafetyGate'
```

This is incident 6 in `.claude/skills/module-split-consistency-sweep/SKILL.md`,
recurring in a package the ADR-017 facade fix did not cover. A static check over
all 40 `src/mousedroid/**/__init__.py` found this as the **only** dangling case.
Every real consumer imports the submodule directly, so the file is a pure
no-op — delete it or bind the names.

**WS-6b — Three wired-but-inert features.** Each is a *decision* (wire it or
remove it), not a deletion:

- **`self._skill_delegator`** — declared `orchestrator/_state.py:142`, built
  `factory/orchestrator.py:326`, injected `:467`, stored
  `orchestrator/orchestrator.py:359`, and **read by no mixin and no test**.
  Verified: every other repo reference is to the *builder*
  (`build_skill_delegator`), never to the stored attribute. A complete dead DI
  path — skill delegation is constructed and inert.
- **`self._dla_enabled`** — `efficiency/tensorrt.py:133` reads
  `cfg.dla_enabled` into it once, and the attribute is never read again. That
  makes schema field `hardware.py:549 dla_enabled` a **no-op** — despite being
  explicitly set in four config files (`config/default.yaml:110`,
  `jetson_production.yaml:45`, `jetson_dual_stream.yaml:23`,
  `jetson_sdcard_64gb.yaml:21`). Operators are configuring something inert.
- **`self._bc_batch_size`** — `learning/offline_rl.py:181` is its only
  reference. The `bc_batch_size` kwarg is plumbed from
  `training/train_offline_rl.py:90,104` and schema-validated, and affects only a
  log line.

Plus 8 smaller write-only attributes (`training/gpu_monitor.py:48`,
`training/observability/mlflow_logger.py:84-85`, `mcp/server.py:112,146`,
`sim/isaaclab/rover_env.py:173,362`, `orchestrator/autonomous.py:47,99,109`,
`orchestrator/mission_lifecycle.py:135`, `constants.py:50`). Three more sit
under the F-008 freeze (`arm/control/primitives.py:43`,
`arm/environments/base.py:52`, `arm/perception/pose_estimator.py:43`) — record
now, remediate when the freeze lifts.

One **behavioural** item, not removable: `safety/three_laws.py:68`
`LawViolation.action_override` is populated with real corrective actions at
`:246` and `:308` and read by nothing. Not a safety bug — enforcement happens
via the in-place `safe[:] = 0.0` at `:248` — but a redundant second channel that
invites a future caller to trust it. Needs a documented decision.

**WS-6c — 11 duplication clusters, ~430 LOC.** Two close invariant violations
outright:

| Cluster | Sites | Shared home |
|---|---|---|
| **sysfs GPU temp/load readers — 3 copies** | `health/monitor.py:37-67,93-102`; `efficiency/profiler.py:35-71`; `training/gpu_monitor.py:50-66,102-104` | new `common/sysfs.py` |
| **Exponential backoff — 4 copies** | `resilience/retry.py:39-117` (canonical); `utils/weights_manager.py:206-238`; `cloud/weight_update_poller.py:362-410`; `hardware/audio/usb_speaker.py:110-136` | `resilience/retry.py` |
| Journal `append` — byte-identical bodies | `harness/journal/jsonl_journal.py:102-124` vs `lmdb_journal.py:105-126` | base class |
| `Settings` built bypassing `load_settings` | 5 sites in `training/`, `scripts/` | `config.loader.load_settings` |
| Static HTML handlers — 3 copies | `telemetry/server/_rest_handlers.py:431-516` | local helper |
| Content-addressed slot `persist` | `growth/slot_store.py:84-109` vs `learning/on_device/slot_store.py:117-147` | `common/hashing.py` |
| Circuit-breaker + retry boilerplate | 6 sites in `resilience/` | private `_guarded_call` |
| `mock_hardware` short-circuit ×6 | `validation/preflight.py:210-379` | decorator |
| `".pt"` slot suffix ×3 | `growth/`, `learning/on_device/`, `factory/` | `constants.py` |

The sysfs cluster closes a **CLAUDE.md Red Flag**: `efficiency/profiler.py:70`
uses bare `open(path)` and `training/gpu_monitor.py:104` uses `read_text()` —
both **without `encoding="utf-8", errors="replace"`**. Only
`health/monitor.py:101` gets it right. The retry cluster closes two hardcoded
values (`weight_update_poller.py:391` `backoff_s = 2.0**attempt`;
`weights_manager.py:176-177,373-374` `max_retries=3, backoff_base=2.0`).

**Negative results worth recording** (so nobody re-audits them): ring-buffer
handling has no extractable duplication — all `deque` sites are schema-sourced
except the one WS-3c covers. Metric label validation is already centralised
(`telemetry/metrics/primitives.py:412`, 13 call sites). Factory `mock_hardware`
branches are the DI pattern, not duplication.

**WS-6d — Dependency hygiene.** Declared-but-unimported: `transformers>=4.40`
(`[vla]`) — and `tests/unit/vla/test_distilled_onnx.py:243` **asserts
`'transformers' not in sys.modules`**, so a declared dependency is actively
contradicted by a pinned test; `anyio>=4.0` (`[mcp]`); `pycuda>=2022.1`
(`[jetson]`); and two **entire unused extras**, `[gcp-training]`
(`pyproject.toml:167-170`) and `[gcp-simulation]` (`:171-174`).

Imported-but-undeclared (all lazily guarded, so nothing crashes — but no
documented `pip install` reaches those paths): `cv2`, `jetson_utils`
(`hardware/camera/jetson_csi.py:30`), `sdnotify` (`health/watchdog.py:58-61`),
`starlette`/`uvicorn` (`mcp/transport.py:215-217`, not in `[mcp]`), `torch2trt`
(`efficiency/tensorrt.py:32`, not in `[jetson]`), `zstandard`
(`cloud/experience_exporter.py:217`, not in `[gcp]`).

**WS-6e — 2 orphaned scripts, 358 LOC.** `scripts/nemoclaw_audit_spike.py`
(90 LOC, zero references; its own usage docstring at `:5` names a *different*
file — residue of a bad rename) and `scripts/benchmark_latency.py` (268 LOC,
absent from `Makefile`, `scripts/ci.sh`, `.github/`, `docs/`, `features.yaml`;
its sibling `benchmark_voice_latency.py` *is* covered by a test). **Zero**
orphaned modules in `src/mousedroid/`.

**WS-6f — Facade tidy-up.** `tests/unit/factory/test_facade_completeness.py:45-60`
pins "the 12 private symbols documented in ADR-017"; the facade actually
re-exports **26**. The 14 extras have no importer outside `factory/` (~42 LOC).
Removing them makes the doc and test claim of "12" true again.

**Policy call, not cleanup:** 19 package `__init__.py` facades (287 LOC) have
zero facade-level consumers — every consumer uses the submodule path. They force
eager imports (`mousedroid.efficiency` pulls `tensorrt`; `mousedroid.growth`
pulls torch student modules), which matters for Jetson cold start. Decide as
policy; do not bulk-delete.

**Gate.** Fold the 7 verified false-positive *classes* into
`scripts/vulture_allowlist.py` — its "Pydantic `@field_validator`" section
(`:47-51`) is currently **empty**, which is why the single zero-reference
finding in the whole run (`skills/builtin/voice.py:40 _exactly_one`, a
`@model_validator`) is a false positive. Extend
`test_facade_completeness.py` to assert the private re-export count.

**Removable: ~850 LOC high-confidence.** **Effort:** M. **Risk:** low for
deletions, M for WS-6b (each needs a wire-or-remove decision).

---

### WS-7 — Coverage integrity

Configuration is better than most: `branch = true` (`pyproject.toml:389`),
`fail_under = 90` (`:405`), and the comment records 92% across 22,524 statements
and 4,696 branches when branch coverage was switched on.

Two structural caveats to close:

**WS-7a — Six hardware drivers are omitted from measurement entirely**
(`pyproject.toml` `[tool.coverage.run] omit`): `hardware/sensors/ultrasonic.py`,
`hardware/camera/imx500.py`, `hardware/lidar/ld19_driver.py`,
`arm/hardware/so_arm100_driver.py`, `hardware/display/ssd1306_face_driver.py`,
plus `main.py`. These are not measured *and* not reported, so the 90% number
says nothing about them. `ld19_driver.py` additionally carries a complexity-14
function (WS-4). Proposal: keep them out of the *gate* but report them, so the
blind spot is visible rather than silent.

**WS-7b — 92 `# pragma: no cover` in `src/`, ungoverned.** Unlike `noqa` and
`type: ignore`, `pragma: no cover` has **no ratchet budget** in
`.claude/workforce.yaml:100-112`. It is the one coverage-suppression marker with
no ceiling and no justification requirement. Add it as a third ratchet, seeded
at 92.

**WS-7c — `exclude_lines` vs `exclude_also`.** `pyproject.toml:407` uses
`exclude_lines`, which **replaces** coverage.py's built-in defaults rather than
adding to them. The current list re-states `pragma: no cover` by hand, so
nothing is broken today — but future upstream default additions will be silently
dropped. Per Context7's coverage.py documentation, `exclude_also` is the
additive form and is the correct choice here.

**WS-7d — Tier distribution.** Two tiers are vestigial: `tests/functional`
(1 file, 54 LOC) and `tests/user_journey` (1 file, 65 LOC). `NEXT_STEPS.md:75-78`
confirms both cover the parked `AutonomousOrchestrator`, not production. Either
fold them into `tests/e2e/` or document them as permanently parked tiers rather
than leaving two near-empty directories implying coverage that is not there.
Regression-pair discipline is partial: 39 `_aqa.py` and 39 `_backwards_compat.py`
files, but 15 `_aqa.py` files have no same-named partner — some legitimately
(e.g. `test_doc_reconciliation`), so this wants a documented rule, not a blanket
pairing mandate.

**Largest test files as split candidates:** `tests/unit/validation/test_validation_runtime.py`
(1,379 LOC), `tests/unit/telemetry/test_telemetry_metrics.py` (1,192),
`tests/unit/telemetry/test_telemetry_server.py` (893).

**Gate.** WS-2b's changed-lines gate is the load-bearing one. Add the
`pragma: no cover` ratchet; switch to `exclude_also`; report-but-don't-gate the
omitted drivers.

**Effort:** M. **Risk:** low.

---

### WS-8 — Enterprise organisation

**WS-8a — Root directory.** 13 root `.md` files; 8 belong elsewhere. Highest
value first:

| File | Finding | Action |
|---|---|---|
| `SMOKE_REPORT.md` | Content dated **2026-05-17**, last touched 2026-07-23, pinned to dead branch `claude/smoke-test-stability-pass`. `:144` still audits `src/mousedroid/factory.py` and `config/schema.py` **in the present tense** — both became packages under ADR-017. | **Archive** |
| `agent.md` | 25-line persona stub whose "Key Invariants" (`:14-25`) restate `CLAUDE.md` and `AGENTS.md:10-45`. Already flagged as drift in `docs/superpowers/plans/2026-06-13-…:30` and never resolved. | **Delete/merge** |
| `progress.md` | 76.7 KB — **3.8×** the repo's own 20 KB doc budget; holds 14 sessions against `HARNESS_SPEC.md:265`'s "~10" rule while `progress-archive/` holds 1. | **Rotate** to `progress-archive/2026-Q3.md` |
| `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` | Accurate; `CODE_OF_CONDUCT.md` has **0 inbound references** | → `.github/` |
| `SKILLS.md` (38.8 KB), `HARNESS_SPEC.md` | Accurate; `SKILLS.md` is 1.9× budget | → `docs/` |

*Not a bug:* `agent.md` vs `AGENTS.md` is **not** a case collision —
`git ls-files | tr A-Z a-z | sort | uniq -d` is empty. It is redundancy, not a
filesystem hazard. (Invariants do, however, now live in four places:
`docs/CHARTER.md` §4, `CLAUDE.md`, `AGENTS.md:10-45`, `agent.md:14-25`.)

**Before moving anything:** `NEXT_STEPS.md` is hard-coded in three gates
(`.github/workflows/ci.yml:400`, `scripts/ci.sh:65`,
`scripts/validations/F-038.sh:21`) and pinned by
`tests/regression/test_f038_aqa.py:62-63`; `features.yaml` is named by
`.claude/workforce.yaml` (`freeze.features_file`). Move paths and gates in the
same commit.

**WS-8b — The cheapest lever in this plan.** `tools/doc_hygiene.py` enforces the
20 KB / 10-done-mark budget but is invoked on **one** file (`NEXT_STEPS.md`).
Run over the other root docs it already reports:

```
WARN: progress.md: 76725 bytes exceeds the 20000-byte budget
WARN: SMOKE_REPORT.md: 33974 bytes exceeds the 20000-byte budget
WARN: SMOKE_REPORT.md: 27 '✅' marks exceed the 10-mark budget
WARN: SKILLS.md: 38796 bytes exceeds the 20000-byte budget
WARN: AGENTS.md: 23118 bytes exceeds the 20000-byte budget
WARN: README.md: 20288 bytes exceeds the 20000-byte budget
```

Widening the default file list closes 5 findings with one config change.

**WS-8c — Grab-bag packages are the DI invariant's escape hatch.**
`scripts/check_subsystem_boundaries.py:83-94` whitelists `mousedroid.common.*`
and `mousedroid.utils.*` as `_SHARED_KERNEL_PREFIXES` — so anything placed there
is **structurally exempt from Factory-First DI enforcement**. Current contents:

- `src/mousedroid/utils/` — 413 LOC, **one module**, `weights_manager.py` (398
  lines of OTA weight management). A package named `utils` holding one specific
  domain concern. Fold into `cloud/` or `learning/`.
- `src/mousedroid/common/` — 1,789 LOC; `common/math/` and `common/time/`
  **shadow stdlib module names**.
- `src/mousedroid/tools/` — 2 files, one of which is `__init__.py`. Note there
  are **four** things named "tools": root `tools/`, `src/mousedroid/tools/`,
  `src/mousedroid/common/tools/`, and the `mcp/` tool bridge.
- `src/mousedroid/config/schema/misc.py` — a literal `misc` module inside the
  schema package.

Shrinking these and narrowing `_SHARED_KERNEL_PREFIXES` *is* a strengthening of
invariant 1, not cosmetics.

**WS-8d — Two parallel per-directory doc systems.** 22 in-package agent docs in
two incompatible formats (confirmed against `pyproject.toml:242-244`, whose
wheel-exclusion comment says "the wheel ships 22 CLAUDE.md / agent.md files"):
5 directories have **both**; 3 have `CLAUDE.md` only; **9** have `agent.md`
only. The `agent.md` stubs carry far less signal — compare
`hardware/agent.md` (13 lines) with `hardware/CLAUDE.md` (26 lines, 6 numbered
invariants). Root `CLAUDE.md`'s Surface Map indexes only the 8 `CLAUDE.md`
files, so **9 subsystems carry per-directory guidance invisible from the root
surface**. Pick one format; index all subsystems.

Also undocumented: `.agents/skills/approval-gate-wiring/SKILL.md` is a top-level
directory parallel to `.claude/skills/`, live in
`tools/validate_skill_commands.py:280`, called "dead" in
`docs/superpowers/plans/2026-07-03-…:401`, and mentioned in no normative doc.

**WS-8e — Doc link rot: 9 broken links, 26 stale path refs.** Notable:

- `docs/architecture/ADR-009-isaac-lab-phase-b.md:131,134,136` → `config/schema.py`,
  split into a package by ADR-017.
- `docs/planning/VALIDATION_REPORT.md:201,242` → an **absolute Windows OneDrive
  path** leaking the pre-rename project slug `Gronk-Droid-Jetson-Nano`.
  Non-portable, and an information leak.
- `SKILLS.md:640` → `` `.github/pull_request_template.md` `` — **case
  mismatch**; the real file is `PULL_REQUEST_TEMPLATE.md`. Breaks on
  case-sensitive filesystems, i.e. Linux CI.
- `docs/architecture/c4-overview.md:150` → `hardware/accelerators/hailo_runtime.py`;
  the real package is `accelerator/` (singular). This is in the "Code-level entry
  points" table — an interface-boundary claim that does not resolve.
- A `tests/unit/` flat→nested reorg left ~122 stale test paths across C4
  diagrams, `AGENTS.md:225`, `SKILLS.md:114,177,179`, and
  `docs/claude/surfaces/usbc-smoke.md:17`.
- Two factual errors in nested surfaces: `telemetry/CLAUDE.md:29` →
  `tests/e2e/test_telemetry_server.py` (wrong tier *and* path; the real file is
  `tests/unit/telemetry/`), and `arm/CLAUDE.md:19` → `mock_arm.py` (real:
  `arm/hardware/mock_arm_driver.py`) — the latter self-declared at
  `NEXT_STEPS.md:79-80` and blocked behind the F-008 freeze.

**WS-8f — Governance artefact accumulation.**
`openspec/changes/` holds 18 directories; **16 have every task checked off**
(241 completed tasks) and only 2 are live. There is no
`openspec/changes/archive/` and no `openspec/specs/`, so the directory reads as
18 items of active work. Add an archive and move the 16.

ADRs: `ADR-017-god-files-decomposition.md` exists on disk but is **absent from
`docs/architecture/adr-log.md`** (the table ends at 016); `docs/README.md:19`
says the log covers "ADR-004…015", two versions stale; and the
`**Status:**` line mandated by `docs/architecture/adr/TEMPLATE.md:3` appears in
only 2 of 15 ADRs.

Evidence traceability: `NEXT_STEPS.md:108` cites
`reports/dead_code/2026-08-16.json` as the evidence for the 447-finding claim,
but `.gitignore:324` excludes `reports/dead_code/` entirely — *"dated JSON
artifacts, never committed"*. The cited evidence can never resolve for a reader.
Either commit a summary or cite the command instead of the artefact.

**Gate.** A link/path-resolution gate over tracked `.md`. The policy already
exists for `SKILL.md` in `tools/validate_skill_commands.py` — generalise it.
That closes 9 broken links + 26 stale refs and stops the next `tests/` reorg
silently re-breaking them.

**Effort:** M (mostly mechanical). **Risk:** low, provided path moves and gate
updates land together.

---

### WS-9 — Release and container hardening

**WS-9a — The release path has never run, and the version records it.**
`pyproject.toml:7` says `version = "0.3.0"`; `CHANGELOG.md:3208` records
`## [v0.4.0] — 2026-05-16`. `src/mousedroid/__init__.py` derives `__version__`
from installed distribution metadata, so the runtime — and any telemetry or
build label carrying it — reports `0.3.0` for a post-0.4.0 tree.

The cause is mechanical, not an oversight in judgement:
`.github/workflows/release.yml:16-18` triggers **only** on
`push: tags: v*.*.*`, and `git ls-remote --tags origin` returns **0 tags**. The
release workflow has never fired, so nothing has ever forced the version bump.

`tests/unit/test_package_version.py` tests only the
`PackageNotFoundError → "0+unknown"` fallback; nothing asserts
pyproject ↔ CHANGELOG agreement. The pattern to copy already exists:
`tests/regression/test_ruff_version_single_source.py`.

**WS-9b — Containers run as root.** Neither `Dockerfile.jetson` nor
`Dockerfile.dev` has a `USER` directive. The Jetson image needs device access
(serial, GPIO, video), so the fix is a non-root user in `dialout`/`gpio`/`video`
rather than a bare `USER nobody` — but root-by-default in the shipped image is
worth closing deliberately.

**WS-9c — 14 silently-swallowed installs in the production image.**
`Dockerfile.jetson` has 14 `|| echo "WARNING: …"` / `|| true` install steps. Two
swallow errors with no message at all: `:120` (`huggingface-hub`, `diskcache`)
and `:134` (`hailort`). Graceful degradation is a deliberate and reasonable
design choice for optional hardware — but `huggingface-hub` is what the
weight-update poller needs, so the image can build "successfully" while a
feature is silently absent. Emit a build-time capability manifest and assert the
expected set, so degradation is *reported* rather than *discovered*.

**WS-9d — Dependencies declared twice.** `Dockerfile.jetson` restates versions
already in `pyproject.toml` (`pyserial>=3.5`, `"aiohttp>=3.9"`,
`"anthropic>=0.40"`, `"smbus2>=0.4"`, `"Pillow>=10.0"`, …) with no single source
of truth. WS-2a's lockfile is the fix: install extras from the locked set.

**WS-9e — No trunk, no tags, 90 branches.** Measured with `git ls-remote`
against the remote itself, so this is independent of the local clone's depth:

| Query | Result |
|---|---|
| `git ls-remote --symref origin HEAD` | `refs/heads/claude/markdown-implementation-plan-aVJ2l` |
| `git ls-remote --heads origin main master trunk develop` | **empty — none exist** |
| `git ls-remote --heads origin \| wc -l` | **90** |
| `git ls-remote --tags origin \| wc -l` | **0** |

A long-lived agent branch is the default branch, and there is no `main`. This is
not cosmetic — three concrete consequences follow:

1. **`features.yaml`'s own contract references a branch that does not exist.**
   Its header comment defines `implemented_in` as *"their landing commit SHA on
   main"*. 39 `done` features carry such a pin.
2. **Nothing but a surviving branch keeps those pins reachable.**
   `scripts/archive_stale_branches.sh:5-6` is explicitly designed to archive each
   stale branch **as a tag** because *"the tag keeps every commit reachable and
   GC-safe"*, and the `deploy-repin` skill exists to create exactly those
   annotated tags. With 0 tags and 90 branches, there is no safety net — which
   is precisely the precondition `pin-reachability-audit` is meant to catch
   **before** `archive_stale_branches.sh --push` ever runs.
3. **No stable base for "is the base branch green" reasoning**, release tagging,
   or `deployments/*.json` re-pinning.

*Deliberately not claimed:* that any pin is currently orphaned. 26 of the 39
pins do not resolve in this audit's clone, but that clone is **shallow** (52
commits), so this says nothing about reachability on the server. Establishing
that requires a full clone — which is the first task of this workstream, not one
of its findings.

**Change.** Run `pin-reachability-audit` in a full clone; create the protective
annotated tags via `deploy-repin` for anything unprotected; tag `v0.4.0`
retroactively so `release.yml` has fired at least once and the version
invariant in WS-9a has something to assert against; then establish `main`
(keeping the current default as an alias until CI, `deployments/*.json`, and
`features.yaml`'s comment are re-pointed); then prune the 90 branches via
`archive_stale_branches.sh`, which is safe only once tags exist.

Order matters here more than anywhere else in the plan: tags before renames,
renames before pruning.

**Also worth keeping:** `.github/workflows/config-compat.yml` is a genuinely
strong practice — it worktrees out the *deployed image SHA* and validates new
`config/*.yaml` against the **old** schema, catching the
"YAML-only PR merges → rover pulls → container crash-loops with `Extra inputs
are not permitted`" failure class. Invariant 6 is enforced, not aspirational.
Do not weaken it.

**Effort:** S–M. **Risk:** low, except WS-9e which needs care around pinned SHAs.

---

## 4. Sequencing

Ordered by (risk retired) ÷ (blast radius), and respecting the ratchet
constraint from §1.

### Wave 0 — Buy ratchet headroom (prerequisite)

All three suppression budgets are at ceiling, so any wave that needs a marker is
blocked until slack exists. Two free reductions:

- `config/migration.py:129,134` — two unjustified `# hardcoded-ok` markers on
  `/ 1000.0` and `* 1000.0`; both duplicate `constants.MILLISECONDS_PER_SECOND`
  (`constants.py:59`). Swap to the constant, reclaim 2 slots.
- `llm_gateway/mission_parser.py:117-124` — `_SPEED_MAP` ClassVar replicating
  `MissionParserConfig.speed_map` verbatim, with **no reference other than its
  own definition** (the constructor at `:133` correctly uses `cfg.speed_map`).
  Delete.

### Wave 1 — Correctness and the two dated deadlines

1. **WS-1** — `S101` + 8 asserts. Highest severity, smallest diff.
2. **WS-2c** — promote `security` to blocking (pip-audit is green today);
   check `test-windows`' green streak. **Due 2026-09-19 / 2026-09-23.**
3. **WS-9e, protective half only** — `pin-reachability-audit` in a **full**
   clone, then `deploy-repin` for anything unprotected. Pulled forward out of
   Wave 5: with 0 remote tags, 39 feature pins rest on surviving branches, and
   creating the tags is cheap, non-destructive, and blocks nothing. The rename
   and the branch prune stay in Wave 5.
4. **WS-6a** — delete the dangling `__all__` (8 lines, verified `AttributeError`).
5. **WS-8b** — widen `doc_hygiene.py`'s file list (one config change, 5 findings).
6. **WS-9a** — tag `v0.4.0` so `release.yml` fires once, then add the
   pyproject ↔ CHANGELOG version test.

### Wave 2 — Make the gates real

6. **WS-2b** — changed-lines coverage gate into the `test` job.
7. **WS-2a** — `uv.lock` + `uv lock --check`.
8. **WS-3a** — fix the four hardcoded-value gate blind spots.
9. **WS-7b/7c** — `pragma: no cover` ratchet; `exclude_also`.
10. **WS-5** duplicate-Protocol-name regression test.
11. **WS-8e** — doc link/path-resolution gate.

Wave 2 before Wave 3 deliberately: a gate landed after its cleanup only
documents the cleanup; landed before, it holds the line.

### Wave 3 — Consolidation

12. **WS-3b–3g** — collapse the three config roots; add the 5 missing schema
    fields; kill the 9 `getattr` bypasses. Each with a paired AQA +
    backwards-compat test.
13. **WS-6c** — sysfs + retry consolidation first (closes the two invariant
    violations); then the remaining 9 clusters.
14. **WS-6b** — wire-or-remove decisions on `_skill_delegator`, `dla_enabled`,
    `bc_batch_size`, and `action_override`.
15. **WS-5a/5b/5c** — de-duplicate the Protocols; add `tests/unit/interfaces/`.
16. **WS-6d/6e/6f** — dependency hygiene, 2 orphan scripts, facade tidy-up.

### Wave 4 — Structure

17. **WS-4** — orchestrator constructor bundles, `tick()` extraction,
    `RoverIsaacLabEnv` split. C901 15 → 12. Sequence
    `hardware/lidar/ld19_driver.py` first (complex **and** coverage-omitted).
18. **WS-8a/8c/8d/8f** — root-file moves, grab-bag dissolution + narrowed
    `_SHARED_KERNEL_PREFIXES`, one per-directory doc format, `openspec` archive.
19. **WS-9b/9c/9d** — container hardening.
20. **WS-2d** — extras-job consolidation.

### Wave 5 — Trunk

21. **WS-9e, remaining half** — establish `main` (old default kept as an alias
    until CI, `deployments/*.json`, and `features.yaml`'s "SHA on main" comment
    are re-pointed), then prune the 90 branches with
    `archive_stale_branches.sh`. Safe only because Wave 1 created the protective
    tags first — **do not reorder these two.**

**Deferred by external gate.** Everything under `src/mousedroid/arm/**` is
blocked by the `freeze_gate.py` PreToolUse hook while F-008 is not `done`:
`symbolic_planner.py:333` (WS-3e), three dead attributes (WS-6b),
`arm/CLAUDE.md:19` (WS-8e). Recorded here, executed when the freeze lifts —
**not** via `MOUSEDROID_WORKFORCE_ALLOW_FROZEN=1`, since none of it is urgent.

---

## 5. New gates, consolidated

The durable output of this plan. Nine gates; five extend something that already
exists.

| Gate | Prevents | Where |
|---|---|---|
| `S101` blocking on `src/` | `assert` stripped under `PYTHONOPTIMIZE=1` | `pyproject.toml` (existing `lint` job) |
| `uv lock --check` | Non-reproducible CI resolution | new step |
| Changed-lines branch coverage ≥90% | New uncovered code | existing `test` job, 3.11 leg |
| `pragma: no cover` ratchet | Ungoverned coverage suppression | `.claude/workforce.yaml` |
| Constructor-arity + class-size ratchets | God-constructor regrowth | `.claude/workforce.yaml` |
| C901 max-complexity 15 → 12 | Complexity creep at the ceiling | `pyproject.toml` |
| Duplicate-Protocol-name check | A third `PromptInjectionFilterProtocol` | new regression test |
| Markdown link/path resolution | Doc rot after a reorg | generalise `tools/validate_skill_commands.py` |
| `doc_hygiene.py` over all root docs | Unbounded doc growth | widen existing invocation |

Plus: extend `test_constants_schema_parity_aqa.py::_MIRRORED_PAIRS`; add a
`GatewayConfig`↔`LLMConfig` parity test; extend
`test_facade_completeness.py` to the private re-export count; add a
pyproject↔CHANGELOG version test; populate the empty Pydantic-validator section
of `scripts/vulture_allowlist.py:47-51`.

---

## 6. Explicitly not debt

Recorded so these do not absorb effort later.

- **`learning/offline_rl.py` (755 LOC)** — 6 cohesive classes, largest 243
  lines. Not a god file.
- **`config/schema/hardware.py` (700 LOC)** — 12 domain config classes.
  Cohesive.
- **~450 of the ~750 raw numeric literals** — HTTP status codes, RFC 6455 close
  codes (already centralised at `constants.py:175-191`), the mel-scale formula
  (`hardware/audio/feature_extractor.py:33,45`), stdlib log levels, numerical
  epsilons, SHA-256's 64 hex chars, torch axis arguments, `/proc/modules`
  (kernel ABI), `/tmp/argus_socket` (fixed NVIDIA path, already justified),
  POSIX exit code 127.
- **`telemetry/mock_source.py` (44 literals), `hardware/camera/mock_camera.py`
  (~30)** — synthetic-waveform and rendering maths. High churn, no behavioural
  value.
- **~347 of the 406 vulture findings** — Pydantic fields/validators, Protocol
  members, `self.x` reads vulture cannot trace, enum-by-value. The 2.7% true
  positive rate is *why* the job is correctly advisory.
- **`shell=True` at `harness/spec.py:225`** — executes the operator-authored
  `validation_command` from `features.yaml`; dev CLI, no network path, and the
  `S602` suppression at `pyproject.toml:301` matches its documented rationale.
- **`factory/cloud.py`'s 5 structurally-similar optional-SDK builders** —
  explicitness there *is* invariant 1. Do not DRY it.
- **`agent.md` / `AGENTS.md`** — not a case collision. Redundancy only.
- **`scripts/check_subsystem_boundaries.py`** — already enforces Factory-First
  DI whole-tree in CI with zero violations. Do not add `import-linter`; tighten
  this instead.
- **`.github/workflows/config-compat.yml`** — validates new YAML against the
  *deployed* schema. Keep as-is.

---

## 7. What this plan does not cover

- **Hardware-blocked work.** F-008 (ESP32 diagnosis, USB-C rover smoke) is bench
  work and the repo's stated top blocker (`NEXT_STEPS.md:37-40`). Nothing here
  unblocks it, and nothing here should wait for it except the `arm/**` items.
- **Operator leftovers** already tracked in `NEXT_STEPS.md` items 1–13 (key
  rotation, image re-pin, Grafana import, bring-up). This plan is code and CI.
- **The `ANTHROPIC_API_KEY` rotation** (`NEXT_STEPS.md:33-36`) — P0, operator
  action, unaffected by anything below it. It stays P0 regardless of this plan's
  sequencing.
- **Advisory→blocking promotions beyond the two dated ones.** `vulture-audit`
  (2026-10-01) and `performance` (2026-10-23) need a decision informed by
  WS-6's triage and by Jetson-vs-runner timing variance respectively.

---

## 8. Effort summary

| Wave | Workstreams | Effort | Retires |
|---|---|---|---|
| 0 | Ratchet headroom | XS | Unblocks everything else |
| 1 | WS-1, 2c, 6a, 8b, 9a, 9e (tags) | S | Production correctness hole; 2 dated deadlines; pin protection |
| 2 | WS-2a, 2b, 3a, 5-gate, 7b, 7c, 8e | M | Makes every later wave self-enforcing |
| 3 | WS-3b–3g, 5a–5c, 6b–6f | L | ~850 LOC; 3 config roots → 1 |
| 4 | WS-4, 8a, 8c, 8d, 8f, 9b–9d | L | God-constructor; enterprise layout |
| 5 | WS-9e (rename + prune) | S | Stable trunk; 90 branches → curated set |

Waves 0–2 are the ones worth doing regardless of appetite for the rest: they are
small, they retire the only finding that behaves differently on the rover than
in CI, they hit two calendar deadlines this week, and they convert five
aspirations into gates.

---

## 9. Method and verification discipline

Six audits ran in parallel — config, security, dead code, tests, CI, docs — each
required to cite `file:line` and to report the commands it ran. Their claims
were then re-verified independently before landing here. Two did not survive,
and are recorded because the same traps will catch the next reader:

1. **`telemetry/metrics_registry.py` was reported as a zero-caller orphan
   recommended for deletion.** It is not. `PrometheusMetricsRegistry` is built
   at `factory/autonomous.py:40-42` (a **function-local** import, which a
   module-path grep misses) and used by 10+ test files across the unit,
   property, e2e, functional and user_journey tiers. Deleting it would redden
   the suite. The real finding is the confusing near-identical naming —
   `telemetry/metrics_registry.py::PrometheusMetricsRegistry` (parked
   ADR-016 path) vs `telemetry/metrics/registry.py::MetricsRegistry`
   (production, 46 importers) — plus its absence from `telemetry/CLAUDE.md`.
   That is a naming and documentation problem, not dead code.

2. **`agent.md` vs `AGENTS.md` was suspected to be a case collision.** It is
   not: `git ls-files | tr A-Z a-z | sort | uniq -d` is empty. Redundancy only.

Severity was also corrected downward in one place: the motor angular-velocity
divergence (WS-3d) reads as a live safety defect until you trace every
`MockMotorController(` construction site and find the single one, on the parked
ADR-016 path. It is still worth fixing; it is not a production 30 Hz bug.

**Two measurement caveats for whoever executes this.** The audit clone is
**shallow** (52 commits), so the `gitleaks` history scan covers a subset of what
CI scans at `fetch-depth: 0` — the working-tree scan is complete, and both were
clean. And no `implemented_in` SHA-reachability claim is made here; run
`pin-reachability-audit` in a full clone before Wave 5 touches refs.

---

## 10. Execution mechanics

Each wave is one PR, reviewed against the checklist in
`.github/PULL_REQUEST_TEMPLATE.md`. Waves 2–4 carry enough scope to warrant an
`openspec/changes/<slug>/` bundle (proposal, design, tasks, specs) and an
F-number reserved per ADR-013 — use the `openspec-author` skill so the house
format is followed and `openspec/project.md`'s registry stays current.

Before any wave merges: `make gates`, then `make test`, then
`bash scripts/ci.sh` as the authoritative local superset. Wave 4's module splits
additionally want the `module-split-consistency-sweep` skill — it catches
MRO-shadowed dead code, wrong `mock.patch` targets, dangling re-exports and
stale doc references, which is precisely the set of things WS-4 and WS-5 risk
introducing. New config fields need the `regression-pair-scaffold` skill's
paired AQA + backwards-compat tests, and `prove-pin-fails` to show each new pin
actually detects what it claims to protect.

No wave should weaken an invariant in `docs/CHARTER.md` §4. If one appears to
need to, that is a §6 escalation and a `charter-carveout`, not an
implementation detail.
