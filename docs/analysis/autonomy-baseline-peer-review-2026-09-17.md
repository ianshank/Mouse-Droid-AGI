# Peer review — external "Physical Autonomy Baseline" OpenSpec proposal (2026-09-17)

- **reviewed_artifact**: a user-supplied synthesis of three model reviews (Kimi K3, GPT-5.6 Sol
  Thinking, Nemotron 3 Ultra) proposing an umbrella OpenSpec change
  `jetson-physical-autonomy-baseline` with four sequential child changes
  (`jetson-safety-runtime-foundation` → `encoderless-localization-and-calibration` →
  `baseline-waypoint-navigation` → `learned-planner-qualification`) plus a cross-cutting
  `autonomy-readiness-evidence`.
- **basis_commit**: `587f85f` (clean worktree, branch `claude/openspec-peer-review-0b49b8`)
- **review_date**: 2026-09-17
- **method**: every load-bearing claim checked against the working tree; evidence cited by
  symbol. Behavioural claims were *executed* against a worktree-local venv rather than read —
  the transcripts are in §7. Hugging Face artefacts were read from the live Hub.
- **confidence vocabulary**: `[Certain]` — verified against the tree or an executed probe;
  `[Likely]` — strong inference; `[Guessing]` — flagged as such inline.
- **outcome**: **REQUEST_CHANGES.** The gap diagnosis is substantially correct. The factual
  base is not: the proposal re-proposes three capabilities that already shipped, mis-states the
  firmware's failsafe, and is sequenced against this repo's own control plane. Separately, this
  review found **six safety defects the proposal did not**, four of them more severe than
  anything in it.

> **Scope note.** This is a findings-only review (ADR-013 audit posture). It reserves no
> F-number, edits no roadmap surface, and proposes no CHARTER carve-out. Two defects it found
> were fixed in the same change as executable pins; everything else is recorded for triage.

---

## 1. The three things that matter most

1. **The proposal's premise — "no firmware host-heartbeat exists" — is false, and the
   correction is a one-line config flip that is already item 3 on `NEXT_STEPS.md`.** F-025
   shipped `CMD_HEART_BEAT_SET`. It is dormant only because `ESP32Config.command_set` defaults
   to `legacy`. Building a host-side supervisor process before flipping that switch would add a
   subsystem to work around a feature the repo already owns. [Certain — §7.2]
2. **The planner cannot drive straight or turn in place.** `MCTSPlanner._generate_candidate_actions`
   broadcast a single `linspace` across all three action axes, so every candidate satisfied
   `vx == vy == omega` and the candidate matrix had **rank 1**. No model in the external review
   found this, and it invalidates the premise of its `learned-planner-qualification` child
   change: there is nothing to qualify while the action set is degenerate. **Fixed in this
   change** behind an opt-in strategy flag. [Certain — §7.1]
3. **`SIGTERM` never stops the wheels.** No signal handler exists anywhere in `src/`. The motor
   halt lives in a `finally` that `docker stop` and `systemctl stop` both bypass. The container
   entrypoint `exec`s specifically so SIGTERM reaches Python — and Python then ignores it. This
   is the *normal* shutdown path, not an edge case, and it is strictly worse than the `kill -9`
   scenario the proposal analysed. [Certain — §4, S-1]

---

## 2. Verdict table — governance and sequencing

All verdicts `[Certain]` unless tagged.

| # | Claim as written | Verdict | Evidence |
|---|---|---|---|
| G1 | Drive the work with `/opsx:explore → propose → apply → verify → archive` | **REFUTED — cannot run here** | `openspec/project.md`: the tree is *"documentation-only: no OpenSpec CLI/tooling is installed in this repository, and nothing in CI validates or consumes this tree."* Zero `opsx` hits repo-wide; `.claude/commands/` is gate-pinned deleted by `test_legacy_commands_dir_stays_deleted`. This exact claim was already refuted once, in `openspec/changes/mouse-droid-claude-workforce/peer-review.md` finding 7 |
| G2 | One umbrella change with four child changes | **REFUTED — unsupported shape** | All 18 bundles under `openspec/changes/` are flat, one change ↔ one F-number. The registry table in `openspec/project.md` has a single `F-number` column and no parent/child concept. The precedent for multi-feature work is one bundle + several catalog entries (Isaac Lab F-043–F-049), not nesting |
| G3 | "Your repo already has an `openspec/` tree and an `openspec-quality-plan` skill, so this drops in directly" | **REFUTED** | There is no `openspec-quality-plan` skill. `.claude/skills/` contains `openspec-change`. The `openspec/` tree exists but is explicitly non-authoritative (G1) |
| G4 | Autonomy is the next program of work | **REFUTED** | `python scripts/select_next.py` → `F-008 USB-C rover smoke passes on the physical Jetson (priority=critical, tier=hardware)`. `NEXT_STEPS.md`: the ESP32 is *"functionally dead on UART, ROM bootloader, AND WiFi AP broadcast across both rover USB-C ports"*. `.claude/workforce.yaml` states the rule: *"hardware readiness preempts all in-flight software streams"* |
| G5 | A "fully autonomous" physical baseline | **CHARTER §3 CONFLICT — human decision required** | §3 places *"Autonomous motion without a human in the loop"* out of scope, default posture no-motion. `.claude/skills/charter-carveout` Q1 is a yes; the learned-planner child answers Q2 as well. F-047 is already `deferred` for exactly this: *"Loading a PPO graph through vla/policy.py would change 30 Hz actuation (CHARTER §3 Q1/Q3)."* Agents may not ratify this |
| G6 | `peer-review.md` with `PR-01…PR-14`, severity + disposition | **PARTIALLY TRUE** | Review rounds are real (`rev. B`, `rev. D`, `Round 1–4`), but `PR-nn` appears nowhere. The house shape is a **Verdict table** + **Corrected-design map** + **What survives review unchanged** + **Load-bearing pins**; the one numbered-finding precedent uses `F-PRn`. `.claude/agents/peer-reviewer.md` requires citing **by symbol, never by line number** |
| G7 | F-number allocation (omitted entirely) | **MISSING — would collide** | ADR-013 burns F-009–F-014 **and F-033**, the latter reserved precisely because `progress.md` used it informally for *"chassis safety sensors"* — which is what `jetson-safety-runtime-foundation` is. Next free is **F-050**. Each feature also needs a per-feature script under `scripts/validations/` asserting `passed > 0`, a regression pair, and a hex-SHA `implemented_in` |
| G8 | New `autonomy-readiness-evidence` change | **LARGELY EXISTS** | `.claude/skills/evidence-commit`, `.claude/agents/hw-evidence-auditor.md`, `.claude/workforce.yaml` `evidence.*` (`stale_after_days: 90`), `reports/`, `smoke-reports/`, F-018 trend journal, F-030, F-035, F-038 |
| G9 | Field gates ("≥200 m between interventions", "≥90/100 missions", "≤3% drift over 20 m") | **UNGROUNDED — and it contradicts numbers the repo already committed** | No drift-% or mission-success-rate figure exists anywhere in the repo. The repo's own unbuilt navigation plan committed to *different* numbers: goal tolerance `0.10` m, mission timeout `60.0` s, *"visit counts shift toward the goal-direction action in ≥80% of seeds"* (`docs/superpowers/plans/2026-05-15-full-stack-training-autonomy-dashboard.md`), plus an endurance target of 5 simulated **and** 5 wall-clock minutes. A new proposal should reconcile with those, not invent a third set |

---

## 3. Verdict table — safety and the motion path

| # | Claim as written | Verdict | Evidence |
|---|---|---|---|
| S1 | "No documented firmware host-heartbeat in the Waveshare JSON set" | **REFUTED** | `src/mousedroid/comms/command_set.py`: `WAVESHARE_CMD_HEART_BEAT_SET`, `heartbeat_window_ms`, `worst_case_command_gap_s`, `WaveshareStockCodec.connect_commands`; `ESP32Config.heartbeat_enabled` / `heartbeat_window_multiple`. Armed at connect via `BaseESP32Driver._arm_command_set`. This is F-025, landed, checked against vendor `json_cmd.h`. The proposal is restating vendor-audit finding **R6**, which that feature closed |
| S2 | Therefore build a lease + independent supervisor process as deliverable one | **PARTIALLY TRUE — right gap, wrong first move** | The failsafe is real but **dormant**: no `config/*.yaml` sets `command_set`, so `LegacyCommandCodec.connect_commands` returns `[]`. Executed proof in §7.2. Genuinely absent: a timestamped/sequenced/expiring lease type, and any supervisor crossing a process boundary (one compose service, one console entrypoint). **Correct order: flip the flag, then decide whether a supervisor is still needed** |
| S3 | Read IMU from `{"T":126}` / `{"T":131}` | **REFUTED — already solved differently** | F-036 parses `r`/`p`/`y` from the polled `T:1001` `FEEDBACK_BASE_INFO` frame; F-040 adds the optional RSSM slot. `T:126` is catalogued and deliberately unused. Note for the record: the original audit *recommended* the `T:131` + `T:142` feedback flow and was overruled in favour of polling without a written rebuttal (`c4-esp32-command-set.md`: *"serial `_query_data` only writes when given a command"*). Re-proposing `T:131` re-litigates a silent reversal — do it explicitly or not at all |
| S4 | 30 Hz plan-act is fiction; needs a multi-rate cadence contract | **CONFIRMED — and stronger than argued** | `tick` → `_select_action` (synchronous) → `MouseDroidNavigationAgent.act` → `planner.plan()` blocks the event loop. **`LoopConfig.planning_hz = 10.0` already exists with zero consumers** — the multi-rate contract was declared and never wired, which is better evidence than anything the proposal cites. `loop_overrun_consecutive_ticks`' own docstring concedes that comparing one sample *"emergency-stops on one slow MCTS plan"*. In-repo precedent for the fix already exists: `_try_vla_action` (budget/timeout/fallback) and `CognitiveCore`'s fast/slow split |
| S5 | `AS-6` static gate on direct motor-driver calls | **CONFIRMED absent** | `scripts/check_subsystem_boundaries.py` analyses **imports**, not call sites, and is structurally blind to `await self._esp32.send_velocity(...)`. Two caveats the proposal must absorb: the gate has to cover **two** actuation stacks (`ESP32CommProtocol` and `MotorControllerProtocol`), and `src/mousedroid/orchestrator/autonomous.py` is parked-but-importable per ADR-016, so a naive gate starts red |
| S6 | Assert `esp32.baud` 115200 vs `lidar.baud` 230400; "preflight fails on unset" | **PARTIALLY TRUE — right risk, unimplementable mechanism** | Pydantic defaults always resolve, so "unset" has no runtime representation; `_apply_command_set_coupling` explicitly rejected `model_fields_set` for this reason. The baud is **auto-derived** by the same flip as S2 — executed proof in §7.2. The real gap is different and worse: `_check_esp32` never opens the port (§4, S-4) |
| S7 | `[HUMAN-GATE]` on every motion task; ODD; physical kill switch | **CONFIRMED, with one correction** | Partially mechanised already: `ESP32Config.smoke_test_allow_motion` (default `False`) and `MOUSEDROID_SMOKE_ALLOW_MOTION`. The `RUN-MOTION` consent phrase is **prose only — no code reads it**, as this repo's own earlier review already recorded. **No ODD document and no physical kill-switch requirement exist anywhere** — those are genuine greenfield, and the strongest unique contribution in the external review |

---

## 4. Defects this review found that the proposal did not

Ordered by severity. Each is a live defect on the shipped configuration.

| ID | Severity | Defect | Evidence |
|---|---|---|---|
| **S-1** | **Critical** | **`SIGTERM` does not stop the motors.** No signal handler exists in `src/`. Motor halt lives in `_run`'s `finally` → `stop()` → `_halt_actuators`, which SIGTERM's default disposition bypasses entirely. `docker stop` and `systemctl stop` both send SIGTERM first. Only `SIGINT` (unreachable for a service-managed rover) unwinds correctly. `scripts/mousedroid_entrypoint.sh` `exec`s *specifically* so SIGTERM reaches Python, documenting an intent the code never implements | `main.py::_run`, `_lifecycle_mixin.py::stop`/`_halt_actuators`, `scripts/mousedroid_entrypoint.sh` |
| **S-2** | **Critical** | **A dead LiDAR fails open.** `_safe_lidar_read` returns `np.ones(feature_dim)` on any exception. Features are normalised range fractions, so all-ones decodes to *maximum range in every sector*: the monitor computes `12.0 m` clearance and sets `lidar_clearance_ok = True`. The only backstop is `min_valid_sensors`, which `config/jetson_production.yaml` **lowers to 1** | `src/mousedroid/sensing/manager.py::_safe_lidar_read`, `src/mousedroid/safety/monitor.py::evaluate`, `config/jetson_production.yaml` |
| **S-3** | **High** | **No battery voltage can trigger an emergency stop on the production overlay.** `battery_critical_v`/`battery_warn_v` are `0.0` (disabled by documented semantics), and an unreadable battery returns a literal `0.0`, which the implausibility guard routes to a log-only branch. A genuinely flat pack reads plausible and trips nothing | `config/jetson_production.yaml`, `src/mousedroid/safety/monitor.py::evaluate`, `src/mousedroid/comms/base_driver.py::get_battery_voltage` |
| **S-4** | **High** | **Preflight reports the ESP32 healthy when it is physically dead.** `_check_esp32` calls `build_esp32_driver` and asserts the result is not `None` — no `connect()`, no frame, no response. With `esp32.enabled: false` the factory returns `MockESP32Driver`, so the check reports `OK detail="driver=MockESP32Driver"`. Compare `_check_lidar`, which genuinely actuates and asserts angular coverage. This runs as `ExecStartPre` on both systemd units | `src/mousedroid/validation/preflight.py::_check_esp32` vs `::_check_lidar` |
| **S-5** | **High** | **Emergency stop has no latch and no operator re-arm.** `is_emergency` is recomputed from `False` every tick and `SafetyContext` is frozen with no acknowledgement field. The instant the triggering condition clears, the next tick resumes driving with no human in the loop — which sits uneasily beside CHARTER §3's no-autonomous-motion posture. The container also stays `healthy` indefinitely while e-stopped, because the heartbeat is touched on any non-raising tick | `src/mousedroid/safety/monitor.py::evaluate`, `src/mousedroid/safety/context.py::SafetyContext`, `scripts/mousedroid_healthcheck.sh` |
| **S-6** | **Medium** | **A single LiDAR read can block the 30 Hz tick for up to 1.0 s** — 30× the control period and 5× `max_loop_time_ms`. `read_all` awaits `read_scan` inline; the acquisition deadline is `max(scan_acquisition_timeout_s, …)` = 1.0 s. Related: all five sensor ring buffers are **write-only** — appended, never read — and `observation.timestamp` is captured *before* the gather, so it understates the age of its own contents | `src/mousedroid/sensing/manager.py::read_all`, `src/mousedroid/hardware/lidar/ld19_driver.py::read_scan` |
| **S-7** | **Medium** | **Skills advertise tools that no registry provides — silently.** Of 11 declared tool names across the four builtin skills, **2 resolve and 9 do not**; `mousedroid-voice` and `mousedroid-world-model` resolve to **zero** tools. `FilteredToolRegistry.names` intersects with the parent, so the capability disappears without error. **Fixed in this change**: now logged, and ratcheted by a gate | `src/mousedroid/skills/builtin/`, `src/mousedroid/skills/registry.py::FilteredToolRegistry` — executed proof in §7.3 |
| **S-8** | **Medium** | **`vy` is transmitted on a chassis that cannot execute it.** `_project_action_to_executable_axes` zeroes lateral velocity only when the codec reports no lateral support — true for `waveshare_stock`, **false for `legacy`**, the shipped default. So on the current configuration a non-zero `vy` is PWM-scaled, sent, and recorded as the executed action on a skid-steer rover, with no warning at any layer. Related: the projection's guard is `action.shape[0] <= 1`, which is the *batch* dimension for the `(1, action_dim)` tensor `plan()` returns | `_action_mixin.py::_project_action_to_executable_axes`, `src/mousedroid/comms/base_driver.py::send_velocity` |
| **S-9** | **Low** | **The serial port is opened non-exclusively.** No `exclusive=True`, no `flock`. `scripts/jetson_smoke_test.sh` opens the same device from a second process and writes a legacy STOP frame; `tests/hardware/test_motor_smoke.py` builds a second driver. In-process, the MCP `set_velocity` tool writes motors outside the tick | `src/mousedroid/comms/serial_driver.py::_open_serial`, `src/mousedroid/common/tools/motor_tools.py` |
| **S-10** | **Low** | **One declared-but-unconsumed safety budget.** `MotorLimitsConfig.watchdog_timeout_s` has no production consumer — its only reference outside the schema is a regression assert. This is the class of defect F-026 ("declared governance budgets have consumers") exists to prevent, so that gate has a blind spot. **Corrected in review:** an earlier draft of this row also named `SafetyConfig.max_velocity_mps`. That was wrong — it *is* consumed, via `src/mousedroid/factory/cognitive.py::build_cognitive_core` into `ConstitutionalRLConfig.speed_ceiling_mps`, where `src/mousedroid/cognitive/constitutional_rl.py::ConstitutionalChecker` clips against it | `src/mousedroid/config/schema/hardware.py`, `src/mousedroid/cognitive/constitutional_rl.py` |

---

## 5. Corrected-design map

| Proposal | Correction |
|---|---|
| `/opsx:*` lifecycle | House format: a flat bundle under `openspec/changes/` (proposal, design, tasks, peer-review) + `features.yaml` entry from **F-050** + `openspec/project.md` registry row |
| Umbrella + 4 child changes | One program, N flat catalog entries (Isaac Lab F-043–F-049 precedent) |
| Build a motion-lease deadman first | **Flip `MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock`** (already `NEXT_STEPS.md` item 3) → the firmware failsafe arms itself and baud auto-derives. *Then* scope a lease for what remains |
| Add a keepalive re-send task | **Do not.** A naive re-send of the last velocity would refresh the chassis heartbeat and **defeat** the failsafe it is meant to complement. This is why the proposal's *lease* framing (expiry resolves to zero) is right and a keepalive is wrong |
| New multi-rate cadence contract | **Wire the existing `LoopConfig.planning_hz`**, which already exists with zero consumers, using the `_try_vla_action` budget/timeout/fallback pattern already in the tree |
| New scenario harness | `src/mousedroid/harness/` already has `TaskSpec` + acceptance predicates + 5-phase tick hooks, with `TickContext` already carrying `proposed_action` and `executed_action`. It is wired and default-OFF. Turn it on; do not build a second one |
| New preflight gating | `src/mousedroid/validation/preflight.py` exists and is good. The gap is **one call site** (nothing invokes it at orchestrator startup) plus S-4 |
| New rollout-error evaluation | `src/mousedroid/training/drift_metrics.py::measure_drift` already does posterior-warmup → prior-only rollout → per-step MSE. Add the naive baselines *inside* it. The honesty template already exists in `CHANGELOG.md`'s BDI entries (mean / marginal / PCA / majority-class) |
| New model-lifecycle states | Extend `src/mousedroid/learning/on_device/regression_gate.py` + the SHA-256 `slot_store`, which already implement promote/revert with a written post-mortem of the self-gaming metric they replaced |
| `learned-planner-qualification` | **Blocked on a prerequisite the proposal never identified**: the candidate action set was rank-1 (§7.1). Qualification of a planner that cannot drive straight measures nothing |

---

## 6. What survives review unchanged

These are correct, well-argued, and should carry into any rev. B:

- **Multi-rate cadence with plan expiry**, and the *lease* (not keepalive) framing for motion commands.
- **Classical local planner as a permanent fallback, comparator and data collector** — the repo has no classical planner at all, and MCTS is not a substitute.
- **The "theater test"**: compare RSSM multi-step rollout error against constant-velocity *and* zero-motion baselines before investing in shadow-mode plumbing. Cheap, offline, falsifiable, and runnable against `measure_drift` today.
- **`[HUMAN-GATE]` discipline on physical-motion tasks**, wheels-raised first, verified E-stop as a prerequisite.
- **A formal ODD** and a **physical kill-switch requirement** — genuine greenfield here; both deserve an ADR (`docs/architecture/`, next free number).
- **Default-OFF config posture** for every new block.
- **Do not wire `src/mousedroid/meta/`, `src/mousedroid/scaling/`, `src/mousedroid/growth/`, `src/mousedroid/arm/`** during this campaign — matches the tree.
- **No ROS 2 / Nav2 migration.** Correct, though not for the cited reason: CHARTER is silent on ROS; the real basis is `NEXT_STEPS.md`'s stock-firmware policy and the repo's protocol-DI investment.
- **Sequencing advice**: interfaces, lease, telemetry and preflight gates *before* the scan matcher.
- **Pre-committing the follow-on** (fiducial relocalisation, not MoE/meta-learning) if scan drift proves limiting.
- **Do not archive on green unit tests** — physical evidence required.

---

## 7. Executed evidence

Run on `587f85f` with a worktree-local venv (Python 3.11.9, torch 2.14.0+cpu, numpy 2.4.6,
pydantic 2.13.5). The global interpreter's editable install resolves `mousedroid` to a
*different* checkout, so every figure below was produced against an interpreter verified to
import from this worktree.

### 7.1 The planner could not drive straight or turn in place

```
n_action_candidates=9  action_dim=3   (MCTSConfig defaults)
matrix_rank = 1  (min(n,dim)=3)
every row has vx==vy==omega : True
contains drive-straight (vx!=0, omega==0) : False
contains turn-in-place  (vx==0, omega!=0) : False
contains full stop      (all zero)        : True
```

Every candidate lay on the main diagonal of the action cube. The only non-arcing primitive
reachable was a full stop. Compounding it, `_expand` always descends to `children[0]` after
expanding regardless of UCB, and `plan()` selects the **most-visited** child, never `mean_value`.

Budget interacts badly with the published latency. `compute_mcts_budget` scales
`base × (1 + surprise)` to a `n_simulations_max` of 200 — 4× base — so a high-surprise tick
targets roughly four times the measured planning cost against a 200 ms loop interlock:

```
surprise= 0.0 -> budget= 50 sims (1.0x)     surprise= 2.0 -> budget=150 sims (3.0x)
surprise= 1.0 -> budget=100 sims (2.0x)     surprise>=3.0 -> budget=200 sims (4.0x)
```

**On the published MCTS latency.** The proposal's p50 ≈ 109 ms / p95 ≈ 125 ms is real but
should be cited precisely: it comes from `mcts/tuned_config.json` in the Hugging Face repo `ianshank/mousedroid-weights` (read from the live Hub on 2026-09-17; not verifiable from this tree offline), produced by `training/warmstart_policy.py::tune_ucb`
at a **fixed 200-simulation benchmark**. It is **not recorded anywhere in this repository**,
there is no `tests/performance/test_mcts_latency.py`, and the 50 ms target
(`DEFAULT_UCB_TARGET_MS`) is a tie-break inside the tuner (`if mean_reward > best_reward and
p50_ms < target_ms`), never a gate. Note also that the selected `best_ucb_c = 1.41` has the
*worst* mean reward of the five candidates (0.145 vs 0.413 at `ucb_c=3.0`) — no candidate met
the 50 ms target, so the tie-break never fired and the value stayed at its initial default.
[Certain]

### 7.2 The firmware deadman is one documented env flip away

```
as shipped:
  command_set='legacy' baud=1000000 codec=LegacyCommandCodec connect=[] window_ms=3000
with MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock   (NEXT_STEPS.md item 3):
  command_set='waveshare_stock' baud=115200 codec=WaveshareStockCodec
  connect=[{'T': 136, 'cmd': 3000}] window_ms=3000
```

The flip simultaneously derives the correct 115200 baud **and** arms the 3000 ms chassis
failsafe. Both `config/default.yaml` and `config/jetson_production.yaml` resolve to `legacy`
today, so `connect_commands()` is empty and no failsafe is armed. `jetson_production.yaml` also
pins `esp32.enabled: false`, so the factory currently resolves `MockESP32Driver`. [Certain]

### 7.3 The skill/tool contract was silently broken

```
skill                       declared  resolved  missing
mousedroid-navigate                2         1  ['query_world_model_pose']
mousedroid-sensor-report           4         1  ['query_health','read_battery','read_distance']
mousedroid-voice                   2         0  ['play_phrase','speak_event']
mousedroid-world-model             3         0  ['episodic_recent_summary',
                                                 'query_world_model_belief',
                                                 'query_world_model_pose']
TOTAL declared=11 resolved=2 dangling=9
```

`query_world_model_pose` is not merely unimplemented — it is **unimplementable**: there is no
pose estimator, odometry source, scan matcher or map anywhere in `src/`, and the chassis is
encoder-less, so `EncoderReading.odometry_*` is structurally zero. The `mousedroid-world-model`
skill's description advertises a *"pose estimate"* that cannot exist. [Certain]

### 7.4 Baseline

`6345 passed, 129 skipped, 1 deselected` on `tests/unit tests/property tests/integration`
(`-m "not hardware"`) at `587f85f` before any edit.

After the changes in §9, each step named by the command that produced it (the repo's
`make test` is an ordered composite of four pytest steps; `behaviour` collects nothing in
this tree):

| Step | Command | Result |
|---|---|---|
| test-cov | `pytest tests/unit tests/property tests/integration -m "not hardware" --import-mode=importlib --cov=src/mousedroid --cov-fail-under=90` | 6374 passed, 129 skipped, 1 deselected — coverage **92.01%** (floor 90) |
| regression | `pytest tests/regression/ -m "not hardware"` | 1422 passed, 33 skipped |
| smoke | `pytest tests/smoke -m "not hardware and not slow"` | 158 passed, 4 skipped |
| behaviour | `pytest tests/behaviour -m "not hardware"` | no tests collected (tier is empty in this tree) |
| e2e | `pytest tests/e2e/ -m "not hardware"` | 22 passed, 5 skipped, 8 deselected |

`mypy --strict` clean over 417 source files; `ruff check` and `ruff format --check` clean;
`tools/validate_skill_commands.py`, `scripts/validate_configs.py`,
`scripts/check_subsystem_boundaries.py` and `scripts/validate.py --tier fast` all pass.

One timing-sensitive `@pytest.mark.slow` case,
`tests/performance/test_instrumentation_overhead.py::test_vlm_progress_instrumentation_within_budget`,
is intermittently red under concurrent load and passes 5/5 in isolation; it exercises
`VLMProgressHead.score`, which these changes do not touch, and the `performance` job is
advisory in CI.

---

## 8. Load-bearing pins for any rev. B

1. **F-008 first.** Any hardware-tier feature inherits its bench blocker; `select_next.py` is the
   arbiter, not prose.
2. **The CHARTER §3 carve-out question is the user's to answer**, explicitly, in writing, either
   way — including a reasoned "no carve-out needed" if that is the conclusion.
3. **Fix the goal path and the action set before qualifying any planner.** `process_mission`
   returns a `GoalVector` to the HTTP caller and nothing feeds it to `_select_action`; the
   candidate set was rank-1. Goal-directed navigation was already specified in the 2026-05-15
   plan (`MissionGoal.navigate_relative`, `MCTSConfig.goal_cost_weight`, `LocalReplanner`) and
   never built — **re-use that spec rather than writing a third one.**
4. **A no-bypass gate must cover both actuation stacks** and decide the status of the parked
   `AutonomousOrchestrator` (ADR-016) before it can go green.
5. **Prefer enabling an existing default-OFF mechanism over building a new one.** Three of the
   proposal's five safety claims describe capabilities that already exist and are switched off.
6. **Software-only work that is genuinely unblocked today**: wire `planning_hz`; add naive
   baselines to `measure_drift`; experience schema v2 (episode id, terminal flag, and the
   proposed/clamped/executed triple — only the executed action is persisted today); a real
   `_check_esp32` probe; a `SIGTERM` handler (S-1).

---

## 9. Changes made alongside this review

Two defects were fixed here because each is software-only, testable offline, and needs no
carve-out. Everything else in §4 is left for triage.

- **`MCTSConfig.action_candidate_strategy`** — new `Literal["shared_axis", "per_axis"]` field,
  default `shared_axis` (byte-identical legacy behaviour, pinned by a test that rebuilds the old
  tensor and asserts equality). `per_axis` emits a spanning set: stop, ±unit along each axis,
  then a deterministic de-duplicated low-discrepancy fill. Dimension-agnostic and built functionally, per
  the world-model subsystem's no-in-place-mutation invariant. **Opt-in**, because changing which
  actions a planner may propose is an actuation change.
- **`mousedroid.skills.contract`** — `unresolved_tool_names` / `log_unresolved_tool_names`,
  called from `SkillRegistry.tools_for`, the one place that sees both halves of the contract.
  **Scope caveat:** `tools_for` currently has no production caller (it is reached only from
  tests), so the warning is latent until the skill-delegation path is wired. The enforcement
  that bites today is the ratcheting regression gate, which records the nine existing dangling
  names with a reason each and fails on any new one.

Test tiers: unit (`tests/unit/world_model/test_mcts_candidate_strategy.py`,
`tests/unit/skills/test_contract.py`), property
(`tests/property/test_mcts_candidates_property.py`), regression pair
(`tests/regression/test_mcts_candidate_strategy_aqa.py`, `tests/regression/test_mcts_candidate_strategy_backwards_compat.py`) and
`tests/regression/test_skill_tool_contract_aqa.py`. The `per_axis` pins were verified to go red
against the reverted implementation, per `.claude/skills/prove-pin-fails`.

### Round 2 — adversarial review of this change

These changes were themselves reviewed adversarially before landing, and the review was
load-bearing. What it caught, and the disposition:

| Finding | Disposition |
|---|---|
| The de-duplication guarding the quasirandom fill was dead code, and the docstring justifying it described a collision that the chosen sequence cannot produce (it was true of the Sobol engine used in an earlier draft, not of the R_d rotation that replaced it) | **Accepted.** Deleted `_drop_colliding_rows` and its two tunables; uniqueness is now asserted by the property tier instead of defended at runtime |
| Four mutations survived the new tests, including scaling the axis primitives to 0.001 — leaving the magnitude, the entire operational payload of `per_axis`, unpinned | **Accepted.** Added a full-scale pin and an interleave-ordering pin; all three reproducible mutations now fail |
| The byte-identical pin built `expected` from `DEFAULT_ACTION_LIMIT`, the same symbol the implementation reads, so re-pointing the constant moved both sides together | **Accepted.** The pin now spells `-1.0, 1.0` as literals, with a second test pinning the constant itself |
| §4 S-10 claimed `SafetyConfig.max_velocity_mps` has no consumer. It does | **Accepted — this was the worst defect.** Corrected in place, and the correction left visible rather than silently edited. It was inherited from a delegated sweep and not re-verified first, which is the exact failure this document attributes to the proposal under review |
| §9 claimed a skill "now warns" at runtime, but `tools_for` has no production caller | **Accepted.** Scope caveat added |
| `per_axis` at the default candidate count narrows single-axis motion to {stop, full scale}, undocumented | **Accepted.** Recorded as an explicit TRADE-OFF in the field description |
| `_low_discrepancy_points` divided by zero at `dim == 0`; `per_axis` emitted signed negative zeros | **Accepted.** Guarded and normalised, both pinned |
| `progress.md` labelled a partial run as the full CI test job | **Accepted.** Corrected to name the steps actually run |

**No CHARTER §3 carve-out needed for these two changes**: both add config-gated or
observability-only behaviour, the default posture stays byte-identical, neither touches the
actuation gate, and neither places inference or training inside the 30 Hz loop.
