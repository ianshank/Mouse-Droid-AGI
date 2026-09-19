# Peer review — external "positioning + safety-boundary" multi-model synthesis (2026-09-19)

- **reviewed_artifact**: a user-supplied synthesis of three model reviews (Gemini 3.8 Flash
  Thinking, Claude Opus 5 Thinking, Kimi K3) arguing that the repo should stop adding cognitive
  pillars, reframe as a safety-bounded edge-autonomy reference platform, close the odometry gap,
  move safety out-of-process with ESP32-side authority, publish falsifiable latency percentiles,
  adopt an explicit ODD, dual-track JetPack 7.2, and ship a LeRobot adapter pair.
- **basis_commit**: `2469588` (branch `claude/mousedroid-peer-review-8lmr6q`, clean at review start)
- **review_date**: 2026-09-19
- **method**: every load-bearing claim checked against the working tree and cited by symbol.
  Behavioural claims were **executed** against an interpreter with the real package on
  `sys.path`, not read — transcripts in §8. External facts (JetPack, LeRobot, USPTO, GitHub,
  Hugging Face) were checked against live sources; anything unconfirmable is marked UNVERIFIED
  rather than asserted.
- **confidence vocabulary**: `[Certain]` — verified against the tree or an executed probe;
  `[Likely]` — strong inference; `[Guessing]` — flagged inline.
- **outcome**: **REQUEST_CHANGES.** The directional advice is largely sound and several
  individual findings are correct. But the synthesis's headline result — "three models
  independently converged" — is not evidence of insight: the thing they converged on is written
  on the first page of `docs/CHARTER.md`, which all three had. Its single highest-priority
  recommendation is blocked on dead hardware it never detected. Its central technical argument
  (odometry) is mis-stated in a way that inverts the correct work order. It cites a gate contract
  from a different repository. And it missed a set of live safety defects that outrank most of
  what it raised — including that **every human-safety mechanism in the tree is unreachable code**.

> **Scope note.** Findings-only review under the ADR-013 audit posture: no F-number reserved, no
> `features.yaml` / `NEXT_STEPS.md` / `CHARTER.md` edit, no carve-out proposed. F-008 remains the
> hardware next feature. Two software-only defects found *during* this review are fixed alongside
> it (§9), matching the precedent set by `2469588` (#228); everything else is recorded for triage.

---

## 1. The five things that matter most

1. **Both compensating controls CHARTER §3 names for the cloud-egress carve-out are broken.** The
   carve-out is the ratified argument for letting rover NL reach `api.anthropic.com`, and it rests
   on two things. *Control 1* — the velocity clamp. `comms/_utils.py::clamp` is
   `max(lo, min(hi, value))`, which returns `hi` for `NaN` because every NaN comparison is False.
   It is the terminal guard on the motor path, shared by **both** codecs. Executed:
   `build_velocity_cmd(nan, nan, nan)` returned `{'T': 1, 'vx': 255, 'vy': 255, 'omega': 255}` —
   full-scale PWM on all three axes, no exception, because the clamp resolved the NaN to `1.0`
   before `int()` saw it. Two siblings shared it: `motor_tools.py::_clamp`, guarding the MCP
   `set_velocity` tool and therefore the only LLM-reachable `send_velocity` in the tree, and
   `_action_mixin.py::_execute_action` on the live 30 Hz tick, which had **no bound at all**.
   *Control 2* — `security/injection_filter.py` is honest that it is *"best-effort … not a
   complete defense"* and says the real protection is that output is *"clamped by
   `LLMConfig.max_vx_norm_mps`/`max_vy_norm_mps`/`max_omega_norm_rads`"*. **Nothing reads those
   three fields.** Executed: with `max_vx_norm_mps=0.01`, `_parse_response` still returns
   `vx_target=1.0` — 100× the documented bound. **All of Control 1 is fixed in this change; Control
   2's false claim is corrected and the fields recorded as unconsumed (§9).** [Certain — §8.4, §8.6]

2. **The convergent diagnosis is a restatement of the project constitution, not a discovery.**
   `docs/CHARTER.md` §1 already reads: *"an edge-AI / robotics engineering project, **not a
   product and not a claim of general intelligence**"*, and names the integration gap itself —
   *"the honest axis is integration … meta-learning and scaling are implemented and tested but
   not yet wired in at all"*. The synthesis presents "stop adding pillars", "reframe away from
   AGI", and "`meta/`, `scaling/` unwired" as a three-model consensus finding. All three are
   pre-existing, ratified, self-disclosed positions. Three models agreeing on the contents of a
   document all three read is not independent corroboration. [Certain — §2]

3. **The one thing it says to do first cannot be done: the ESP32 is physically dead.**
   Kimi K3's closing line — *"If you do only one thing this month, do the ESP32 watchdog and
   independent E-stop"* — is the synthesis's strongest single recommendation, and it targets a
   microcontroller that `NEXT_STEPS.md` P0 item 2 records as needing *"diagnosis + repair"*, and
   that the prior review recorded as *"functionally dead on UART, ROM bootloader, AND WiFi AP
   broadcast across both rover USB-C ports"*. `features.yaml` F-008 is `status: "todo"`,
   `priority: "critical"`, `tier: "hardware"`. No firmware heartbeat, command TTL, boot-time
   motor disable or independent E-stop can be implemented on a board that does not enumerate.
   The review's "cheapest item on the list" is gated on bench work none of the three models
   detected. [Certain — §3, C4]

4. **The odometry argument is mis-stated, and the correction reverses the work order.** All three
   models call missing odometry "the top physical blocker" because drift *"breaks RSSM/MCTS
   displacement prediction"*. The RSSM predicts no displacement: `RSSM.imagine_step` returns
   `(new_h, new_z, predicted_reward)` and `RSSM.decode` reconstructs an observation embedding,
   not a pose. The real defect is worse and is upstream of localization: **`RSSM.reward_head` is
   never on any loss graph.** `RSSM.train_sequence` computes `loss = recon + kl_beta * kl`; the
   repo states the consequence in its own words at `learning/on_device/rssm_refiner.py` —
   *"`allow_unused=True` is MANDATORY: the RSSM `reward_head` … are not on the recon/KL graph, so
   their grads come back `None`"*. `MCTSPlanner._rollout` backs up exactly that head's output.
   **MCTS therefore selects actions by maximising a randomly-initialised, never-trained
   `nn.Linear`.** Perfect odometry would not change the action distribution by one bit. Compounding
   it: `MouseDroidNavigationAgent.act(h, z, safety_ctx)` takes **no goal argument at all**, and
   `GoalVector` is a velocity triple, not a position — so there is nothing for a localization
   estimate to be measured against yet. Fix the objective and the goal type before the sensor.
   [Certain — §8.1]

5. **Every human-safety path in the repo is unreachable code.** `ObservationProtocol` and
   `MouseDroidObservationBundle` define no `human_detected` / `human_dist_m` field;
   `MouseDroidSafetyMonitor.evaluate` reads them via
   `getattr(observation, "human_detected", False)`, and nothing in `src/` ever assigns either. So
   `SafetyContext.human_detected` is permanently `False` on the rover, which makes the Law-1 stop
   in `MouseDroidNavigationAgent.act`, the human-proximity clamp in
   `GeometricSafetyProjector.project`, and the human-proximity branch of `evaluate` itself all
   dead. `SafetyProjectorConfig.human_keepout_m`, `human_proximity_speed_mps` and
   `ThreeLawsConfig.human_safety_radius_m` are declared budgets with no consumable input. The
   `getattr` default is what hides it — a typed protocol field would have failed `mypy --strict`.
   **And a green test actively vouched for the opposite.**
   `tests/unit/test_coverage_gaps.py::test_safety_monitor_human_detection_emergency` set
   `obs.human_detected = True` on a `MagicMock`, which auto-creates any attribute — so the
   `getattr` read the mock's value and the human-proximity emergency passed, while no
   production observation type had the field at all. The test proved the monitor's
   arithmetic and nothing about reachability. That is the most instructive part of this
   defect: it was not merely unnoticed, it was certified. Found while fixing it (§9).
   A review recommending a firmware motion lease while `human_detected` cannot ever be `True` has
   its priorities inverted. [Certain — §5, D-1]

---

## 2. Verdict table — positioning, naming and distribution

| # | Claim as written | Verdict | Evidence |
|---|---|---|---|
| N1 | "Stop adding cognitive pillars; ship an integration-proven slice" | **CORRECT, ALREADY RATIFIED** | `docs/CHARTER.md` §1 and §3 already declare the wired / factory-instantiated-default-OFF / not-yet-wired / parked split. Agreement with the charter is not a finding |
| N2 | "`meta/`, `scaling/`, `arm/` unwired" | **PARTIALLY TRUE — one-third wrong, and it missed the biggest one** | `meta/` (211 LOC) and `scaling/` (220 LOC) are genuinely unwired: no factory builder, no config block. `arm/` (4,018 LOC) is **not** unwired — `factory/arm.py` exposes five builders and it is *deliberately* frozen by a mechanised gate (`.claude/workforce.yaml` `freeze.frozen_paths`, unfreeze on F-008) plus `tests/regression/test_import_graph_freeze.py`. Calling a governed park "unwired" conflates policy with drift. **Missed: `efficiency/` (633 LOC) is the largest genuinely-unwired surface** — 3× meta+scaling combined; `build_tensorrt_compiler` has zero production call sites and `optimized_inference.py` + `profiler.py` have zero references anywhere in the tree |
| N3 | "Breadth exceeds integrated capability" | **TRUE, but the real number is a default-OFF posture question** | WIRED-LIVE is 42,528 / 72,123 LOC (59%). The tail is 26% WIRED-OPTIONAL (15 packages, all default-OFF), 8% offline tooling, 5.6% parked, **1.5% actually unwired**. The sharper finding is *inside* the 59%: with stock defaults `cognitive.enabled=False` and `vla.backend="none"` both short-circuit, so 4,297 LOC of orchestrator and 3,356 LOC of world model funnel into `agents/navigation.py` — 168 LOC |
| N4 | "Reframe away from 'AGI mouse droid'" | **ALREADY DONE IN-REPO AND REGRESSION-GATED; STILL LIVE ON GITHUB** | Of 55 repo-wide `\bAGI\b` hits, **50 are the `Mouse-Droid-AGI` URL token** and 4 of the remaining 5 are the test that removed the framing. `tests/regression/test_portfolio_reframe_aqa.py::test_headline_docs_drop_agi_framing` bans it in README / CHARTER / CLAUDE.md. `pyproject.toml` name is `mousedroid`; `CITATION.cff` claims "edge-AI / robotics portfolio project". **But the gate reads files, so it cannot see GitHub's own surfaces**: the repo description is still `" MouseDroidAGI: Star Wars MSE-6 Agentic World Model"` — the exact token the gate bans in-tree is the public one-liner. That inconsistency is worse than either state alone |
| N5 | "Extract the runtime under a neutral name **before** you publish packages, a HF org, or weights repos" | **PARTLY OVERTAKEN — and aimed at the wrong token** | Already public: `ianshank/mousedroid-weights` (20 LFS files, ~100 MB, 0 downloads) and `ianshank/mousedroid-dual-stream-rssm` (placeholder). PyPI `mousedroid` is **404** (unpublished; `release.yml` built, never fired), 0 releases, 0 tags. Still cheap — nobody has consumed any of it. But **"AGI" never reached HF or PyPI at all**; what actually spread is `mousedroid`, hardcoded in `config/default.yaml`, `config/jetson_production.yaml`, `config/schema/world_model.py`, `scripts/download_weights.sh`, `scripts/export_dual_stream_rssm_onnx.py`. Renaming to strip "AGI" while keeping "mousedroid" pays the migration cost and leaves the higher-exposure token in place. Note `mouse-droid` on PyPI is **already taken** (unrelated 2015 package) |
| N6 | "1 star, 0 forks" | **TRUE, and generous** | Live API: `stargazers_count: 1` (sole stargazer is `ianshank` — a self-star), `forks_count: 0`, `subscribers_count: 0`, both open issues are Dependabot PRs. **Missed, and cheaper than anything the review proposes: `default_branch` is `claude/markdown-implementation-plan-aVJ2l`, not `main`**, and `topics: []` with `homepage: null`. Every clone, Code-tab landing, raw URL and badge resolves against an agent-generated branch name, and the repo is invisible to topic search. "1 star" is partly a distribution fact, not only a quality signal |
| N7 | "`three_laws.py` is a credibility tax" | **TRUE — the one live, ungated overclaim** | 466 LOC of genuine, unit- and property-tested constraint arithmetic wearing Asimov branding. `README.md` asserts it *"encodes Asimov's Three Laws as hard constraints"*; `_check_law1` is a proximity + forward-clearance + acceleration envelope with no harm model and no inaction reasoning. The gap is purely nominal and a rename costs an afternoon. See also D-2: it is not wired into the motion path at all |
| N8 | Lucasfilm "Droid" Reg. No. 2553167 | **CONFIRMED, scope overstated** | DROID, Lucasfilm Entertainment Company Ltd., serial 75652542, filed 1999-03-03, registered 2002-03-26, "Registered and Renewed". The 1999 filing is the software/CD-ROM class; the wireless-handset class the review implies is a **different, later** Lucasfilm registration |
| N9 | "$20M Shepperton replica judgment" as a Droid-trademark precedent | **NUMBER REAL, FRAMING WRONG** | The 2006 C.D. Cal. $20m default judgment against Ainsworth / Shepperton Design Studios is a **Stormtrooper prop-replica copyright** matter (plus misrepresentation of authorship), not trademark, and has nothing to do with the word "droid". The correct characterisation is arguably *more* on-point for a physical MSE-6 replica — but it is a different legal theory and citing it as trademark precedent is an error. Reporting only; not legal advice |

---

## 3. Verdict table — safety architecture and the motion path

| # | Claim as written | Verdict | Evidence |
|---|---|---|---|
| C1 | "Your `safety/` + `orchestrator/` already **is** a Simplex architecture; you simply haven't named or argued it" | **REFUTED — the pieces are not Simplex** | "Unnamed" is right (`grep -rni simplex` → zero hits), but naming it would document an architecture the tree does not implement. The fallback chain in `_ActionMixin._select_action` switches on **exception / timeout / shape mismatch**, not on a safety verdict. The hard override in `tick` short-circuits `safety_ctx.is_emergency` to `emergency_stop()` — a switch whose "baseline" is zero velocity, not a controller. The one true decision module, `GeometricSafetyProjector.project`, is **disabled in every shipped config** (D-4). Decisive: **no safety verdict selects the controller** |
| C2 | "The monitor lives in-loop; an orchestrator stall takes it with it" | **CONFIRMED** | `MouseDroidOrchestrator.tick` calls `self._safety_monitor.evaluate(...)` **synchronously** — same coroutine, event loop, thread and process. `health/watchdog.py`'s notifiers have **no actuator authority**; they signal liveness only. The sole deadline is `asyncio.wait_for(self.tick(), tick_timeout_s)` in the same task, which cannot help if the event loop itself blocks. `docker-compose.jetson.yml` defines one app service |
| C3 | "≥100 Hz, own process, unconditional ESP32 authority, log **every** decision not just overrides" | **PREMISE MOSTLY CONFIRMED** | Rate: one `evaluate` per 30 Hz tick; **no separate-rate field exists** in `SafetyConfig`. Authority: the monitor has **none** — `SafetyMonitorProtocol.evaluate` returns a frozen `SafetyContext` and the orchestrator decides. Logging: `evaluate` *does* emit `safety_evaluate_result` every call, but at `_log.debug`, and `config/default.yaml` pins `logging.level: "INFO"` → **suppressed in every shipped config**; even at DEBUG it carries only `is_emergency` and `valid_sensors`, omitting clearance, LiDAR distance, battery and loop time. So "only overrides are logged" is true in production, and the DEBUG record is not a replayable decision record either |
| C4 | "Final authority should be the ESP32: heartbeat, command TTL, firmware-side clamps, motors disabled on boot, E-stop independent of Jetson/Docker/Python/Wi-Fi/LLM" | **RIGHT TARGET, BLOCKED HARDWARE, AND ONE PART IS FREE TODAY** | (a) heartbeat **exists but is still dormant**: `ESP32Config.command_set` defaults `"legacy"` and **no `config/*.yaml` sets it**, so `LegacyCommandCodec.connect_commands` returns `[]`. (b) command TTL / motion lease: **does not exist** — `_last_velocity` has no expiry. (c) firmware-side clamps: **no firmware sources exist in this repo at all** (zero `*.ino` / `*.cpp` / `*.h`); every clamp is host-side and pre-transmit, and there is **no acceleration limit on the production path**. (d) motors disabled on boot: **does not exist** — `_LifecycleMixin.start` connects and never issues a stop; `orchestrator/CLAUDE.md` states the consequence outright (*"the rover can be moving throughout bring-up"*). **This is the one C4 item that needs no firmware and no working ESP32 — one host-side stop after `connect()`.** (e) hardware E-stop: **does not exist**; `emergency_stop` is a software command over the same serial link the Jetson owns. All of (a)–(c) and (e) are gated on F-008 |
| C5 | "Safety must move out-of-process with independent authority" | **CONFIRMED as a gap — with a correction the review inverts** | No independent authority exists. But the review frames this as purely absent; the repo ships something *worse than absent* in the same slot — see D-5, where the only out-of-process authority that does exist (`systemd`) **automatically restarts the motion system** |
| C6 | "Adopt an explicit ODD with declared exclusions; ISO 3691-4 / UL 4600 vocabulary" | **CONFIRMED ABSENT — genuine greenfield, and the strongest unique contribution in the synthesis** | Repo-wide search for `ODD` / `operational design domain` / `operating zone` / `restricted zone` / `geofence` → nothing but the prior review's own note that none exists. `ISO ?3691` / `UL ?4600` / `GSN` / `assurance case` / `ISO ?13849` / `IEC ?61508` → **zero hits repo-wide**. Nearest analogue is `RoboticsLawChecker._check_law2`'s `allowed_zone_min/max`, which clips the **action vector**, not a position, and is never populated — flagging it because it is the symbol someone will reach for first |
| C7 | "Constrain the LLM to a typed, allowlisted mission schema with bounded parameter ranges and operator approval" | **RIGHT CONCLUSION, WRONG PATH — the threat model is aimed at a seam that does not actuate** | All three models worry about the LLM holding motor authority via the *mission* path. It does not: no LLM-derived `GoalVector` reaches `send_velocity` anywhere in the production orchestrator (§8.4). The path that genuinely gives a model motor authority is the MCP `set_velocity` tool, which none of the three audited — and which carried the fail-open clamp (D-19). Operator approval is already CHARTER §3's default posture. And the bounded-range argument cuts the other way from how the review uses it: the existing clamps were **the thing manufacturing the maximum command** (D-17/D-18/D-19). The review would have scored all of this as "already handled — values are clamped" |

---

## 4. Verdict table — evidence, performance and toolchain

| # | Claim as written | Verdict | Evidence |
|---|---|---|---|
| P1 | "Publish falsifiable p50/p90/p99 + deadline-miss ratio, not '30 Hz' as an assertion" | **CONFIRMED on the conclusion, wrong on the premise** | The premise that nothing is instrumented is false: `set_loop_time_ms` writes a gauge **and** a histogram (`mousedroid_loop_latency_ms`, 10 buckets to 200 ms), plus an 8-phase `mousedroid_tick_phase_ms` and a `mousedroid_tick_overruns_total` counter. Percentiles and a miss ratio are **derivable in one PromQL expression**. What is true: no recording rule or alert consumes either series, and **no tracked artifact under `reports/` or `smoke-reports/` contains a single control-loop latency number**, at any staleness. Note the repo's own convention is p50/**p95**/p99 (`validation/latency_stats.py`); **p90 appears nowhere** |
| P2 | "Reuse your Hex-vision gate contract (`G-LATENCY` with hardware tags, `G-ENGINE-COMPAT`)" | **FACTUAL ERROR — the review is citing a different repository** | Zero repo-wide hits for `G-LATENCY`, `G-ENGINE-COMPAT`, `ENGINE_COMPAT`, `hex-vision` or `hexvision`. No such naming convention exists here. This repo's gate vocabulary is per-feature `scripts/validations/F-0NN.sh` + `features.yaml` acceptance entries + named CI jobs. Any rev. B must drop this or re-derive it |
| P3 | "30 Hz" is an assertion | **CONFIRMED UNBACKED** | Asserted in `README.md` (4×), `CLAUDE.md`, `docs/CHARTER.md` (7×), 47 files under `docs/`, and `orchestrator/CLAUDE.md` ("Strict 33.3 ms Cadence"). `SMOKE_REPORT.md`'s only "30 Hz" content is a **config echo** (`control_hz=30.0` from a preflight check), not a measurement — and it is 125 days stale with camera FAIL, LiDAR FAIL and ESP32 fallen back to mock. `reports/endurance/` is **empty** (`.gitkeep` only) — a declared-but-never-populated evidence family |
| P4 | CI `performance` job covers this | **NO — it is a tripwire, not a benchmark** | `pytest tests/performance -m "not hardware"` on `ubuntu-latest`; the `-m "not hardware"` filter excludes `test_jetson_endurance.py`, **the only file that times the orchestrator loop**. Remaining tests are component micro-benchmarks at a deliberately loosened 2.0× budget (relaxed from 1.15× because runner contention alone measured 1.17–1.26×). Promotion window due 2026-10-23, not overdue |
| P5 | Prior defects S-6 / `planning_hz` | **BOTH STILL OPEN** | `_safe_lidar_read` awaits `read_scan()` with **no `asyncio.wait_for`**; the acquisition deadline resolves to 1.0 s. Worse than the prior review stated: it is bounded by `LoopConfig.tick_timeout_s`, whose default is **also 1.0 s**, so the guard does not protect the 33.3 ms budget at all — it converts the overrun into `_log.critical` + `emergency_stop()`. And because `read_scan` uses `asyncio.to_thread`, cancelling the tick does **not** stop the worker, which keeps the serial handle past the timeout. `LoopConfig.planning_hz` still has zero consumers |
| P6 | "JetPack 7.2 dual-track; Dockerfile pins L4T r36.4.0 / CUDA 12.6, a generation back" | **REPO FACTS CONFIRMED; EXTERNAL FACTS MOSTLY CONFIRMED; THE RT-KERNEL ARGUMENT IS OVERSOLD** | Pins confirmed verbatim (`dustynv/l4t-pytorch:r36.4.0`). JetPack 7.2 (June 2026, L4T 39.2, Ubuntu 24.04, kernel 6.8) does cover Orin Nano, and 7.2.1 is current `[Likely]` — but **CUDA is 13.2.x, not "13.0"**, and whether the preemptible RT kernel is production-grade for 7.2 is **UNVERIFIED**. The RT-kernel→30 Hz argument does not hold here: this repo's 30 Hz is a *soft cadence*, an overrun merely increments a counter, `tick_timeout_s` is 30× the period, and there is **no `SCHED_FIFO` / `chrt` / `isolcpus` / `taskset` anywhere in `src/` or `scripts/`**. Nothing in the design depends on kernel preemption. Yocto is irrelevant — there is no Yocto surface in the repo |
| P7 | "TensorRT engines must be rebuilt per GPU and TensorRT version" | **CONFIRMED, AND THE REVIEW UNDERSTATED IT — was a live defect; now closed** | See D-7, closed across two changes. It was not a currency observation: it fired on a GPU swap or a base-image bump regardless of whether anyone ever upgraded JetPack. `_runtime_identity()` now puts the TensorRT, CUDA, torch, torch2trt and compute-capability identity in the cache key |
| P8 | "Ship a LeRobotDataset exporter + `LeRobotPolicy` adapter" | **DIRECTIONALLY FINE, MATERIALLY FARTHER THAN ARGUED; two citation errors** | Zero LeRobot integration exists (one docstring mention repo-wide). Native format is LMDB + msgpack. The container change is ~a day; **the blocker is that no raw frames are persisted at all** — only a 256-d embedding survives the loop, so a LeRobotDataset camera schema cannot be populated from any existing store and historical data is not back-fillable. Corrections: v0.6's third world model is **LingBot-VA**, not "LaWAM" (zero occurrences in the release material); and "16K+ datasets / 2.2K+ contributors" is a **September 2025** snapshot quoted as current |

---

## 5. Defects this review found that the synthesis did not

Ordered by severity. Each is live on `2469588` unless marked fixed.

| ID | Severity | Defect | Evidence |
|---|---|---|---|
| **D-25** | **Medium** (latent High) | ~~**The TensorRT engine cache unpickles arbitrary code behind a permission control that does not exist.**~~ **FIXED IN THIS CHANGE (§9.2).** `efficiency/tensorrt.py::load_compiled` falls back to `torch.load(..., weights_only=False)` -- arbitrary pickle, i.e. code execution on load -- behind a comment asserting the cache directory *"should have restricted permissions (0700)"*. Nothing enforced it: `mkdir` passed no `mode=` (executed: the directory lands at **0o755**, group- and other-readable), there is no `chmod` anywhere in `src/`, no validator on `JetsonConfig.tensorrt_cache_dir`, and `docker-compose.jetson.yml:80` bind-mounts `/opt/mousedroid` **from the host**. The fallback fires precisely when the file is *not* what was expected. **This is the D-21 class for the third time** -- after the injection filter's `max_vx_norm_mps` and D-1's `human_detected`: a security claim written in a comment, relied on by a real decision, never implemented. **Rated latent, and the rating is the finding's twin.** An earlier draft of this row called it a live RCE; tracing it shows `OptimizedInference` -- the only caller of `compile_model` -- is **constructed nowhere in `src/`**, `build_tensorrt_compiler` has **no caller**, and `vulture` already reports both as unused. That is the D-0 misrating repeated after this review had written the lesson down; see §9.2 | `efficiency/tensorrt.py::load_compiled`, `::_save_compiled`, `config/schema/hardware.py::JetsonConfig`, `docker-compose.jetson.yml`; executed in §8.10 |
| **D-26** | **Medium** | ~~**`tensorrt_enabled` declares an acceleration that never runs.**~~ **CORRECTED IN THIS CHANGE (§9.2).** Schema default `True` (`config/schema/hardware.py:563`) and set `true` explicitly in **four** shipped configs -- `default.yaml:107`, `jetson_production.yaml:61`, `jetson_dual_stream.yaml:20`, `jetson_sdcard_64gb.yaml:18`. Nothing constructs a compiler, so an operator reading any of them concludes TensorRT acceleration is on while the models run in eager PyTorch. **Fourth instance of the S-11 class in this review**, after D-1 (`human_detected`), D-2 (`ThreeLawsConfig.enabled`) and D-21 (`max_vx_norm_mps`). The recurrence is the actionable part: this repo's config surface systematically over-promises relative to its wiring. Fixed as a description correction, not a default flip -- the flag expresses intent correctly for the day the seam is wired | `config/schema/hardware.py::JetsonConfig.tensorrt_enabled`, `config/*.yaml` |
| **D-27** | **Medium** | **A `PRE_ACTION` hook cannot change the action that executes.** `orchestrator.py:622` assigns `ctx.proposed_action` and `:623` runs the `PRE_ACTION` hooks, but `:626` then does `action = executable` -- so a hook that mutates `proposed_action` has no effect on what reaches the motors. `ctx.safety_ctx` is likewise exposed to hooks (`orchestrator.py:525`, declared `harness/protocol.py:155`) but read-only in practice. The harness advertises a mutation seam it does not honour -- the same declared-but-inert class as D-1 and D-2. Found while tracing D-6; **not fixed here** | `orchestrator/orchestrator.py`, `harness/protocol.py` |
| **D-24** | **Critical** | ~~**A dead LiDAR still reported an affirmative all-clear ring, and the `lidar_unavailable_policy` was never even consulted.**~~ **FIXED IN THIS CHANGE (§9).** S-2 removed the `np.ones(feature_dim)` substitute from `SensorManager._safe_lidar_read`'s *exception* path. The identical vector was still manufactured one layer down, on a path that **never raises**: `LD19LidarDriver.read_scan` returns `empty_scan()` when the serial port was never opened or no valid frame arrived, and `LidarFeatureExtractor.extract` maps a zero-point scan to `np.ones(n_sectors)`. Features are normalised range fractions, so that is *maximum range in every sector*. Executed: an unplugged LD19 yields `lidar_min_dist_m=12.0`, `lidar_clearance_ok=True`, `is_emergency=False` **with `lidar_unavailable_policy='emergency'` set** — because the features were "present", the policy branch was unreachable. This makes D-3 worse than D-3 states: arming the interlock would have handed an operator an interlock that cannot fire for an unplugged cable or a stalled motor, which is the exact failure mode `test_unknown_policy_is_rejected_at_load` exists to prevent in its config form. Found while implementing D-3; it is a **blocker** for D-3, not a sibling | `hardware/lidar/ld19_driver.py::read_scan`, `hardware/lidar/feature_extractor.py::extract`, `sensing/manager.py::_safe_lidar_read`; executed in §8.9 |
| **D-22** | **High** | **Sim and rover pack different physical quantities into the same `motor_state` slot — a train/serve mismatch.** `training/rover_obs_adapter.py` packs `[vx_body_mps, 0.0, omega_rads, battery_v]` for Isaac-Lab pretraining: body-frame linear velocity and an angular *rate*. `sensing/manager.py::_safe_motor_read` packs `[left_velocity_mps, right_velocity_mps, heading_for_motor(), battery_v]`: per-wheel speeds and an absolute *angle*. So an RSSM pretrained in sim learns slot 2 as a bounded angular rate and is then fed an unbounded heading angle on the rover, and slots 0/1 change frame entirely. Both docstrings are accurate for their own module, which is why a documentation sweep alone would have closed D-11 and left this standing. Found while correcting D-11; **not fixed here** — reconciling the layouts is a modelling decision, not a wording one | `training/rover_obs_adapter.py`, `sensing/manager.py::_safe_motor_read` |
| **D-23** | **Medium** | **Three different "human radius" values are live at once, and the most conservative is the one that is off by default.** `safety.min_forward_clearance_m` = 0.20 m is what `monitor.py::evaluate`'s human branch actually compares against; `three_laws.human_safety_radius_m` = 0.50 m is what `MouseDroidNavigationAgent.act` uses; `SafetyProjectorConfig.human_keepout_m` = 1.00 m is what the projector uses — and the projector is `enabled: False` by default (D-4). The day a detector is wired, one human produces three different stop distances, with the 1.00 m one inert. Using an *obstacle*-clearance threshold as a *human* threshold in `evaluate` is also wrong on its face. Latent until D-1's seam is fed, which is exactly why it is recorded now | `safety/monitor.py::evaluate`, `agents/navigation.py::act`, `safety/projector.py::project`, `config/schema/reward_safety.py` |
| **D-21** | **High** | ~~**The blast-radius bound the injection filter names does not exist.**~~ **CORRECTED IN THIS CHANGE (§9).** `security/injection_filter.py` is candid that it is *"a literal-pattern denylist, not a semantic classifier … best-effort against a motivated adversary"*, and located the real protection elsewhere: *"parsed mission output is still clamped by `LLMConfig.max_vx_norm_mps`/`max_vy_norm_mps`/`max_omega_norm_rads`"*. Those three fields are declared in **two** schemas, validated `gt=0`, copied into `GatewayConfig` by `factory.build_llm_gateway`, and asserted in a unit test — and an exhaustive repo-wide grep finds **no read anywhere**. Executed: with `max_vx_norm_mps=0.01`, `_parse_response` returns `vx_target=1.0`, 100× the documented bound. Two aggravations: it is the S-11 class again (an operator reading `config/*.yaml` sees `max_vx_norm_mps: 0.5` and concludes LLM output is limited to 0.5 m/s), and the fields are **dimensionally incoherent** with what they claimed to bound — declared in m/s and rad/s against a `GoalVector` documented as normalised `[-1, 1]`, so wiring them naively would silently halve the achievable command. The false claim is corrected here; whether to delete the fields or give them coherent semantics is a maintainer decision, pinned so it cannot drift back | `security/injection_filter.py`, `config/schema/llm.py`, `llm_gateway/config.py`, `factory/llm_gateway.py`; executed in §8.6 |
| ~~D-20~~ | — | **WITHDRAWN.** A draft of this review recorded `hardware/motor_controller.py`'s two `max(min(val, max_val), -max_val)` sites as carrying the same defect. They do not: both are preceded by an explicit `if math.isnan(val) or math.isinf(val): return 0.0`. The sweep that found them matched the clamp line and missed the guard above it. Recorded rather than deleted because the correction is the point — `motor_controller.py` and `lidar_driver.py::_sanitize_scan` both got this right, which makes the ESP32 path's omission an **internal inconsistency** rather than an unforeseeable gap, and strengthens D-17..D-19 | `hardware/motor_controller.py`, `hardware/lidar_driver.py` |
| **D-17** | **Critical** | ~~**The terminal bound before the motors fails open on `NaN`.**~~ **FIXED IN THIS CHANGE (§9).** `comms/_utils.py::clamp` is `max(lo, min(hi, value))` and is the single shared guard for **both** codecs. Executed: `build_velocity_cmd(nan, nan, nan)` → `{'T': 1, 'vx': 255, 'vy': 255, 'omega': 255}` — full-scale PWM on every axis, no exception, because the clamp resolved the NaN to `1.0` before `int()` saw it. `WaveshareStockCodec.build_velocity` emits `max_velocity_mps` / `max_omega_rads` the same way, and that is the codec `NEXT_STEPS.md` item 3 is about to switch to. The repo's own on-device learning with hot-swappable weight slots is a plausible NaN source | `comms/_utils.py::clamp`, `::build_velocity_cmd`, `comms/command_set.py::WaveshareStockCodec.build_velocity`; executed in §8.4 |
| **D-18** | **High** | ~~**`_execute_action` had no bound at all on the live 30 Hz tick.**~~ **FIXED IN THIS CHANGE (§9).** `vx = float(action[0]) * max_v` with no range or finiteness check; the docstring says the action is *"assumed already restricted"*, and nothing on the path enforced it. `MouseDroidNavigationAgent.act` clamps with `torch.max(torch.min(...))`, but the cognitive and VLA branches are separate action sources with their own guarantees, and an out-of-range action scaled straight through | `orchestrator/_action_mixin.py::_execute_action` |
| **D-19** | **High** | ~~**The only LLM-reachable `send_velocity` in the tree failed open on `NaN`.**~~ **FIXED IN THIS CHANGE (§9).** `motor_tools.py::_clamp` carried the same idiom and guards the MCP `set_velocity` tool, whose arguments come from a model. Executed: `_clamp(nan, lower=-0.5, upper=0.5)` → `0.5`; `_clamp(nan, lower=-1.5, upper=1.5)` → `1.5`. Gated by a default-OFF toggle (`mcp.enabled: false` in both shipped configs, `Settings.mcp = None`), which is why this is High rather than Critical — it is one opt-in flip from live | `common/tools/motor_tools.py::_clamp`, `::_set_velocity` |
| **D-0** | **Medium** (latent High) | ~~**The LLM gateways' clamp turned `NaN` into the upper bound.**~~ **FIXED IN THIS CHANGE (§9).** Same idiom, same mechanism: `json.loads` accepts the bare `NaN` literal, the decoded payload is a well-formed dict so the `isinstance` guard passes, `float()` does not raise, and the clamp returns `1.0`. **Correction — an earlier draft of this row rated this Critical and described it as "a full-scale motion command". That was wrong.** Tracing every `vx_target` consumer shows the LLM-derived `GoalVector` is consumed *only* by structlog fields, the REST `POST /api/v1/mission` response body, and an MCP tool result: **no production path actuates on it.** It is applied as a velocity only in `orchestrator/autonomous.py`, which is parked with zero production callers (ADR-016) — and even there via the *other*, pydantic-validated `GoalVector`. The live blast radius is therefore an observability and API surface, not motion: a `NaN` comes back to a language model as a confident `1.0` (same family as D-14). It is latent High because it sits directly on the seam the roadmap plans to connect to actuation — goal-conditioned planning is exactly what §1 item 4 recommends building | `llm_gateway/{gateway,anthropic_gateway,openai_compatible}.py`; executed in §8.2, trace in §8.4 |
| **D-1** | **Critical** | ~~**Human detection is structurally impossible; five safety mechanisms and three config budgets are dead code.**~~ **PLUMBING FIXED (§9)** — the gap is now typed and announced; no detector ships, and that is now said out loud rather than hidden. No `human_detected` / `human_dist_m` field exists on `ObservationProtocol` or `MouseDroidObservationBundle`; `evaluate` reads them through `getattr(..., False)` and nothing ever assigns them. A typed protocol field would have failed `mypy --strict`; the `getattr` default is what hides it | `sensing/protocol.py`, `sensing/bundle.py`, `safety/monitor.py::evaluate`, `agents/navigation.py::act`, `safety/projector.py::project` |
| **D-2** | **High** | **`three_laws.py` is never called on the motion path — and wiring it would not help.** `factory/cognitive.py::build_cognitive_core` constructs `ConstitutionalChecker` with **no `law_checker` argument**, so `check` takes the `else: safe = action.copy()` branch in every production build; the only wiring site is offline training. A second, independent gate sits behind it: `CognitiveCore.tick_fast` filters context to the hard whitelist `("battery_v", "obstacle_dist_m", "mcts_sims")`, dropping every key Laws 1–3 need. `ThreeLawsConfig.enabled` defaults `True`, so config reads "enforcement on" while nothing enforces — the S-11 defect class again | `factory/cognitive.py`, `cognitive/cognitive_core.py::tick_fast`, `cognitive/constitutional_rl.py` |
| **D-3** | **High** | ~~**The S-2 LiDAR fail-closed fix shipped inert on exactly the rig that needs it.**~~ **FIXED IN THIS CHANGE (§9)**, in `config/jetson_lidar_only.yaml` and nowhere else — and the scoping is the finding. An earlier draft of the fix set the policy on `jetson_production.yaml` too. That would have **braked or emergency-stopped the production rover permanently**: `jetson_lidar_only.yaml` is an overlay *stacked on* production, production inherits `model.lidar_dim: 0` from `default.yaml` (D-13), and `_evaluate_lidar_clearance` takes its "available" branch only when `len(lidar_features) > 0`, so an armed policy there fires on every tick forever. **D-13 is a blocker for D-3 on production, not a sibling defect** — and raising `lidar_dim` is an RSSM input-dimension change that invalidates every checkpoint trained without the modality, i.e. a retraining decision. Original finding: `SafetyConfig.lidar_unavailable_policy` defaults `"ignore"` and **no YAML sets it** — including `config/jetson_lidar_only.yaml`, which also sets `ultrasonic: null`. On that stack `_safe_distance_read` returns `distance_fallback_m` = **999.0**, so `forward_clearance_ok` is *always* True, and `_evaluate_lidar_clearance` returns `(inf, True, False)` for a dead LiDAR. `c6f701d` built the mechanism correctly and left every shipped deployment on the fail-open branch | `config/schema/reward_safety.py`, `config/jetson_lidar_only.yaml`, `sensing/manager.py`, `safety/monitor.py` |
| **D-4** | **High** | ~~**`GeometricSafetyProjector` never runs in any shipped configuration.**~~ **FIXED IN THIS CHANGE (§9)** on the lidar-only stack only, for the same `lidar_dim` reason as D-3. Flagged as a **different risk class from the rest of this change**: it alters commanded actions in *normal* operation, not only on a failure path, and its thresholds have never been validated against this rover (F-008 blocks bench validation). Original finding: `SafetyProjectorConfig.enabled` defaults `False` and no `config/*.yaml` sets it. The forward-brake, human-keepout and tight-quarters clamps are the **only graded (non-binary) safety response in the tree**. With D-3, the production LiDAR-only rig's entire LiDAR interlock reduces to one binary test at 0.20 m, and only while the LiDAR is alive | `config/schema/reward_safety.py`, `safety/projector.py` |
| **D-5** | **High** | ~~**The production units automatically restart a motion system,**~~ **FIXED (§9)** — `safety.emergency_latch` now holds an emergency stop across ticks and across restarts, cleared only by `python -m mousedroid.cli.rearm`. Ships default-OFF; the ratchet to on is a separate change. Original finding: the production units automatically restart a motion system, which is the single most quotable ISO 3691-4 prohibition — in the standard the review itself recommends adopting.** `scripts/mousedroid.service` sets `Restart=on-failure`, `RestartSec=5`, `WatchdogSec=30`; `scripts/mousedroid-docker.service` sets `Restart=on-failure`, `RestartSec=10`. Combined with S-5 (e-stop has no latch and no persisted state — a restart is a fresh process), a fault can be cleared by a restart with no human in the loop. `WatchdogSec=30` is also **900 ticks** at 30 Hz. And because the watchdog is notified on any non-raising tick, an orchestrator that is e-stopped forever reports healthy forever | `scripts/mousedroid.service`, `scripts/mousedroid-docker.service`, `orchestrator/_lifecycle_mixin.py::run` |
| **D-6** | **High** | **The primary action path is never shown the safety context -- and the trace is worse than this row first recorded.** Confirmed by exhaustive classification: `safety_ctx` is referenced exactly once in `_select_action`'s body (`_action_mixin.py:66`), and that branch is the **exception-fallback** path, reached only when `_try_cognitive_action` returns `None`. `_try_vla_action` is stronger still -- it does `del observation` on the first line of its body (`:130`) and sees only latent state. `cognitive_core.py:104` narrows context to the literal whitelist `("battery_v", "obstacle_dist_m", "mcts_sims")`, so `human_detected` / `human_dist_m` -- which `ConstitutionalChecker.check` documents as accepted keys -- are stripped before arrival. And `lidar_clearance_ok` is read at **exactly one site in all of `src/`**: `telemetry/frame_builder.py:191`, pure telemetry. Zero reads of either LiDAR field are policy inputs. Original finding: `_ActionMixin._select_action` receives `safety_ctx` and forwards it **only** to `self._agents[0].act(...)`. `_try_cognitive_action` and `_try_vla_action` never see it. `config/jetson_production.yaml` sets `cognitive.enabled: true`, making the cognitive branch **primary** — and its only safety input is `obstacle_dist_m = observation.distance_m`, which on the lidar-only stack is the 999.0 fallback (D-3). LiDAR-derived clearance reaches the *policy* through nothing at all | `orchestrator/_action_mixin.py` |
| **D-7** | **High** | ~~**The TensorRT engine cache has no version identity and survives image rebuilds.**~~ **CLOSED ACROSS TWO CHANGES — neither of which annotated this row, which is why it still read as live.** The version-identity half was fixed by #234: `_runtime_identity()` now folds `torch`, `cuda`, `tensorrt`, `torch2trt` and the GPU compute capability into the key, each resolved through `_safe_version`, and `cache_dir_is_private` makes the unpickle fail closed. The persistence half changed shape rather than going away: F-051 replaced the host bind-mount with the named volume `mousedroid_tensorrt_cache`, so the cache still outlives the image and `--force-recreate` — which is the point — but is no longer a bind-mount of a host path under `/opt/mousedroid`. **As originally written:** `efficiency/tensorrt.py::_model_fingerprint` hashed only class name, architecture string, input shapes, precision and parameter count — **not** TensorRT version, CUDA version, GPU compute capability or driver. `compile_model` treated a stale engine as a **cache hit**; `tensorrt_cache_dir` defaulted under `/opt/mousedroid`, which `docker-compose.jetson.yml` then **bind-mounted from the host**, so the cache outlived the image; and `load_compiled`'s `_load_sync` falls through a **bare `except Exception:`** to `torch.load(..., weights_only=False)`, swallowing the exact deserialization error a version mismatch would raise. Fires on a GPU swap or base-image bump, not only a JetPack upgrade. Docs claim TRT 10.4; JetPack 7.2.1 ships 10.16.2 | `efficiency/tensorrt.py`, `docker-compose.jetson.yml`, `config/schema/hardware.py::JetsonConfig` |
| **D-8** | **Medium** | ~~**Valid-JSON-but-not-an-object crashed the default LLM gateway.**~~ **FIXED IN THIS CHANGE (§9).** `LLMGateway._parse_response` caught `(JSONDecodeError, KeyError, TypeError)` — but `dict.get` cannot raise `KeyError`, and a non-dict always raises `AttributeError`, which was **not** named. So `[1,2,3]`, `null`, `7` and `"go forward"` raised while plain garbage (`"not json at all"`) was handled safely. Its two sibling implementations both guard this and `OpenAICompatibleLLMGateway` documents a "never raises" invariant — the **default** backend was the one without the guard, and `llama_cpp` is also the documented off-network fallback, so the degraded path is where it would bite | `llm_gateway/gateway.py`; executed in §8.2 |
| **D-9** | **Medium** | ~~**The loop-latency histogram is survivorship-biased, so any published p99 would be misleading.**~~ **ADDRESSED (§9)** — the bias is intentional per telemetry invariant 5 and stays; what was missing was the denominator, now recorded, and the caveat, now documented. `_finish_tick_timing` latches the duration on every path but **returns before recording when `ok is False`**. A tick that raises — including one cancelled by `asyncio.wait_for(self.tick(), tick_timeout_s)` — contributes no histogram sample **and** no `tick_overruns` increment. `histogram_quantile(0.99, ...)` is therefore a p99 *of successful ticks*, structurally blind to the 1.0 s timeout class in P5 | `orchestrator/_telemetry_experience_mixin.py::_finish_tick_timing` |
| **D-10** | **Medium** | **`ESP32Config.chassis_has_wheel_encoders` defaults `True` on an encoder-less chassis, and nothing consumes it.** The review states it is `false`; it is not. Its own `description` records that the WAVE ROVER ships encoder-less (vendor audit R3), yet the default asserts the opposite on every shipped overlay, and the only branch on it is a hardware-tier test skipped in CI. A capability declaration that is both wrong and inert | `config/schema/hardware.py::ESP32Config` |
| **D-11** | **Medium** | **`motor_state` layout is documented wrong in three places.** `ModelConfig.motor_state_dim`, `sensing/bundle.py` and `training/drift_metrics.py` all say `[vx, vy, omega, battery]`. What `SensorManager._safe_motor_read` actually packs is `[left_velocity_mps, right_velocity_mps, heading_for_motor(), battery_v]` — slots 0/1 are **per-wheel speeds** and slot 2 is an **angle**, not an angular rate. Any analysis reading the documented layout (the external review's included) is reasoning about a vector that does not exist | `config/schema/`, `sensing/manager.py::_safe_motor_read` |
| **D-12** | **Medium** | **`power_chain.assert_power_chain`'s `estop_latency_ms` does not measure stopping.** It times `await driver.emergency_stop()`, which returns once the stop frame is **written to serial** — not when motion ceases. F-008's own acceptance criteria cite `power_chain_probe_complete` events, so the repo's one critical-priority hardware gate is verified by a number whose name implies a safety property it does not measure. This matters precisely because the review asks for "measured stopping distance per speed tier": the repo has a metric that looks like it already answers that | `diagnostics/power_chain.py`, `comms/base_driver.py::emergency_stop`, `features.yaml` F-008 |
| **D-13** | **Medium** | **LiDAR is disabled in the world model on the production overlay.** `config/default.yaml` sets `model.lidar_dim: 0`; overlays deep-merge onto it and `config/jetson_production.yaml` has **no `model:` block**, so it inherits `lidar_dim: 0` while its `lidar:` block enables the hardware. The 360° sensor feeds safety interlocks but never the RSSM | `config/default.yaml`, `config/jetson_production.yaml`, `config/loader.py` |
| **D-14** | **Low** | **`motor_tools.py::_read_encoders` is an LLM hallucination surface.** Registered with the description *"Read latest wheel encoder reading and odometry pose"* and returns `odometry_x_m` / `odometry_y_m` that are **structurally 0.0** (`WaveshareStockCodec.parse_encoders` never populates them) — handed to a language model as if measured. `query_world_model_pose` is declared on two skill SPECs and is likewise unimplementable | `common/tools/motor_tools.py`, `skills/builtin/{navigate,world_model}.py` |
| **D-15** | **Low** | **`docs/planning/NEXT_STEPS.md` "Known Limitations" states *"Odometry drift accumulates over long runs; reset via landmarks."*** Both halves are false — there is no odometry and there are no landmarks. A doc-honesty defect that would mislead the next reviewer exactly as it did this one | `docs/planning/NEXT_STEPS.md` |
| **D-16** | **High** (raised from Low) | ~~**`cfg.loop.max_miss_pct` is phantom config, and the CI test named for deadline-miss does not measure one.**~~ **FIXED (§9). Severity raised on review:** `LoopConfig` is a `StrictBaseModel` (`extra="forbid"`), so an operator who believed the documentation and set `max_miss_pct` in YAML got a `ValidationError` at settings load — **the rover failed to boot.** Not "the fallback quietly wins". `LoopConfig` has no `max_miss_pct` field, so `tests/hardware/test_e2e_sense_plan_act.py`'s `getattr` fallback always wins. Separately `tests/integration/test_e2e_5sec_run.py::test_deadline_miss_rate_below_threshold` — which **does** run in CI — computes a p90 over 10 mock ticks against a ~3333 ms budget and computes **no miss rate at all**. Same class as S-10 | `tests/hardware/`, `tests/integration/` |

---

## 6. Corrected-design map

| Synthesis says | Correction |
|---|---|
| "Reframe as a safety-bounded edge-autonomy reference platform" | Already CHARTER §1. The actionable residue is three GitHub-surface edits the in-tree gate cannot see: repo **description**, `default_branch` → `main`, and `topics` |
| "If you do only one thing this month, do the ESP32 watchdog and independent E-stop" | Blocked on F-008 bench repair. **The one part that is free today** is C4(d): issue a zero-velocity stop immediately after `connect()` in `_LifecycleMixin.start`. No firmware, no working ESP32, closes the documented "rover can be moving throughout bring-up" window |
| "Flip on the firmware heartbeat" (implied by C4(a)) | Still just the documented env flip `MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock` — already `NEXT_STEPS.md` item 3, still taken up by no `config/*.yaml`. Unchanged from the prior review |
| "Close state estimation first (LiDAR odometry + optical flow)" | **Reverse the order.** Train `reward_head` (or replace the MCTS objective), then give `MouseDroidNavigationAgent.act` a goal argument and make `GoalVector` expressible as a pose — *then* localization has something to be measured against. AprilTags remain the honest v0.1 answer after that |
| "Name your Simplex architecture" | Do not. Build the missing half first: a safety verdict that *selects a controller*, and a decision module that is enabled in a shipped config (D-4) |
| "Publish p50/p90/p99" | Use the repo's own p50/**p95**/p99 convention, and fix D-9 first — the histogram excludes failed and timed-out ticks, so today's p99 is a p99 of *successful* ticks. Close P5 (LiDAR block, `planning_hz`) before publishing, not after |
| "Reuse `G-LATENCY` / `G-ENGINE-COMPAT`" | These do not exist here. Use a per-feature `scripts/validations/F-0NN.sh` + `features.yaml` acceptance entry |
| "Dual-track JetPack 7.2 now; RT kernel serves the 30 Hz story" | Drop the RT-kernel argument — nothing in the design depends on kernel preemption. Keep the *testable* residue: promote the CUDA/TensorRT/L4T triple to a schema contract (the `esp32.command_set` precedent, which has a **raising** validator) and fix D-7 (now done — see that row), both of which were worth doing whether or not you ever upgrade |
| "Constrain the LLM to a typed schema for injection/motor-authority reasons" | Right conclusion, wrong reason — and the first step was repairing the clamp the CHARTER already relies on (D-0, fixed in §9) |
| "Ship a LeRobotDataset exporter" | Not a weekend adapter. No raw frames are persisted anywhere, so an exporter written today emits action/state parquet with empty video. This is a `SCHEMA_VERSION = 2` capture change — a feature with a config field and a regression pair, per invariant 6 |
| "Kimi K3's Core/Research split with maturity labels" | Already exists in prose in CHARTER §1/§3 (wired / default-OFF / not-wired / parked). What is missing is the *machine-readable* form, which is the genuinely new half of that idea |
| "Rename before publishing" | Aim it at `mousedroid` + "Star Wars MSE-6", not "AGI" — the AGI token never reached HF or PyPI, and `mouse-droid` on PyPI is already taken |

---

## 7. What survives review unchanged

Correct, well-argued, and worth carrying into any rev. B.

Two things the **repo** already got right, recorded because this review spent most of its length
on what is wrong:

- **`hardware/lidar_driver.py::_sanitize_scan` and `hardware/motor_controller.py` both guard
  non-finite input before clamping** — an explicit `isnan`/`isinf` check, exactly the pattern the
  ESP32 path lacked. They are the in-repo precedent, and they are now pinned by the §9 gate so the
  precedent cannot erode.
- **No credential material is in the repository.** `NEXT_STEPS.md` item 1 treats the
  `ANTHROPIC_API_KEY` as compromised; no `sk-ant-*` string appears in the working tree or anywhere
  in git history (`git log --all -S`). The rotation is an operator action on the rover, not a repo
  leak — recorded so the next reader does not re-investigate.

And from the external review:

- **An explicit ODD with declared exclusions**, and ISO 3691-4 / UL 4600 *vocabulary* with an
  explicitly compliance-**shaped**, not compliance-**certified** posture. Genuine greenfield
  (C6), and the strongest unique contribution in the synthesis.
- **A GSN-structured assurance case** linking safety goals to CI evidence. Cheap, and it is the
  artifact that most distinguishes this from a hobby rover repo.
- **The Simplex *insight*** — you verify the switch, not the learned policy — even though the
  claim that this repo already implements it is refuted (C1).
- **A machine-readable release manifest** (image digest, toolchain versions, config hash, model
  hashes, latency profile, ODD). This is also the right home for the D-7 version triple.
- **Falsifiable percentiles + a deadline-miss ratio instead of an asserted "30 Hz"** (P1, P3).
- **Fault-injection matrix** (LiDAR dropout, camera occlusion, Jetson overload, network loss,
  ESP32 link loss, low battery, restart recovery) — none of which exists today; note that
  "restart recovery" collides directly with D-5.
- **Battery-under-load verification, thermal soak, and measured stopping distance per speed
  tier** as the only physical work worth doing now — but see D-12 before reusing
  `estop_latency_ms`.
- **Defer** Yocto, FCC/CE pre-scan, OTA fleet management, on-device online learning in the
  control path, enclosure/BMS/pogo-charging manufacturing, and any hardware sale.
- **Do not wire `meta/` or `scaling/`** during this campaign — matches the tree, and matches the
  charter.
- **The MSE-6 shell as a thermal trap** is a real physical concern and cheap to address; the
  injection-moldable CAD is not.

---

## 8. Executed evidence

Run on `2469588` with `python3 3.11.15`, `pydantic 2.13.5`, `pytest 9.1.1`, the real package on
`sys.path`. Reproductions live in the session scratchpad; the assertions below are re-derivable
from the two regression files added in §9.

### 8.1 MCTS maximises a never-trained head

`RSSM.reward_head` is defined as `nn.Linear(cfg.hidden_dim + cfg.latent_dim, 1)` and referenced
in exactly one place — `imagine_step`, which returns it as `predicted_reward`.
`RSSM.train_sequence` computes `loss = recon + self._cfg.kl_beta * kl` and returns; there is no
reward term. `MCTSPlanner._rollout` consumes `imagine_step(...)` and backs up that value. The
repo states the consequence itself, in `learning/on_device/rssm_refiner.py`:

> `allow_unused=True` is MANDATORY: the RSSM `reward_head` (+ `prior` / `observation_decoder`)
> are not on the recon/KL graph, so their grads come back `None` and the loop skips them.

Corroborated a second time by `learning/on_device/scoring.py`, which records that an
imagined-return metric was **retired** because it "summed the model's OWN `reward_head` and so
self-gamed on reward-head inflation… an unused head". [Certain]

### 8.2 The clamp manufactured a full-scale command

Against the shipped parsers, before the §9 fix:

```
PRODUCTION clamp helpers, NaN input:
  anthropic_gateway._clamp_unit(nan)  = 1.0
json.loads of a bare NaN literal -> {'vx': nan} | isinstance(doc, dict) = True
  (so the isinstance guard in both hardened gateways PASSES and NaN reaches the clamp)

Validated (BaseModel) GoalVector vs NaN:              REJECTED by pydantic -> ValidationError
Dataclass GoalVector (the type on the LLM motion path): ACCEPTED: GoalVector(vx_target=1.0, ...)
```

Two same-named `GoalVector` types exist: `interfaces/protocols.py::GoalVector` is a pydantic
`BaseModel` with `ge=-1.0, le=1.0` that **rejects NaN**, and
`llm_gateway/protocol.py::GoalVector` is an unvalidated frozen dataclass. The motion path uses
the unvalidated one. [Certain]

And the `except` clause on the default backend, before the fix:

```
JSON list          -> *** UNCAUGHT AttributeError: 'list' object has no attribute 'get'
JSON null          -> *** UNCAUGHT AttributeError: 'NoneType' object has no attribute 'get'
JSON bare int      -> *** UNCAUGHT AttributeError: 'int' object has no attribute 'get'
JSON bare string   -> *** UNCAUGHT AttributeError: 'str' object has no attribute 'get'
garbage            -> vx=0.0  vy=0.0  omega=0.0        <- non-JSON was handled safely
```

Note the inversion: input that is *not* JSON was safe; input that *is* valid JSON crashed.
[Certain]

### 8.3 The pins detect the defect

With the new symbols kept but pre-fix *semantics* restored, the two regression files go
**14 failed, 18 passed** — the 18 passes being exactly the finite-clamp cases that must not
change. With the fix in place: **32 passed**. Full gateway suite: **231 passed**.
`ruff check` clean, `ruff format --check` clean, `mypy --strict` clean on all four changed
modules. [Certain]

### 8.4 Where a velocity actually goes

Three `send_velocity` call sites exist in `src/`: `_action_mixin.py::_execute_action` (the 30 Hz
tick, fed by the policy action tensor), `motor_tools.py::_set_velocity` (the MCP tool), and
`power_chain.py` (the smoke helper). **Not one is fed by a `GoalVector`.** Every `vx_target`
consumer in the tree is a structlog field, a REST response body (`_rest_handlers.py`), or an MCP
tool result (`common/tools/registry.py`) — never an actuation call. `orchestrator/autonomous.py`
is the only module that applies a goal as a velocity, it is parked with zero production callers
(ADR-016), and it uses the *other* `GoalVector` (the pydantic `BaseModel`) fed by
`CompositeLLMGateway`, whose `_dispatch_primary_translation` is itself a stub returning
rule-based mock output from `constants.py` literals. [Certain]

The terminal chain, executed:

```
clamp(benign 0.2) -> 0.2    -> int(*MAX_PWM) = 51
clamp(over   9.9) -> 1.0    -> int(*MAX_PWM) = 255
clamp(NaN       ) -> 1.0    -> int(*MAX_PWM) = 255   <== FULL-SCALE PWM ON THE WIRE, no exception
```

and end-to-end through the real codec, pre-fix:

```
build_velocity_cmd(nan, nan, nan) -> {'T': 1, 'vx': 255, 'vy': 255, 'omega': 255}
```

Post-fix the same call returns `{'T': 1, 'vx': 0, 'vy': 0, 'omega': 0}`. Note a reader might
assume `int(nan)` would have raised `ValueError` and failed safe; it could not, because the clamp
resolved the NaN to `1.0` before `int()` ever saw it. That case is pinned. [Certain]

### 8.5 The actuation pins detect the defect

With pre-fix semantics restored and the symbols kept, the two new regression files go
**22 failed, 19 passed** — the 19 being exactly the finite and saturation cases that must not
change. With the fix: **41 passed**. Regression + comms tiers: **1527 passed**; the 10 failures
and 21 collection errors reproduce identically on a clean tree (`torch` / `cv2` / `hypothesis`
absent from this sandbox). `ruff` and `mypy --strict` clean. [Certain]

### 8.6 The bound the filter named is never read

```
configured max_vx_norm_mps   = 0.01
parsed goal from the model   = GoalVector(vx_target=1.0, vy_target=1.0, omega_target=1.0)
vx_target 1.0 EXCEEDS the documented bound 0.01 by 100x
```

Repo-wide, the only attribute accesses of `max_vx_norm_mps` / `max_vy_norm_mps` /
`max_omega_norm_rads` are `factory/llm_gateway.py`'s copy of `LLMConfig` into `GatewayConfig`, and
the `injection_filter.py` docstring naming them. No consumer. [Certain]

### 8.7 The idiom's NaN behaviour is argument-order-dependent

Builtin `min`/`max` keep the **first** operand when the comparison is False, and every comparison
against NaN is False. So the same code shape has opposite failure modes:

```
max(-1.0, min(1.0, nan))  = 1.0   <- fabricates the UPPER BOUND   (comms/_utils.py form)
max(min(nan, 12.0), 0.0)  = nan   <- PROPAGATES NaN               (hardware/ form)
```

This is why the source-level gate in §9 matches both orders and carries a self-test. The gate's
first version matched the literal substrings `min(hi, value)` / `min(upper, value)`, so it was
coupled to the variable names in the two files it was written against and flagged **neither**
synthetic sample. A gate that has never been shown to fail is not evidence. [Certain]

### 8.9 A dead LiDAR reported full clearance, with the policy armed

`LD19LidarDriver.read_scan()` on a driver whose serial port was never opened — an unplugged
cable — against `SafetyConfig(lidar_unavailable_policy="emergency", lidar_unavailable_grace_s=0.0)`:

```
read_scan raised?           False
scan.n_points               0
features (unique values)    [1.]
policy=ignore     lidar_min_dist_m=12.0     clearance_ok=True is_emergency=False
policy=emergency  lidar_min_dist_m=12.0     clearance_ok=True is_emergency=False
```

Identical output under both policies is the finding: the policy is not weak here, it is
**unreachable**. `_evaluate_lidar_clearance` only consults it when `len(lidar_features) == 0`, and
the features were present — all-ones, i.e. `1.0 * lidar_max_range_m` = 12.0 m in every sector.

After the fix, through the real `SensorManager` and a factory-built monitor:

```
_safe_lidar_read -> features=None ok=False
obs.lidar_features is None : True
ctx.lidar_min_dist_m       : 0.0
ctx.lidar_clearance_ok     : False
ctx.is_emergency           : True
```

The counter-case that shaped the fix, and the reason it is a `sensor_responding` flag rather than
a blanket `n_points == 0` check. On the LD19 a no-return beam reports `0` mm, which is below
`min_range_m`, so a room with nothing inside `max_range_m` **legitimately** yields zero points. A
naive fix would have emergency-stopped a rover standing in an open space:

```
open-room scan: n_points=0 sensor_responding=True
obs.lidar_features is None : False
ctx.lidar_min_dist_m       : 12.0
ctx.is_emergency           : False
```

Frames arrived, so the sensor is answering; all-ones is a true reading of that room.

---

### 8.10 The cache key that could not see its own runtime, and the 0700 that was not

The permission claim, tested rather than read:

```
umask-derived mode: 0o755    group/other readable: True
```

`mkdir(parents=True, exist_ok=True)` passes no `mode=`, and `grep -rn "chmod\|0o700\|0o600" src/`
returns nothing. The control the `SECURITY:` comment names does not exist anywhere in the tree.

The cache key, before and after. `_runtime_identity()` on this host:

```
('torch=2.14.0+cu130', 'cuda=13.0', 'tensorrt=unavailable',
 'torch2trt=unavailable', 'compute=unavailable')
```

Each component moves the fingerprint; before the fix none of them did:

```
fingerprint          : 3f5096e526fee2da
torch changed        : 1942f87828394401
tensorrt now present : 03a89de485b10949
```

Absent components are **recorded** as `unavailable` rather than omitted. Omitting them would make
the key of a host that cannot report a component collide with one that can — the same collision
being fixed.

The guard, end to end:

```
after save, cache mode: 0o700  private=True
loosened to           : 0o755  private=False
direct load_compiled  : refused -> refusing to deserialize .../engine_deadbeef.pth ...
compile_model         : returned TopLevelTracedModule (no exception)
self-healed mode      : 0o700  private=True
```

Note the fourth line. `compile_model` **recompiles** rather than raising, and the recompile's save
restores the permissions. A cache is an optimization: declining to trust an entry must mean rebuild
it, never fail. Raising would have dropped every rig to eager PyTorch through
`OptimizedInference._ensure_compiled`'s `except Exception` — a silent performance cliff.

One negative result worth recording, because it shaped the tests. The first attempt to demonstrate
the guard produced `direct load_compiled : LOADED (guard did not fire)` — correctly. Without
torch2trt installed the compiler JIT-traces, so the cached engine **is** TorchScript,
`torch.jit.load` succeeds and the pickle branch is never entered. The pickle path is reachable only
for a genuine torch2trt engine, which is saved with `torch.save`. The guard costs nothing on any
host without TensorRT, and a test that did not write a non-TorchScript file would have been
vacuous.

---

### 8.8 Baseline

`python scripts/select_next.py` → `F-008  USB-C rover smoke passes on the physical Jetson
(priority=critical, tier=hardware)`. Unchanged by this review.

---

## 9. Changes made alongside this review

Five software-only defects fixed and one false claim corrected, all strictly fail-safer, none
needing a CHARTER carve-out (a carve-out gates *expansion* of the no-motion posture; every one of
these reduces commanded motion or reduces what the docs assert). One shared rule now holds
everywhere a velocity is bounded: **a non-finite velocity is a malformed velocity, and a malformed
velocity means no motion.**

**D-21 — the filter no longer claims a bound that does not exist.**
`security/injection_filter.py`'s blast-radius paragraph now states what actually bounds a parsed
goal (`clamp_unit`'s `[-1, 1]`, then `ESP32Config.max_velocity_mps` and `comms._utils.clamp`
downstream), records the three `*_norm_*` fields as unconsumed with their unit mismatch, and notes
that no production path actuates on an LLM-derived `GoalVector` at all. The fields are **not**
wired up here: they are declared in m/s against a normalised `[-1, 1]` quantity, so wiring them
naively would change actuation scale, and that is a maintainer decision rather than a bug fix. Two
pins in `test_goal_vector_clamp_aqa.py` hold the line — one asserting the false claim does not
return, one that fails loudly *if* someone consumes the fields, so the docstring and the unit
question get revisited together.

**D-17 — the terminal bound before the motors.** `comms/_utils.py::clamp` returns `0.0` for
non-finite input and logs `velocity_clamp_non_finite` at error. Every finite result is unchanged,
the signature is unchanged, and all five call sites were checked first — they are the two
`build_velocity` paths and nothing else, so no caller depended on the NaN-to-bound behaviour.
This is the highest-stakes edit in the change: it is what emitted PWM 255.

**D-18 — the live 30 Hz tick.** `_action_mixin.py::_execute_action` now zeroes **every** axis when
any of `vx`/`vy`/`omega` is non-finite, and logs `action_non_finite_zeroed` at error. All three
axes rather than only the offending one: a NaN in any component means the policy output is
untrustworthy, and driving the remaining axes on that basis is not safer. Error level because a
non-finite action is a model-health signal, not a value to drop silently.

**D-19 — the MCP tool.** `motor_tools.py::_clamp` gets the same guard, logging
`motor_tool_non_finite_velocity`. This is the only LLM-reachable `send_velocity` in the tree.

**D-0 — NaN no longer becomes full scale.** `llm_gateway/protocol.py` gains `GOAL_VECTOR_MIN`,
`GOAL_VECTOR_MAX` and a NaN-safe `clamp_unit()`, defined once beside the type it constrains. All
three `LLMGatewayProtocol` implementations now import it instead of carrying their own
`max(MIN, min(MAX, value))` — the defect shipped in three places precisely because the expression
was duplicated three times. Non-finite input resolves to `0.0`, matching how these parsers
already treat every other malformed field. **Deliberate behaviour change:** `±Infinity`
previously clamped to `±1.0` and is now `0.0`; pinned explicitly so it is not incidental.

**D-8 — the default backend now upholds the "never raises" invariant its siblings document.**
`LLMGateway._parse_response` gains the `isinstance(doc, dict)` guard that
`AnthropicLLMGateway._parse_goal_vector` and `OpenAICompatibleLLMGateway._parse_goal_vector`
already had, and its `except` now names `(TypeError, ValueError)` instead of the unreachable
`KeyError`.

Pinned by two regression pairs per the house convention:

- `tests/regression/test_velocity_clamp_actuation_{aqa,backwards_compat}.py` — the actuation
  contract, exercising the real codec entry points end to end rather than only the helper, both
  codecs, the MCP tool and the tick guard. Includes a source-level gate over every module that
  bounds an actuation quantity, matching **both** operand orderings (§8.7) and carrying a
  self-test that asserts it flags a known-bad sample in each — the gate's first version matched
  literal substrings tied to two files' variable names and would have flagged neither. The
  already-correct `hardware/motor_controller.py` and `hardware/lidar_driver.py` are in the gate's
  list precisely because they are the precedent worth keeping. Also an explicit pin that the
  legacy path never raised (a reader might assume `int(nan)` would have failed safe; it could
  not, because the clamp resolved the NaN before `int()` saw it).
- `tests/regression/test_goal_vector_clamp_aqa.py` — the contract (non-finite → no motion; NaN is
  not the upper bound; finite behaviour unchanged; the bare two-sided idiom may not reappear in
  any gateway module). That last gate found a **fourth** inline clamp during authoring —
  `mission_parser.py::_extract_rotation_magnitude` — which on inspection is a *one-sided*
  normalisation of a regex-extracted `(\d+)` degree count, so it can be neither signed nor
  non-finite. The gate was narrowed to the two-sided idiom and the exclusion documented rather
  than the finding dropped.
- `tests/regression/test_goal_vector_clamp_backwards_compat.py` — what must not have changed
  (every finite value, the four pre-existing parser behaviours, non-numeric fields), plus the
  five valid-JSON-non-object inputs that used to raise.

Both pairs were proved against pre-fix semantics rather than assumed: the gateway pair goes
14 red / 18 green, the actuation pair 22 red / 19 green, the greens in each being exactly the
finite cases that must not change.

**A note on how this change reached its final shape, because it bears on how the review should be
read.** It took three passes, and each one corrected the previous. The first found the idiom in
the LLM gateways, rated it Critical, and fixed it there. The second traced every `send_velocity`
call site and found the gateways are the one place it does *not* actuate, while three paths that
do carried it untouched — D-0's row records that correction in place. The third found that the
gate shipped by the second was coupled to two files' variable names and would have caught neither
form of the idiom, that `hardware/motor_controller.py` had been wrongly accused (D-20, withdrawn),
and that the *other* control CHARTER §3 leans on was a phantom (D-21).

The generalisable lesson is the one this review levels at the external synthesis in §1 item 2: a
finding that has not been traced to its consumers is a hypothesis, not a result — and that applies
to the reviewer's findings first.

---

### 9.1 Second batch — arming the LiDAR interlock, and the defect that blocked it

**D-24 — a dead LiDAR no longer reports an all-clear ring.** `LidarScan` gains
`sensor_responding: bool = True`; `LD19LidarDriver` sets it `False` on its two "no frames at all"
paths (unopened serial port, no valid frame) and leaves it `True` on the "frames arrived, every
point out of range" path; `SensorManager._safe_lidar_read` reports the modality absent when it is
`False`. The field is a **bool and not a tunable threshold** on purpose: a `min_scan_points` knob
would only be a way to configure the fail-closed path back open, the same reasoning as
`EmergencyLatchConfig`'s deliberately absent fail-open knob.

Two observables change on existing `ignore` deployments and are pinned rather than left to be
discovered: a dead LiDAR's `lidar_min_dist_m` goes from a fabricated `lidar_max_range_m` to `inf`,
and its `valid_mask` slot from 1 to 0. Neither changes a safety decision under `ignore`. Both are
what the `read_all`, `_safe_lidar_read` and `frame_builder` docstrings already *said* happened.

**D-3 / D-4 — the interlock and the projector armed, on the one stack with a LiDAR.**
`config/jetson_lidar_only.yaml` gains `lidar_unavailable_policy: emergency`,
`lidar_unavailable_grace_s: 0.5` and `projector.enabled: true`. It is the only shipped file whose
behaviour changes. `jetson_production.yaml` gains a **comment and no keys**, recording why the
apparent gap must stay open — closing it there would e-stop that rig on every tick.

`emergency` rather than `degrade` because that stack sets `ultrasonic: null`, so LiDAR is the only
obstacle sensor and braking-without-halting would be driving blind at reduced speed. The grace is
sized against relations, not taste, and the AQA pins the relations: blind travel during the grace
(`grace × max_velocity_mps` = 0.25 m) must stay within `projector.lidar_brake_distance_m` (0.30 m),
and the grace must stay *below* `lidar.scan_acquisition_timeout_s` — sizing it to ride out a whole
failed acquisition would need 0.5 m of blind travel, 2.5× `min_forward_clearance_m`. After D-24
that is also unnecessary: a *slow* scan comes back partial-but-real and never reaches the policy.
The honest residue is stated in the YAML: 0.25 m of blind travel still exceeds the 0.20 m
forward-clearance threshold, and none of it is bench-validated while F-008 is open.

**D-4 is a different risk class from everything above it** and is called out as such rather than
filed under "strictly fail-safer": the projector changes commanded actions in *normal* operation.
`human_keepout_m` stays inert regardless, because nothing feeds D-1's `HumanPresenceProtocol` seam.

Pinned across three tiers, each proved red-then-green against pre-fix semantics:

- `tests/unit/sensing/test_lidar_not_responding.py` — 9 tests, 3 red pre-fix. The driver's three
  empty paths told apart, the sensing boundary, and the open-room counter-case.
- `tests/regression/test_lidar_not_responding_backwards_compat.py` — 21 tests. Invariant 6 on the
  new field (defaulted, last in the signature, still frozen), the mock path unaffected, every
  `ignore` overlay's decision unchanged, and the two changed observables recorded.
- `tests/regression/test_lidar_interlock_overlay_aqa.py` — 37 tests, 3 red pre-fix. The stacked
  load, the derived budgets, the three-way `emergency` + latch + zero-grace footgun, and
  `test_production_alone_stays_inert` — the pin that would have caught the draft described in
  D-3's row, which makes it the most valuable test in the batch.
- `tests/integration/test_lidar_interlock_overlay_integration.py` — 8 tests, 6 red pre-fix.
  Factory-built monitor and projector on the real stacked settings, the grace exercised against
  `observation.timestamp` rather than wall time, and a real `LD19LidarDriver` with no serial port
  driving the whole path.

**How this batch reached its shape.** The plan for it was rewritten once before any code was
written, because the first version set the policy on `jetson_production.yaml`. Then D-24 was found
while writing the tests for the corrected version — which means the corrected plan was *still*
wrong, in a quieter way: it would have shipped an interlock that resolves correctly in config and
cannot fire on the two most likely hardware failures. Both corrections come from the same habit and
neither came from re-reading the plan: trace the value to its consumers, then execute it.

---

### 9.2 Third batch — the TensorRT engine cache, and a misrating repeated

**D-7 — the cache key now names the runtime that built the engine.**
`_model_fingerprint` gains `_runtime_identity()`: torch version, `torch.version.cuda`, TensorRT
version, torch2trt version, GPU compute capability. Each resolves through one guarded accessor that
returns an explicit `"unavailable"` sentinel — **recorded, never omitted**, because an omitted
component collides a host that cannot report it with one that can. That sentinel path is the CI and
dev path, not an edge case: neither `tensorrt` nor `torch2trt` is installed on either.

This **invalidates every previously cached engine by design** — old keys never match, so a stale
engine is ignored rather than mis-loaded. The cost is one recompile per model after an upgrade.

**D-25 — the 0700 claim is now true, enforced, and fails to a rebuild.** `_save_compiled` creates
the directory `mode=stat.S_IRWXU` *and* `chmod`s it, because `mkdir`'s mode is umask-masked and
mode alone would have left the same false claim in a new place. `load_compiled` refuses to unpickle
from a group- or world-accessible directory, raising the named `UntrustedEngineCacheError` — a
named exception, not an `assert`, because `PYTHONOPTIMIZE=1` in `Dockerfile.jetson` strips asserts
and would turn the guard into no guard. The `except Exception` around `torch.jit.load` is narrowed
to `RuntimeError`, with the honest note that narrowing alone does **not** separate a
version-mismatched engine from a hostile pickle — both arrive as `RuntimeError`. D-7 prevents the
first, this guard the second.

`compile_model` treats an untrusted cache as a **MISS and recompiles**, and the recompile self-heals
the permissions. This is a correction to the first design: raising would have dropped every existing
rig to eager PyTorch via `OptimizedInference._ensure_compiled`'s `except Exception`, logged once at
warning level — a performance cliff nobody would notice. The permission mode is deliberately **not
configurable**, following `EmergencyLatchConfig`'s absent fail-open knob and
`LidarScan.sensor_responding` being a bool rather than a threshold.

**D-26 — `tensorrt_enabled` now says it is inert.** Four shipped configs set it `true` while
nothing constructs a compiler. Corrected as a description, not a default flip: the flag expresses
the right intent for the day the seam is wired.

Pinned across the unit tier and a regression pair — 81 tests, 11 red against pre-fix semantics:

- `tests/unit/efficiency/test_tensorrt.py` — the identity table (one named row per component, so a
  component dropped from the tuple names itself on failure), the sentinel and raising-probe paths,
  directory mode, miss-and-rebuild, self-heal, direct-load refusal, and a pin that a TorchScript
  engine never reaches the pickle path at all.
- `tests/regression/test_tensorrt_cache_identity_aqa.py` — a structural `ast` gate that every
  `weights_only=False` call sits inside a function that consults `cache_dir_is_private`, **with a
  self-test in both directions** (it must flag a known-bad sample and accept a known-good one),
  because this review's earlier NaN-clamp gate shipped coupled to two files' variable names and
  flagged neither of its own samples.
- `tests/regression/test_tensorrt_cache_identity_backwards_compat.py` — no new config field, every
  shipped overlay loads, the cache-dir default and the digest format are untouched,
  `tensorrt_enabled: false` is still a pure pass-through, and **two pins that nothing in `src/`
  constructs `OptimizedInference` or calls `build_tensorrt_compiler`** — the tests that make D-25's
  latent rating revisitable the day someone wires the seam, instead of leaving it to be rediscovered.

**The misrating, recorded because it is the third instance of the same lesson.** A first draft of
this batch called D-25 a live RCE and "the sharpest thing left". It is not live: nothing constructs
`OptimizedInference`, nothing calls `build_tensorrt_compiler`, and `vulture` already reports both as
unused. That is exactly the D-0 error — rating a mechanism without tracing it to its consumers —
committed *after* this review had written the lesson into its own §9 and into the pull request. The
generalisable form is narrower than "trace to consumers", which I evidently can recite without
applying: **the trace has to happen before the severity is written down, because a severity, once
written, is what the rest of the analysis anchors to.**

A second, smaller correction from the same pass: I also reasoned that raising on a loose cache would
"brick every existing rig". It would not — `optimized_inference.py:108` catches it and falls back to
eager PyTorch. The right objection was subtler (a silent performance cliff), and the fix is the
miss-and-rebuild above.

Also recorded, not fixed: **D-27**, a `PRE_ACTION` hook cannot change the executing action, and the
sharpened **D-6** trace. Both found while tracing this batch.

Not fixed here, recorded for triage: **D-1 through D-7 and D-9 through D-16.** D-1 (human
detection) and D-5 (automatic restart of a motion system) are the two that should be triaged
first; C4(d) — a zero-velocity stop after `connect()` — is the cheapest safety win in the whole
document and needs no working ESP32.
